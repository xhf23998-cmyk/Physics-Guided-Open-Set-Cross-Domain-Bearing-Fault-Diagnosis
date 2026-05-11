"""Step 5/6: 跨工况闭集迁移 + 开集诊断训练脚本 (v2 - 修复类索引映射)"""
import os
import sys
import time
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel, EVTHead
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner, build_batch_template
from utils import set_seed, compute_closed_set_metrics, compute_open_set_metrics


# ── 标签重映射 ──────────────────────────────────────────────────────

_ALL_CLASS_MAP = {"Health": 0, "Inner": 1, "Outer": 2, "Ball": 3}


def remap_labels(labels, known_classes):
    """将原始标签重映射到 0..K-1 范围

    例: known_classes=["Health","Outer","Ball"]
    原始: 0,2,3 → 重映射: 0,1,2
    """
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    return np.array([label_map.get(int(l), -1) for l in labels], dtype=np.int64)


# ── 特征提取 ────────────────────────────────────────────────────────

def extract_all_features(model, loader, M_phy, device, known_classes=None):
    """提取整个数据集的特征，可选地只保留已知类"""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, _ = batch
            x = x.to(device)
            feats, _ = model.extract_features(x, M_phy)
            all_feats.append(feats.cpu())
            all_labels.append(y)

    feats = torch.cat(all_feats, 0)
    labels = torch.cat(all_labels, 0)

    if known_classes is not None:
        known_set = set(_ALL_CLASS_MAP[c] for c in known_classes)
        mask = np.array([int(l) in known_set for l in labels])
        feats = feats[mask]
        labels = labels[mask]
        # 重映射到 0..K-1
        labels_np = remap_labels(labels, known_classes)
        labels = torch.from_numpy(labels_np)

    return feats, labels


# ── 源域预训练 ──────────────────────────────────────────────────────

def train_source(model, source_loader, M_phy_dict, criterion, optimizer, scaler, device, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y, _ in source_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            M_phy_batch = build_batch_template(M_phy_dict, y, device)
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy_batch, return_features=True)
                loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Source Epoch {epoch}/{epochs} | Loss: {total_loss / len(source_loader):.4f}")


# ── 跨域对齐训练 ────────────────────────────────────────────────────

def train_alignment(model, source_loader, target_loader, M_phy_dict, aligner,
                    criterion, optimizer, scaler, device, epochs, feature_dim):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (src_batch, tgt_batch) in zip(source_loader, target_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            M_phy_src = build_batch_template(M_phy_dict, src_y, device)
            M_phy_tgt = build_batch_template(M_phy_dict, torch.zeros(len(tgt_x), dtype=torch.long), device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy_src, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy_tgt)

                # 类原型 (src_y 已经是 0..K-1)
                num_classes = int(src_y.max().item()) + 1
                prototypes = []
                for c in range(num_classes):
                    mask = src_y == c
                    if mask.sum() > 0:
                        prototypes.append(src_feats[mask].mean(dim=0))
                    else:
                        prototypes.append(torch.zeros(feature_dim, device=device))
                prototypes = torch.stack(prototypes)

                L_align, L_sep, _, _ = aligner.alignment_loss(
                    src_feats, src_y, tgt_feats, prototypes)

                L_cls, _ = criterion(src_logits, src_y, features=src_feats,
                                     attn_weights=src_attn, M_phy=M_phy_src,
                                     L_align=L_align, L_sep=L_sep)
                total_loss += L_cls.item()

            scaler.scale(L_cls).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 10 == 0:
            print(f"  Align Epoch {epoch}/{epochs} | Loss: {total_loss / len(source_loader):.4f}")


# ── 主函数 ──────────────────────────────────────────────────────────

def main():
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    source_domain = cfg.experiment.source_domain
    target_domain = cfg.experiment.target_domain
    is_open_set = (cfg.experiment.task == "open_set")
    unknown_class = cfg.experiment.unknown_class

    print(f"\n任务: {cfg.experiment.task}")
    print(f"源域: {source_domain} → 目标域: {target_domain}")
    if is_open_set:
        print(f"未知类别: {unknown_class}")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    if is_open_set:
        known_classes = [c for c in all_classes if c != unknown_class]
    else:
        known_classes = all_classes

    K = len(known_classes)
    cfg.model.num_classes = K

    # 1. 加载数据
    print("\n加载数据...")
    source_ds = SelfCollectedDataset(
        data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size,
        overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins,
    )
    target_ds = SelfCollectedDataset(
        data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes if is_open_set else known_classes,
        window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins,
    )

    print(f"源域: {len(source_ds)} samples, 目标域: {len(target_ds)} samples")
    if len(source_ds) == 0 or len(target_ds) == 0:
        print("[ERROR] 数据集为空")
        return

    # 重映射源域标签到 0..K-1
    source_labels_remapped = remap_labels(source_ds.labels, known_classes)
    source_ds.labels = source_labels_remapped

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    print(f"频率维度: {cfg.model.freq_input_dim}, 已知类数: {K}")

    source_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size,
                                shuffle=True, num_workers=0, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size,
                                shuffle=True, num_workers=0, drop_last=True)

    # 2. 物理频率模板 (使用源域实际RPM)
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    template_builder = FrequencyTemplateBuilder(
        sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle},
    )
    # 合并模板 (所有故障频率, 开集闭集通用)
    template_fft = template_builder.build_template(rpm=source_rpm)
    if cfg.model.freq_input_dim > cfg.signal.fft_bins:
        pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
        combined_tmpl = torch.from_numpy(np.concatenate([template_fft, pad])).float()
    else:
        combined_tmpl = torch.from_numpy(template_fft[:cfg.model.freq_input_dim]).float()
    M_phy_dict = {i: combined_tmpl for i in range(K)}  # 所有类用同一模板

    # 3. 模型 (重建以匹配新的 num_classes)
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    # 4. 阶段一：源域预训练
    print("\n[阶段1] 源域预训练...")
    pretrain_epochs = 80 if is_open_set else 30
    train_source(model, source_loader, M_phy_dict, criterion, optimizer, scaler, device, epochs=pretrain_epochs)

    # 5. 阶段二：跨域对齐 (仅闭集任务)
    if is_open_set:
        print("\n[阶段2] 开集任务: 跳过跨域对齐 (源域预训练+能量分数)")
    else:
        print("\n[阶段2] 跨域对齐训练...")
        aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=K).to(device)
        optimizer_align = torch.optim.AdamW(
            list(model.parameters()) + list(aligner.parameters()),
            lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay,
        )
        train_alignment(model, source_loader, target_loader, M_phy_dict, aligner,
                        criterion, optimizer_align, scaler, device, epochs=50,
                        feature_dim=cfg.model.feature_dim)

    # 6. 评估
    print("\n[评估] ...")

    if is_open_set:
        # 开集评估：使用源域能量分数校准 + 阈值扫描
        model.eval()
        # 评估时使用类 0 模板 (最中性)
        M_eval = M_phy_dict[0].unsqueeze(0).to(device)

        # 先收集源域能量分数（用于阈值校准）
        src_energies = []
        with torch.no_grad():
            for batch in source_loader:
                x, _, _ = batch
                x = x.to(device)
                logits, _ = model(x, M_eval)
                energy = torch.logsumexp(logits, dim=1)
                src_energies.append(energy.cpu().numpy())
        src_energies = np.concatenate(src_energies)

        # 目标域能量分数
        tgt_logits_all, tgt_true_all = [], []
        with torch.no_grad():
            for batch in target_loader:
                x, y, _ = batch
                x = x.to(device)
                logits, _ = model(x, M_eval)
                tgt_logits_all.append(logits.cpu())
                tgt_true_all.append(y)

        tgt_logits = torch.cat(tgt_logits_all, 0)
        tgt_true = torch.cat(tgt_true_all, 0)
        tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
        tgt_preds = tgt_logits.argmax(dim=1).numpy()

        # 真标签
        unknown_orig_label = _ALL_CLASS_MAP[unknown_class]
        label_map_known = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
        true_labels = np.array([
            -1 if int(l) == unknown_orig_label else label_map_known.get(int(l), -1)
            for l in tgt_true.numpy()
        ])

        # 阈值扫描找最佳 H-score
        best_h, best_result = 0, None
        for pct in range(1, 100, 2):
            thresh = np.percentile(src_energies, pct)
            p = tgt_preds.copy()
            p[tgt_energy < thresh] = -1
            m = compute_open_set_metrics(true_labels, p, -tgt_energy)
            if m["h_score"] > best_h:
                best_h = m["h_score"]
                best_result = (pct, thresh, m)

        pct, energy_thresh, metrics = best_result
        print(f"\n开集诊断结果 ({source_domain} → {target_domain}, Unknown={unknown_class}):")
        print(f"  最佳百分位:  {pct}%")
        print(f"  能量阈值:    {energy_thresh:.4f}")
        print(f"  源域能量:    mean={src_energies.mean():.4f}, std={src_energies.std():.4f}")
        print(f"  Known Acc:   {metrics['known_acc']:.4f}")
        print(f"  Unknown Acc: {metrics['unknown_acc']:.4f}")
        print(f"  H-score:     {metrics['h_score']:.4f}")
        print(f"  AUROC:       {metrics['auroc']:.4f}")
    else:
        # 闭集评估 (使用类 0 模板作为默认)
        model.eval()
        M_eval = M_phy_dict[0].unsqueeze(0).to(device)
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in target_loader:
                x, y, _ = batch
                x = x.to(device)
                logits, _ = model(x, M_eval)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.numpy())

        metrics = compute_closed_set_metrics(all_true, all_preds)
        print(f"\n闭集迁移结果 ({source_domain} → {target_domain}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    # 保存
    tag = f"{source_domain}_to_{target_domain}"
    if is_open_set:
        tag += f"_unk-{unknown_class}"
    torch.save({
        "model_state_dict": model.state_dict(),
        "task": cfg.experiment.task,
        "known_classes": known_classes,
    }, os.path.join(cfg.paths.checkpoint, f"cross_domain_{tag}.pth"))
    print(f"模型已保存: cross_domain_{tag}.pth")


if __name__ == "__main__":
    main()
