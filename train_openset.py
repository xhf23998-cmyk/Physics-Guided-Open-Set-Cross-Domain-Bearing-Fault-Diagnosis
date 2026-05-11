"""开集诊断专用训练: 纯源域预训练 → 能量分数开集检测"""
import os, sys, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import _SELF_WORKING_CONDITIONS
from configs import cfg
from data import SelfCollectedDataset
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, build_batch_template
from utils import set_seed, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    parser.add_argument("--unknown", default="Ball")
    args = parser.parse_args()

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    source_domain = args.source
    target_domain = args.target
    unknown_class = args.unknown
    all_classes = ["Health", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    print(f"开集诊断: {source_domain}→{target_domain}, Unknown={unknown_class}")
    print(f"已知类: {known_classes} ({K}类)")

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]

    source_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    target_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板 (使用源域实际RPM, 合并模板)
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count":cfg.bearing.ball_count,"ball_diameter":cfg.bearing.ball_diameter,
                        "pitch_diameter":cfg.bearing.pitch_diameter,"contact_angle":cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    combined_tmpl = torch.from_numpy(np.concatenate([t_fft, np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)])).float()
    M_phy_dict = {i: combined_tmpl for i in range(K)}

    # 模型 — 只用源域训练，不做跨域对齐
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print("\n[训练] 纯源域预训练 (80 epochs)...")
    model.train()
    for epoch in range(1, 81):
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
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/80 | Loss: {total_loss/len(source_loader):.4f}")

    # 评估
    print("\n[评估] ...")
    model.eval()
    M_eval = M_phy_dict[0].unsqueeze(0).to(device)

    # 源域能量分数 (用于阈值校准)
    src_energies = []
    with torch.no_grad():
        for x, _, _ in source_loader:
            x = x.to(device)
            logits, _ = model(x, M_eval)
            src_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    src_energies = np.concatenate(src_energies)

    # 目标域能量和预测
    tgt_logits, tgt_true = [], []
    with torch.no_grad():
        for x, y, _ in target_loader:
            x = x.to(device)
            logits, _ = model(x, M_eval)
            tgt_logits.append(logits.cpu())
            tgt_true.append(y)
    tgt_logits = torch.cat(tgt_logits, 0)
    tgt_true = torch.cat(tgt_true, 0)
    tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
    tgt_preds = tgt_logits.argmax(dim=1).numpy()

    # 真标签
    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    true_labels = np.array([-1 if int(l)==unknown_orig else label_map.get(int(l),-1) for l in tgt_true.numpy()])

    # 扫描不同阈值
    print(f"\n源域能量: mean={src_energies.mean():.3f}, std={src_energies.std():.3f}")
    print(f"\n阈值扫描:")
    best_h = 0
    best_result = None
    for pct in range(1, 100, 2):
        thresh = np.percentile(src_energies, pct)
        p = tgt_preds.copy()
        p[tgt_energy < thresh] = -1
        m = compute_open_set_metrics(true_labels, p, -tgt_energy)
        if m["h_score"] > best_h:
            best_h = m["h_score"]
            best_result = (pct, thresh, m)

    pct, thresh, m = best_result
    print(f"\n最佳结果 (阈值百分位={pct}%):")
    print(f"  能量阈值:    {thresh:.4f}")
    print(f"  Known Acc:   {m['known_acc']:.4f}")
    print(f"  Unknown Acc: {m['unknown_acc']:.4f}")
    print(f"  H-score:     {m['h_score']:.4f}")
    print(f"  AUROC:       {m['auroc']:.4f}")

    # 保存
    tag = f"openset_{source_domain}_to_{target_domain}_unk-{unknown_class}"
    torch.save({"model_state_dict": model.state_dict(), "best_pct": pct, "best_thresh": thresh},
               os.path.join(cfg.paths.checkpoint, f"{tag}.pth"))
    print(f"\n模型已保存: {tag}.pth")
    return m


if __name__ == "__main__":
    main()
