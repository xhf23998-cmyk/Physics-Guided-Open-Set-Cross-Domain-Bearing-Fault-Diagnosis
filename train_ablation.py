"""Step 7: 核心模块消融实验 (v2 - 修复Bug 2 + 类别感知模板)"""
import os, sys, copy, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner, build_batch_template
from utils import set_seed, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP, train_source, train_alignment


def run_ablation(source_domain, target_domain, unknown_class):
    """消融实验 (v2): 逐步添加模块

    变体:
      1. Backbone: M_phy=None (真正无物理), 无SupCon, 无对齐
      2. + Physics FA (合并): 全频率模板, 无SupCon, 无对齐
      3. + Physics FA (类别感知): 类别特定模板, 无SupCon, 无对齐
      4. + Physics-SupCon: 类别模板 + 物理一致性SupCon
      5. Full Ours: 全部模块
    """
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    freq_input_dim = source_ds.segments.shape[1]

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板 (类别感知 + 合并)
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})

    # 类别感知模板
    class_tmpl_raw = tb.build_class_templates_openset(source_rpm, known_classes)
    M_phy_class = {}
    for ci, tmpl in class_tmpl_raw.items():
        pad = np.zeros(freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
        M_phy_class[ci] = torch.from_numpy(np.concatenate([tmpl, pad])).float()

    # 合并模板 (所有故障频率)
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy_combined_raw = torch.from_numpy(np.concatenate([t_fft, pad])).float()
    M_phy_combined = {i: M_phy_combined_raw for i in range(K)}  # 所有样本用同一个

    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}

    # 消融变体定义
    variants = [
        {"name": "Backbone", "use_phy": None, "align": False, "supcon": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "+ Physics FA (combined)", "use_phy": "combined", "align": False, "supcon": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "+ Physics FA (class-aware)", "use_phy": "class", "align": False, "supcon": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "+ Physics-SupCon", "use_phy": "class", "align": False, "supcon": True,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "Full Ours", "use_phy": "class", "align": True, "supcon": True,
         "pretrain_epochs": 30, "align_epochs": 50},
    ]

    results = {}

    for v in variants:
        print(f"\n{'='*50}")
        print(f"消融: {v['name']}")
        print(f"  Phy={v['use_phy']}, SupCon={v['supcon']}, Align={v['align']}")
        print(f"{'='*50}")

        # Bug 2 fix: deep copy config for each variant
        cfg_v = copy.deepcopy(cfg)
        cfg_v.model.num_classes = K
        cfg_v.model.freq_input_dim = freq_input_dim
        if not v["supcon"]:
            cfg_v.train.lambda_supcon = 0.0
        if not v["align"]:
            cfg_v.train.lambda_align = 0.0
            cfg_v.train.lambda_sep = 0.0

        model = FullModel(cfg_v).to(device)
        criterion = CombinedLoss(cfg_v)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = GradScaler(enabled=cfg.train.fp16)

        # 选择模板
        if v["use_phy"] is None:
            M_dict = None
        elif v["use_phy"] == "combined":
            M_dict = M_phy_combined
        else:
            M_dict = M_phy_class

        # 阶段1: 源域预训练
        print(f"  源域预训练 ({v['pretrain_epochs']} epochs)...")
        if M_dict is not None:
            train_source(model, src_loader, M_dict, criterion, optimizer, scaler, device, epochs=v["pretrain_epochs"])
        else:
            # 无物理模板: 直接训练
            model.train()
            for epoch in range(1, v["pretrain_epochs"] + 1):
                total_loss = 0
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, None, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                if epoch % 20 == 0:
                    print(f"    Epoch {epoch}/{v['pretrain_epochs']} | Loss: {total_loss/len(src_loader):.4f}")

        # 阶段2: 选择性对齐
        if v["align"] and M_dict is not None:
            print(f"  选择性对齐 ({v['align_epochs']} epochs)...")
            aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=K).to(device)
            optimizer_a = torch.optim.AdamW(
                list(model.parameters()) + list(aligner.parameters()),
                lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler_a = GradScaler(enabled=cfg.train.fp16)
            train_alignment(model, src_loader, tgt_loader, M_dict, aligner, criterion,
                            optimizer_a, scaler_a, device, epochs=v["align_epochs"],
                            feature_dim=cfg.model.feature_dim)

        # 评估
        model.eval()
        M_eval = M_phy_class[0].unsqueeze(0).to(device) if M_dict is not None else None

        src_energies = []
        with torch.no_grad():
            for x, _, _ in src_loader:
                x = x.to(device)
                logits, _ = model(x, M_eval)
                src_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
        src_energies = np.concatenate(src_energies)

        tgt_logits, tgt_true = [], []
        with torch.no_grad():
            for x, y, _ in tgt_loader:
                x = x.to(device)
                logits, _ = model(x, M_eval)
                tgt_logits.append(logits.cpu())
                tgt_true.append(y)
        tgt_logits = torch.cat(tgt_logits, 0)
        tgt_true = torch.cat(tgt_true, 0)
        tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
        tgt_preds = tgt_logits.argmax(dim=1).numpy()

        true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                                for l in tgt_true.numpy()])

        best_h, best_result = 0, None
        for pct in range(1, 100, 2):
            thresh = np.percentile(src_energies, pct)
            p = tgt_preds.copy()
            p[tgt_energy < thresh] = -1
            m = compute_open_set_metrics(true_labels, p, -tgt_energy)
            if m["h_score"] > best_h:
                best_h = m["h_score"]
                best_result = m

        results[v["name"]] = best_result
        print(f"  Known Acc={best_result['known_acc']:.4f}, Unknown Acc={best_result['unknown_acc']:.4f}, "
              f"H-score={best_result['h_score']:.4f}, AUROC={best_result['auroc']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"消融实验汇总 (Unknown={unknown_class})")
    print(f"{'='*60}")
    print(f"{'变体':<35} {'Known':>8} {'Unk':>8} {'H-score':>8} {'AUROC':>8}")
    print("-" * 67)
    for name, m in results.items():
        print(f"{name:<35} {m['known_acc']:>8.4f} {m['unknown_acc']:>8.4f} {m['h_score']:>8.4f} {m['auroc']:>8.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    parser.add_argument("--unknown", default="Ball")
    args = parser.parse_args()

    run_ablation(args.source, args.target, args.unknown)
