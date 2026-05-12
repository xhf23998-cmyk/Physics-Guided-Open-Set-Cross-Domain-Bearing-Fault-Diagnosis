"""开集诊断专用训练 (v2 - 修复阈值泄漏)

修复: 阈值在源域验证集上确定, 目标域只做一次性评估
训练数据: 源域90%
验证数据: 源域10% (用于阈值校准)
测试数据: 目标域全集 (一次性评估, 不扫描)
"""
import os, sys, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder
from utils import set_seed, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    parser.add_argument("--unknown", default="Ball")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="源域验证集比例")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--fixed-pct", type=int, default=5,
                        help="源域验证集百分位(1-99), 5=接受95%已知样本")
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

    print(f"开集诊断(v2): {source_domain}→{target_domain}, Unknown={unknown_class}")
    print(f"已知类: {known_classes} ({K}类)")
    print(f"阈值校准: 源域验证集 {args.val_ratio*100:.0f}%, 百分位={args.fixed_pct}%")

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]

    # 源域分割: 90%训练 + 10%验证 (信号级别, 用索引顺序分割)
    n_src = len(source_ds)
    n_val = int(n_src * args.val_ratio)
    n_train = n_src - n_val

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_src))

    train_ds = Subset(source_ds, train_indices)
    val_ds = Subset(source_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    print(f"源域训练: {n_train}样本, 验证: {n_val}样本, 目标域: {len(target_ds)}样本")

    # 物理模板
    source_rpm = 1200 if source_domain == "W1" else 1800
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    M_phy = torch.from_numpy(np.concatenate([t_fft, np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)])).float().to(device)

    # 模型训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print(f"\n[训练] 源域预训练 ({args.pretrain_epochs} epochs)...")
    model.train()
    for epoch in range(1, args.pretrain_epochs + 1):
        total_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy, return_features=True)
                loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{args.pretrain_epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # ── 阶段2: 阈值校准 (在源域验证集上) ──────────────────────────
    print("\n[阈值校准] 源域验证集...")
    model.eval()

    val_energies = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            val_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    val_energies = np.concatenate(val_energies)

    # 用验证集确定阈值: 第p百分位 (p越低越严格, 越多已知被拒)
    threshold = np.percentile(val_energies, args.fixed_pct)
    print(f"  验证集能量: mean={val_energies.mean():.3f}, std={val_energies.std():.3f}")
    print(f"  阈值(第{args.fixed_pct}百分位): {threshold:.4f}")

    # ── 阶段3: 目标域一次性评估 ──────────────────────────────────
    print("\n[评估] 目标域 (一次性, 无阈值扫描)...")
    tgt_logits, tgt_true = [], []
    with torch.no_grad():
        for x, y, _ in target_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            tgt_logits.append(logits.cpu())
            tgt_true.append(y)
    tgt_logits = torch.cat(tgt_logits, 0)
    tgt_true = torch.cat(tgt_true, 0)
    tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
    tgt_preds = tgt_logits.argmax(dim=1).numpy()

    # 真标签
    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1) for l in tgt_true.numpy()])

    # 固定阈值评估
    predictions = tgt_preds.copy()
    predictions[tgt_energy < threshold] = -1

    m = compute_open_set_metrics(true_labels, predictions, -tgt_energy)

    print(f"\n{'='*50}")
    print(f"开集诊断结果 (阈值校准: val_{args.fixed_pct}pct)")
    print(f"{'='*50}")
    print(f"  阈值:         {threshold:.4f}")
    print(f"  Known Acc:    {m['known_acc']:.4f}")
    print(f"  Unknown Acc:  {m['unknown_acc']:.4f}")
    print(f"  H-score:      {m['h_score']:.4f}")
    print(f"  AUROC:        {m['auroc']:.4f}")

    # 同时报告多个百分位的结果 (供参考, 不用于选择)
    print(f"\n[参考] 不同百分位的性能 (仅报告, 不用于阈值选择):")
    print(f"  {'百分位':>6} {'阈值':>10} {'Known':>8} {'Unk':>8} {'H-score':>8}")
    for pct in [1, 5, 10, 20, 30, 50]:
        t = np.percentile(val_energies, pct)
        p = tgt_preds.copy()
        p[tgt_energy < t] = -1
        r = compute_open_set_metrics(true_labels, p, -tgt_energy)
        print(f"  {pct:>5}% {t:>10.4f} {r['known_acc']:>8.4f} {r['unknown_acc']:>8.4f} {r['h_score']:>8.4f}")

    # 保存
    tag = f"openset_v2_{source_domain}_to_{target_domain}_unk-{unknown_class}"
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "val_pct": args.fixed_pct,
        "metrics": m,
    }, os.path.join(cfg.paths.checkpoint, f"{tag}.pth"))
    print(f"\n模型已保存: {tag}.pth")
    return m


if __name__ == "__main__":
    main()
