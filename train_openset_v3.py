"""开集诊断 (v3 - 源域验证集校准 + 自适应域偏移校正)

阈值策略:
  1. 源域训练集训练模型
  2. 源域验证集计算能量分布 (已知类)
  3. 目标域(无标签)计算能量分布 (混合已知+未知)
  4. 自适应阈值 = 源域验证p百分位 × (目标域中位数/源域中位数)
     这用到了目标域的无标签统计量(合法), 但不用任何标签

报告两个版本:
  - Fixed: 纯源域p百分位 (最严格, 无目标域信息)
  - Adaptive: 源域p百分位 × 域偏移校正 (用无标签目标域统计量)
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
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--pct", type=int, default=5)
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

    print(f"开集诊断(v3): {source_domain}→{target_domain}, Unknown={unknown_class}")

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]

    # 源域分割
    n_src = len(source_ds)
    n_val = int(n_src * args.val_ratio)
    train_indices = list(range(n_src - n_val))
    val_indices = list(range(n_src - n_val, n_src))

    train_ds = Subset(source_ds, train_indices)
    val_ds = Subset(source_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = 1200 if source_domain == "W1" else 1800
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    M_phy = torch.from_numpy(np.concatenate([t_fft, np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)])).float().to(device)

    # 训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print(f"\n[训练] {args.pretrain_epochs} epochs...")
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

    model.eval()

    # 源域验证集能量
    val_energies = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            val_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    val_energies = np.concatenate(val_energies)

    # 目标域能量和预测
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

    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1) for l in tgt_true.numpy()])

    # ── 阈值方案1: 纯源域固定百分位 ──────────────────────────────
    fixed_thresh = np.percentile(val_energies, args.pct)
    p_fixed = tgt_preds.copy()
    p_fixed[tgt_energy < fixed_thresh] = -1
    m_fixed = compute_open_set_metrics(true_labels, p_fixed, -tgt_energy)

    # ── 阈值方案2: 自适应域偏移校正 ──────────────────────────────
    # 原理: 源域p百分位 × (目标域能量中位数 / 源域能量中位数)
    # 只用了目标域无标签统计量, 合法
    src_median = np.median(val_energies)
    tgt_median = np.median(tgt_energy)
    shift_ratio = tgt_median / (src_median + 1e-8)
    adaptive_thresh = fixed_thresh * shift_ratio

    p_adaptive = tgt_preds.copy()
    p_adaptive[tgt_energy < adaptive_thresh] = -1
    m_adaptive = compute_open_set_metrics(true_labels, p_adaptive, -tgt_energy)

    # ── 报告 ──────────────────────────────────────────────────────
    print(f"\n源域能量: mean={val_energies.mean():.3f}, median={src_median:.3f}")
    print(f"目标域能量: mean={tgt_energy.mean():.3f}, median={tgt_median:.3f}")
    print(f"域偏移比: {shift_ratio:.3f}")

    print(f"\n{'='*55}")
    print(f"方案1: 纯源域固定阈值 (val_{args.pct}pct={fixed_thresh:.4f})")
    print(f"{'='*55}")
    print(f"  Known Acc:    {m_fixed['known_acc']:.4f}")
    print(f"  Unknown Acc:  {m_fixed['unknown_acc']:.4f}")
    print(f"  H-score:      {m_fixed['h_score']:.4f}")
    print(f"  AUROC:        {m_fixed['auroc']:.4f}")

    print(f"\n{'='*55}")
    print(f"方案2: 自适应域偏移校正 (thresh={adaptive_thresh:.4f})")
    print(f"{'='*55}")
    print(f"  Known Acc:    {m_adaptive['known_acc']:.4f}")
    print(f"  Unknown Acc:  {m_adaptive['unknown_acc']:.4f}")
    print(f"  H-score:      {m_adaptive['h_score']:.4f}")
    print(f"  AUROC:        {m_adaptive['auroc']:.4f}")

    # 多百分位参考表
    print(f"\n[参考] 不同百分位 × 自适应校正:")
    print(f"  {'pct':>4} {'fixed_th':>10} {'adapt_th':>10} {'Known':>8} {'Unk':>8} {'H':>8} {'AUROC':>8}")
    for pct in [1, 5, 10, 20, 30, 50]:
        ft = np.percentile(val_energies, pct)
        at = ft * shift_ratio
        p = tgt_preds.copy()
        p[tgt_energy < at] = -1
        r = compute_open_set_metrics(true_labels, p, -tgt_energy)
        print(f"  {pct:>3}% {ft:>10.4f} {at:>10.4f} {r['known_acc']:>8.4f} {r['unknown_acc']:>8.4f} {r['h_score']:>8.4f} {r['auroc']:>8.4f}")

    return m_fixed, m_adaptive


if __name__ == "__main__":
    main()
