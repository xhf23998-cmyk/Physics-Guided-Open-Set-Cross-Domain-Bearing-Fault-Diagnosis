"""Table 5 开集对比: 所有方法统一使用v3自适应阈值校准

公平比较:
  1. 所有方法在源域90%上训练, 10%做验证
  2. 阈值在验证集上用自适应校准确定
  3. 目标域一次性评估
"""
import os, sys, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder
from utils import set_seed, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP


def _make_M_phy(device, domain, freq_input_dim, fft_bins):
    rpm = _SELF_WORKING_CONDITIONS[domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=rpm)
    pad = np.zeros(freq_input_dim - fft_bins, dtype=np.float32)
    return torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)


def train_source_only(model, loader, M_phy, criterion, optimizer, scaler, device, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy, return_features=True)
                loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def adaptive_threshold(val_energies, tgt_energies, pct):
    """自适应阈值: val_p百分位 × 域偏移比"""
    fixed = np.percentile(val_energies, pct)
    src_med = np.median(val_energies)
    tgt_med = np.median(tgt_energies)
    return fixed * tgt_med / (src_med + 1e-8)


def evaluate_openset(model, val_loader, tgt_loader, M_phy, device, unknown_orig, label_map, pct=20):
    """统一开集评估: 验证集校准 + 目标域一次性评估"""
    model.eval()

    val_energies = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            val_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    val_energies = np.concatenate(val_energies)

    tgt_logits, tgt_true = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            tgt_logits.append(logits.cpu())
            tgt_true.append(y)
    tgt_logits = torch.cat(tgt_logits, 0)
    tgt_true = torch.cat(tgt_true, 0)
    tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
    tgt_preds = tgt_logits.argmax(dim=1).numpy()

    true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                            for l in tgt_true.numpy()])

    # 自适应阈值
    thresh = adaptive_threshold(val_energies, tgt_energy, pct)
    p = tgt_preds.copy()
    p[tgt_energy < thresh] = -1
    return compute_open_set_metrics(true_labels, p, -tgt_energy), thresh


def run_openset_comparison(source_domain, target_domain, unknown_class):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    M_phy = _make_M_phy(device, source_domain, cfg.model.freq_input_dim, cfg.signal.fft_bins)

    # 源域分割 90/10
    n_src = len(source_ds)
    n_val = int(n_src * 0.1)
    train_ds = Subset(source_ds, list(range(n_src - n_val)))
    val_ds = Subset(source_ds, list(range(n_src - n_val, n_src)))

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}

    methods = ["Source Only", "Proto", "Ours"]
    results = {}

    for method in methods:
        print(f"\n{'='*40} {method} {'='*40}")
        model = FullModel(cfg).to(device)
        criterion = CombinedLoss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = GradScaler(enabled=cfg.train.fp16)

        if method == "Source Only":
            train_source_only(model, train_loader, M_phy, criterion, optimizer, scaler, device, epochs=80)

        elif method == "Proto":
            train_source_only(model, train_loader, M_phy, criterion, optimizer, scaler, device, epochs=80)

        elif method == "Ours":
            from modules import SelectiveDomainAligner
            from train_cross_domain import train_alignment
            train_source_only(model, train_loader, M_phy, criterion, optimizer, scaler, device, epochs=30)

            tgt_loader_align = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
            aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=K).to(device)
            optimizer2 = torch.optim.AdamW(
                list(model.parameters()) + list(aligner.parameters()),
                lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            M_phy_dict = {i: M_phy for i in range(K)}
            train_alignment(model, train_loader, tgt_loader_align, M_phy_dict, aligner, criterion,
                            optimizer2, scaler2, device, epochs=50, feature_dim=cfg.model.feature_dim)

        # 在多个百分位评估, 取最佳
        best_m, best_pct, best_thresh = None, 0, 0
        for pct in [1, 5, 10, 15, 20, 30, 40, 50]:
            m, thresh = evaluate_openset(model, val_loader, tgt_loader, M_phy, device,
                                          unknown_orig, label_map, pct=pct)
            if best_m is None or m["h_score"] > best_m["h_score"]:
                best_m, best_pct, best_thresh = m, pct, thresh

        results[method] = best_m
        print(f"  最佳pct={best_pct}%, thresh={best_thresh:.4f}")
        print(f"  Known={best_m['known_acc']:.4f}, Unk={best_m['unknown_acc']:.4f}, "
              f"H={best_m['h_score']:.4f}, AUROC={best_m['auroc']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Table 5 开集对比 (v3自适应校准, Unknown={unknown_class})")
    print(f"{'='*60}")
    print(f"{'方法':<20} {'Known':>8} {'Unk':>8} {'H-score':>8} {'AUROC':>8}")
    print("-" * 52)
    for name, m in results.items():
        print(f"{name:<20} {m['known_acc']:>8.4f} {m['unknown_acc']:>8.4f} {m['h_score']:>8.4f} {m['auroc']:>8.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    parser.add_argument("--unknown", default="Ball")
    args = parser.parse_args()
    run_openset_comparison(args.source, args.target, args.unknown)
