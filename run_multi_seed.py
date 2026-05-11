"""统计显著性: 多次seed重跑核心实验, 报告mean±std"""
import os, sys, copy, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner
from utils import set_seed, compute_closed_set_metrics, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP, train_source, train_alignment


def run_closed_set_with_seed(source_domain, target_domain, seed):
    """闭集迁移, 指定seed"""
    set_seed(seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(all_classes)

    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    M_phy_dict = {}
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy_raw = torch.from_numpy(np.concatenate([t_fft, pad])).float()
    M_phy = M_phy_raw.unsqueeze(0).to(device)
    M_phy_dict = {i: M_phy_raw for i in range(4)}

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    train_source(model, src_loader, M_phy_dict, criterion, optimizer, scaler, device, epochs=30)

    aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=4).to(device)
    optimizer2 = torch.optim.AdamW(
        list(model.parameters()) + list(aligner.parameters()),
        lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
    scaler2 = GradScaler(enabled=cfg.train.fp16)
    train_alignment(model, src_loader, tgt_loader, M_phy_dict, aligner, criterion,
                    optimizer2, scaler2, device, epochs=50, feature_dim=cfg.model.feature_dim)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(y.numpy())

    m = compute_closed_set_metrics(trues, preds)
    return m


def run_openset_with_seed(source_domain, target_domain, unknown_class, seed):
    """开集诊断, 指定seed"""
    set_seed(seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy_raw = torch.from_numpy(np.concatenate([t_fft, pad])).float()
    M_phy = M_phy_raw.unsqueeze(0).to(device)
    M_phy_dict = {i: M_phy_raw for i in range(K)}

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}

    # Ours: 源域+对齐+能量分数
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    train_source(model, src_loader, M_phy_dict, criterion, optimizer, scaler, device, epochs=30)

    aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=K).to(device)
    optimizer2 = torch.optim.AdamW(
        list(model.parameters()) + list(aligner.parameters()),
        lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
    scaler2 = GradScaler(enabled=cfg.train.fp16)
    train_alignment(model, src_loader, tgt_loader, M_phy_dict, aligner, criterion,
                    optimizer2, scaler2, device, epochs=50, feature_dim=cfg.model.feature_dim)

    model.eval()
    src_energies = []
    with torch.no_grad():
        for x, _, _ in src_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            src_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    src_energies = np.concatenate(src_energies)

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

    best_h, best_result = 0, None
    for pct in range(1, 100, 2):
        thresh = np.percentile(src_energies, pct)
        p = tgt_preds.copy()
        p[tgt_energy < thresh] = -1
        m = compute_open_set_metrics(true_labels, p, -tgt_energy)
        if m["h_score"] > best_h:
            best_h = m["h_score"]
            best_result = m
    return best_result


if __name__ == "__main__":
    seeds = [42, 123, 2024]

    # 闭集迁移
    print("=" * 60)
    print("闭集迁移 多seed结果")
    print("=" * 60)
    for src, tgt in [("W1", "W2"), ("W2", "W1")]:
        accs, f1s = [], []
        for seed in seeds:
            m = run_closed_set_with_seed(src, tgt, seed)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
            print(f"  {src}→{tgt} seed={seed}: Acc={m['accuracy']:.4f}")
        print(f"  {src}→{tgt} Final: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    # 开集诊断
    print("\n" + "=" * 60)
    print("开集诊断 多seed结果")
    print("=" * 60)
    for unk in ["Ball", "Inner", "Outer"]:
        hs, aus = [], []
        for seed in seeds:
            m = run_openset_with_seed("W1", "W2", unk, seed)
            hs.append(m["h_score"])
            aus.append(m["auroc"])
            print(f"  Unk={unk} seed={seed}: H={m['h_score']:.4f}, AUROC={m['auroc']:.4f}")
        print(f"  Unk={unk} Final: H={np.mean(hs):.4f}±{np.std(hs):.4f}, AUROC={np.mean(aus):.4f}±{np.std(aus):.4f}")
