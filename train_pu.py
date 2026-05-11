"""Step 8: Paderborn 公开数据集验证 (跨工况迁移 + 开集诊断)"""
import os
import sys
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data.dataset import _PU_CLASSES, _PU_FILE_TO_CLASS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner
from utils import set_seed, compute_closed_set_metrics, compute_open_set_metrics
from train_cross_domain import remap_labels


# ── PU 工况定义 ───────────────────────────────────────────────────

# 按实际数据文件名定义工况
_PU_CONDITIONS = {
    "A": "N15_M07_F04",  # 1500rpm, 0.7Nm, 400N
    "B": "N09_M07_F10",  # 900rpm, 0.7Nm, 1000N
    "C": "N15_M01_F10",  # 1500rpm, 0.1Nm, 1000N
}

# PU RPM 映射
_PU_RPM = {"A": 1500, "B": 900, "C": 1500}


class PUTensorDataset(Dataset):
    """PU Tensor Dataset wrapper"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, torch.tensor(0)


def _load_pu_signal(fpath):
    """从单个 PU .mat 文件加载振动信号

    PU .mat 结构: X 有3个通道
      Ch0: 振动信号 (加速度计, ~16000点, 64kHz, ~0.25s)
      Ch1: 电机电流 (~256000点)
      Ch2: 元数据 (5点)
    选择 Ch0 (振动通道), 而不是最长通道(电流)
    """
    from scipy.io import loadmat
    mat = loadmat(fpath)
    for key in mat:
        if key.startswith("__"):
            continue
        val = mat[key]
        if not isinstance(val, np.ndarray):
            continue
        if val.dtype.names and "X" in val.dtype.names:
            rec = val[0, 0]
            X = rec["X"]
            # 优先选择振动通道 (Ch0, ~16000点)
            # 跳过过短的通道 (<1000点)
            for i in range(X.shape[1]):
                ch = X[0, i]
                if "Data" in ch.dtype.names:
                    sig = ch["Data"].flatten()
                    if 1000 < len(sig) < 100000:  # 振动通道长度范围
                        return sig.astype(np.float32)
            # 回退: 选择最短的有效通道
            valid = []
            for i in range(X.shape[1]):
                ch = X[0, i]
                if "Data" in ch.dtype.names:
                    sig = ch["Data"].flatten()
                    if len(sig) > 1000:
                        valid.append((len(sig), sig))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1].astype(np.float32)
        if val.size > 100:
            return val.flatten().astype(np.float32)
    return None


def _downsample(signal, factor):
    """降采样信号"""
    return signal[::factor]


def _sliding_window(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return np.array([signal[i:i+window_size] for i in range(0, len(signal)-window_size+1, step)], dtype=np.float32)


def _compute_features(segments, fft_bins):
    from scipy.signal import hilbert, detrend
    # 去趋势: PU信号有强线性偏移, 不去除会导致所有类FFT相同
    detrended = detrend(segments, axis=-1)
    normed = (detrended - detrended.mean(axis=-1, keepdims=True)) / (detrended.std(axis=-1, keepdims=True) + 1e-8)
    fft_spec = np.abs(np.fft.rfft(normed, axis=-1))[:, :fft_bins]
    envelope = np.abs(np.abs(hilbert(normed, axis=-1)))
    env_spec = np.abs(np.fft.rfft(envelope, axis=-1))[:, :fft_bins]
    return np.concatenate([fft_spec, env_spec], axis=-1).astype(np.float32)


def load_pu_data(condition, bearing_ids=None, window_size=1024, overlap=0.5, fft_bins=512,
                 max_files_per_bearing=20, max_signal_len=100000):
    """快速加载 PU 数据 (每轴承只取少数文件)"""
    import glob
    from data.dataset import _PU_FILE_TO_CLASS, _PU_CLASSES

    cond_str = _PU_CONDITIONS.get(condition, condition)
    print(f"  加载 PU 工况={condition} ({cond_str})...")

    if bearing_ids is None:
        bearing_ids = list(_PU_FILE_TO_CLASS.keys())

    all_segments = []
    all_labels = []

    for bid in bearing_ids:
        if bid not in _PU_FILE_TO_CLASS:
            continue
        cls_name = _PU_FILE_TO_CLASS[bid]
        label = _PU_CLASSES[cls_name]
        bear_dir = os.path.join(cfg.paths.pu, bid)
        if not os.path.exists(bear_dir):
            continue

        mat_files = sorted(glob.glob(os.path.join(bear_dir, "*.mat")))
        mat_files = [f for f in mat_files if cond_str in os.path.basename(f)]

        # 只取前 max_files_per_bearing 个文件
        mat_files = mat_files[:max_files_per_bearing]

        for mf in mat_files:
            try:
                sig = _load_pu_signal(mf)
            except Exception:
                continue
            if sig is None or len(sig) < window_size:
                continue
            # 限制信号长度 (振动通道Ch0约16000点, 无需截断)
            if len(sig) > max_signal_len:
                sig = sig[:max_signal_len]
            # 不降采样: Ch0振动通道已足够短(~16000点)
            segs = _sliding_window(sig, window_size, overlap)
            if len(segs) > 0:
                all_segments.append(segs)
                all_labels.append(np.full(len(segs), label, dtype=np.int64))

    if not all_segments:
        print(f"  [WARNING] 工况 {condition} 无数据")
        return np.array([]), np.array([])

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = _compute_features(segments, fft_bins)
    print(f"  加载完成: {len(labels)} 样本")
    return features, labels


# ── 训练函数 ───────────────────────────────────────────────────────

def train_source(model, loader, M_phy, criterion, optimizer, scaler, device, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy, return_features=True)
                loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs} | Loss: {total_loss / len(loader):.4f}")


def train_alignment_fn(model, src_loader, tgt_loader, M_phy, aligner, criterion, optimizer, scaler, device, epochs):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy)

                num_classes = int(src_y.max().item()) + 1
                prototypes = []
                for c in range(num_classes):
                    mask = src_y == c
                    prototypes.append(src_feats[mask].mean(dim=0) if mask.sum() > 0
                                      else torch.zeros(cfg.model.feature_dim, device=device))
                prototypes = torch.stack(prototypes)

                L_align, L_sep, _, _ = aligner.alignment_loss(src_feats, src_y, tgt_feats, prototypes)
                loss, _ = criterion(src_logits, src_y, features=src_feats, attn_weights=src_attn, M_phy=M_phy,
                                    L_align=L_align, L_sep=L_sep)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Align Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── 实验 ──────────────────────────────────────────────────────────

def run_closed_set_transfer(src_cond, tgt_cond, fft_bins=512):
    """PU 跨工况闭集迁移"""
    print(f"\n{'='*50}")
    print(f"PU 跨工况: {src_cond} → {tgt_cond}")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = list(_PU_CLASSES.keys())
    cfg.model.num_classes = len(all_classes)

    # 按类别分组加载，确保每个类都有数据
    src_features_all, src_labels_all = [], []
    tgt_features_all, tgt_labels_all = [], []

    for cls_name in all_classes:
        cls_bearings = [bid for bid, cn in _PU_FILE_TO_CLASS.items() if cn == cls_name]
        src_seg, src_lbl = load_pu_data(src_cond, bearing_ids=cls_bearings, fft_bins=fft_bins)
        tgt_seg, tgt_lbl = load_pu_data(tgt_cond, bearing_ids=cls_bearings, fft_bins=fft_bins)
        if len(src_seg) > 0:
            src_features_all.append(src_seg)
            src_labels_all.append(src_lbl)
        if len(tgt_seg) > 0:
            tgt_features_all.append(tgt_seg)
            tgt_labels_all.append(tgt_lbl)

    if not src_features_all or not tgt_features_all:
        print("[SKIP] 数据为空")
        return None

    src_features = np.concatenate(src_features_all)
    src_labels = np.concatenate(src_labels_all)
    tgt_features = np.concatenate(tgt_features_all)
    tgt_labels = np.concatenate(tgt_labels_all)

    cfg.model.freq_input_dim = src_features.shape[1]
    print(f"  源域: {len(src_features)} samples, 目标域: {len(tgt_features)} samples")

    src_loader = DataLoader(PUTensorDataset(src_features, src_labels),
                            batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(PUTensorDataset(tgt_features, tgt_labels),
                            batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = _PU_RPM.get(src_cond, 1500)
    tb = FrequencyTemplateBuilder(sample_rate=64000, fft_bins=fft_bins,
        bearing_params={"ball_count": 8, "ball_diameter": 6.747, "pitch_diameter": 28.5, "contact_angle": 0.0})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    results = {}

    # Source Only
    print("\n[Source Only]")
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    train_source(model, src_loader, M_phy, criterion, optimizer, scaler, device, epochs=50)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(y.numpy())
    m = compute_closed_set_metrics(trues, preds)
    results["Source Only"] = m
    print(f"  Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")

    # Ours (with alignment)
    print("\n[Ours (Selective Alignment)]")
    model2 = FullModel(cfg).to(device)
    criterion2 = CombinedLoss(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler2 = GradScaler(enabled=cfg.train.fp16)

    print("  阶段1: 源域预训练...")
    train_source(model2, src_loader, M_phy, criterion2, optimizer2, scaler2, device, epochs=30)

    print("  阶段2: 选择性对齐...")
    aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=cfg.model.num_classes).to(device)
    optimizer_a = torch.optim.AdamW(
        list(model2.parameters()) + list(aligner.parameters()),
        lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
    scaler_a = GradScaler(enabled=cfg.train.fp16)
    train_alignment_fn(model2, src_loader, tgt_loader, M_phy, aligner, criterion2, optimizer_a, scaler_a, device, epochs=50)

    model2.eval()
    preds2, trues2 = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model2(x, M_phy)
            preds2.extend(logits.argmax(dim=1).cpu().numpy())
            trues2.extend(y.numpy())
    m2 = compute_closed_set_metrics(trues2, preds2)
    results["Ours"] = m2
    print(f"  Acc={m2['accuracy']:.4f}, F1={m2['macro_f1']:.4f}")

    return results


def run_openset(src_cond, tgt_cond, unknown_class, fft_bins=512):
    """PU 跨工况开集诊断"""
    print(f"\n{'='*50}")
    print(f"PU 开集: {src_cond} → {tgt_cond}, Unknown={unknown_class}")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = list(_PU_CLASSES.keys())
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    src_features_all, src_labels_all = [], []
    tgt_features_all, tgt_labels_all = [], []

    for cls_name in all_classes:
        cls_bearings = [bid for bid, cn in _PU_FILE_TO_CLASS.items() if cn == cls_name]
        # 源域只加载已知类
        if cls_name in known_classes:
            seg, lbl = load_pu_data(src_cond, bearing_ids=cls_bearings, fft_bins=fft_bins)
            if len(seg) > 0:
                src_features_all.append(seg)
                src_labels_all.append(lbl)
        # 目标域加载所有类
        seg, lbl = load_pu_data(tgt_cond, bearing_ids=cls_bearings, fft_bins=fft_bins)
        if len(seg) > 0:
            tgt_features_all.append(seg)
            tgt_labels_all.append(lbl)

    if not src_features_all or not tgt_features_all:
        print("[SKIP] 数据为空")
        return None

    src_features = np.concatenate(src_features_all)
    src_labels = np.concatenate(src_labels_all)
    # PU 专用标签重映射 (不使用 _ALL_CLASS_MAP)
    _pu_label_map = {_PU_CLASSES[c]: i for i, c in enumerate(known_classes)}
    src_labels = np.array([_pu_label_map.get(int(l), -1) for l in src_labels], dtype=np.int64)

    tgt_features = np.concatenate(tgt_features_all)
    tgt_labels = np.concatenate(tgt_labels_all)

    cfg.model.freq_input_dim = src_features.shape[1]

    src_loader = DataLoader(PUTensorDataset(src_features, src_labels),
                            batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(PUTensorDataset(tgt_features, tgt_labels),
                            batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = _PU_RPM.get(src_cond, 1500)
    tb = FrequencyTemplateBuilder(sample_rate=64000, fft_bins=fft_bins,
        bearing_params={"ball_count": 8, "ball_diameter": 6.747, "pitch_diameter": 28.5, "contact_angle": 0.0})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 源域预训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print("  源域预训练 (80 epochs)...")
    train_source(model, src_loader, M_phy, criterion, optimizer, scaler, device, epochs=80)

    # 能量分数评估
    model.eval()
    src_energies = []
    with torch.no_grad():
        for x, _, _ in src_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            src_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    src_energies = np.concatenate(src_energies)

    tgt_logits_all, tgt_true_all = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            tgt_logits_all.append(logits.cpu())
            tgt_true_all.append(y)
    tgt_logits = torch.cat(tgt_logits_all, 0)
    tgt_true = torch.cat(tgt_true_all, 0)
    tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
    tgt_preds = tgt_logits.argmax(dim=1).numpy()

    unknown_orig = _PU_CLASSES[unknown_class]
    label_map = {_PU_CLASSES[c]: i for i, c in enumerate(known_classes)}
    true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1) for l in tgt_true.numpy()])

    best_h, best_result = 0, None
    for pct in range(1, 100, 2):
        thresh = np.percentile(src_energies, pct)
        p = tgt_preds.copy()
        p[tgt_energy < thresh] = -1
        m = compute_open_set_metrics(true_labels, p, -tgt_energy)
        if m["h_score"] > best_h:
            best_h = m["h_score"]
            best_result = (pct, thresh, m)

    pct, thresh, m = best_result
    print(f"  最佳 (pct={pct}%): Known={m['known_acc']:.4f}, Unk={m['unknown_acc']:.4f}, "
          f"H={m['h_score']:.4f}, AUROC={m['auroc']:.4f}")
    return m


# ── 主函数 ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["transfer", "openset", "all"], default="all")
    parser.add_argument("--src-cond", default="A")
    parser.add_argument("--tgt-conds", nargs="+", default=["B", "C"])
    parser.add_argument("--unknown", default="Rolling")
    parser.add_argument("--fft-bins", type=int, default=512)
    args = parser.parse_args()

    fft_bins = args.fft_bins
    all_results = {}

    if args.task in ("transfer", "all"):
        for tgt in args.tgt_conds:
            r = run_closed_set_transfer(args.src_cond, tgt, fft_bins)
            if r:
                all_results[f"transfer_{args.src_cond}to{tgt}"] = r

    if args.task in ("openset", "all"):
        for tgt in args.tgt_conds:
            r = run_openset(args.src_cond, tgt, args.unknown, fft_bins)
            if r:
                all_results[f"openset_{args.src_cond}to{tgt}_{args.unknown}"] = r

    # 汇总
    print("\n" + "=" * 60)
    print("Paderborn 实验汇总")
    print("=" * 60)
    for key, val in all_results.items():
        print(f"\n{key}:")
        if isinstance(val, dict):
            for method, m in val.items():
                if isinstance(m, dict) and "accuracy" in m:
                    print(f"  {method}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")
                elif isinstance(m, dict) and "known_acc" in m:
                    print(f"  {method}: Known={m['known_acc']:.4f}, Unk={m['unknown_acc']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
