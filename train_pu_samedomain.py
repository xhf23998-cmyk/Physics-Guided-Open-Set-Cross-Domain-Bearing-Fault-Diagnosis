"""Step 8 (修订版): Paderborn 同工况闭集分类 + 跨工况迁移

策略:
  - 同工况闭集: 在单一工况内做4类分类, 证明方法在PU复杂数据上有效
  - 跨工况迁移: 用更小的域偏移组合, 或增加训练轮数
"""
import os, sys, glob, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
from scipy.signal import hilbert, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner
from utils import set_seed, compute_closed_set_metrics


# PU 类别映射
_PU_CLASSES = {"Healthy": 0, "Inner": 1, "Outer": 2, "Rolling": 3}

_PU_FILE_TO_CLASS = {
    "K001": "Healthy", "K002": "Healthy", "K003": "Healthy", "K004": "Healthy", "K005": "Healthy",
    "K006": "Healthy",
    "KA01": "Outer", "KA03": "Outer", "KA04": "Outer", "KA05": "Outer", "KA07": "Outer",
    "KA08": "Outer", "KA09": "Outer", "KA15": "Outer", "KA16": "Outer", "KA22": "Outer",
    "KA30": "Outer",
    "KI01": "Inner", "KI03": "Inner", "KI04": "Inner", "KI05": "Inner", "KI07": "Inner",
    "KI08": "Inner", "KI14": "Inner", "KI16": "Inner", "KI17": "Inner", "KI18": "Inner",
    "KI21": "Inner",
    "KB23": "Rolling", "KB24": "Rolling", "KB27": "Rolling",
}

_PU_CONDITIONS = {
    "A": "N15_M07_F04",  # 1500rpm, 0.7Nm, 400N
    "B": "N09_M07_F10",  # 900rpm, 0.7Nm, 1000N
    "C": "N15_M01_F10",  # 1500rpm, 0.1Nm, 1000N
}

_PU_RPM = {"A": 1500, "B": 900, "C": 1500}

# PU 6203 轴承参数
_PU_BEARING = {"ball_count": 8, "ball_diameter": 6.747, "pitch_diameter": 28.5, "contact_angle": 0.0}


class PUTensorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(0)


def _load_pu_signal(fpath):
    """从PU .mat文件加载振动信号

    PU数据结构:
      X字段: 时间基准 (非测量数据)
      Y字段: 实际测量 (vibration_1, force, speed, torque等)
    """
    try:
        mat = loadmat(fpath)
    except Exception:
        return None
    for key in mat:
        if key.startswith("__"):
            continue
        val = mat[key]
        if not isinstance(val, np.ndarray):
            continue
        # 结构化数组(size=1)包含子字段，不按size过滤
        if not val.dtype.names and val.size < 100:
            continue
        if val.dtype.names and "Y" in val.dtype.names:
            rec = val[0, 0]
            Y = rec["Y"]
            # 优先找 vibration_1 通道
            best_signal, best_len = None, 0
            for i in range(Y.shape[1]):
                ch = Y[0, i]
                if "Name" in ch.dtype.names and "Data" in ch.dtype.names:
                    name = str(ch["Name"].flatten()[0]) if ch["Name"].size > 0 else ""
                    data = ch["Data"].flatten().astype(np.float32)
                    # 优先 vibration, 其次选最长的非零信号
                    if "vibration" in name.lower():
                        return detrend(data)
                    if len(data) > best_len and data.std() > 1e-6:
                        best_signal = data
                        best_len = len(data)
            if best_signal is not None:
                return detrend(best_signal)
        # 降级: 尝试X字段
        if val.dtype.names and "X" in val.dtype.names:
            rec = val[0, 0]
            X = rec["X"]
            for i in range(X.shape[1]):
                ch = X[0, i]
                if "Data" in ch.dtype.names:
                    data = ch["Data"].flatten().astype(np.float32)
                    if data.std() > 1e-6:
                        return detrend(data)
    return None


def _sliding_window(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return np.array([signal[i:i+window_size] for i in range(0, len(signal)-window_size+1, step)], dtype=np.float32)


def _compute_features(segments, fft_bins):
    normed = (segments - segments.mean(axis=-1, keepdims=True)) / (segments.std(axis=-1, keepdims=True) + 1e-8)
    fft_spec = np.abs(np.fft.rfft(normed, axis=-1))[:, :fft_bins]
    envelope = np.abs(np.abs(hilbert(normed, axis=-1)))
    env_spec = np.abs(np.fft.rfft(envelope, axis=-1))[:, :fft_bins]
    return np.concatenate([fft_spec, env_spec], axis=-1).astype(np.float32)


def load_pu_condition(condition, window_size=1024, overlap=0.5, fft_bins=512, max_files_per_bearing=5, max_signal_len=200000):
    """加载单个工况的所有类别数据"""
    cond_str = _PU_CONDITIONS.get(condition, condition)
    print(f"  加载 PU 工况={condition} ({cond_str})...")

    all_segments, all_labels = [], []

    for bid, cls_name in _PU_FILE_TO_CLASS.items():
        label = _PU_CLASSES[cls_name]
        bear_dir = os.path.join(cfg.paths.pu, bid)
        if not os.path.exists(bear_dir):
            continue

        mat_files = sorted(glob.glob(os.path.join(bear_dir, "*.mat")))
        mat_files = [f for f in mat_files if cond_str in os.path.basename(f)]
        mat_files = mat_files[:max_files_per_bearing]

        for mf in mat_files:
            sig = _load_pu_signal(mf)
            if sig is None or len(sig) < window_size:
                continue
            if len(sig) > max_signal_len:
                sig = sig[:max_signal_len]
            segs = _sliding_window(sig, window_size, overlap)
            if len(segs) > 0:
                all_segments.append(segs)
                all_labels.append(np.full(len(segs), label, dtype=np.int64))

    if not all_segments:
        return np.array([]), np.array([])

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = _compute_features(segments, fft_bins)

    # 统计各类样本数
    for cls_name, cls_label in _PU_CLASSES.items():
        n = (labels == cls_label).sum()
        if n > 0:
            print(f"    {cls_name}: {n} samples")

    print(f"  总计: {len(labels)} samples")
    return features, labels


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
            print(f"  Epoch {epoch}/{epochs} | Loss: {total_loss/len(loader):.4f}")


# ── 实验 ──────────────────────────────────────────────────────────

def run_samedomain_classify(condition, fft_bins=512):
    """同工况闭集分类 (80/20 split)"""
    print(f"\n{'='*50}")
    print(f"PU 同工况闭集: 工况 {condition}")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    cfg.model.num_classes = 4
    features, labels = load_pu_condition(condition, fft_bins=fft_bins)
    if len(features) == 0:
        print("[SKIP] 无数据")
        return None

    cfg.model.freq_input_dim = features.shape[1]

    # 80/20 split
    n = len(features)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_ds = PUTensorDataset(features[train_idx], labels[train_idx])
    test_ds = PUTensorDataset(features[test_idx], labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    rpm = _PU_RPM.get(condition, 1500)
    tb = FrequencyTemplateBuilder(sample_rate=64000, fft_bins=fft_bins, bearing_params=_PU_BEARING)
    t_fft = tb.build_template(rpm=rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print("  训练中 (50 epochs)...")
    train_source(model, train_loader, M_phy, criterion, optimizer, scaler, device, epochs=50)

    # 评估
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(y.numpy())

    m = compute_closed_set_metrics(all_true, all_preds)
    print(f"  结果: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    return m


def run_cross_condition_transfer(src_cond, tgt_cond, fft_bins=512):
    """跨工况闭集迁移"""
    print(f"\n{'='*50}")
    print(f"PU 跨工况: {src_cond} → {tgt_cond}")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    cfg.model.num_classes = 4

    src_features, src_labels = load_pu_condition(src_cond, fft_bins=fft_bins)
    tgt_features, tgt_labels = load_pu_condition(tgt_cond, fft_bins=fft_bins)

    if len(src_features) == 0 or len(tgt_features) == 0:
        print("[SKIP] 数据为空")
        return None

    cfg.model.freq_input_dim = src_features.shape[1]

    src_loader = DataLoader(PUTensorDataset(src_features, src_labels),
                            batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(PUTensorDataset(tgt_features, tgt_labels),
                            batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    rpm = _PU_RPM.get(src_cond, 1500)
    tb = FrequencyTemplateBuilder(sample_rate=64000, fft_bins=fft_bins, bearing_params=_PU_BEARING)
    t_fft = tb.build_template(rpm=rpm)
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
    aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=4).to(device)
    optimizer_a = torch.optim.AdamW(
        list(model2.parameters()) + list(aligner.parameters()),
        lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
    scaler_a = GradScaler(enabled=cfg.train.fp16)

    model2.train()
    for epoch in range(1, 51):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
            optimizer_a.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model2(src_x, M_phy, return_features=True)
                tgt_feats, _ = model2.extract_features(tgt_x, M_phy)
                num_classes = int(src_y.max().item()) + 1
                prototypes = []
                for c in range(num_classes):
                    mask = src_y == c
                    prototypes.append(src_feats[mask].mean(dim=0) if mask.sum() > 0
                                      else torch.zeros(cfg.model.feature_dim, device=device))
                prototypes = torch.stack(prototypes)
                L_align, L_sep, _, _ = aligner.alignment_loss(src_feats, src_y, tgt_feats, prototypes)
                loss, _ = criterion2(src_logits, src_y, features=src_feats, attn_weights=src_attn,
                                     M_phy=M_phy, L_align=L_align, L_sep=L_sep)
            scaler_a.scale(loss).backward()
            scaler_a.step(optimizer_a)
            scaler_a.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Align Epoch {epoch}/50 | Loss: {total_loss/len(src_loader):.4f}")

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["samedomain", "transfer", "all"], default="all")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--fft-bins", type=int, default=512)
    args = parser.parse_args()

    all_results = {}

    if args.task in ("samedomain", "all"):
        for cond in args.conditions:
            r = run_samedomain_classify(cond, args.fft_bins)
            if r:
                all_results[f"samedomain_{cond}"] = r

    if args.task in ("transfer", "all"):
        # 跨工况迁移组合
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        for src, tgt in pairs:
            r = run_cross_condition_transfer(src, tgt, args.fft_bins)
            if r:
                all_results[f"transfer_{src}to{tgt}"] = r

    # 汇总
    print(f"\n{'='*60}")
    print("Paderborn 实验汇总")
    print(f"{'='*60}")
    for key, val in all_results.items():
        print(f"\n{key}:")
        if isinstance(val, dict):
            for method, m in val.items():
                print(f"  {method}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")
        elif isinstance(val, dict) and "accuracy" in val:
            print(f"  Acc={val['accuracy']:.4f}, F1={val['macro_f1']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
