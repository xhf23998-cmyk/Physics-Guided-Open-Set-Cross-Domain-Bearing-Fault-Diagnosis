"""Step 9: CWRU benchmark 实验 (跨负载迁移 + 开集 + 抗噪声)"""
import os
import sys
import glob
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner
from utils import set_seed, compute_closed_set_metrics, compute_open_set_metrics


def _remap_cwru_labels(labels, known_classes):
    """CWRU 标签重映射"""
    label_map = {_CWRU_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    return np.array([label_map.get(int(l), -1) for l in labels], dtype=np.int64)


# ── CWRU 数据加载 ─────────────────────────────────────────────────

_CWRU_CLASS_MAP = {"Normal": 0, "Inner": 1, "Outer": 2, "Ball": 3}

# CWRU RPM 与负载映射
_CWRU_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def _load_cwru_mat(fpath):
    """从 CWRU .mat 文件加载振动信号"""
    mat = loadmat(fpath)
    for key in mat:
        if key.startswith("X") and "_DE_time" in key:
            return mat[key].flatten().astype(np.float32)
    for key in mat:
        if not key.startswith("__"):
            data = mat[key]
            if isinstance(data, np.ndarray) and data.size > 100:
                return data.flatten().astype(np.float32)
    raise ValueError(f"无法从 {fpath} 读取信号")


def _sliding_window(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return np.array([signal[i:i + window_size] for i in range(0, len(signal) - window_size + 1, step)], dtype=np.float32)


def _compute_features(segments, fft_bins):
    """FFT + 包络谱"""
    normed = (segments - segments.mean(axis=-1, keepdims=True)) / (segments.std(axis=-1, keepdims=True) + 1e-8)
    fft_spec = np.abs(np.fft.rfft(normed, axis=-1))[:, :fft_bins]
    envelope = np.abs(np.abs(hilbert(normed, axis=-1)))
    env_spec = np.abs(np.fft.rfft(envelope, axis=-1))[:, :fft_bins]
    return np.concatenate([fft_spec, env_spec], axis=-1).astype(np.float32)


def load_cwru_dataset(data_root, load_hp, classes=None, window_size=1024, overlap=0.5, fft_bins=512,
                      fault_size="0007", outer_position="Centered"):
    """加载指定负载的 CWRU 数据

    Args:
        data_root: CWRU 数据根目录
        load_hp: 负载 (0/1/2/3)
        classes: 要加载的类别，默认全部4类
        fault_size: 故障尺寸目录 (0007/0014/0021/0028)
        outer_position: 外圈故障位置 (Centered/Opposite/Orthogonal)
    """
    if classes is None:
        classes = ["Normal", "Inner", "Outer", "Ball"]

    all_segments = []
    all_labels = []

    for cls_name in classes:
        label = _CWRU_CLASS_MAP[cls_name]
        signals = []

        if cls_name == "Normal":
            fpath = os.path.join(data_root, "Normal Baseline", f"normal_{load_hp}.mat")
            if os.path.exists(fpath):
                signals.append(_load_cwru_mat(fpath))
        else:
            fault_base = os.path.join(data_root, "12k Drive End Bearing Fault Data")
            if cls_name == "Inner":
                search_dir = os.path.join(fault_base, "Inner Race", fault_size)
                pattern = f"IR*_{load_hp}.mat"
            elif cls_name == "Outer":
                search_dir = os.path.join(fault_base, "Outer Race", outer_position, fault_size)
                pattern = f"OR*_{load_hp}.mat"
            elif cls_name == "Ball":
                search_dir = os.path.join(fault_base, "Ball", fault_size)
                pattern = f"B*_{load_hp}.mat"
            else:
                continue

            if os.path.exists(search_dir):
                for fpath in sorted(glob.glob(os.path.join(search_dir, pattern))):
                    signals.append(_load_cwru_mat(fpath))

        for sig in signals:
            segs = _sliding_window(sig, window_size, overlap)
            if len(segs) > 0:
                all_segments.append(segs)
                all_labels.append(np.full(len(segs), label, dtype=np.int64))

    if not all_segments:
        return np.array([]), np.array([])

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    features = _compute_features(segments, fft_bins)
    return features, labels


class CWRUTensorDataset(Dataset):
    """CWRU Tensor Dataset wrapper"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, torch.tensor(0)


# ── 训练/评估函数 ──────────────────────────────────────────────────

def train_source_only(model, loader, M_phy, criterion, optimizer, scaler, device, epochs):
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


def train_with_alignment(model, src_loader, tgt_loader, M_phy, aligner, criterion, optimizer, scaler, device, epochs):
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
                    prototypes.append(src_feats[mask].mean(dim=0) if mask.sum() > 0 else torch.zeros(cfg.model.feature_dim, device=device))
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


def add_noise(signal, snr_db):
    """给信号加高斯噪声"""
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise.astype(np.float32)


# ── 实验 ──────────────────────────────────────────────────────────

def run_closed_set_transfer(source_load, target_load, fft_bins=512):
    """跨负载闭集迁移"""
    print(f"\n{'='*50}")
    print(f"CWRU 跨负载迁移: {source_load}hp → {target_load}hp")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    classes = ["Normal", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(classes)

    src_features, src_labels = load_cwru_dataset(cfg.paths.cwru, source_load, classes=classes, fft_bins=fft_bins)
    tgt_features, tgt_labels = load_cwru_dataset(cfg.paths.cwru, target_load, classes=classes, fft_bins=fft_bins)

    if len(src_features) == 0 or len(tgt_features) == 0:
        print("[SKIP] 数据为空")
        return None

    cfg.model.freq_input_dim = src_features.shape[1]
    src_ds = CWRUTensorDataset(src_features, src_labels)
    tgt_ds = CWRUTensorDataset(tgt_features, tgt_labels)

    src_loader = DataLoader(src_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = _CWRU_RPM[source_load]
    tb = FrequencyTemplateBuilder(sample_rate=12000, fft_bins=fft_bins,
        bearing_params={"ball_count": 9, "ball_diameter": 7.94, "pitch_diameter": 39.04, "contact_angle": 0.0})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 方法列表
    results = {}

    # 1. Source Only
    print("\n[Source Only]")
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    train_source_only(model, src_loader, M_phy, criterion, optimizer, scaler, device, epochs=50)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model(x, M_phy)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_true.extend(y.numpy())
    m = compute_closed_set_metrics(all_true, all_preds)
    results["Source Only"] = m
    print(f"  Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")

    # 2. Ours (with alignment)
    print("\n[Ours (Selective Alignment)]")
    model2 = FullModel(cfg).to(device)
    criterion2 = CombinedLoss(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler2 = GradScaler(enabled=cfg.train.fp16)

    # 阶段1: 源域预训练
    print("  阶段1: 源域预训练...")
    train_source_only(model2, src_loader, M_phy, criterion2, optimizer2, scaler2, device, epochs=30)

    # 阶段2: 选择性对齐
    print("  阶段2: 选择性对齐...")
    aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=cfg.model.num_classes).to(device)
    optimizer_align = torch.optim.AdamW(
        list(model2.parameters()) + list(aligner.parameters()),
        lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
    scaler_align = GradScaler(enabled=cfg.train.fp16)
    train_with_alignment(model2, src_loader, tgt_loader, M_phy, aligner, criterion2, optimizer_align, scaler_align, device, epochs=50)

    model2.eval()
    all_preds2, all_true2 = [], []
    with torch.no_grad():
        for x, y, _ in tgt_loader:
            x = x.to(device)
            logits, _ = model2(x, M_phy)
            all_preds2.extend(logits.argmax(dim=1).cpu().numpy())
            all_true2.extend(y.numpy())
    m2 = compute_closed_set_metrics(all_true2, all_preds2)
    results["Ours"] = m2
    print(f"  Acc={m2['accuracy']:.4f}, F1={m2['macro_f1']:.4f}")

    return results


def run_openset(source_load, target_load, unknown_class, fft_bins=512):
    """跨负载开集诊断"""
    print(f"\n{'='*50}")
    print(f"CWRU 开集: {source_load}hp → {target_load}hp, Unknown={unknown_class}")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Normal", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    src_features, src_labels = load_cwru_dataset(cfg.paths.cwru, source_load, classes=known_classes, fft_bins=fft_bins)
    src_labels = _remap_cwru_labels(src_labels, known_classes)

    tgt_features, tgt_labels = load_cwru_dataset(cfg.paths.cwru, target_load, classes=all_classes, fft_bins=fft_bins)

    if len(src_features) == 0 or len(tgt_features) == 0:
        print("[SKIP] 数据为空")
        return None

    cfg.model.freq_input_dim = src_features.shape[1]
    src_ds = CWRUTensorDataset(src_features, src_labels)
    tgt_ds = CWRUTensorDataset(tgt_features, tgt_labels)

    src_loader = DataLoader(src_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = _CWRU_RPM[source_load]
    tb = FrequencyTemplateBuilder(sample_rate=12000, fft_bins=fft_bins,
        bearing_params={"ball_count": 9, "ball_diameter": 7.94, "pitch_diameter": 39.04, "contact_angle": 0.0})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 源域预训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print("  源域预训练 (80 epochs)...")
    train_source_only(model, src_loader, M_phy, criterion, optimizer, scaler, device, epochs=80)

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

    # 真标签
    unknown_orig = _CWRU_CLASS_MAP[unknown_class]
    label_map = {_CWRU_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}
    true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1) for l in tgt_true.numpy()])

    # 阈值扫描
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
    print(f"  最佳 (pct={pct}%): Known Acc={m['known_acc']:.4f}, Unknown Acc={m['unknown_acc']:.4f}, "
          f"H-score={m['h_score']:.4f}, AUROC={m['auroc']:.4f}")
    return m


def run_noise_robustness(test_load, snr_levels, fft_bins=512):
    """抗噪声鲁棒性测试"""
    print(f"\n{'='*50}")
    print(f"CWRU 抗噪声: load={test_load}hp, SNR={snr_levels}dB")
    print(f"{'='*50}")

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    classes = ["Normal", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(classes)

    # 用 clean 数据训练
    train_features, train_labels = load_cwru_dataset(cfg.paths.cwru, test_load, classes=classes, fft_bins=fft_bins)
    if len(train_features) == 0:
        print("[SKIP] 数据为空")
        return None

    cfg.model.freq_input_dim = train_features.shape[1]

    # 80/20 划分
    n_train = int(0.8 * len(train_features))
    indices = np.random.permutation(len(train_features))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train_ds = CWRUTensorDataset(train_features[train_idx], train_labels[train_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # 物理模板
    source_rpm = _CWRU_RPM[test_load]
    tb = FrequencyTemplateBuilder(sample_rate=12000, fft_bins=fft_bins,
        bearing_params={"ball_count": 9, "ball_diameter": 7.94, "pitch_diameter": 39.04, "contact_angle": 0.0})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 训练
    model = FullModel(cfg).to(device)
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)

    print("  训练中 (50 epochs)...")
    train_source_only(model, train_loader, M_phy, criterion, optimizer, scaler, device, epochs=50)

    # 在不同 SNR 下测试
    model.eval()
    results = {}
    # 加载原始信号用于加噪
    raw_signals = {}
    for cls_name in classes:
        fpath = os.path.join(cfg.paths.cwru, "Normal Baseline", f"normal_{test_load}.mat") if cls_name == "Normal" else None
        if cls_name != "Normal":
            fault_base = os.path.join(cfg.paths.cwru, "12k Drive End Bearing Fault Data")
            if cls_name == "Inner":
                search = os.path.join(fault_base, "Inner Race", "0007")
                pattern = f"IR*_{test_load}.mat"
            elif cls_name == "Outer":
                search = os.path.join(fault_base, "Outer Race", "Centered", "0007")
                pattern = f"OR*_{test_load}.mat"
            else:
                search = os.path.join(fault_base, "Ball", "0007")
                pattern = f"B*_{test_load}.mat"
            files = sorted(glob.glob(os.path.join(search, pattern)))
            fpath = files[0] if files else None
        if fpath and os.path.exists(fpath):
            raw_signals[cls_name] = _load_cwru_mat(fpath)

    for snr_db in snr_levels:
        print(f"\n  SNR = {snr_db} dB:")
        all_features, all_labels = [], []
        for cls_name in classes:
            if cls_name not in raw_signals:
                continue
            label = _CWRU_CLASS_MAP[cls_name]
            noisy = add_noise(raw_signals[cls_name], snr_db)
            segs = _sliding_window(noisy, cfg.signal.window_size, cfg.signal.overlap)
            if len(segs) > 0:
                feats = _compute_features(segs, fft_bins)
                all_features.append(feats)
                all_labels.append(np.full(len(segs), label, dtype=np.int64))

        if not all_features:
            continue
        test_features = np.concatenate(all_features)
        test_labels = np.concatenate(all_labels)
        test_ds = CWRUTensorDataset(test_features, test_labels)
        test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device)
                logits, _ = model(x, M_phy)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_true.extend(y.numpy())

        m = compute_closed_set_metrics(all_true, all_preds)
        results[snr_db] = m
        print(f"    Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")

    return results


# ── 主函数 ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["transfer", "openset", "noise", "all"], default="all")
    parser.add_argument("--source-load", type=int, default=0)
    parser.add_argument("--target-load", type=int, default=None)
    parser.add_argument("--unknown", default="Ball")
    parser.add_argument("--snr", nargs="+", type=int, default=[12, 6, 0])
    parser.add_argument("--fft-bins", type=int, default=512)
    args = parser.parse_args()

    fft_bins = args.fft_bins
    all_results = {}

    if args.task in ("transfer", "all"):
        # 跨负载迁移: 0 → 1, 2, 3
        for tgt in [1, 2, 3]:
            r = run_closed_set_transfer(args.source_load, tgt, fft_bins)
            if r:
                all_results[f"transfer_{args.source_load}to{tgt}"] = r

    if args.task in ("openset", "all"):
        # 开集: Unknown=Ball/Inner/Outer
        for unk in ["Ball", "Inner", "Outer"]:
            r = run_openset(args.source_load, args.source_load, unk, fft_bins)
            if r:
                all_results[f"openset_{unk}"] = r

    if args.task in ("noise", "all"):
        r = run_noise_robustness(args.source_load, args.snr, fft_bins)
        if r:
            all_results["noise"] = r

    # 汇总
    print("\n" + "=" * 60)
    print("CWRU 实验汇总")
    print("=" * 60)
    for key, val in all_results.items():
        print(f"\n{key}:")
        if isinstance(val, dict):
            for method, m in val.items():
                if isinstance(m, dict) and "accuracy" in m:
                    print(f"  {method}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}")
                elif isinstance(m, dict) and "known_acc" in m:
                    print(f"  {method}: Known={m['known_acc']:.4f}, Unk={m['unknown_acc']:.4f}, H={m['h_score']:.4f}")
        elif isinstance(val, dict) and "accuracy" in val:
            print(f"  Acc={val['accuracy']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
