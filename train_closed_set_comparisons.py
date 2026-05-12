"""Table 2 对比方法: SVM / 1D-CNN / ResNet-1D / Ours
修复: 按信号级别分割(非窗口级别), 避免滑窗重叠导致的数据泄漏
"""
import os, sys, glob, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from models import FullModel
from models.backbone import ResNet1D as ResNet1DBackbone
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder
from utils import set_seed


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _read_csv_signal(fpath):
    data = np.loadtxt(fpath, delimiter=",", skiprows=1)
    if data.ndim == 2:
        return data[:, 0].astype(np.float32)
    return data.astype(np.float32)


def _sliding_window(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return np.array([signal[i:i+window_size] for i in range(0, len(signal)-window_size+1, step)], dtype=np.float32)


def _compute_features(segments, fft_bins):
    normed = (segments - segments.mean(axis=-1, keepdims=True)) / (segments.std(axis=-1, keepdims=True) + 1e-8)
    fft_spec = np.abs(np.fft.rfft(normed, axis=-1))[:, :fft_bins]
    envelope = np.abs(np.abs(hilbert(normed, axis=-1)))
    env_spec = np.abs(np.fft.rfft(envelope, axis=-1))[:, :fft_bins]
    return np.concatenate([fft_spec, env_spec], axis=-1).astype(np.float32)


_CLASS_MAP = {"Health": 0, "Inner": 1, "Outer": 2, "Ball": 3}
_SPEED_MAP = {"W1": 1200, "W2": 1800}
_LOAD_MAP = {"W1": 0, "W2": 50}


class TensorDS(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(0)


def load_self_collected_signal_split(domain, classes, window_size, overlap, fft_bins, train_ratio=0.8):
    """按原始信号级别分割, 然后分别做滑窗, 避免数据泄漏"""
    speed = _SPEED_MAP[domain]
    load = _LOAD_MAP[domain]
    data_root = cfg.paths.self_collected

    train_feats, train_labels = [], []
    test_feats, test_labels = [], []

    for cls_name in classes:
        label = _CLASS_MAP[cls_name]
        fpath = os.path.join(data_root, f"{cls_name}_{speed}_{load}.csv")
        if not os.path.exists(fpath):
            fpath_alt = os.path.join(data_root, f"{cls_name}_{speed} _{load}.csv")
            if os.path.exists(fpath_alt):
                fpath = fpath_alt
            else:
                continue

        signal = _read_csv_signal(fpath)
        # 信号级别分割: 前80%训练, 后20%测试 (不打乱, 保持时间顺序)
        n = len(signal)
        split = int(n * train_ratio)
        train_sig = signal[:split]
        test_sig = signal[split:]

        # 分别做滑窗 + 特征提取
        train_segs = _sliding_window(train_sig, window_size, overlap)
        test_segs = _sliding_window(test_sig, window_size, overlap)

        if len(train_segs) > 0:
            train_feats.append(_compute_features(train_segs, fft_bins))
            train_labels.append(np.full(len(train_segs), label, dtype=np.int64))
        if len(test_segs) > 0:
            test_feats.append(_compute_features(test_segs, fft_bins))
            test_labels.append(np.full(len(test_segs), label, dtype=np.int64))

    train_features = np.concatenate(train_feats)
    train_labels = np.concatenate(train_labels)
    test_features = np.concatenate(test_feats)
    test_labels = np.concatenate(test_labels)

    return train_features, train_labels, test_features, test_labels


class Simple1DCNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 7, padding=3), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, 5, padding=2), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, 3, padding=1), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x, M_phy=None):
        h = self.features(x).squeeze(-1)
        return self.classifier(h), h


def eval_model(model, test_loader, device, M_phy=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            out = model(x, M_phy)
            logits = out[0] if isinstance(out, tuple) else out
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(y.numpy())
    return compute_metrics(trues, preds)


def run_comparison(domain="W1"):
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    classes = ["Health", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(classes)
    cfg.model.freq_input_dim = 1024  # FFT 512 + Envelope 512

    # 信号级别分割加载
    print(f"加载数据 (信号级别分割, {domain}→{domain})...")
    train_features, train_labels, test_features, test_labels = load_self_collected_signal_split(
        domain, classes, cfg.signal.window_size, cfg.signal.overlap, cfg.signal.fft_bins, train_ratio=0.8)

    print(f"  训练集: {len(train_features)} 样本, 测试集: {len(test_features)} 样本")
    print(f"  训练集类别: {np.bincount(train_labels)}")
    print(f"  测试集类别: {np.bincount(test_labels)}")

    train_ds = TensorDS(train_features, train_labels)
    test_ds = TensorDS(test_features, test_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    rpm = _SPEED_MAP[domain]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=rpm)
    pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    results = {}

    # 1. SVM
    print("\n[1/4] SVM...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, train_labels)
    results["SVM"] = compute_metrics(test_labels, svm.predict(X_test))
    print(f"  Acc={results['SVM']['accuracy']:.4f}, F1={results['SVM']['macro_f1']:.4f}")

    # 2. 1D-CNN
    print("\n[2/4] 1D-CNN...")
    criterion = torch.nn.CrossEntropyLoss()
    cnn = Simple1DCNN(cfg.model.freq_input_dim, cfg.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler_amp = GradScaler(enabled=cfg.train.fp16)
    cnn.train()
    for epoch in range(1, 51):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, _ = cnn(x)
                loss = criterion(logits, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
    results["1D-CNN"] = eval_model(cnn, test_loader, device)
    print(f"  Acc={results['1D-CNN']['accuracy']:.4f}, F1={results['1D-CNN']['macro_f1']:.4f}")

    # 3. ResNet-1D (无注意力)
    print("\n[3/4] ResNet-1D...")
    resnet = ResNet1DBackbone(feature_dim=cfg.model.feature_dim).to(device)
    cls_head = torch.nn.Linear(cfg.model.feature_dim, cfg.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(list(resnet.parameters()) + list(cls_head.parameters()),
                                   lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler_amp = GradScaler(enabled=cfg.train.fp16)
    for epoch in range(1, 51):
        resnet.train(); cls_head.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                feat = resnet(x)
                logits = cls_head(feat)
                loss = criterion(logits, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

    class ResNetWrapper(torch.nn.Module):
        def __init__(self, bb, cls_h): super().__init__(); self.bb = bb; self.cls = cls_h
        def forward(self, x, M_phy=None): return self.cls(self.bb(x)), self.bb(x)
    results["ResNet-1D"] = eval_model(ResNetWrapper(resnet, cls_head), test_loader, device)
    print(f"  Acc={results['ResNet-1D']['accuracy']:.4f}, F1={results['ResNet-1D']['macro_f1']:.4f}")

    # 4. Ours
    print("\n[4/4] Ours (Physics FA + ResNet)...")
    model = FullModel(cfg).to(device)
    criterion_ours = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler_amp = GradScaler(enabled=cfg.train.fp16)
    model.train()
    for epoch in range(1, 51):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy, return_features=True)
                loss, _ = criterion_ours(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
    results["Ours"] = eval_model(model, test_loader, device, M_phy)
    print(f"  Acc={results['Ours']['accuracy']:.4f}, F1={results['Ours']['macro_f1']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Table 2 ({domain}→{domain}, 信号级别分割)")
    print(f"{'='*60}")
    print(f"{'方法':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")

    return results


if __name__ == "__main__":
    run_comparison("W1")
