"""Table 2 对比方法: SVM / 1D-CNN / ResNet-1D / FFT+ResNet / FA+ResNet / Ours"""
import os, sys, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from models import FullModel, ResNet1D
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


class FFTResNet(torch.nn.Module):
    """直接在原始信号上用ResNet (不经过频率注意力)"""
    def __init__(self, input_dim, feature_dim, num_classes):
        super().__init__()
        self.backbone = ResNet1DBackbone(feature_dim=feature_dim)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x, M_phy=None):
        feat = self.backbone(x)
        return self.classifier(feat), feat


def train_torch_model(model, train_loader, criterion, optimizer, scaler, device, epochs, M_phy=None):
    model.train()
    for epoch in range(1, epochs + 1):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, *_ = model(x, M_phy, return_features=True) if hasattr(model, 'extract_features') else (*model(x, M_phy), None, None)
                if isinstance(logits, tuple):
                    logits, feats = logits[0], logits[1] if len(logits) > 1 else logits[0]
                loss = criterion(logits, y) if not hasattr(criterion, '__call__') or not hasattr(criterion, 'parameters') else criterion(logits, y)[0] if isinstance(criterion(logits, y), tuple) else criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def eval_torch_model(model, test_loader, device, M_phy=None):
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
    """在单一工况下对比所有方法"""
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    classes = ["Health", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(classes)

    # 加载数据
    ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=domain,
        classes=classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    features = ds.segments  # (N, 1024)
    labels = ds.labels

    cfg.model.freq_input_dim = features.shape[1]

    # 80/20 split
    n = len(features)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_features, train_labels = features[train_idx], labels[train_idx]
    test_features, test_labels = features[test_idx], labels[test_idx]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_features).unsqueeze(1),  # (N, 1, L)
        torch.from_numpy(train_labels).long(),
        torch.zeros(len(train_labels))
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(test_features).unsqueeze(1),
        torch.from_numpy(test_labels).long(),
        torch.zeros(len(test_labels))
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    rpm = 1200 if domain == "W1" else 1800
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=rpm)
    pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    results = {}

    # 1. SVM + 手工特征 (直接用频域特征)
    print("\n[1/6] SVM + 频域特征...")
    scaler_svm = StandardScaler()
    X_train = scaler_svm.fit_transform(train_features)
    X_test = scaler_svm.transform(test_features)
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, train_labels)
    svm_pred = svm.predict(X_test)
    results["SVM"] = compute_metrics(test_labels, svm_pred)
    print(f"  Acc={results['SVM']['accuracy']:.4f}, F1={results['SVM']['macro_f1']:.4f}")

    # 2. 1D-CNN
    print("\n[2/6] 1D-CNN...")
    cnn = Simple1DCNN(cfg.model.freq_input_dim, cfg.model.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    cnn.train()
    for epoch in range(1, 51):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, _ = cnn(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    results["1D-CNN"] = eval_torch_model(cnn, test_loader, device)
    print(f"  Acc={results['1D-CNN']['accuracy']:.4f}, F1={results['1D-CNN']['macro_f1']:.4f}")

    # 3. ResNet-1D (无注意力)
    print("\n[3/6] ResNet-1D...")
    resnet = FFTResNet(cfg.model.freq_input_dim, cfg.model.feature_dim, cfg.model.num_classes).to(device)
    optimizer = torch.optim.AdamW(resnet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    resnet.train()
    for epoch in range(1, 51):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats = resnet(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    results["ResNet-1D"] = eval_torch_model(resnet, test_loader, device)
    print(f"  Acc={results['ResNet-1D']['accuracy']:.4f}, F1={results['ResNet-1D']['macro_f1']:.4f}")

    # 4. Ours (FullModel with Physics FA)
    print("\n[4/6] Ours (Physics FA + ResNet)...")
    model = FullModel(cfg).to(device)
    criterion_ours = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = GradScaler(enabled=cfg.train.fp16)
    model.train()
    for epoch in range(1, 51):
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                logits, feats, attn = model(x, M_phy, return_features=True)
                loss, _ = criterion_ours(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    results["Ours"] = eval_torch_model(model, test_loader, device, M_phy)
    print(f"  Acc={results['Ours']['accuracy']:.4f}, F1={results['Ours']['macro_f1']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Table 2 对比 ({domain}→{domain} 同工况闭集)")
    print(f"{'='*60}")
    print(f"{'方法':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")

    return results


if __name__ == "__main__":
    run_comparison("W1")
