"""Step 3: 自采同工况闭集分类训练脚本"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder
from utils import set_seed, compute_closed_set_metrics
from torch.utils.data import DataLoader, random_split


def train_one_epoch(model, loader, optimizer, criterion, scaler, M_phy, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with autocast(enabled=cfg.train.fp16):
            logits, features, attn_weights = model(x, M_phy, return_features=True)
            loss, loss_dict = criterion(logits, y, features=features,
                                        attn_weights=attn_weights, M_phy=M_phy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_closed_set_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, M_phy, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        x, y, _ = batch
        x = x.to(device)
        logits, _ = model(x, M_phy)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return compute_closed_set_metrics(all_labels, all_labels if len(all_preds) == 0 else all_preds)


def main():
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("加载自采数据集...")
    dataset = SelfCollectedDataset(
        data_root=cfg.paths.self_collected,
        domain="W1",
        classes=["Health", "Inner", "Outer", "Ball"],
        window_size=cfg.signal.window_size,
        overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins,
    )
    print(f"样本数: {len(dataset)}, 特征维度: {dataset.segments.shape[1] if len(dataset.segments) > 0 else 0}")

    if len(dataset) == 0:
        print("[ERROR] 数据集为空，请检查数据路径和格式")
        return

    # 自动检测频率维度
    cfg.model.freq_input_dim = dataset.segments.shape[1]
    print(f"频率输入维度: {cfg.model.freq_input_dim}")

    # 2. 划分训练/测试集
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                      generator=torch.Generator().manual_seed(cfg.train.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 3. 构建物理频率模板 (使用默认转速 1200 rpm)
    template_builder = FrequencyTemplateBuilder(
        sample_rate=cfg.signal.sample_rate,
        fft_bins=cfg.signal.fft_bins,
        bearing_params={
            "ball_count": cfg.bearing.ball_count,
            "ball_diameter": cfg.bearing.ball_diameter,
            "pitch_diameter": cfg.bearing.pitch_diameter,
            "contact_angle": cfg.bearing.contact_angle,
        },
    )
    template_fft = template_builder.build_template(rpm=1200)
    # 如果输入包含 FFT + 包络谱，拼接模板
    if cfg.model.freq_input_dim > cfg.signal.fft_bins:
        template_env = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
        M_phy = torch.from_numpy(np.concatenate([template_fft, template_env])).float().to(device)
    else:
        M_phy = torch.from_numpy(template_fft[:cfg.model.freq_input_dim]).float().to(device)
    print(f"物理频率模板已构建, 维度: {M_phy.shape}")

    # 4. 创建模型
    model = FullModel(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {param_count:,}")

    # 5. 损失函数和优化器
    criterion = CombinedLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = GradScaler(enabled=cfg.train.fp16)

    # 6. 训练循环
    best_acc = 0
    print(f"\n开始训练 ({cfg.train.epochs} epochs)...")
    print("-" * 60)

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, M_phy, device
        )
        test_metrics = evaluate(model, test_loader, M_phy, device)
        scheduler.step()

        elapsed = time.time() - t0

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(cfg.paths.checkpoint, "best_closed_set.pth"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{cfg.train.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']:.4f} | "
                  f"Best: {best_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")

    print("-" * 60)
    print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")
    print(f"模型已保存至: {cfg.paths.checkpoint}/best_closed_set.pth")


if __name__ == "__main__":
    main()
