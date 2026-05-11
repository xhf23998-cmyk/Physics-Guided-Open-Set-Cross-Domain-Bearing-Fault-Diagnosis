"""Step 10: 复杂度与推理效率分析"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from models import FullModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, input_shape, M_phy=None):
    """估算 FLOPs (简化: 统计乘加操作)"""
    # 使用 thop 如果可用
    try:
        from thop import profile
        dummy = torch.randn(*input_shape).to(next(model.parameters()).device)
        if M_phy is not None:
            flops, _ = profile(model, inputs=(dummy, M_phy), verbose=False)
        else:
            flops, _ = profile(model, inputs=(dummy, None), verbose=False)
        return flops
    except ImportError:
        pass

    # 手动估算
    total_flops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            out_elements = m.out_channels  # simplified
            total_flops += m.weight.numel() * 2  # multiply-add
        elif isinstance(m, torch.nn.Linear):
            total_flops += m.in_features * m.out_features * 2
    return total_flops


def measure_inference_time(model, input_shape, M_phy=None, n_runs=100, device='cuda'):
    """测量单样本推理时间"""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    if M_phy is not None:
        M_phy = M_phy.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if M_phy is not None:
                model(dummy, M_phy)
            else:
                model(dummy, None)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            if M_phy is not None:
                model(dummy, M_phy)
            else:
                model(dummy, None)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    freq_input_dim = 1024
    num_classes = 4
    feature_dim = 128

    # 构建我们的模型
    cfg.model.freq_input_dim = freq_input_dim
    cfg.model.num_classes = num_classes
    model = FullModel(cfg).to(device)

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    # 构建 M_phy
    from modules import FrequencyTemplateBuilder
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=1200)
    pad = np.zeros(freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # FLOPs
    flops = estimate_flops(model, (1, 1, freq_input_dim), M_phy)

    # 推理时间
    mean_time, std_time = measure_inference_time(model, (1, 1, freq_input_dim), M_phy, n_runs=200, device=str(device))

    print("=" * 60)
    print("复杂度分析结果")
    print("=" * 60)
    print(f"\nOurs (ResNet-1D + Physics FA):")
    print(f"  总参数量:      {total_params:,}")
    print(f"  可训练参数:    {trainable_params:,}")
    print(f"  FLOPs:         {flops:,.0f}" if flops > 0 else "  FLOPs:         (需安装thop)")
    print(f"  推理时间:      {mean_time:.3f} +/- {std_time:.3f} ms")

    # 对比方法参数量 (简化估计)
    from models.backbone import ResNet1D

    # ResNet-1D (无注意力)
    backbone = ResNet1D(feature_dim=feature_dim)
    classifier = torch.nn.Linear(feature_dim, num_classes)
    resnet_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in classifier.parameters())

    # 1D-CNN
    class Simple1DCNN(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv1d(1, 32, 7, padding=3), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(32, 64, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
                torch.nn.Conv1d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = torch.nn.Linear(128, num_classes)

        def forward(self, x, M_phy=None):
            h = self.features(x).squeeze(-1)
            return self.classifier(h), h

    cnn1d = Simple1DCNN(freq_input_dim, num_classes)
    cnn1d_params = sum(p.numel() for p in cnn1d.parameters())

    print(f"\n对比方法参数量:")
    print(f"  1D-CNN:        {cnn1d_params:,}")
    print(f"  ResNet-1D:     {resnet_params:,}")
    print(f"  Ours:          {total_params:,}")

    # DANN 额外参数 (域判别器)
    dann_disc = torch.nn.Sequential(torch.nn.Linear(feature_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    dann_extra = sum(p.numel() for p in dann_disc.parameters())
    print(f"  DANN域判别器:  +{dann_extra:,} (额外)")
    print(f"  DANN总计:      {total_params + dann_extra:,}")

    # OpenMax 无额外参数 (后处理方法)
    print(f"  OpenMax:       {total_params:,} (无额外参数)")

    # 推理时间对比
    class ResNetClassifier(torch.nn.Module):
        def __init__(self, bb, cls):
            super().__init__()
            self.bb = bb
            self.cls = cls
        def forward(self, x):
            return self.cls(self.bb(x))

    resnet_model = ResNetClassifier(backbone, classifier).to(device)
    # ResNet 推理时间
    resnet_model.eval()
    dummy = torch.randn(1, 1, freq_input_dim).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            backbone(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times_resnet = []
    with torch.no_grad():
        for _ in range(200):
            start = time.perf_counter()
            backbone(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_resnet.append(time.perf_counter() - start)
    resnet_time = np.mean(times_resnet) * 1000

    print(f"\n推理时间 (batch=1):")
    print(f"  ResNet-1D:     {resnet_time:.3f} ms")
    print(f"  Ours:          {mean_time:.3f} ms")

    # 保存结果到文件
    results = {
        "ours_params": total_params,
        "ours_flops": flops,
        "ours_time_ms": mean_time,
        "resnet_params": resnet_params,
        "cnn1d_params": cnn1d_params,
        "resnet_time_ms": resnet_time,
    }

    print(f"\n{'='*60}")
    print("Table 9 数据:")
    print(f"{'方法':<20} {'参数量':>12} {'FLOPs':>12} {'推理时间(ms)':>12} {'需源数据':>8} {'支持Unknown':>10}")
    print("-" * 76)
    print(f"{'1D-CNN':<20} {cnn1d_params:>12,} {'-':>12} {'-':>12} {'否':>8} {'否':>10}")
    print(f"{'ResNet-1D':<20} {resnet_params:>12,} {'-':>12} {resnet_time:>12.3f} {'否':>8} {'否':>10}")
    print(f"{'DANN':<20} {total_params+dann_extra:>12,} {'-':>12} {'-':>12} {'是':>8} {'否':>10}")
    print(f"{'OpenMax':<20} {total_params:>12,} {'-':>12} {mean_time:>12.3f} {'否':>8} {'是':>10}")
    print(f"{'Ours':<20} {total_params:>12,} {'-':>12} {mean_time:>12.3f} {'否':>8} {'是':>10}")


if __name__ == "__main__":
    main()
