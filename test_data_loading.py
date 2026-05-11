"""验证数据加载是否正常工作"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset, CWRUDataset, PUDataset


def test_self_collected():
    print("=" * 50)
    print("测试自采数据集加载...")
    try:
        ds = SelfCollectedDataset(
            data_root=cfg.paths.self_collected,
            domain="W1",
            classes=["Health", "Inner", "Outer", "Ball"],
            window_size=cfg.signal.window_size,
            overlap=cfg.signal.overlap,
            fft_bins=cfg.signal.fft_bins,
        )
        print(f"  样本数: {len(ds)}")
        if len(ds) > 0:
            x, y, raw = ds[0]
            print(f"  频域特征维度: {x.shape}")
            print(f"  标签: {y.item()}")
            print(f"  原始信号维度: {raw.shape if raw is not None else 'None'}")
            print("  [OK] 自采数据加载成功")
        else:
            print("  [WARNING] 数据集为空")
    except Exception as e:
        print(f"  [ERROR] {e}")


def test_cwru():
    print("=" * 50)
    print("测试 CWRU 数据集加载...")
    try:
        ds = CWRUDataset(
            data_root=cfg.paths.cwru,
            load_hp=0,
            window_size=cfg.signal.window_size,
            overlap=cfg.signal.overlap,
            fft_bins=cfg.signal.fft_bins,
        )
        print(f"  样本数: {len(ds)}")
        if len(ds) > 0:
            x, y, raw = ds[0]
            print(f"  频域特征维度: {x.shape}")
            print(f"  标签: {y.item()}")
            print("  [OK] CWRU 数据加载成功")
        else:
            print("  [WARNING] 数据集为空")
    except Exception as e:
        print(f"  [ERROR] {e}")


def test_pu():
    print("=" * 50)
    print("测试 Paderborn 数据集加载...")
    try:
        ds = PUDataset(
            data_root=cfg.paths.pu,
            bearing_ids=["K001", "KI01", "KA04", "KB23"],
            window_size=cfg.signal.window_size,
            overlap=cfg.signal.overlap,
            fft_bins=cfg.signal.fft_bins,
        )
        print(f"  样本数: {len(ds)}")
        if len(ds) > 0:
            x, y, raw = ds[0]
            print(f"  频域特征维度: {x.shape}")
            print(f"  标签: {y.item()}")
            print("  [OK] Paderborn 数据加载成功")
        else:
            print("  [WARNING] 数据集为空")
    except Exception as e:
        print(f"  [ERROR] {e}")


def test_model():
    print("=" * 50)
    print("测试模型前向传播...")
    import torch
    from models import FullModel
    from modules import FrequencyTemplateBuilder

    try:
        # 自动检测输入维度
        test_ds = SelfCollectedDataset(
            data_root=cfg.paths.self_collected,
            domain="W1",
            classes=["Health", "Inner", "Outer", "Ball"],
            window_size=cfg.signal.window_size,
            overlap=cfg.signal.overlap,
            fft_bins=cfg.signal.fft_bins,
        )
        if len(test_ds) > 0:
            cfg.model.freq_input_dim = test_ds.segments.shape[1]
        else:
            # CWRU fallback
            test_ds = CWRUDataset(data_root=cfg.paths.cwru, load_hp=0,
                                  window_size=cfg.signal.window_size,
                                  overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins)
            if len(test_ds) > 0:
                cfg.model.freq_input_dim = test_ds.segments.shape[1]

        model = FullModel(cfg)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {param_count:,}")

        # 测试前向传播
        B = 4
        freq_dim = cfg.model.freq_input_dim
        x = torch.randn(B, 1, freq_dim)

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
        if freq_dim > cfg.signal.fft_bins:
            template_env = np.zeros(freq_dim - cfg.signal.fft_bins, dtype=np.float32)
            M_phy = torch.from_numpy(np.concatenate([template_fft, template_env])).float()
        else:
            M_phy = torch.from_numpy(template_fft[:freq_dim]).float()

        logits, features, attn = model(x, M_phy, return_features=True)
        print(f"  logits shape: {logits.shape}")
        print(f"  features shape: {features.shape}")
        print(f"  attn shape: {attn.shape}")
        print("  [OK] 模型前向传播成功")
    except Exception as e:
        print(f"  [ERROR] {e}")


if __name__ == "__main__":
    print("数据与模型验证测试")
    print("=" * 50)
    test_self_collected()
    test_cwru()
    test_pu()
    test_model()
    print("\n" + "=" * 50)
    print("测试完成!")
