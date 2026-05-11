"""Step 4: 频率注意力与故障特征频率对齐分析

验证模型的频率注意力是否真的关注了轴承故障相关频带
输出: 验证结果文本 (用于Table 3数据)
"""
import os, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from modules import FrequencyTemplateBuilder
from utils import set_seed


def compute_fault_freqs(rpm, bearing_params):
    """计算理论故障特征频率"""
    n = bearing_params["ball_count"]
    d = bearing_params["ball_diameter"]
    D = bearing_params["pitch_diameter"]
    a = bearing_params["contact_angle"]
    f_r = rpm / 60.0
    bpfo = (n / 2) * f_r * (1 - d / D * np.cos(a))
    bpfi = (n / 2) * f_r * (1 + d / D * np.cos(a))
    bsf  = (D / (2 * d)) * f_r * (1 - (d / D * np.cos(a)) ** 2)
    ftf  = (1 / 2) * f_r * (1 - d / D * np.cos(a))
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def get_freq_bins(fft_bins, sample_rate, window_size):
    """获取每个FFT bin对应的频率"""
    return np.arange(fft_bins) * sample_rate / window_size


def find_peak_alignment(attention_weights, freq_axis, target_freqs, harmonics=3):
    """检查注意力峰值是否与目标频率(含倍频)对齐

    返回: 命中率 = 注意力Top-K峰值中与目标频率±容差匹配的比例
    """
    tolerance = 10.0  # Hz 容差
    top_k = 10
    top_indices = np.argsort(attention_weights)[-top_k:]
    top_freqs = freq_axis[top_indices]

    # 构建目标频率列表 (基频 + 倍频)
    targets = []
    for name, f in target_freqs.items():
        for h in range(1, harmonics + 1):
            targets.append((name, f * h))

    hits = 0
    matched = []
    for tf_name, tf in targets:
        for pf in top_freqs:
            if abs(pf - tf) < tolerance:
                hits += 1
                matched.append((tf_name, tf, pf))
                break

    hit_rate = hits / len(targets) if targets else 0
    return hit_rate, matched, top_freqs


def main():
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    classes = ["Health", "Inner", "Outer", "Ball"]
    source_rpm = _SELF_WORKING_CONDITIONS["W1"]["speed"]

    # 加载数据
    dataset = SelfCollectedDataset(
        data_root=cfg.paths.self_collected, domain="W1",
        classes=classes, window_size=cfg.signal.window_size,
        overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins,
    )
    cfg.model.freq_input_dim = dataset.segments.shape[1]
    cfg.model.num_classes = len(classes)

    # 物理模板
    tb = FrequencyTemplateBuilder(
        sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle},
    )
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy = torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)

    # 加载模型
    model = FullModel(cfg).to(device)
    ckpt_path = os.path.join(cfg.paths.checkpoint, "best_closed_set.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"模型已加载: {ckpt_path}")
    else:
        print("[ERROR] 未找到训练好的闭集模型，请先运行 Step 3")
        return

    model.eval()

    # 理论故障频率
    bearing_params = {"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                      "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle}
    fault_freqs = compute_fault_freqs(source_rpm, bearing_params)

    # FFT频率轴
    freq_bins_fft = cfg.signal.fft_bins
    freq_axis_fft = get_freq_bins(freq_bins_fft, cfg.signal.sample_rate, cfg.signal.window_size)

    print(f"\n轴承参数: n={cfg.bearing.ball_count}, d={cfg.bearing.ball_diameter}, D={cfg.bearing.pitch_diameter}")
    print(f"转速: {source_rpm} rpm (f_r={source_rpm/60:.2f} Hz)")
    print(f"理论故障频率:")
    for name, f in fault_freqs.items():
        print(f"  {name}: {f:.1f} Hz")

    # 每个类取一个样本，提取注意力权重
    print(f"\n{'='*60}")
    print(f"频率注意力对齐分析")
    print(f"{'='*60}")

    results = {}
    for cls_idx, cls_name in enumerate(classes):
        # 找该类的第一个样本
        mask = dataset.labels == cls_idx
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0][0]
        x = torch.from_numpy(dataset.segments[idx]).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            _, attn = model(x, M_phy)

        # attn: (1, 1, F) — 全部F=1024维度
        attn_np = attn.squeeze().cpu().numpy()  # (F,)

        # 只看FFT部分 (前fft_bins个)
        attn_fft = attn_np[:freq_bins_fft]

        # 针对不同故障类，检查对应频率的对齐
        if cls_name == "Inner":
            check_freqs = {"BPFI": fault_freqs["BPFI"]}
        elif cls_name == "Outer":
            check_freqs = {"BPFO": fault_freqs["BPFO"]}
        elif cls_name == "Ball":
            check_freqs = {"BSF": fault_freqs["BSF"]}
        else:
            check_freqs = fault_freqs  # Health: 检查所有

        hit_rate, matched, top_freqs = find_peak_alignment(attn_fft, freq_axis_fft, check_freqs)

        results[cls_name] = {"hit_rate": hit_rate, "matched": matched, "top_freqs": top_freqs}

        print(f"\n[{cls_name}]")
        print(f"  检查频率: {', '.join(f'{k}={v:.1f}Hz' for k,v in check_freqs.items())}")
        print(f"  命中率: {hit_rate:.2%}")
        if matched:
            print(f"  匹配: {[(m[0], f'{m[1]:.1f}Hz', f'{m[2]:.1f}Hz') for m in matched]}")
        print(f"  Top-5注意频率: {sorted(top_freqs)[:5]}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Table 3 数据汇总")
    print(f"{'='*60}")
    avg_hit = np.mean([r["hit_rate"] for r in results.values()])
    print(f"平均Top-K频率命中率: {avg_hit:.2%}")
    print(f"\n结论: 频率注意力{'正确关注了' if avg_hit > 0.3 else '未能充分关注'}故障特征频率")


if __name__ == "__main__":
    main()
