"""EVT分数分布诊断 — 找出 Unknown Acc=0 的根因"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from models import FullModel, EVTHead
from modules import FrequencyTemplateBuilder

# 直接导入 train_cross_domain 中的工具函数
from train_cross_domain import remap_labels, extract_all_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 配置 ──
source_domain, target_domain, unknown_class = "W1", "W2", "Ball"
all_classes = ["Health", "Inner", "Outer", "Ball"]
known_classes = [c for c in all_classes if c != unknown_class]
cfg.model.num_classes = len(known_classes)
cfg.model.freq_input_dim = 1024

# ── 数据 ──
source_ds = SelfCollectedDataset(
    data_root=cfg.paths.self_collected, domain=source_domain,
    classes=known_classes, window_size=cfg.signal.window_size,
    overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins,
)
target_ds = SelfCollectedDataset(
    data_root=cfg.paths.self_collected, domain=target_domain,
    classes=all_classes, window_size=cfg.signal.window_size,
    overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins,
)
source_ds.labels = remap_labels(source_ds.labels, known_classes)

from torch.utils.data import DataLoader
source_loader = DataLoader(source_ds, batch_size=128, shuffle=False)
target_loader = DataLoader(target_ds, batch_size=128, shuffle=False)

# ── 物理模板 ──
template_builder = FrequencyTemplateBuilder(
    sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
    bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                    "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle},
)
t_fft = template_builder.build_template(rpm=1200)
t_env = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
M_phy = torch.from_numpy(np.concatenate([t_fft, t_env])).float().to(device)

# ── 加载模型 ──
model = FullModel(cfg).to(device)
tag = f"{source_domain}_to_{target_domain}_unk-{unknown_class}"
ckpt = torch.load(os.path.join(cfg.paths.checkpoint, f"cross_domain_{tag}.pth"), weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# ── 提取特征 ──
src_feats, src_labels = extract_all_features(model, source_loader, M_phy, device, known_classes)
tgt_feats_all, tgt_labels_all = extract_all_features(model, target_loader, M_phy, device)

print(f"源域特征: {src_feats.shape}, 标签范围: [{src_labels.min()}-{src_labels.max()}]")
print(f"目标域特征: {tgt_feats_all.shape}")

# ── 拟合 EVT ──
evt = EVTHead(tail_size=0.15)
evt.fit(src_feats, src_labels)

# ── 分析 ──
scores, nearest = evt.compute_unknown_score(tgt_feats_all)
scores = scores.numpy()
nearest = nearest.numpy()

unknown_orig_label = all_classes.index(unknown_class)
true_labels = tgt_labels_all.numpy()
is_unknown = (true_labels == unknown_orig_label)
is_known = ~is_unknown

print(f"\n=== 分数分布 ===")
print(f"已知样本 (n={is_known.sum()}):")
print(f"  score: mean={scores[is_known].mean():.4f}, std={scores[is_known].std():.4f}")
print(f"  min={scores[is_known].min():.4f}, max={scores[is_known].max():.4f}")
print(f"  percentiles: 50th={np.percentile(scores[is_known],50):.4f}, "
      f"90th={np.percentile(scores[is_known],90):.4f}, "
      f"95th={np.percentile(scores[is_known],95):.4f}")

print(f"\n未知样本 (n={is_unknown.sum()}):")
print(f"  score: mean={scores[is_unknown].mean():.4f}, std={scores[is_unknown].std():.4f}")
print(f"  min={scores[is_unknown].min():.4f}, max={scores[is_unknown].max():.4f}")
print(f"  percentiles: 50th={np.percentile(scores[is_unknown],50):.4f}, "
      f"90th={np.percentile(scores[is_unknown],90):.4f}, "
      f"95th={np.percentile(scores[is_unknown],95):.4f}")

# 按类别
print(f"\n=== 各类别分数 ===")
for i, name in enumerate(all_classes):
    mask = true_labels == i
    s = scores[mask]
    tag = " [UNK]" if i == unknown_orig_label else " [KNOWN]"
    print(f"  {name}{tag}: mean={s.mean():.4f}, std={s.std():.4f}")

# 自动阈值
auto_thresh = evt._auto_threshold()
print(f"\n自动阈值: {auto_thresh:.4f}")
print(f"已知样本被误判为未知: {(scores[is_known] > auto_thresh).sum()}/{is_known.sum()} "
      f"= {(scores[is_known] > auto_thresh).mean():.4f}")
print(f"未知样本被正确拒识: {(scores[is_unknown] > auto_thresh).sum()}/{is_unknown.sum()} "
      f"= {(scores[is_unknown] > auto_thresh).mean():.4f}")

# 特征空间统计
print(f"\n=== 特征空间 ===")
for i, name in enumerate(known_classes):
    mask = src_labels == i
    proto = src_feats[mask].mean(0)
    dists = torch.norm(src_feats[mask] - proto, dim=1)
    print(f"  源域 {name}: proto_norm={proto.norm():.3f}, "
          f"intra_dist mean={dists.mean():.3f}, max={dists.max():.3f}")
