"""诊断EVT分数分布，找出Unknown Acc为0的原因"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from models import FullModel, EVTHead
from modules import FrequencyTemplateBuilder
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载已训练的模型
source_domain, target_domain, unknown_class = "W1", "W2", "Inner"
all_classes = ["Health", "Inner", "Outer", "Ball"]
known_classes = [c for c in all_classes if c != unknown_class]
unknown_label = all_classes.index(unknown_class)

cfg.model.freq_input_dim = 1024

# 物理模板
template_builder = FrequencyTemplateBuilder(
    sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
    bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                    "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle},
)
template_fft = template_builder.build_template(rpm=1200)
template_env = np.zeros(cfg.model.freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
M_phy = torch.from_numpy(np.concatenate([template_fft, template_env])).float().to(device)

# 数据
source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
                                  classes=known_classes, window_size=cfg.signal.window_size,
                                  overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins)
target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
                                  classes=all_classes, window_size=cfg.signal.window_size,
                                  overlap=cfg.signal.overlap, fft_bins=cfg.signal.fft_bins)

source_loader = DataLoader(source_ds, batch_size=128, shuffle=False)
target_loader = DataLoader(target_ds, batch_size=128, shuffle=False)

# 从checkpoint加载
model = FullModel(cfg).to(device)
ckpt = torch.load(os.path.join(cfg.paths.checkpoint, f"cross_domain_{source_domain}_to_{target_domain}.pth"),
                   weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def extract_feats(loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            f, _ = model.extract_features(x, M_phy)
            feats.append(f.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

print("提取特征...")
src_feats, src_labels = extract_feats(source_loader)
tgt_feats, tgt_labels = extract_feats(target_loader)

print(f"源域: {src_feats.shape[0]} samples")
print(f"目标域: {tgt_feats.shape[0]} samples")

# 计算类原型
prototypes = {}
for c in range(len(known_classes)):
    mask = src_labels == c
    prototypes[c] = src_feats[mask].mean(dim=0)
    print(f"  类{c}({known_classes[c]}): {mask.sum()} samples, proto_norm={prototypes[c].norm():.3f}")

# 计算距离
dist_to_proto = []
for i in range(len(tgt_feats)):
    min_d = float('inf')
    for c in prototypes:
        d = (tgt_feats[i] - prototypes[c]).norm().item()
        min_d = min(min_d, d)
    dist_to_proto.append(min_d)
dist_to_proto = np.array(dist_to_proto)

# 按类别分析距离
print("\n目标域各类距离统计:")
true_labels = tgt_labels.numpy()
for c_idx, c_name in enumerate(all_classes):
    mask = true_labels == c_idx
    if mask.sum() == 0:
        continue
    d = dist_to_proto[mask]
    is_unk = " [UNKNOWN]" if c_idx == unknown_label else ""
    print(f"  {c_name}{is_unk}: mean={d.mean():.3f}, std={d.std():.3f}, min={d.min():.3f}, max={d.max():.3f}")

# 源域距离分布 (用于Weibull拟合参考)
print("\n源域各类距离统计:")
src_dist = []
for i in range(len(src_feats)):
    c = src_labels[i].item()
    d = (src_feats[i] - prototypes[c]).norm().item()
    src_dist.append(d)
src_dist = np.array(src_dist)
for c_idx, c_name in enumerate(known_classes):
    mask = src_labels.numpy() == c_idx
    d = src_dist[mask]
    print(f"  {c_name}: mean={d.mean():.3f}, std={d.std():.3f}, 90th={np.percentile(d, 90):.3f}, 99th={np.percentile(d, 99):.3f}")

# 分析特征空间
print(f"\n特征空间统计:")
print(f"  源域特征norm: mean={src_feats.norm(dim=1).mean():.3f}, std={src_feats.norm(dim=1).std():.3f}")
print(f"  目标域特征norm: mean={tgt_feats.norm(dim=1).mean():.3f}, std={tgt_feats.norm(dim=1).std():.3f}")

# 检查特征是否坍缩
print(f"  源域特征总方差: {src_feats.var(dim=0).sum():.3f}")
print(f"  目标域特征总方差: {tgt_feats.var(dim=0).sum():.3f}")
