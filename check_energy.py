import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import cfg
from data import SelfCollectedDataset
from models import FullModel
from modules import FrequencyTemplateBuilder
from train_cross_domain import remap_labels
from torch.utils.data import DataLoader
from utils import compute_open_set_metrics

device = torch.device('cuda')
all_classes = ['Health','Inner','Outer','Ball']
known_classes = ['Health','Inner','Outer']
cfg.model.num_classes = 3
cfg.model.freq_input_dim = 1024

target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain='W2',
    classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
    fft_bins=cfg.signal.fft_bins)
target_loader = DataLoader(target_ds, batch_size=256, shuffle=False)

tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
    bearing_params={'ball_count':cfg.bearing.ball_count,'ball_diameter':cfg.bearing.ball_diameter,
                    'pitch_diameter':cfg.bearing.pitch_diameter,'contact_angle':cfg.bearing.contact_angle})
t_fft = tb.build_template(rpm=1200)
M_phy = torch.from_numpy(np.concatenate([t_fft, np.zeros(512, dtype=np.float32)])).float().to(device)

model = FullModel(cfg).to(device)
ckpt = torch.load(os.path.join(cfg.paths.checkpoint, 'cross_domain_W1_to_W2_unk-Ball.pth'), weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_logits, all_true = [], []
with torch.no_grad():
    for x, y, _ in target_loader:
        x = x.to(device)
        logits, _ = model(x, M_phy)
        all_logits.append(logits.cpu())
        all_true.append(y)

all_logits = torch.cat(all_logits, 0)
all_true = torch.cat(all_true, 0)

for i, name in enumerate(all_classes):
    mask = all_true == i
    if mask.sum() == 0: continue
    lg = all_logits[mask]
    energy = torch.logsumexp(lg, dim=1)
    pred = lg.argmax(1)
    pred_dist = [(pred==j).sum().item() for j in range(3)]
    tag = ' [UNK]' if name=='Ball' else ''
    print(f'{name}{tag}: energy={energy.mean():.3f}+/-{energy.std():.3f}, '
          f'range=[{energy.min():.3f},{energy.max():.3f}], pred=H{pred_dist[0]} I{pred_dist[1]} O{pred_dist[2]}')

# 尝试不同阈值下的 Known/Unknown Acc
unknown_label = 3
true_labels = all_true.numpy()
known_mask = true_labels != unknown_label
unk_mask = true_labels == unknown_label
energy = torch.logsumexp(all_logits, dim=1).numpy()
preds = all_logits.argmax(dim=1).numpy()

print(f'\n不同能量阈值下的性能:')
for pct in [5, 10, 15, 20, 25, 30, 40, 50]:
    thresh = np.percentile(energy[known_mask], pct)
    p = preds.copy()
    p[energy < thresh] = -1
    # Known Acc
    k_correct = (p[known_mask] != -1).sum() if True else 0
    # Actually: Known Acc = known samples classified as known AND correctly
    k_total = known_mask.sum()
    k_correct_classify = ((p != -1) & known_mask & (p == true_labels)).sum()
    known_acc = k_correct_classify / k_total
    # Unknown Acc
    u_total = unk_mask.sum()
    u_correct = (p[unk_mask] == -1).sum()
    unk_acc = u_correct / u_total
    h = 2*known_acc*unk_acc/(known_acc+unk_acc+1e-8)
    print(f'  pct={pct:2d}% thresh={thresh:.3f}: Known={known_acc:.4f} Unknown={unk_acc:.4f} H={h:.4f}')
