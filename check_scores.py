import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import cfg
from data import SelfCollectedDataset
from models import FullModel
from modules import FrequencyTemplateBuilder
from torch.utils.data import DataLoader

device = torch.device('cuda')
all_classes = ['Health','Inner','Outer','Ball']
known_classes = ['Health','Inner','Outer']
cfg.model.num_classes = 3
cfg.model.freq_input_dim = 1024

target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain='W2',
    classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
    fft_bins=cfg.signal.fft_bins)
target_loader = DataLoader(target_ds, batch_size=256, shuffle=False)

template_builder = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
    bearing_params={'ball_count':cfg.bearing.ball_count,'ball_diameter':cfg.bearing.ball_diameter,
                    'pitch_diameter':cfg.bearing.pitch_diameter,'contact_angle':cfg.bearing.contact_angle})
t_fft = template_builder.build_template(rpm=1200)
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
    probs = torch.softmax(lg, dim=1)
    max_prob = probs.max(dim=1)[0]
    pred = lg.argmax(dim=1)
    pred_dist = torch.bincount(pred, minlength=3).float()
    entropy = -(probs * torch.log(probs+1e-8)).sum(1)
    energy = torch.logsumexp(lg, dim=1)
    tag = ' [UNK]' if name == 'Ball' else ' [KNOWN]'
    print(f'{name}{tag}: max_prob={max_prob.mean():.3f}+/-{max_prob.std():.3f}, '
          f'entropy={entropy.mean():.3f}+/-{entropy.std():.3f}, '
          f'energy={energy.mean():.3f}+/-{energy.std():.3f}')
    print(f'  pred dist: Health={pred_dist[0]:.0f} Inner={pred_dist[1]:.0f} Outer={pred_dist[2]:.0f}')
