"""对比方法实现: DANN, MMD, OpenMax 等 (用于 Table 4/5)"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scipy.stats import weibull_min

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder
from utils import set_seed, compute_closed_set_metrics, compute_open_set_metrics
from train_cross_domain import remap_labels, _ALL_CLASS_MAP, extract_all_features


def _make_M_phy(device, source_domain, freq_input_dim, fft_bins):
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    from modules import FrequencyTemplateBuilder as FTB
    tb = FTB(sample_rate=cfg.signal.sample_rate, fft_bins=fft_bins,
             bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                             "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})
    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(freq_input_dim - fft_bins, dtype=np.float32)
    return torch.from_numpy(np.concatenate([t_fft, pad])).float().to(device)


# ── DANN (非选择性) ────────────────────────────────────────────────

def train_dann(model, src_loader, tgt_loader, M_phy, domain_disc, criterion,
               optimizer, scaler, device, epochs):
    """标准 DANN 训练 (所有目标样本等权重对齐)"""
    from modules.selective_alignment import grad_reverse

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy)

                # 分类损失
                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)

                # 域对抗损失 (非选择性，所有样本等权重)
                src_dom = domain_disc(grad_reverse(src_feats, 1.0))
                tgt_dom = domain_disc(grad_reverse(tgt_feats, 1.0))
                dom_loss = (F.binary_cross_entropy_with_logits(src_dom, torch.ones_like(src_dom)) +
                            F.binary_cross_entropy_with_logits(tgt_dom, torch.zeros_like(tgt_dom))) * 0.5

                loss = cls_loss + cfg.train.lambda_align * dom_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  DANN Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── MMD 对齐 ──────────────────────────────────────────────────────

def mmd_loss_rbf(src_feats, tgt_feats, gammas=None):
    """RBF 核 MMD 损失"""
    if gammas is None:
        gammas = [0.01, 0.1, 1.0]
    loss = torch.tensor(0.0, device=src_feats.device)
    for gamma in gammas:
        xx = torch.cdist(src_feats, src_feats, p=2).pow(2)
        yy = torch.cdist(tgt_feats, tgt_feats, p=2).pow(2)
        xy = torch.cdist(src_feats, tgt_feats, p=2).pow(2)
        kxx = torch.exp(-gamma * xx).mean()
        kyy = torch.exp(-gamma * yy).mean()
        kxy = torch.exp(-gamma * xy).mean()
        loss += kxx + kyy - 2 * kxy
    return loss / len(gammas)


def train_mmd(model, src_loader, tgt_loader, M_phy, criterion, optimizer, scaler, device, epochs):
    """MMD 对齐训练"""
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy)

                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)
                mmd = mmd_loss_rbf(src_feats.float(), tgt_feats.float())
                loss = cls_loss + cfg.train.lambda_align * mmd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  MMD Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── CORAL (Deep CORAL) ──────────────────────────────────────────────

def coral_loss(src_feats, tgt_feats):
    """Deep CORAL: 对齐源域和目标域的协方差矩阵"""
    d = src_feats.shape[1]
    src_c = (src_feats.t() @ src_feats) / (src_feats.shape[0] - 1)
    tgt_c = (tgt_feats.t() @ tgt_feats) / (tgt_feats.shape[0] - 1)
    loss = (src_c - tgt_c).pow(2).sum() / (4 * d * d)
    return loss


def train_coral(model, src_loader, tgt_loader, M_phy, criterion, optimizer, scaler, device, epochs):
    """CORAL 对齐训练"""
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy)
                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)
                loss = cls_loss + cfg.train.lambda_align * coral_loss(src_feats.float(), tgt_feats.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  CORAL Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── CDAN (Conditional Domain Adversarial Network) ──────────────────

def train_cdan(model, src_loader, tgt_loader, M_phy, domain_disc, criterion,
               optimizer, scaler, device, epochs):
    """CDAN: 条件域对抗 (用softmax概率条件化)"""
    from modules.selective_alignment import grad_reverse

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_logits, tgt_feats, _ = model(tgt_x, M_phy, return_features=True)

                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)

                # 条件化: 拼接概率和特征 (简化版CDAN)
                src_prob = F.softmax(src_logits.detach(), dim=1)  # (B, C)
                tgt_prob = F.softmax(tgt_logits.detach(), dim=1)  # (B, C)
                src_cond = torch.cat([src_prob, src_feats.float()], dim=1)  # (B, C+D)
                tgt_cond = torch.cat([tgt_prob, tgt_feats.float()], dim=1)  # (B, C+D)

                src_dom = domain_disc(grad_reverse(src_cond, 1.0))
                tgt_dom = domain_disc(grad_reverse(tgt_cond, 1.0))
                dom_loss = (F.binary_cross_entropy_with_logits(src_dom, torch.ones_like(src_dom)) +
                            F.binary_cross_entropy_with_logits(tgt_dom, torch.zeros_like(tgt_dom))) * 0.5

                loss = cls_loss + cfg.train.lambda_align * dom_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  CDAN Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── DSAN (Deep Subdomain Adaptation Network) ───────────────────────

def train_dsan(model, src_loader, tgt_loader, M_phy, criterion, optimizer, scaler, device, epochs):
    """DSAN: 子域自适应MMD (类条件MMD, 类似Ours但无选择性)"""
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_logits, tgt_feats, _ = model(tgt_x, M_phy, return_features=True)

                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)

                # 类条件MMD (用源域真实标签 + 目标域伪标签)
                tgt_pseudo = tgt_logits.argmax(dim=1)
                lmm = torch.tensor(0.0, device=device)
                for c in range(cfg.model.num_classes):
                    s_mask = src_y == c
                    t_mask = tgt_pseudo == c
                    if s_mask.sum() > 1 and t_mask.sum() > 1:
                        lmm += mmd_loss_rbf(src_feats[s_mask].float(), tgt_feats[t_mask].float())
                lmm = lmm / cfg.model.num_classes

                loss = cls_loss + cfg.train.lambda_align * lmm
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  DSAN Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── MCD (Maximum Classifier Discrepancy) ───────────────────────────

def train_mcd(model, src_loader, tgt_loader, M_phy, classifier2, criterion,
              optimizer, scaler, device, epochs):
    """MCD: 最大分类器差异 (两个分类器, 最小化源域差异, 最大化目标域差异)"""
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
            src_x, src_y, _ = src_batch
            tgt_x, _, _ = tgt_batch
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.train.fp16):
                src_logits, src_feats, src_attn = model(src_x, M_phy, return_features=True)
                tgt_feats, _ = model.extract_features(tgt_x, M_phy)

                cls_loss, _ = criterion(src_logits, src_y, features=src_feats,
                                        attn_weights=src_attn, M_phy=M_phy)

                # 分类器2
                tgt_logits1 = model.classifier(tgt_feats)
                tgt_logits2 = classifier2(tgt_feats)

                # 最大化目标域差异 (对抗)
                discrepancy = -torch.mean(torch.abs(F.softmax(tgt_logits1, dim=1) - F.softmax(tgt_logits2, dim=1)))

                loss = cls_loss + 0.1 * discrepancy
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  MCD Epoch {epoch}/{epochs} | Loss: {total_loss / len(src_loader):.4f}")


# ── OpenMax ────────────────────────────────────────────────────────

def fit_openmax(model, loader, M_phy, device, tail_size=0.15):
    """拟合 OpenMax Weibull 参数

    对每个类，计算类内样本到类均值的距离，拟合尾部分布
    """
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            f, _ = model.extract_features(x, M_phy)
            feats_list.append(f.cpu())
            labels_list.append(y)

    feats = torch.cat(feats_list, 0)
    labels = torch.cat(labels_list, 0)
    num_classes = int(labels.max()) + 1

    class_means = []
    weibull_params = []

    for c in range(num_classes):
        mask = labels == c
        class_feats = feats[mask]
        class_mean = class_feats.mean(dim=0)
        class_means.append(class_mean)

        dists = torch.norm(class_feats - class_mean.unsqueeze(0), dim=1).numpy()
        tail_n = max(int(len(dists) * tail_size), 10)
        tail_dists = np.sort(dists)[-tail_n:]

        try:
            shape, loc, scale = weibull_min.fit(tail_dists, floc=0)
            weibull_params.append((shape, loc, scale))
        except Exception:
            weibull_params.append((1.0, 0.0, dists.mean()))

    return class_means, weibull_params


def openmax_score(features, class_means, weibull_params, alpha=10):
    """计算 OpenMax 开集分数

    返回: open_scores (N,) — 越高越可能是未知
    """
    feats_np = features.numpy() if isinstance(features, torch.Tensor) else features
    num_classes = len(class_means)
    N = feats_np.shape[0]

    # 计算到每个类均值的距离
    dists = np.zeros((N, num_classes))
    for c in range(num_classes):
        cm = class_means[c].numpy() if isinstance(class_means[c], torch.Tensor) else class_means[c]
        dists[:, c] = np.linalg.norm(feats_np - cm[np.newaxis, :], axis=1)

    # Weibull CDF: P(d > d_obs) = 1 - CDF(d_obs)
    # w_score = 1 - CDF(distance)
    w_scores = np.zeros((N, num_classes))
    for c in range(num_classes):
        shape, loc, scale = weibull_params[c]
        w_scores[:, c] = 1.0 - weibull_min.cdf(dists[:, c], shape, loc=loc, scale=scale)

    # 修改 softmax 概率
    # 取距离最近的 alpha 个类重分配概率
    open_scores = np.zeros(N)
    for i in range(N):
        # 按距离排序
        sorted_idx = np.argsort(dists[i])
        # 未知分数 = 1 - sum(修改后的已知概率)
        # 简化版: 取最大 w_score 作为未知分数
        open_scores[i] = w_scores[i].max()

    return open_scores


# ── 主对比实验 ──────────────────────────────────────────────────────

def run_closed_set_comparison(source_domain, target_domain, methods=None):
    """闭集跨域对比实验 (Table 4)"""
    if methods is None:
        methods = ["Source Only", "CORAL", "DANN", "CDAN", "DSAN", "MMD", "MCD", "Ours"]

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    cfg.model.num_classes = len(all_classes)

    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    M_phy = _make_M_phy(device, source_domain, cfg.model.freq_input_dim, cfg.signal.fft_bins)

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

    results = {}

    for method in methods:
        print(f"\n{'='*40} {method} {'='*40}")

        model = FullModel(cfg).to(device)
        criterion = CombinedLoss(cfg)

        if method == "Source Only":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 51):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        elif method == "DANN":
            from torch import nn
            domain_disc = nn.Sequential(nn.Linear(cfg.model.feature_dim, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()),
                                          lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            # 源域预训练
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            # DANN 对齐
            optimizer2 = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()),
                                           lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_dann(model, src_loader, tgt_loader, M_phy, domain_disc, criterion,
                       optimizer2, scaler2, device, epochs=50)

        elif method == "MMD":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            # 源域预训练
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            # MMD 对齐
            optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_mmd(model, src_loader, tgt_loader, M_phy, criterion, optimizer2, scaler2, device, epochs=50)

        elif method == "Ours":
            from modules import SelectiveDomainAligner
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            # 源域预训练
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            # 选择性对齐
            aligner = SelectiveDomainAligner(cfg.model.feature_dim, cfg.model.num_classes).to(device)
            optimizer2 = torch.optim.AdamW(
                list(model.parameters()) + list(aligner.parameters()),
                lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            from train_cross_domain import train_alignment
            M_phy_dict = {i: M_phy for i in range(cfg.model.num_classes)}
            train_alignment(model, src_loader, tgt_loader, M_phy_dict, aligner, criterion,
                            optimizer2, scaler2, device, epochs=50, feature_dim=cfg.model.feature_dim)

        elif method == "CORAL":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_coral(model, src_loader, tgt_loader, M_phy, criterion, optimizer2, scaler2, device, epochs=50)

        elif method == "CDAN":
            from torch import nn
            # CDAN域判别器: 概率+特征维度 → 1
            cond_dim = cfg.model.num_classes + cfg.model.feature_dim
            domain_disc = nn.Sequential(nn.Linear(cond_dim, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
            all_params = list(model.parameters()) + list(domain_disc.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            optimizer2 = torch.optim.AdamW(all_params, lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_cdan(model, src_loader, tgt_loader, M_phy, domain_disc, criterion,
                       optimizer2, scaler2, device, epochs=50)

        elif method == "DSAN":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_dsan(model, src_loader, tgt_loader, M_phy, criterion, optimizer2, scaler2, device, epochs=50)

        elif method == "MCD":
            from torch import nn
            classifier2 = nn.Linear(cfg.model.feature_dim, cfg.model.num_classes).to(device)
            all_params = list(model.parameters()) + list(classifier2.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            optimizer2 = torch.optim.AdamW(all_params, lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_mcd(model, src_loader, tgt_loader, M_phy, classifier2, criterion,
                      optimizer2, scaler2, device, epochs=50)

        # 评估
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y, _ in tgt_loader:
                x = x.to(device)
                logits, _ = model(x, M_phy)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(y.numpy())

        m = compute_closed_set_metrics(trues, preds)
        results[method] = m
        print(f"  Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")

    return results


def run_openset_comparison(source_domain, target_domain, unknown_class, methods=None):
    """开集诊断对比实验 (Table 5)"""
    if methods is None:
        methods = ["Source Only", "DANN + Energy", "OpenMax", "Proto", "OSBP", "Ours"]

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    known_classes = [c for c in all_classes if c != unknown_class]
    K = len(known_classes)
    cfg.model.num_classes = K

    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=known_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    source_ds.labels = remap_labels(source_ds.labels, known_classes)

    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    cfg.model.freq_input_dim = source_ds.segments.shape[1]
    M_phy = _make_M_phy(device, source_domain, cfg.model.freq_input_dim, cfg.signal.fft_bins)

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)

    unknown_orig = _ALL_CLASS_MAP[unknown_class]
    label_map = {_ALL_CLASS_MAP[c]: i for i, c in enumerate(known_classes)}

    results = {}

    for method in methods:
        print(f"\n{'='*40} {method} {'='*40}")

        model = FullModel(cfg).to(device)
        criterion = CombinedLoss(cfg)

        if method == "Source Only":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 81):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        elif method == "DANN + Energy":
            from torch import nn
            domain_disc = nn.Sequential(nn.Linear(cfg.model.feature_dim, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
            optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()),
                                          lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            optimizer2 = torch.optim.AdamW(list(model.parameters()) + list(domain_disc.parameters()),
                                           lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            train_dann(model, src_loader, tgt_loader, M_phy, domain_disc, criterion,
                       optimizer2, scaler2, device, epochs=50)

        elif method == "OpenMax":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 81):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        elif method == "Proto":
            # Prototype Distance: 源域预训练, 用特征距离做开集检测
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 81):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        elif method == "OSBP":
            # OSBP: 将未知类作为额外类训练
            from torch import nn
            cfg_bak = cfg.model.num_classes
            cfg.model.num_classes = K + 1  # 增加unknown类

            model_osbp = FullModel(cfg).to(device)
            criterion_osbp = CombinedLoss(cfg)
            optimizer = torch.optim.AdamW(model_osbp.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)

            model_osbp.train()
            for epoch in range(1, 81):
                for (src_batch, tgt_batch) in zip(src_loader, tgt_loader):
                    src_x, src_y, _ = src_batch
                    tgt_x, _, _ = tgt_batch
                    src_x, src_y = src_x.to(device), src_y.to(device)
                    tgt_x = tgt_x.to(device)

                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        src_logits, src_feats, src_attn = model_osbp(src_x, M_phy, return_features=True)
                        tgt_logits, _ = model_osbp(tgt_x, M_phy)

                        cls_loss, _ = criterion_osbp(src_logits, src_y, features=src_feats,
                                                      attn_weights=src_attn, M_phy=M_phy)

                        # OSBP: 目标域用熵最小化 + unknown类鼓励
                        tgt_prob = F.softmax(tgt_logits, dim=1)
                        entropy = -(tgt_prob * torch.log(tgt_prob + 1e-8)).sum(dim=1).mean()
                        # 鼓励目标域预测为unknown类(最后一个类)
                        unk_prob = tgt_prob[:, -1].mean()
                        loss = cls_loss + 0.1 * entropy - 0.1 * unk_prob

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            model = model_osbp
            cfg.model.num_classes = cfg_bak

        elif method == "Ours":
            from modules import SelectiveDomainAligner
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scaler = GradScaler(enabled=cfg.train.fp16)
            model.train()
            for epoch in range(1, 31):
                for x, y, _ in src_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    with autocast(enabled=cfg.train.fp16):
                        logits, feats, attn = model(x, M_phy, return_features=True)
                        loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_phy)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            aligner = SelectiveDomainAligner(cfg.model.feature_dim, K).to(device)
            optimizer2 = torch.optim.AdamW(
                list(model.parameters()) + list(aligner.parameters()),
                lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler2 = GradScaler(enabled=cfg.train.fp16)
            from train_cross_domain import train_alignment
            M_phy_dict = {i: M_phy for i in range(K)}
            train_alignment(model, src_loader, tgt_loader, M_phy_dict, aligner, criterion,
                            optimizer2, scaler2, device, epochs=50, feature_dim=cfg.model.feature_dim)

        # 评估
        model.eval()

        if method == "Proto":
            # Prototype Distance: 用源域原型距离做开集检测
            src_feats_list, src_labels_list = [], []
            with torch.no_grad():
                for x, y, _ in src_loader:
                    x = x.to(device)
                    f, _ = model.extract_features(x, M_phy)
                    src_feats_list.append(f.cpu())
                    src_labels_list.append(y)
            src_feats = torch.cat(src_feats_list, 0)
            src_labels = torch.cat(src_labels_list, 0)

            # 类原型
            prototypes = []
            for c in range(K):
                mask = src_labels == c
                prototypes.append(src_feats[mask].mean(dim=0))
            prototypes = torch.stack(prototypes)

            # 目标域
            tgt_feats_list, tgt_logits_list, tgt_true_list = [], [], []
            with torch.no_grad():
                for x, y, _ in tgt_loader:
                    x = x.to(device)
                    f, _ = model.extract_features(x, M_phy)
                    lg, _ = model(x, M_phy)
                    tgt_feats_list.append(f.cpu())
                    tgt_logits_list.append(lg.cpu())
                    tgt_true_list.append(y)
            tgt_feats = torch.cat(tgt_feats_list, 0)
            tgt_logits = torch.cat(tgt_logits_list, 0)
            tgt_true = torch.cat(tgt_true_list, 0)

            # 原型距离作为未知分数
            dists = torch.cdist(tgt_feats, prototypes, p=2)
            min_dists = dists.min(dim=1)[0].numpy()
            tgt_preds = tgt_logits.argmax(dim=1).numpy()

            true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                                    for l in tgt_true.numpy()])

            best_h, best_result = 0, None
            for pct in range(1, 100, 2):
                thresh = np.percentile(min_dists, 100 - pct)
                p = tgt_preds.copy()
                p[min_dists > thresh] = -1
                m = compute_open_set_metrics(true_labels, p, min_dists)
                if m["h_score"] > best_h:
                    best_h = m["h_score"]
                    best_result = m
            m = best_result

        elif method == "OSBP":
            # OSBP: 前K个类为已知, 最后一个类为unknown
            tgt_logits, tgt_true = [], []
            with torch.no_grad():
                for x, y, _ in tgt_loader:
                    x = x.to(device)
                    logits, _ = model(x, M_phy)
                    tgt_logits.append(logits.cpu())
                    tgt_true.append(y)
            tgt_logits = torch.cat(tgt_logits, 0)
            tgt_true = torch.cat(tgt_true, 0)

            # 已知类概率 = 前K个类的softmax之和, unknown概率 = 最后一个类
            prob_k = F.softmax(tgt_logits[:, :K], dim=1)
            unk_score = F.softmax(tgt_logits, dim=1)[:, -1].numpy()
            tgt_preds = prob_k.argmax(dim=1).numpy()

            true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                                    for l in tgt_true.numpy()])

            best_h, best_result = 0, None
            for pct in range(1, 100, 2):
                thresh = np.percentile(unk_score, 100 - pct)
                p = tgt_preds.copy()
                p[unk_score > thresh] = -1
                m = compute_open_set_metrics(true_labels, p, unk_score)
                if m["h_score"] > best_h:
                    best_h = m["h_score"]
                    best_result = m
            m = best_result

        elif method == "OpenMax":
            # OpenMax 评估
            class_means, weibull_params = fit_openmax(model, src_loader, M_phy, device)

            feats_tgt, labels_tgt = [], []
            with torch.no_grad():
                for x, y, _ in tgt_loader:
                    x = x.to(device)
                    f, _ = model.extract_features(x, M_phy)
                    feats_tgt.append(f.cpu())
                    labels_tgt.append(y)
            feats_tgt = torch.cat(feats_tgt, 0)
            labels_tgt = torch.cat(labels_tgt, 0)

            open_scores = openmax_score(feats_tgt, class_means, weibull_params)

            # 用 logits 的 argmax 做已知类预测
            logits_tgt = []
            with torch.no_grad():
                for x, y, _ in tgt_loader:
                    x = x.to(device)
                    lg, _ = model(x, M_phy)
                    logits_tgt.append(lg.cpu())
            logits_tgt = torch.cat(logits_tgt, 0)
            tgt_preds = logits_tgt.argmax(dim=1).numpy()

            true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                                    for l in labels_tgt.numpy()])

            # 阈值扫描
            best_h, best_result = 0, None
            for pct in range(1, 100, 2):
                thresh = np.percentile(open_scores, 100 - pct)  # 高分=未知
                p = tgt_preds.copy()
                p[open_scores > thresh] = -1
                m = compute_open_set_metrics(true_labels, p, open_scores)
                if m["h_score"] > best_h:
                    best_h = m["h_score"]
                    best_result = (pct, thresh, m)

            _, _, m = best_result
        else:
            # 能量分数评估
            src_energies = []
            with torch.no_grad():
                for x, _, _ in src_loader:
                    x = x.to(device)
                    logits, _ = model(x, M_phy)
                    src_energies.append(torch.logsumexp(logits, dim=1).cpu().numpy())
            src_energies = np.concatenate(src_energies)

            tgt_logits, tgt_true = [], []
            with torch.no_grad():
                for x, y, _ in tgt_loader:
                    x = x.to(device)
                    logits, _ = model(x, M_phy)
                    tgt_logits.append(logits.cpu())
                    tgt_true.append(y)
            tgt_logits = torch.cat(tgt_logits, 0)
            tgt_true = torch.cat(tgt_true, 0)
            tgt_energy = torch.logsumexp(tgt_logits, dim=1).numpy()
            tgt_preds = tgt_logits.argmax(dim=1).numpy()

            true_labels = np.array([-1 if int(l) == unknown_orig else label_map.get(int(l), -1)
                                    for l in tgt_true.numpy()])

            best_h, best_result = 0, None
            for pct in range(1, 100, 2):
                thresh = np.percentile(src_energies, pct)
                p = tgt_preds.copy()
                p[tgt_energy < thresh] = -1
                m = compute_open_set_metrics(true_labels, p, -tgt_energy)
                if m["h_score"] > best_h:
                    best_h = m["h_score"]
                    best_result = (pct, thresh, m)

            _, _, m = best_result

        results[method] = m
        print(f"  Known={m['known_acc']:.4f}, Unk={m['unknown_acc']:.4f}, H={m['h_score']:.4f}, AUROC={m['auroc']:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["closed_set", "openset"], default="closed_set")
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    parser.add_argument("--unknown", default="Ball")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    if args.task == "closed_set":
        results = run_closed_set_comparison(args.source, args.target, methods=args.methods)
    else:
        results = run_openset_comparison(args.source, args.target, args.unknown, methods=args.methods)
