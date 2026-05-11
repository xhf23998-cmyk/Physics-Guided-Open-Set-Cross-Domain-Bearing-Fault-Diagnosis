"""消融实验 - 闭集版本: 在跨域闭集任务上评估各模块贡献"""
import os, sys, copy, numpy as np, torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import cfg
from data import SelfCollectedDataset
from data.dataset import _SELF_WORKING_CONDITIONS
from models import FullModel
from losses import CombinedLoss
from modules import FrequencyTemplateBuilder, SelectiveDomainAligner
from utils import set_seed, compute_closed_set_metrics
from train_cross_domain import _ALL_CLASS_MAP


def run_closedset_ablation(source_domain, target_domain):
    """闭集消融: 逐步添加模块，评估跨域闭集准确率"""
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    all_classes = ["Health", "Inner", "Outer", "Ball"]
    K = len(all_classes)
    cfg.model.num_classes = K

    # 数据
    source_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=source_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)
    target_ds = SelfCollectedDataset(data_root=cfg.paths.self_collected, domain=target_domain,
        classes=all_classes, window_size=cfg.signal.window_size, overlap=cfg.signal.overlap,
        fft_bins=cfg.signal.fft_bins)

    freq_input_dim = source_ds.segments.shape[1]
    cfg.model.freq_input_dim = freq_input_dim

    src_loader = DataLoader(source_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(target_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    # 物理模板
    source_rpm = _SELF_WORKING_CONDITIONS[source_domain]["speed"]
    tb = FrequencyTemplateBuilder(sample_rate=cfg.signal.sample_rate, fft_bins=cfg.signal.fft_bins,
        bearing_params={"ball_count": cfg.bearing.ball_count, "ball_diameter": cfg.bearing.ball_diameter,
                        "pitch_diameter": cfg.bearing.pitch_diameter, "contact_angle": cfg.bearing.contact_angle})

    t_fft = tb.build_template(rpm=source_rpm)
    pad = np.zeros(freq_input_dim - cfg.signal.fft_bins, dtype=np.float32)
    M_phy_raw = torch.from_numpy(np.concatenate([t_fft, pad])).float()
    M_phy_dict = {i: M_phy_raw for i in range(K)}

    variants = [
        {"name": "Backbone", "use_phy": False, "supcon": False, "align": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "+ Physics FA", "use_phy": True, "supcon": False, "align": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "+ SupCon", "use_phy": True, "supcon": True, "align": False,
         "pretrain_epochs": 80, "align_epochs": 0},
        {"name": "Full Ours", "use_phy": True, "supcon": True, "align": True,
         "pretrain_epochs": 30, "align_epochs": 50},
    ]

    results = {}

    for v in variants:
        print(f"\n{'='*50}")
        print(f"消融: {v['name']}")
        print(f"  Phy={v['use_phy']}, SupCon={v['supcon']}, Align={v['align']}")
        print(f"{'='*50}")

        cfg_v = copy.deepcopy(cfg)
        cfg_v.model.num_classes = K
        cfg_v.model.freq_input_dim = freq_input_dim
        if not v["supcon"]:
            cfg_v.train.lambda_supcon = 0.0
        if not v["align"]:
            cfg_v.train.lambda_align = 0.0
            cfg_v.train.lambda_sep = 0.0

        model = FullModel(cfg_v).to(device)
        criterion = CombinedLoss(cfg_v)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = GradScaler(enabled=cfg.train.fp16)

        M_eval = M_phy_raw.unsqueeze(0).to(device) if v["use_phy"] else None
        M_dict = M_phy_dict if v["use_phy"] else None

        # 阶段1: 源域预训练
        print(f"  源域预训练 ({v['pretrain_epochs']} epochs)...")
        model.train()
        for epoch in range(1, v["pretrain_epochs"] + 1):
            total_loss = 0
            for x, y, _ in src_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with autocast(enabled=cfg.train.fp16):
                    logits, feats, attn = model(x, M_eval, return_features=True)
                    loss, _ = criterion(logits, y, features=feats, attn_weights=attn, M_phy=M_eval)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}/{v['pretrain_epochs']} | Loss: {total_loss/len(src_loader):.4f}")

        # 阶段2: 选择性对齐
        if v["align"]:
            print(f"  选择性对齐 ({v['align_epochs']} epochs)...")
            from train_cross_domain import train_alignment
            aligner = SelectiveDomainAligner(feature_dim=cfg.model.feature_dim, num_classes=K).to(device)
            optimizer_a = torch.optim.AdamW(
                list(model.parameters()) + list(aligner.parameters()),
                lr=cfg.train.lr * 0.1, weight_decay=cfg.train.weight_decay)
            scaler_a = GradScaler(enabled=cfg.train.fp16)
            train_alignment(model, src_loader, tgt_loader, M_dict, aligner, criterion,
                            optimizer_a, scaler_a, device, epochs=v["align_epochs"],
                            feature_dim=cfg.model.feature_dim)

        # 闭集评估
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y, _ in tgt_loader:
                x = x.to(device)
                logits, _ = model(x, M_eval)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.numpy())

        m = compute_closed_set_metrics(all_true, all_preds)
        results[v["name"]] = m
        print(f"  Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"闭集消融汇总 ({source_domain} → {target_domain})")
    print(f"{'='*60}")
    print(f"{'变体':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="W1")
    parser.add_argument("--target", default="W2")
    args = parser.parse_args()

    run_closedset_ablation(args.source, args.target)
