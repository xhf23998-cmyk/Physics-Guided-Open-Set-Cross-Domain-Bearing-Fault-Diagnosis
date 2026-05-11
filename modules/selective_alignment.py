"""模块④：未知感知选择性跨域对齐 (v3)

v1 (原版): class-wise MMD, P_unk用batch统计量, 对齐权重未使用
v2 (DANN): per-sample DANN+GRL, 闭集降级到50%
v3 (当前): 恢复class-wise MMD, 修复P_unk和权重, 加样本级选择性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return _GradReverse.apply(x, alpha)


class SelectiveDomainAligner(nn.Module):
    """未知感知选择性域对齐 v3

    使用 class-wise MMD 保持闭集性能 (99%)，
    同时通过改进的 P_unk 估计和样本选择性减少开集负迁移。
    """

    def __init__(self, feature_dim, num_classes, margin=1.0):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes

    def compute_unknown_probability(self, features, prototypes):
        """估计目标样本属于未知故障的概率

        改进: 用类间距离均值归一化，不用 batch 统计量
        """
        dists = torch.cdist(features, prototypes, p=2)  # (N, C)
        min_dist = dists.min(dim=1)[0]  # (N,)

        # 归一化: 用类间距离均值
        if prototypes.shape[0] > 1:
            inter_proto_dist = torch.cdist(prototypes, prototypes, p=2)
            scale = inter_proto_dist[inter_proto_dist > 0].mean() + 1e-8
        else:
            scale = min_dist.mean() + 1e-8

        normalized_dist = min_dist / scale
        P_unk = torch.sigmoid(2.0 * (normalized_dist - 0.5))
        return P_unk

    def class_wise_mmd(self, source_features, source_labels, target_features, target_labels,
                        sample_weights=None):
        """类别级别 MMD 距离 (支持样本加权)"""
        loss = torch.tensor(0.0, device=source_features.device)
        classes = torch.unique(source_labels)

        for c in classes:
            src_mask = source_labels == c
            tgt_mask = target_labels == c

            if src_mask.sum() < 2 or tgt_mask.sum() < 2:
                continue

            src_cls = source_features[src_mask]
            tgt_cls = target_features[tgt_mask]

            src_mean = src_cls.mean(dim=0)
            tgt_mean = tgt_cls.mean(dim=0)

            if sample_weights is not None:
                # 加权目标均值
                w = sample_weights[tgt_mask]
                w = w / (w.sum() + 1e-8)
                tgt_mean = (tgt_cls * w.unsqueeze(1)).sum(dim=0)

            mmd = F.mse_loss(src_mean, tgt_mean)
            loss += mmd

        return loss / max(len(classes), 1)

    def alignment_loss(self, source_features, source_labels,
                       target_features, prototypes):
        """选择性对齐损失

        Args:
            source_features: (Ns, D)
            source_labels: (Ns,) 0..K-1
            target_features: (Nt, D)
            prototypes: (C, D)
        Returns:
            L_align, L_sep, P_unk, align_weights
        """
        P_unk = self.compute_unknown_probability(target_features, prototypes)
        align_weights = 1.0 - P_unk.detach()

        # 伪标签
        dists = torch.cdist(target_features, prototypes, p=2)
        pseudo_labels = dists.argmin(dim=1)

        # 加权 class-wise MMD
        L_align = self.class_wise_mmd(
            source_features, source_labels,
            target_features, pseudo_labels,
            sample_weights=align_weights,
        )

        # 分离损失
        min_dists = dists.min(dim=1)[0]
        L_sep = (P_unk * F.relu(self.margin - min_dists)).mean()

        return L_align, L_sep, P_unk, align_weights
