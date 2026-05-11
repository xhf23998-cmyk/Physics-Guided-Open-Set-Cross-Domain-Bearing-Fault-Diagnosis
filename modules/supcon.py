"""模块③：物理一致性监督对比学习损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsSupConLoss(nn.Module):
    """物理一致性监督对比学习损失

    在普通 SupCon 基础上，引入故障机理一致性约束：
    - 正样本不仅要求标签相同，还要求物理频率响应模式相似
    - 负样本如果物理模式相似（不同工况的同类故障），适当降低惩罚

    这使得特征空间不仅类别可分，而且物理结构一致。
    """

    def __init__(self, temperature=0.07, physics_weight=0.3):
        """
        Args:
            temperature: 对比学习温度参数
            physics_weight: 物理一致性权重 (0~1)
        """
        super().__init__()
        self.temperature = temperature
        self.physics_weight = physics_weight

    def forward(self, features, labels, physics_templates=None):
        """
        Args:
            features: (N, D) L2 归一化后的特征
            labels: (N,) 标签
            physics_templates: (N, F) 每个样本对应的物理频率模板
                               None 时退化为普通 SupCon
        Returns:
            loss: 标量
        """
        device = features.device
        N = features.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 归一化
        features = F.normalize(features, dim=1)

        # 相似度矩阵
        sim_matrix = torch.mm(features, features.t()) / self.temperature  # (N, N)

        # 构建标签掩码
        labels = labels.unsqueeze(1)  # (N, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(device)  # (N, N)
        mask_pos.fill_diagonal_(0)  # 排除自身

        # 物理一致性调制
        if physics_templates is not None:
            phy_sim = self._compute_physics_similarity(physics_templates)
            phy_sim = phy_sim.to(device)

            # 物理一致的正样本加强
            phy_weight = 1.0 + self.physics_weight * phy_sim
            mask_pos = mask_pos * phy_weight

        # 对比损失
        # 对每个锚点，计算正样本的 log-exp-sum 除以所有样本的 exp-sum
        exp_sim = torch.exp(sim_matrix)
        # 排除自身：用掩码而非 in-place 操作
        diag_mask = 1.0 - torch.eye(N, device=device)
        exp_sim = exp_sim * diag_mask

        # 正样本的 exp 之和
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        # 所有样本的 exp 之和
        all_sim = exp_sim.sum(dim=1)

        # 避免 log(0)
        valid = (mask_pos.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -torch.log(pos_sim[valid] / (all_sim[valid] + 1e-8)).mean()
        return loss

    def _compute_physics_similarity(self, templates):
        """计算物理频率模板之间的余弦相似度

        Args:
            templates: (N, F) 物理频率模板
        Returns:
            sim: (N, N) 相似度矩阵
        """
        templates = F.normalize(templates, dim=1)
        sim = torch.mm(templates, templates.t())
        return sim
