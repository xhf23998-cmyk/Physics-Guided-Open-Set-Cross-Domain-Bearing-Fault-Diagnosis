"""模块② 物理一致性正则化损失"""
import torch
import torch.nn as nn


class PhysicsRegularizationLoss(nn.Module):
    """频率注意力物理一致性正则损失

    L_phy = L_band + L_harmonic + L_smooth

    - L_band: 鼓励注意力集中在故障敏感频带
    - L_harmonic: 鼓励倍频结构一致
    - L_smooth: 避免频率权重剧烈震荡
    """

    def __init__(self, lambda_band=1.0, lambda_harmonic=0.5, lambda_smooth=0.1):
        super().__init__()
        self.lambda_band = lambda_band
        self.lambda_harmonic = lambda_harmonic
        self.lambda_smooth = lambda_smooth

    def forward(self, attn_weights, M_phy):
        """
        Args:
            attn_weights: (B, 1, F) 频率注意力权重
            M_phy: (F,) 或 (B, F) 物理频率模板
        Returns:
            L_phy: 标量
        """
        A = attn_weights.squeeze(1)  # (B, F)

        if M_phy.dim() == 1:
            M_phy = M_phy.unsqueeze(0).expand_as(A)

        # L_band: 注意力与物理模板的一致性
        L_band = ((A - M_phy) ** 2).mean()

        # L_smooth: 相邻频率权重平滑
        if A.shape[1] > 1:
            diff = A[:, 1:] - A[:, :-1]
            L_smooth = (diff ** 2).mean()
        else:
            L_smooth = torch.tensor(0.0, device=A.device)

        # L_harmonic: 简化版 — 倍频位置的注意力一致性
        # 取奇数/偶数位置的注意力，要求它们统计特性一致
        if A.shape[1] > 2:
            even = A[:, ::2]
            odd = A[:, 1::2]
            min_len = min(even.shape[1], odd.shape[1])
            L_harmonic = ((even[:, :min_len] - odd[:, :min_len]) ** 2).mean()
        else:
            L_harmonic = torch.tensor(0.0, device=A.device)

        L_phy = (self.lambda_band * L_band +
                 self.lambda_harmonic * L_harmonic +
                 self.lambda_smooth * L_smooth)

        return L_phy
