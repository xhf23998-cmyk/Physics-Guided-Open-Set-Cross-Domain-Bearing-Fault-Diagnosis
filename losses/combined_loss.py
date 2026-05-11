"""组合总损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_regularization import PhysicsRegularizationLoss
from modules.supcon import PhysicsSupConLoss


class CombinedLoss(nn.Module):
    """总损失函数

    L_total = λ_ce * L_ce + λ_supcon * L_pc_supcon + λ_phy * L_phy
              + λ_align * L_align + λ_sep * L_sep
    """

    def __init__(self, cfg):
        super().__init__()
        self.lambda_ce = cfg.train.lambda_ce
        self.lambda_supcon = cfg.train.lambda_supcon
        self.lambda_phy = cfg.train.lambda_phy
        self.lambda_align = cfg.train.lambda_align
        self.lambda_sep = cfg.train.lambda_sep

        self.ce_loss = nn.CrossEntropyLoss()
        self.supcon_loss = PhysicsSupConLoss(
            temperature=cfg.train.supcon_temperature
        )
        self.phy_reg_loss = PhysicsRegularizationLoss()

    def forward(self, logits, labels, features=None, attn_weights=None,
                M_phy=None, physics_templates=None, L_align=None, L_sep=None):
        """
        Args:
            logits: (B, C) 分类输出
            labels: (B,) 真实标签
            features: (B, D) 特征向量（用于 SupCon）
            attn_weights: (B, 1, F) 注意力权重（用于物理正则）
            M_phy: (F,) 物理频率模板
            L_align: 已计算的对齐损失（可选）
            L_sep: 已计算的分离损失（可选）
        """
        total_loss = torch.tensor(0.0, device=logits.device)
        loss_dict = {}

        # 分类损失
        L_ce = self.ce_loss(logits, labels)
        total_loss += self.lambda_ce * L_ce
        loss_dict["L_ce"] = L_ce.item()

        # 物理一致性 SupCon 损失
        if features is not None:
            L_supcon = self.supcon_loss(features, labels)
            total_loss += self.lambda_supcon * L_supcon
            loss_dict["L_supcon"] = L_supcon.item()

        # 物理正则损失
        if attn_weights is not None and M_phy is not None:
            L_phy = self.phy_reg_loss(attn_weights, M_phy)
            total_loss += self.lambda_phy * L_phy
            loss_dict["L_phy"] = L_phy.item()

        # 域对齐损失
        if L_align is not None:
            total_loss += self.lambda_align * L_align
            loss_dict["L_align"] = L_align.item()

        if L_sep is not None:
            total_loss += self.lambda_sep * L_sep
            loss_dict["L_sep"] = L_sep.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict
