"""完整模型：串联所有模块"""
import torch
import torch.nn as nn

from .backbone import ResNet1D, ConvNeXt1D
from .frequency_attention import PhysicsFrequencyAttention


class ClassificationHead(nn.Module):
    """分类头"""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FullModel(nn.Module):
    """完整诊断模型

    模块① 频率模板构建 → 模块② 频率注意力 → 模块③ 特征提取+对比学习
    → 模块④ 选择性对齐 → 模块⑥ EVT（部分在推理阶段使用）

    注意：模块④和⑥的部分逻辑在训练循环中实现，不在 nn.Module 中。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 模块②：频率注意力
        self.freq_attention = PhysicsFrequencyAttention(
            freq_bins=cfg.model.freq_input_dim,
            hidden_dim=64,
            heads=cfg.model.freq_attention_heads,
            alpha_init=cfg.model.alpha_init,
        )

        # 模块③：深度特征提取主干
        if cfg.model.backbone == "resnet1d":
            self.backbone = ResNet1D(feature_dim=cfg.model.feature_dim)
        elif cfg.model.backbone == "convnext1d":
            self.backbone = ConvNeXt1D(feature_dim=cfg.model.feature_dim)
        else:
            raise ValueError(f"未知主干网络: {cfg.model.backbone}")

        # 分类头
        self.classifier = ClassificationHead(cfg.model.feature_dim, cfg.model.num_classes)

    def forward(self, x, M_phy=None, return_features=False):
        """
        Args:
            x: (B, 1, L) 输入频域信号
            M_phy: (L,) 物理频率模板
            return_features: 是否返回中间特征
        Returns:
            logits: (B, num_classes)
            features: (B, feature_dim) if return_features
            attn_weights: (B, 1, L) 注意力权重
        """
        # 频率注意力净化
        x_purified, attn_weights = self.freq_attention(x, M_phy)

        # 特征提取
        features = self.backbone(x_purified)

        # 分类
        logits = self.classifier(features)

        if return_features:
            return logits, features, attn_weights
        return logits, attn_weights

    def extract_features(self, x, M_phy=None):
        """仅提取特征，不经过分类头"""
        x_purified, attn_weights = self.freq_attention(x, M_phy)
        features = self.backbone(x_purified)
        return features, attn_weights
