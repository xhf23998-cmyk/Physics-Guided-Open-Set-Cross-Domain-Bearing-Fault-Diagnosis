"""模块②：机制约束频率注意力净化模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsFrequencyAttention(nn.Module):
    """物理频率先验约束的注意力机制

    将轴承故障特征频率 (BPFO/BPFI/BSF/FTF) 构建的物理模板
    与数据驱动的注意力自适应融合，对频谱进行净化。

    输入: X (B, 1, F) — 频域输入
          M_phy (F,) 或 (B, F) — 物理频率模板
    输出: X' (B, 1, F) — 净化后频谱
          A (B, 1, F) — 注意力权重（可用于可解释性输出）
    """

    def __init__(self, freq_bins=512, hidden_dim=64, heads=4, alpha_init=0.5):
        super().__init__()
        self.freq_bins = freq_bins
        self.heads = heads

        # 数据驱动频率编码
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 多头频率注意力
        self.query = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.key = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.value = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.attn_proj = nn.Conv1d(hidden_dim, 1, 1)

        # 物理模板投影
        self.phy_proj = nn.Sequential(
            nn.Linear(freq_bins, freq_bins),
            nn.Sigmoid(),
        )

        # 自适应融合参数 α
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv1d(1, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, X, M_phy=None):
        """
        Args:
            X: (B, 1, F) 频域输入
            M_phy: (F,) 或 (B, F) 物理频率模板，None 时只用数据驱动注意力
        Returns:
            X_purified: (B, 1, F) 净化频谱
            A: (B, 1, F) 注意力权重
        """
        B = X.shape[0]

        # 1. 浅层频谱编码
        H = self.encoder(X)  # (B, hidden, F)

        # 2. 数据驱动注意力
        Q = self.query(H)
        K = self.key(H)
        V = self.value(H)
        attn_weights = torch.sigmoid(self.attn_proj(Q))  # (B, 1, F)
        A_learned = attn_weights

        # 3. 物理先验注意力
        if M_phy is not None and M_phy.abs().sum() > 0:
            if M_phy.dim() == 1:
                M_phy = M_phy.unsqueeze(0).expand(B, -1)  # (B, F)
            A_phy = self.phy_proj(M_phy).unsqueeze(1)  # (B, 1, F)

            # 4. 自适应融合
            alpha = torch.sigmoid(self.alpha)
            A = alpha * A_learned + (1 - alpha) * A_phy
        else:
            A = A_learned

        # 5. 频谱净化
        X_purified = A * X

        return X_purified, A
