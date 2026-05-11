"""深度特征提取主干网络：ResNet-1D 和 ConvNeXt-1D"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """ResNet-1D 基本残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet1D(nn.Module):
    """ResNet-1D 特征提取主干

    输入: (B, 1, L) — 单通道频域序列
    输出: (B, feature_dim) — 特征向量
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt-1D 块：DWConv + LN + FFN"""

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + self.drop_path(x)


class ConvNeXt1D(nn.Module):
    """ConvNeXt-1D 特征提取主干"""

    def __init__(self, feature_dim=128, dims=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], 7, stride=2, padding=3),
            nn.BatchNorm1d(dims[0]),
        )
        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            stage = nn.Sequential(
                nn.Conv1d(dims[i], dims[i + 1], 2, stride=2),  # downsample
                ConvNeXtBlock1D(dims[i + 1]),
                ConvNeXtBlock1D(dims[i + 1]),
            )
            self.stages.append(stage)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dims[-1], feature_dim)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)
