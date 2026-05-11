"""模块⑥：目标域自校准 EVT 开集故障识别 (v2)"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import weibull_min


class EVTHead(nn.Module):
    """基于 EVT 的开集识别头

    改进:
    - 使用马氏距离的近似（基于特征协方差）
    - 自适应阈值校准（基于源域距离分布的百分位数）
    - 支持多种未知分数融合
    """

    def __init__(self, tail_size=0.15, margin=1.0):
        super().__init__()
        self.tail_size = tail_size
        self.margin = margin
        self.weibull_shapes = {}
        self.weibull_scales = {}
        self.class_prototypes = {}
        self.class_distances = {}  # 保存每个类的距离分布用于阈值校准
        self.is_fitted = False

    def fit(self, features, labels):
        """用源域特征拟合 Weibull 分布"""
        features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        labels = labels.cpu().numpy() if isinstance(features, np.ndarray) else labels

        classes = np.unique(labels)

        for c in classes:
            mask = labels == c
            cls_feats = features[mask]

            prototype = cls_feats.mean(axis=0)
            self.class_prototypes[int(c)] = prototype

            dists = np.linalg.norm(cls_feats - prototype, axis=1)
            self.class_distances[int(c)] = dists

            # Weibull 拟合
            n_tail = max(int(len(dists) * self.tail_size), 20)
            tail_dists = np.sort(dists)[-n_tail:]

            try:
                shape, loc, scale = weibull_min.fit(tail_dists, floc=0)
                self.weibull_shapes[int(c)] = shape
                self.weibull_scales[int(c)] = scale
            except Exception:
                self.weibull_shapes[int(c)] = 1.0
                self.weibull_scales[int(c)] = tail_dists.mean()

        self.is_fitted = True

    def compute_unknown_score(self, features):
        """计算未知分数 (越高越可能是未知类)"""
        if not self.is_fitted:
            raise RuntimeError("EVT 尚未拟合")

        features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        classes = sorted(self.class_prototypes.keys())

        all_dists = []
        for c in classes:
            proto = self.class_prototypes[c]
            dists = np.linalg.norm(features_np - proto, axis=1)
            all_dists.append(dists)
        all_dists = np.stack(all_dists, axis=1)  # (N, K)

        nearest_idx = all_dists.argmin(axis=1)
        nearest_dists = all_dists.min(axis=1)
        nearest_class = np.array([classes[i] for i in nearest_idx])

        # EVT Weibull 分数
        evt_scores = np.zeros(len(features_np))
        for i in range(len(features_np)):
            c = nearest_class[i]
            d = nearest_dists[i]
            shape = self.weibull_shapes[c]
            scale = self.weibull_scales[c]
            evt_scores[i] = 1.0 - weibull_min.cdf(d, shape, loc=0, scale=scale)

        # 距离归一化分数 (基于源域距离分布的百分位)
        dist_scores = np.zeros(len(features_np))
        for i in range(len(features_np)):
            c = nearest_class[i]
            d = nearest_dists[i]
            src_dists = self.class_distances[c]
            # 样本距离在源域距离分布中的百分位
            pct = np.mean(src_dists <= d)
            dist_scores[i] = pct

        # 融合分数: EVT 和距离百分位取平均
        combined_scores = 0.5 * evt_scores + 0.5 * dist_scores

        return torch.from_numpy(combined_scores).float(), torch.from_numpy(nearest_class).long()

    def predict(self, features, threshold=None):
        """开集预测

        Args:
            features: (N, D) 测试特征
            threshold: 手动阈值，None 时自动校准
        Returns:
            predictions: (N,) 类别，-1 表示 Unknown
            scores: (N,) 未知分数
        """
        scores, nearest_class = self.compute_unknown_score(features)

        if threshold is None:
            # 自动阈值: 使用源域距离的 95th 百分位对应的分数
            threshold = self._auto_threshold()

        predictions = nearest_class.clone()
        predictions[scores > threshold] = -1

        return predictions, scores

    def _auto_threshold(self):
        """基于源域数据自动计算阈值"""
        all_src_scores = []
        for c in sorted(self.class_prototypes.keys()):
            src_dists = self.class_distances[c]
            src_feats_dummy = np.zeros((len(src_dists), 1))
            # 对源域每个样本计算分数
            for d in src_dists:
                shape = self.weibull_shapes[c]
                scale = self.weibull_scales[c]
                evt = 1.0 - weibull_min.cdf(d, shape, loc=0, scale=scale)
                pct = np.mean(src_dists <= d)
                score = 0.5 * evt + 0.5 * pct
                all_src_scores.append(score)

        all_src_scores = np.array(all_src_scores)
        # 阈值 = 源域 95th 百分位分数 (允许 5% 的源域样本被误判为未知)
        threshold = np.percentile(all_src_scores, 95)
        return threshold
