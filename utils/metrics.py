"""评估指标"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


def compute_closed_set_metrics(y_true, y_pred):
    """闭集分类指标

    Returns:
        dict: {accuracy, macro_f1, precision, recall}
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_open_set_metrics(y_true, y_pred, y_scores, known_classes=None):
    """开集诊断指标

    Args:
        y_true: (N,) 真实标签
        y_pred: (N,) 预测标签 (-1=Unknown)
        y_scores: (N,) 未知分数
        known_classes: 已知类列表，None 时自动从 y_true 推断
    Returns:
        dict: {known_acc, unknown_acc, h_score, auroc, aupr, oscr}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    if known_classes is None:
        known_classes = [c for c in np.unique(y_true) if c >= 0]

    # Known / Unknown 掩码
    is_known_true = np.isin(y_true, known_classes)
    is_unknown_true = ~is_known_true

    # Known Acc: 已知样本中正确分类的比例
    if is_known_true.sum() > 0:
        known_mask = is_known_true
        known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask])
    else:
        known_acc = 0.0

    # Unknown Acc: 未知样本中正确拒识的比例
    if is_unknown_true.sum() > 0:
        unknown_acc = (y_pred[is_unknown_true] == -1).mean()
    else:
        unknown_acc = 0.0

    # H-score
    h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-8)

    # AUROC: 将开集检测视为二分类 (known vs unknown)
    binary_true = is_unknown_true.astype(int)
    try:
        auroc = roc_auc_score(binary_true, y_scores)
    except ValueError:
        auroc = 0.0

    return {
        "known_acc": known_acc,
        "unknown_acc": unknown_acc,
        "h_score": h_score,
        "auroc": auroc,
    }
