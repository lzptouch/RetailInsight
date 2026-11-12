"""特征工程通用工具（多处复用的基础算子）。"""
import numpy as np
import pandas as pd


def entropy(counts: pd.Series) -> float:
    """香农熵（以 2 为底），衡量分布的分散程度。

    用于「品类熵」「品牌熵」等标签：熵越高说明购买越分散、品类越杂。
    """
    s = counts.astype(float)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(-(p * np.log2(p + 1e-12)).sum())


def cv(values: pd.Series) -> float:
    """变异系数（标准差 / 均值），衡量波动幅度且无量纲。"""
    s = values.astype(float)
    m = s.mean()
    if not np.isfinite(m) or abs(m) < 1e-12:
        return 0.0
    return float(s.std() / m)
