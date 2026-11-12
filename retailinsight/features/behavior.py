"""行为模式特征（行为维度的一次性实现）。

这三类特征与 RFM 的「消费体量」维度互补：
- RFM 只看「买多少、多久买一次」，看不出「什么时候买、买不买折扣」
- 生成器埋的 8 类先验人群里，「促销敏感 / 周末囤货 / 季节性」三类
  在 RFM 上几乎不可分，只能靠这里的特征区分

画像标签（``features/tags.py``）与聚类特征（``features/cluster_features.py``）
都从这里取数，保证同一指标只有一处实现（计划 §3.1 单一真相源纪律）。
"""
import numpy as np
import pandas as pd


def _member_txn(txn: pd.DataFrame) -> pd.DataFrame:
    """会员且无逻辑错误的交易（口径与画像层保持一致）。"""
    return txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])].copy()


def weekend_ratio(txn: pd.DataFrame) -> pd.Series:
    """周末订单占比（周六 + 周日）。

    口径：周末订单数 / 该会员全部订单数。定义与画像层一致。
    """
    m = _member_txn(txn)
    is_weekend = m["txn_ts"].dt.day_of_week >= 5
    total = m.groupby("member_id").size()
    wk = m[is_weekend].groupby("member_id").size()
    # 没有任何周末订单的会员应为 0，而不是 NaN
    out = (wk / total).rename("weekend_ratio").fillna(0.0).round(4)
    return out


def discount_sensitivity(txn: pd.DataFrame) -> pd.Series:
    """折扣敏感度：折扣总额 / 消费总额（绝对值口径）。

    口径与画像层一致：sum(discount) / |sum(net_amount)|。
    """
    m = _member_txn(txn)
    g = m.groupby("member_id")
    disc = g["discount"].sum()
    # 与画像层口径逐位一致：先对净额求和并保留 2 位，再取绝对值做分母
    amt = g["net_amount"].sum().round(2).abs()
    return (disc / (amt + 1e-9)).rename("discount_sensitivity").round(4)


def seasonality_strength(txn: pd.DataFrame) -> pd.Series:
    """季节性强弱：会员月度订单数的变异系数（std / mean）。

    变异系数越大说明购买越集中在少数月份（如囤货型），
    越接近 0 说明购买均匀分布（如日常刚需型）。
    """
    m = _member_txn(txn).copy()
    m["month"] = m["txn_ts"].dt.to_period("M")
    cnt = m.groupby(["member_id", "month"]).size().unstack(fill_value=0)
    mean = cnt.mean(axis=1).replace(0, np.nan)
    cv = (cnt.std(axis=1) / mean).fillna(0.0)
    return cv.rename("seasonality_strength").round(4)


def behavior_features(txn: pd.DataFrame) -> pd.DataFrame:
    """一次性返回三个行为特征（index=member_id）。"""
    return pd.DataFrame({
        "weekend_ratio": weekend_ratio(txn),
        "discount_sensitivity": discount_sensitivity(txn),
        "seasonality_strength": seasonality_strength(txn),
    })
