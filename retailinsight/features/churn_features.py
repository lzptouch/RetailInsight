"""流失预测特征工程（观察窗内一次性实现，31 维）。

**防泄漏铁律**：所有特征只能使用 ``[cutoff - OBS_WINDOW_DAYS, cutoff]`` 区间内的数据。
标签只能使用 ``(cutoff, cutoff + LABEL_WINDOW_DAYS]`` 区间。

特征分组：
- 体量维度（12）：频次/金额/客单/最近购买/会龄
- 节奏维度（6）：购买间隔统计、趋势斜率、近端窗口强度
- 广度维度（4）：品类/品牌熵、最大品类占比、品类数
- 风险维度（4）：退货笔数、退货金额占比、折扣依赖、跨店程度
- 其他维度（2）：周末占比、季节性强弱

所有特征只在这里实现一次，模型与 notebook 共用（计划 §3.1 单一真相源纪律）。
"""
import numpy as np
import pandas as pd

from retailinsight.features.common import cv, entropy
from retailinsight.settings import LABEL_WINDOW_DAYS, OBS_WINDOW_DAYS

#: 31 维特征清单（顺序即建模列顺序）
CHURN_FEATURES = [
    # 体量维度
    "recency_days", "frequency", "monetary", "avg_basket", "tenure_days",
    "total_amount", "avg_txn_amount", "n_txn",
    # 节奏维度
    "gap_mean", "gap_std", "interval_cv", "trend_90d", "freq_slope", "amt_slope",
    "freq_30d", "freq_90d", "amt_30d", "amt_90d", "days_since_2nd_last",
    # 广度维度
    "n_categories", "items_per_txn", "category_entropy", "brand_entropy",
    "top1_category_share",
    # 风险维度
    "refund_cnt", "refund_amt_ratio", "discount_ratio", "coupon_used_rate", "store_cnt",
    # 其他维度
    "weekend_ratio", "seasonality_strength",
]


def _window(obs: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    """取 (cutoff - days, cutoff] 区间内的观察窗数据。"""
    return obs[obs["txn_ts"] > cutoff - pd.Timedelta(days=days)]


def build_features(txn: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame,
                   members: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """构造观察窗内的特征矩阵（member_id + CHURN_FEATURES）。

    参数
    ----
    txn: 交易表（清洗后），需含 txn_id / member_id / txn_ts / net_amount / discount /
         store_id / member_type / is_return / is_logic_error
    items: 交易行明细，需含 txn_id / sku / qty
    products: 商品维度，需含 sku / category / brand
    members: 会员维度，需含 member_id / reg_date
    cutoff: 观察窗截止日（特征只能用到这一天）
    """
    obs_start = cutoff - pd.Timedelta(days=OBS_WINDOW_DAYS)
    t = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    obs = t[(t["txn_ts"] > obs_start) & (t["txn_ts"] <= cutoff)]

    g = obs.groupby("member_id")
    feat = pd.DataFrame(index=g.size().index)

    # ---- 体量维度 ----
    feat["frequency"] = obs.groupby("member_id")["txn_ts"].apply(
        lambda s: s.dt.normalize().nunique())
    feat["monetary"] = g["net_amount"].sum()
    feat["avg_basket"] = g["net_amount"].mean()
    feat["total_amount"] = g["net_amount"].sum()
    feat["avg_txn_amount"] = g["net_amount"].mean()
    feat["n_txn"] = g.size()
    feat["recency_days"] = (cutoff - g["txn_ts"].max()).dt.days

    # ---- 节奏维度 ----
    dates = obs[["member_id", "txn_ts"]].sort_values(["member_id", "txn_ts"])
    gaps = dates.groupby("member_id")["txn_ts"].diff().dt.days
    feat["gap_mean"] = gaps.groupby(dates["member_id"]).mean()
    feat["gap_std"] = gaps.groupby(dates["member_id"]).std().fillna(0)
    feat["interval_cv"] = (feat["gap_std"] / (feat["gap_mean"].abs() + 1e-9)).round(4)

    last90 = _window(obs, cutoff, 90)
    prev90 = obs[(obs["txn_ts"] > cutoff - pd.Timedelta(days=180))
                 & (obs["txn_ts"] <= cutoff - pd.Timedelta(days=90))]
    n_last90 = last90.groupby("member_id").size()
    n_prev90 = prev90.groupby("member_id").size()
    feat["trend_90d"] = (n_last90.reindex(feat.index, fill_value=0)
                         / (n_prev90.reindex(feat.index, fill_value=0) + 1))
    feat["freq_slope"] = ((n_last90.reindex(feat.index, fill_value=0)
                           - n_prev90.reindex(feat.index, fill_value=0))
                          / (n_prev90.reindex(feat.index, fill_value=0) + 1)).round(4)

    w30 = _window(obs, cutoff, 30)
    feat["freq_30d"] = w30.groupby("member_id").size().reindex(feat.index, fill_value=0)
    feat["freq_90d"] = n_last90.reindex(feat.index, fill_value=0)
    feat["amt_30d"] = (w30.groupby("member_id")["net_amount"].sum()
                       .reindex(feat.index, fill_value=0)).round(2)
    feat["amt_90d"] = (last90.groupby("member_id")["net_amount"].sum()
                       .reindex(feat.index, fill_value=0)).round(2)
    amt_prev90 = prev90.groupby("member_id")["net_amount"].sum().reindex(feat.index, fill_value=0)
    feat["amt_slope"] = ((feat["amt_90d"] - amt_prev90) / (amt_prev90.abs() + 1e-9)).round(4)

    last_two = obs.groupby("member_id")["txn_ts"].apply(lambda s: s.nlargest(2).tolist())
    feat["days_since_2nd_last"] = last_two.apply(
        lambda v: (cutoff - v[1]).days if len(v) >= 2 else np.nan)

    # ---- 广度维度 ----
    it = items[items["txn_id"].isin(obs["txn_id"])].merge(
        obs[["txn_id", "member_id"]], on="txn_id")
    # 品牌是可选维度：部分数据源没有 brand 列，此时退化为单品牌（熵恒为 0）
    prod_cols = ["sku", "category"] + (["brand"] if "brand" in products.columns else [])
    it = it.merge(products[prod_cols], on="sku")
    if "brand" not in it.columns:
        it["brand"] = "UNKNOWN"
    cat_cnt = it.groupby(["member_id", "category"]).size().rename("n").reset_index()
    feat["n_categories"] = cat_cnt.groupby("member_id")["category"].nunique()
    feat["items_per_txn"] = it.groupby("member_id").size() / feat["n_txn"]
    feat["category_entropy"] = cat_cnt.groupby("member_id")["n"].apply(entropy).round(3)
    brand_cnt = it.groupby(["member_id", "brand"]).size().rename("n").reset_index()
    feat["brand_entropy"] = brand_cnt.groupby("member_id")["n"].apply(entropy).round(3)
    feat["top1_category_share"] = (
        cat_cnt.groupby("member_id")["n"].max() / it.groupby("member_id").size()).round(4)

    # ---- 风险维度 ----
    feat["refund_cnt"] = obs[obs["is_return"]].groupby("member_id").size().reindex(
        feat.index, fill_value=0)
    refund_amt = obs[obs["is_return"]].groupby("member_id")["net_amount"].apply(
        lambda s: s.abs().sum()).reindex(feat.index, fill_value=0)
    feat["refund_amt_ratio"] = (refund_amt / (feat["total_amount"].abs() + 1e-9)).round(4)
    feat["discount_ratio"] = (g["discount"].sum()
                              / (g["net_amount"].sum().abs() + 1e-9)).round(4)
    feat["coupon_used_rate"] = (obs.assign(_c=obs["discount"] > 0).groupby("member_id")["_c"]
                                .mean().reindex(feat.index, fill_value=0)).round(4)
    feat["store_cnt"] = g["store_id"].nunique()

    # ---- 其他维度 ----
    feat["weekend_ratio"] = (obs[obs["txn_ts"].dt.day_of_week >= 5].groupby("member_id").size()
                             / feat["n_txn"]).fillna(0).round(4)
    mo = obs.assign(_m=obs["txn_ts"].dt.to_period("M"))
    monthly = mo.groupby(["member_id", "_m"]).size().unstack(fill_value=0)
    feat["seasonality_strength"] = monthly.apply(cv, axis=1).reindex(feat.index).fillna(0).round(4)

    # ---- 会龄 ----
    reg = members.set_index("member_id")["reg_date"]
    feat["tenure_days"] = (cutoff - reg.reindex(feat.index)).dt.days

    feat = feat.replace([float("inf"), float("-inf")], np.nan)
    return feat.reset_index().rename(columns={"index": "member_id"})


def build_labels(txn: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """构造标签：标签窗内 0 交易 = 流失（1）。

    标签窗严格在 cutoff 之后，与观察窗物理隔离。
    """
    label_end = cutoff + pd.Timedelta(days=LABEL_WINDOW_DAYS)
    t = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    label = t[(t["txn_ts"] > cutoff) & (t["txn_ts"] <= label_end)]
    active_ids = label["member_id"].unique()
    return pd.DataFrame({"member_id": active_ids})
