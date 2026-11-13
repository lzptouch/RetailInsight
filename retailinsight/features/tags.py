"""会员画像标签（P0 共 20 个 + P1 扩展 18 个 + CLV 4 列）。"""
import numpy as np
import pandas as pd

from retailinsight.features.behavior import discount_sensitivity, weekend_ratio
from retailinsight.features.p1_tags import build_p1_tags

HOUR_BANDS = {"凌晨(0-6)": (0, 6), "上午(6-12)": (6, 12), "下午(12-18)": (12, 18), "晚间(18-24)": (18, 24)}


def _hour_band(h: int) -> str:
    for name, (lo, hi) in HOUR_BANDS.items():
        if lo <= h < hi:
            return name
    return "晚间(18-24)"


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def build_profile(members: pd.DataFrame, txn: pd.DataFrame, items: pd.DataFrame,
                  products: pd.DataFrame, rfm_seg: pd.DataFrame,
                  churn: pd.DataFrame | None = None) -> pd.DataFrame:
    """输出画像宽表：基础属性 5 + 消费能力 5 + 行为偏好 5 + 生命周期 5。

    参数
    ----
    churn:
        可选。流失得分表（需含 ``member_id`` 与 ``churn_prob``）。传入后会合并
        ``churn_prob`` 与 ``churn_covered`` 两列，补齐 P0 第 20 个标签。

        注意：流失得分只覆盖末期 cohort 观察窗内的活跃会员，覆盖率有限
        （实测约 63%），未覆盖会员的 ``churn_prob`` 保留 NaN 并由
        ``churn_covered=False`` 标注，避免用均值/0 填充引入偏差。
    """
    end_date = txn["txn_ts"].max().normalize() + pd.Timedelta(days=1)
    m_txn = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])].copy()

    # ---- 消费能力 ----
    g = m_txn.groupby("member_id")
    cap = pd.DataFrame({
        "avg_basket": g["net_amount"].mean().round(2),          # 客单价
        "total_amount": g["net_amount"].sum().round(2),
        "total_discount": g["discount"].sum(),
    })
    months = max((end_date - m_txn["txn_ts"].min()).days / 30.4, 1)
    cap["monthly_amount"] = (cap["total_amount"] / months).round(2)
    cap["discount_sensitivity"] = discount_sensitivity(txn).reindex(cap.index)
    ret = g.apply(lambda x: abs(x.loc[x["is_return"], "amount"].sum()), include_groups=False)
    pos = g.apply(lambda x: x.loc[~x["is_return"], "amount"].sum(), include_groups=False)
    cap["return_rate"] = (ret / (pos + 1e-9)).round(4)
    cap = cap.drop(columns=["total_discount"])

    # ---- 行为偏好 ----
    m_txn["hour"] = m_txn["txn_ts"].dt.hour
    m_txn["band"] = m_txn["hour"].map(_hour_band)
    band_mode = (m_txn.groupby(["member_id", "band"]).size().rename("n").reset_index()
                 .sort_values("n", ascending=False).drop_duplicates("member_id")
                 .set_index("member_id")["band"].rename("active_band"))
    wk_ratio = weekend_ratio(txn).reindex(m_txn.groupby("member_id").size().index)

    # 品类偏好：items × txn × products
    it = items.merge(m_txn[["txn_id", "member_id"]], on="txn_id")
    it = it.merge(products[["sku", "category", "brand"]], on="sku")
    cat_cnt = it.groupby(["member_id", "category"]).size().rename("n").reset_index()
    top_cat = (cat_cnt.sort_values("n", ascending=False).groupby("member_id")
               .head(3).groupby("member_id")["category"].agg("/".join).rename("top_categories"))
    cat_entropy = cat_cnt.groupby("member_id")["n"].agg(_entropy).rename("category_entropy").round(3)
    top_brand = (it.groupby(["member_id", "brand"]).size().rename("n").reset_index()
                 .sort_values("n", ascending=False).groupby("member_id")
                 .head(3).groupby("member_id")["brand"].agg("/".join).rename("top_brands"))
    items_per_txn = (it.groupby("member_id").size() / g.size()).rename("items_per_txn").round(2)

    # ---- 生命周期 / 价值分层 ----
    rfm = rfm_seg.set_index("member_id")
    m_tercile = pd.qcut(rfm["monetary"].rank(method="first"), 3, labels=["低价值", "中价值", "高价值"])

    profile = members[["member_id", "gender", "age_band", "city_tier", "store_id", "reg_date"]].copy()
    profile["tenure_days"] = (end_date - profile["reg_date"]).dt.days
    profile = profile.drop(columns=["reg_date"]).set_index("member_id")
    profile = profile.join([cap, band_mode, wk_ratio, top_cat, top_brand,
                            cat_entropy, items_per_txn])
    profile["active_band"] = profile["active_band"].fillna("无消费")
    profile["weekend_ratio"] = profile["weekend_ratio"].fillna(0)
    profile = profile.join(rfm[["recency_days", "frequency", "monetary",
                                "r_score", "f_score", "m_score", "rfm_segment"]])
    profile["yearly_freq"] = (
        profile["frequency"] / (profile["tenure_days"] / 365.0).clip(lower=0.08)).round(1)
    profile["value_tier"] = m_tercile

    # ---- 生命周期第 5 项：流失概率（P0 标签 20/20）----
    if churn is not None:
        c = churn[["member_id", "churn_prob"]].drop_duplicates("member_id").set_index("member_id")
        profile = profile.join(c)
        profile["churn_covered"] = profile["churn_prob"].notna()
        profile["churn_prob"] = profile["churn_prob"].astype("float64")

    # ---- P1 扩展标签（计划 §7.3 的 20 个）----
    # 由 features/p1_tags.py 一次性实现，画像层与聚类层共用（单一真相源）
    p1 = build_p1_tags(txn, items, products, rfm_seg)
    if len(p1.columns):
        profile = profile.join(p1)

    return profile.reset_index()
