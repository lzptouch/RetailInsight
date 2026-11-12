"""CLV 客户终身价值（计划 §8.4 / R-6）。

**口径声明（重要）**：本项目实现的是**简化法**，不是 BG/NBD + Gamma-Gamma。
简化法的好处是每个输入都来自已有产物、可逐行追溯；代价是假设了
「留存期服从几何分布」，对现金流的时间分布做了强假设。

    月流失率   m = 1 - (1 - p_90d) ^ (1/3)      # p_90d 是 90 天标签窗的流失概率
    预期留存月数 L = 1 / m
    CLV = 月均消费 × L × 毛利率

毛利率是**假设值**（零售行业常见 25%~35%，本项目取 30%），必须标注。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 毛利率假设值（零售行业常见区间 25%~35%）
CLV_GROSS_MARGIN = 0.30

# 流失概率的标签窗长度（月）—— 与 models/churn.py 的 LABEL_WINDOW_DAYS=90 对齐
CHURN_LABEL_MONTHS = 3


def monthly_churn_rate(churn_prob_90d: pd.Series | float) -> pd.Series | float:
    """由 90 天流失概率换算月流失率（几何留存假设）。"""
    p = np.clip(pd.Series(churn_prob_90d, dtype="float64"), 0.0, 0.999999)
    return 1.0 - (1.0 - p) ** (1.0 / CHURN_LABEL_MONTHS)


def expected_lifetime_months(monthly_churn: pd.Series) -> pd.Series:
    """预期留存月数 = 1 / 月流失率（月流失率趋于 0 时封顶 120 个月）。"""
    return (1.0 / monthly_churn.clip(lower=1e-6)).clip(upper=120.0)


def compute_clv(profile: pd.DataFrame, churn: pd.DataFrame | None = None,
                gross_margin: float = CLV_GROSS_MARGIN,
                fallback_segment: str = "rfm_segment") -> pd.DataFrame:
    """给画像宽表补 CLV 相关列。

    参数
    ----
    profile: 画像宽表（需含 ``monthly_amount``、``churn_prob``）
    churn:   流失得分表（当 profile 没有 ``churn_prob`` 时从这里取）
    gross_margin: 毛利率假设值
    fallback_segment: 未被流失模型覆盖的会员，用该客群的**中位留存月数**兜底

    返回
    ----
    含 ``clv`` / ``expected_lifetime_months`` / ``monthly_churn_rate`` / ``clv_source`` 的副本
    """
    df = profile.copy()

    if churn is not None and "churn_prob" not in df.columns:
        c = churn[["member_id", "churn_prob"]].drop_duplicates("member_id")
        df = df.merge(c, on="member_id", how="left")

    if "churn_prob" not in df.columns:
        raise ValueError("compute_clv 需要 churn_prob 列：请先传入 churn 得分表")

    df["monthly_churn_rate"] = monthly_churn_rate(df["churn_prob"])
    df["expected_lifetime_months"] = expected_lifetime_months(df["monthly_churn_rate"])

    # 未被模型覆盖的会员：用同客群的中位留存月数兜底（并标注来源）
    df["clv_source"] = "model"
    missing = df["expected_lifetime_months"].isna()
    if missing.any() and fallback_segment in df.columns:
        seg_median = (df.groupby(fallback_segment)["expected_lifetime_months"]
                      .transform("median"))
        df.loc[missing, "expected_lifetime_months"] = seg_median[missing]
        df.loc[missing, "clv_source"] = "segment_median"
    df["expected_lifetime_months"] = df["expected_lifetime_months"].fillna(
        df["expected_lifetime_months"].median())

    arpu = df["monthly_amount"] if "monthly_amount" in df.columns else df.get("monetary")
    df["clv"] = (arpu * df["expected_lifetime_months"] * gross_margin).round(2)
    df["clv_tier"] = pd.qcut(df["clv"].rank(method="first"), 4,
                              labels=["低CLV", "中低CLV", "中高CLV", "高CLV"])
    return df


def clv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """CLV 分层汇总（写进指标文档与画像卡片）。"""
    return (df.groupby("clv_tier", observed=True)
            .agg(members=("member_id", "count"),
                 avg_clv=("clv", "mean"),
                 total_clv=("clv", "sum"),
                 avg_lifetime_months=("expected_lifetime_months", "mean"))
            .round(2)
            .sort_values("avg_clv"))
