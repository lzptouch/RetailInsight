"""CLV 客户终身价值（计划 §8.4 / R-6）。

**两种口径，主口径是 BG/NBD**（v3.7 起）：

1. **BG/NBD + Gamma-Gamma**（``models/clv_bgnbd.py``）—— 主口径。
   用个体购买节奏推断「还在不在」与「多久买一次」，给出未来 12 个月期望毛利，
   并有 91 天时间外回测做同台比较（金额 MAE 229.27 vs 270.81，降 15.3%，
   见 `reports/metrics_clv.json`）。
2. **简化法**（本模块）—— 对照口径，也是 lifetimes 缺失时的降级路径。

简化法的好处是每个输入都来自已有产物、可逐行追溯；代价是假设了
「留存期服从几何分布、月均消费恒定」，同一客群的人会被给出几乎相同的价值。

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
    # 下限裁到 0：净额含退货，会把「退得比买得多」的人算成负 CLV。
    # 客户价值的语义下限是「不贡献」，不是「倒扣」。
    # 无交易会员的 `monthly_amount` 是 NaN，CLV 应当是 0（注册未购），不是 NaN ——
    # 否则下游 groupby 会把这些人整行丢掉，「客群 CLV 均值」只统计了买过的人。
    df["clv_simple"] = (arpu * df["expected_lifetime_months"] * gross_margin
                        ).round(2).clip(lower=0).fillna(0.0)
    # 默认口径先落在简化法上；接入 BG/NBD 后由 apply_bgnbd_clv 覆盖。
    # 「clv 永远是当前采用的口径」这条约定让卡片 / 看板 / Excel 都不用关心来源。
    df["clv"] = df["clv_simple"]
    df["clv_method"] = "simple_fallback"
    df["clv_tier"] = _clv_tier(df["clv"])
    return df


def _clv_tier(clv: pd.Series) -> pd.Series:
    """按 CLV 四分位分层（低 / 中低 / 中高 / 高）。"""
    return pd.qcut(clv.rank(method="first"), 4,
                   labels=["低CLV", "中低CLV", "中高CLV", "高CLV"])


def apply_bgnbd_clv(profile: pd.DataFrame, clv_bgnbd: pd.Series) -> pd.DataFrame:
    """把 BG/NBD 的 CLV 接成**主口径**，简化法保留为对照列。

    参数
    ----
    profile: ``compute_clv`` 的输出（已含 ``clv_simple``）
    clv_bgnbd: BG/NBD 预测的 CLV，索引为 member_id

    落在画像里没有任何 BG/NBD 值的会员（新客、只买过一次且被模型判为
    极低频的人）会退回 ``clv_simple``，并在 ``clv_method`` 逐行标注 ——
    口径混用必须可追溯，不能只看一列数字。
    """
    if "clv_simple" not in profile.columns:
        raise ValueError("apply_bgnbd_clv 需要先跑 compute_clv（缺 clv_simple 列）")

    df = profile.copy()
    bg = clv_bgnbd.rename("clv_bgnbd").reset_index()
    bg.columns = ["member_id", "clv_bgnbd"]
    df = df.merge(bg, on="member_id", how="left")

    covered = df["clv_bgnbd"].notna()
    df["clv"] = np.where(covered, df["clv_bgnbd"], df["clv_simple"]).round(2)
    df["clv_method"] = np.where(
        covered, "bgnbd",
        np.where(df["clv_simple"] > 0, "simple_fallback", "no_purchase"))
    df["clv_tier"] = _clv_tier(df["clv"])
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
