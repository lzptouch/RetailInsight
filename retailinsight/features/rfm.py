"""RFM 指标与 8 类客群分群（指标单一真相源，notebook/看板只调用不重算）。

口径（写入 docs/指标体系.md）：
- 基准日 = 数据最大交易日期 + 1 天（描述性分析，覆盖全量 2 年）
- R = 基准日 - 最近一次消费日期（天）
- F = 观察窗内消费次数，同日多单合并计 1 次
- M = 观察窗内净消费额（实付 - 退货），退货单参与净额但不参与 F
- 分群：R/F/M 各自五分位打分(1~5)，≥3 记「高」，按 2x2x2 规则映射 8 类客群
"""
import pandas as pd

SEGMENT_RULES = {
    (1, 1, 1): "重要价值客户",
    (1, 0, 1): "重要发展客户",
    (0, 1, 1): "重要保持客户",
    (0, 0, 1): "重要挽留客户",
    (1, 1, 0): "一般价值客户",
    (1, 0, 0): "潜力客户",
    (0, 1, 0): "一般保持客户",
    (0, 0, 0): "流失风险客户",
}


def compute_rfm(txn: pd.DataFrame, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """txn 需为清洗后交易表（含 member_type / is_return / net_amount / is_logic_error）。
    仅统计会员、非逻辑异常交易。"""
    df = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])].copy()
    df["txn_date"] = df["txn_ts"].dt.normalize()
    if end_date is None:
        end_date = df["txn_ts"].max().normalize() + pd.Timedelta(days=1)

    g = df.groupby("member_id")
    rfm = pd.DataFrame({
        "last_date": g["txn_date"].max(),
        "frequency": g["txn_date"].nunique(),          # 同日多单合并
        "monetary": g["net_amount"].sum(),              # 净额（退货为负）
        "txn_count": g.size(),
        "return_amount": g.apply(lambda x: x.loc[x["is_return"], "amount"].sum(), include_groups=False),
    })
    rfm["recency_days"] = (end_date - rfm["last_date"]).dt.days
    rfm = rfm.drop(columns=["last_date"])
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm.reset_index()


def _quintile_score(s: pd.Series, reverse: bool = False) -> pd.Series:
    """五分位打分 1~5；reverse=True 时值越小分越高（用于 R）。"""
    ranks = s.rank(method="first")
    q = pd.qcut(ranks, 5, labels=False) + 1
    return (6 - q) if reverse else q


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    df = rfm.copy()
    df["r_score"] = _quintile_score(df["recency_days"], reverse=True).astype(int)
    df["f_score"] = _quintile_score(df["frequency"]).astype(int)
    df["m_score"] = _quintile_score(df["monetary"]).astype(int)
    return df


def assign_segment(scored: pd.DataFrame, high_threshold: int = 4) -> pd.DataFrame:
    """五分位打分 1~5，≥4（前 40%）记「高」，2x2x2 映射 8 类客群。"""
    df = scored.copy()
    key = list(zip(df["r_score"] >= high_threshold,
                   df["f_score"] >= high_threshold,
                   df["m_score"] >= high_threshold, strict=False))
    df["rfm_segment"] = [SEGMENT_RULES[(int(r), int(f), int(m))] for r, f, m in key]
    return df
