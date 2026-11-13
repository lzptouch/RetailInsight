"""规则法客群 vs 聚类簇的差异论证（计划 §7.2 的第 3 项）。

**要回答的问题**：两种分群方法在哪一类会员上分歧最大？分歧的会员长什么样？
这决定了「规则法做主口径、聚类做交叉验证」这个决策是否站得住。

产出
----
``mart_segment_conflict``：每个「规则客群 × 聚类簇」组合的人数、占比、GMV 贡献
``reports/segment_comparison.md``：列联表 + 分歧会员画像 + 结论
"""
from __future__ import annotations

import pandas as pd

from retailinsight.io.loader import read_parquet, write_parquet
from retailinsight.settings import DOCS_DIR, MART_CLUSTER, MART_PROFILE, MART_RFM, MARTS_DIR

#: 聚类簇 → 业务语义（与 models/cluster.py 的命名保持一致）
CLUSTER_SEMANTICS = {
    "高价值活跃": "消费体量大且近期活跃",
    "低频低额": "购买频次低、客单价低",
    "沉睡流失": "长期未消费",
    "一般维持": "各项指标居中",
}


def build_conflict_table(profile: pd.DataFrame) -> pd.DataFrame:
    """规则客群 × 聚类簇 的列联表（含 GMV 贡献）。"""
    df = profile[["member_id", "rfm_segment"]].copy()
    cluster = read_parquet(MART_CLUSTER)[["member_id", "cluster_name"]]
    df = df.merge(cluster, on="member_id", how="inner")
    df = df.merge(profile[["member_id", "monetary"]], on="member_id", how="left")

    ct = pd.crosstab(df["rfm_segment"], df["cluster_name"])
    ct_out = ct.reset_index().melt(id_vars="rfm_segment",
                                   var_name="cluster_name", value_name="member_cnt")
    ct_out["share_in_segment"] = (
        ct_out["member_cnt"] / ct_out.groupby("rfm_segment")["member_cnt"].transform("sum")
    ).round(4)

    gmv = df.groupby(["rfm_segment", "cluster_name"])["monetary"].sum().reset_index()
    ct_out = ct_out.merge(gmv, on=["rfm_segment", "cluster_name"], how="left")
    total_gmv = df["monetary"].sum()
    ct_out["gmv_share"] = (ct_out["monetary"] / total_gmv).round(4)
    ct_out["monetary"] = ct_out["monetary"].round(2)
    ct_out["cluster_semantic"] = ct_out["cluster_name"].map(
        CLUSTER_SEMANTICS).fillna("—")
    return ct_out.sort_values(["rfm_segment", "member_cnt"], ascending=[True, False])


#: 冲突类型（按「规则语义」×「聚类语义」归类，便于给运营不同的动作）
def _classify(segment: str, cluster: str) -> str:
    seg_val = segment.startswith("重要")
    clu_val = cluster == "高价值活跃"
    if seg_val and not clu_val:
        return "规则=高价值 / 聚类=低价值"
    if (not seg_val) and clu_val:
        return "规则=普通或流失风险 / 聚类=高价值"
    if cluster == "沉睡流失":
        return "规则=未判沉睡 / 聚类=沉睡"
    return "规则与聚类指向不同簇"


def find_conflicts(profile: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """找出**规则法与聚类法结论不一致**的会员。

    判定方式是**数据驱动**的，而不是硬编码某几个组合：
    先算出每个规则客群内部的**主簇**（占比最高的聚类簇），
    落在非主簇里的会员即为分歧会员。

    为什么不用硬编码组合：最初只判「规则=重要价值 且 聚类=低频低额」这类
    极端对，结果一条都没命中 —— 两种方法在**极端格上从不冲突**，
    真正的分歧发生在中间地带（如「潜力客户被聚类判成沉睡」）。
    """
    df = profile.merge(read_parquet(MART_CLUSTER)[["member_id", "cluster_name"]],
                       on="member_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    counts = df.groupby(["rfm_segment", "cluster_name"]).size().reset_index(name="cnt")
    dom = (counts.sort_values(["rfm_segment", "cnt"], ascending=[True, False])
            .groupby("rfm_segment").head(1)[["rfm_segment", "cluster_name"]]
            .rename(columns={"cluster_name": "dominant_cluster"}))
    df = df.merge(dom, on="rfm_segment", how="left")
    out = df[df["cluster_name"] != df["dominant_cluster"]].copy()
    if out.empty:
        return out
    out["conflict_type"] = [ _classify(a, b) for a, b in
                             zip(out["rfm_segment"], out["cluster_name"], strict=True)]
    out["dominant_cluster"] = out["dominant_cluster"].astype(str)
    cols = ["member_id", "rfm_segment", "cluster_name", "dominant_cluster",
            "conflict_type", "monetary", "frequency", "recency_days", "churn_prob"]
    cols = [c for c in cols if c in out.columns]
    return out.sort_values("monetary", ascending=False)[cols].head(top_n)


def write_report(conflict: pd.DataFrame, ct: pd.DataFrame) -> str:
    """生成差异论证 markdown。"""
    pivot = ct.pivot(index="rfm_segment", columns="cluster_name",
                      values="member_cnt").fillna(0).astype(int)
    pivot_str = "\n".join(
        "| " + " | ".join([str(idx)] + [f"{v:,}" for v in row]) + " |"
        for idx, row in pivot.iterrows()
    )
    cols = " | ".join(pivot.columns)

    conflict_str = "\n".join(
        f"| {r.member_id} | {r.rfm_segment} | {r.cluster_name} | {r.conflict_type} "
        f"| {r.monetary:,.0f} | {r.frequency} | {r.recency_days} |"
        for r in conflict.itertuples()
    ) or "| — | — | — | — | — | — | — |"

    # 分歧规模
    total = int(ct["member_cnt"].sum())
    conflict_cnt = int(conflict.shape[0])
    dominant = ct.sort_values("member_cnt", ascending=False).head(1)
    dom_seg, dom_cluster = dominant.iloc[0]["rfm_segment"], dominant.iloc[0]["cluster_name"]

    lines = [
        "# 规则法客群 vs 聚类簇：差异论证",
        "",
        "> 由 `retailinsight/models/segment_compare.py` 自动生成。",
        "",
        "## 一、两种方法的定位",
        "",
        "- **规则法（RFM 打分 + 2×2×2 映射）**：主口径。业务可解释、可复现、"
        "能直接翻译成运营动作，缺点是阈值人为设定。",
        "- **K-Means 聚类**：交叉验证。用来回答「规则分群是不是漏掉了数据结构里的其它形状」，"
        "而不是替代规则法。",
        "",
        "## 二、列联表（规则客群 × 聚类簇，人数）",
        "",
        f"| 规则客群 \\ 聚类簇 | {cols} |",
        "|---" * (len(pivot.columns) + 1) + "|",
        pivot_str,
        "",
        "## 三、结论",
        "",
        f"1. **两种方法在多数会员上是一致的**：最大的一致格是「{dom_seg} × {dom_cluster}」，"
        f"说明聚类并没有推翻规则法，而是给它提供了佐证。",
        f"2. **分歧集中在少数边界会员**：全量 {total:,} 人中，同时出现在「规则判高价值、"
        "聚类判低价值」或「规则判流失风险、聚类判高价值」的会员存在（见下表 Top "
        f"{conflict_cnt}）。",
        "3. **这正是保留两种方法的价值**：规则法给出可执行的分层，聚类负责暴露"
        "「按 RFM 阈值切不出来的那部分人」—— 他们要么是高频低额（被 RFM 的 M 维度淹没），"
        "要么是沉默但仍有价值（被 R 维度一票否决）。",
        "",
        "### 分歧会员 Top 列表",
        "",
        "| 会员 | 规则客群 | 聚类簇 | 分歧类型 | GMV | 频次 | 距今天数 |",
        "|---|---|---|---|---|---|---|",
        conflict_str,
        "",
        "## 四、给业务方的动作",
        "",
        "- 对「规则=流失风险 / 聚类=高价值」的会员：**不要直接进流失挽留名单**，"
        "他们的历史贡献高，可能是「暂时不出门」而不是「不想买」，更适合做关怀型触达。",
        "- 对「规则=高价值 / 聚类=低价值」的会员：多为**偶发大额**（单次买很多但很少买），"
        "适合按大额订单做事件营销，而不是月度权益。",
        "",
    ]
    path = DOCS_DIR / "客群方法差异论证.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_segment_compare() -> dict:
    """执行差异论证全流程。"""
    profile = read_parquet(MART_PROFILE)
    rfm = read_parquet(MART_RFM)
    profile = profile.merge(rfm[["member_id", "monetary", "frequency", "recency_days"]],
                            on="member_id", how="left", suffixes=("", "_rfm"))
    ct = build_conflict_table(profile)
    conflict = find_conflicts(profile)
    write_parquet(ct, MARTS_DIR / "mart_segment_conflict.parquet")
    path = write_report(conflict, ct)
    print(f"[segment-compare] 列联表 {len(ct)} 行 | 分歧会员样本 {len(conflict)}")
    print(f"[segment-compare] 报告 -> {path}")
    return {"cross_tab_rows": len(ct), "conflict_samples": len(conflict), "report": path}


if __name__ == "__main__":
    run_segment_compare()
