"""Streamlit 看板：经营驾驶舱 / 客群分析 / 商品洞察 / 流失预警。

启动：.venv/bin/streamlit run dashboard/app.py
数据源只读 data/marts/，不在看板里重算指标（口径单一真相源在 retailinsight/features）。

交互设计（计划 §6.2 / §九 / R-4）
--------------------------------
- **plotly 散点（R × M，气泡=频次）**：hover 显示会员摘要
- **点选下钻**：点击任意散点 → 右侧展示该会员完整画像 + 同客群对比
- **客群筛选联动**：左侧筛选客群，散点与明细同步过滤
"""
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"

# ---- 首屏计时（计划 §11 W5 验收：首屏 < 3 秒）----
# Streamlit 每次交互都会重跑脚本，所以计时口径是「脚本从开始跑到首屏内容
# 渲染完成」的耗时；@st.cache_data 命中缓存时这个值会显著变小。
_T0 = time.perf_counter()

st.set_page_config(page_title="RetailInsight 商超会员画像看板", layout="wide")


@st.cache_data(ttl=3600)
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(MARTS / f"{name}.parquet")


@st.cache_data(ttl=3600)
def load_optional(name: str) -> pd.DataFrame | None:
    """可选产物（不存在时返回 None，让看板在不同产出组合下都能起）。"""
    p = MARTS / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


st.title("RetailInsight · 商超会员用户画像看板")
tab1, tab2, tab3, tab4 = st.tabs(["经营驾驶舱", "客群分析", "商品洞察", "流失预警"])

# 首屏计时（放在标题之后、内容之前取一个中间值，最终值在脚本末尾刷新）
_st = st.sidebar.empty()

with tab1:
    st.subheader("核心指标")
    rfm = load("mart_rfm")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("分析会员数", f"{len(rfm):,}")
    c2.metric("总净消费额（万元）", f"{rfm['monetary'].sum() / 1e4:,.1f}")
    c3.metric("人均消费（元）", f"{rfm['monetary'].mean():,.0f}")
    c4.metric("人均消费频次（天）", f"{rfm['frequency'].mean():.1f}")

    st.subheader("月度 GMV 趋势（含环比/同比）")
    monthly = load("mart_monthly")
    st.line_chart(monthly.set_index(monthly["month"].astype(str))["gmv"])
    st.dataframe(monthly.assign(month=monthly["month"].astype(str)),
                 use_container_width=True)

    st.subheader("季度 GMV（含环比/同比）")
    quarterly = load("mart_quarterly")
    st.bar_chart(quarterly.set_index(quarterly["quarter"].astype(str))["gmv"])
    st.dataframe(quarterly.assign(quarter=quarterly["quarter"].astype(str)),
                 use_container_width=True)

with tab2:
    st.subheader("RFM 客群结构")
    seg = (rfm.groupby("rfm_segment")
           .agg(会员数=("member_id", "count"), 人均净消费=("monetary", "mean"),
                平均未消费天数=("recency_days", "mean"), 人均频次=("frequency", "mean"))
           .round(1).sort_values("会员数", ascending=False))
    st.dataframe(seg, use_container_width=True)
    st.bar_chart(seg["会员数"])

    # ---------------- 交互散点 + 点选下钻（R-4）----------------
    st.subheader("会员分布交互图（点击散点下钻）")
    profile = load("mart_member_profile")
    cluster = load_optional("mart_cluster")

    left, right = st.columns([3, 2])
    with left:
        seg_filter = st.multiselect(
            "按客群筛选（联动下方散点与明细）",
            sorted(profile["rfm_segment"].dropna().unique()))
        view = profile if not seg_filter else profile[profile["rfm_segment"].isin(seg_filter)]

        plot_df = view.dropna(subset=["recency_days", "monetary", "frequency"]).copy()
        plot_df["客群"] = plot_df["rfm_segment"]
        if cluster is not None and "cluster_name" in cluster.columns:
            plot_df = plot_df.merge(cluster[["member_id", "cluster_name"]],
                                    on="member_id", how="left")
            color_col = "cluster_name"
        else:
            color_col = "客群"

        fig = px.scatter(
            plot_df, x="recency_days", y="monetary", color=color_col,
            size="frequency", hover_name="member_id",
            hover_data={"客群": True, "recency_days": ":,.0f",
                        "monetary": ":,.0f", "frequency": ":,.0f"},
            log_y=True,
            title="R 值（距上次消费天数）× M 值（净消费额），气泡大小=消费频次",
            labels={"recency_days": "R 值（距上次消费天数）",
                    "monetary": "M 值（净消费额，元，对数轴）",
                    "frequency": "消费频次", color_col: color_col})
        fig.update_traces(marker={"opacity": 0.65, "line": {"width": 0.5}})
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                                use_container_width=True)

        st.dataframe(view.sample(min(500, len(view)), random_state=42),
                     use_container_width=True)

    with right:
        st.subheader("会员下钻详情")
        picked = None
        sel = (event or {}).get("selection", {}) if event else {}
        points = sel.get("points", []) if isinstance(sel, dict) else []
        if points:
            mid = points[0].get("customdata")
            if mid is not None:
                picked = view[view["member_id"] == mid].head(1)

        if picked is None:
            st.info("在上方散点图中**点击任意一个点**，即可查看该会员的完整画像与同客群对比。")
            picked = view.head(1)

        row = picked.iloc[0]
        st.markdown(f"**会员 {int(row['member_id'])}**")
        kv1, kv2 = st.columns(2)
        kv1.metric("客群", str(row.get("rfm_segment", "-")))
        kv2.metric("流失概率", f"{row['churn_prob']:.1%}"
                   if pd.notna(row.get("churn_prob")) else "未覆盖")
        kv1.metric("净消费额（元）", f"{row.get('monetary', 0):,.0f}")
        kv2.metric("消费频次", f"{row.get('frequency', 0):.0f}")
        kv1.metric("距上次消费（天）", f"{row.get('recency_days', 0):.0f}")
        kv2.metric("CLV（元）", f"{row.get('clv', 0):,.0f}"
                   if pd.notna(row.get("clv")) else "-")
        st.caption(f"偏好品类：{row.get('top_categories', '-')}　"
                   f"偏好品牌：{row.get('top_brands', '-')}　"
                   f"折扣敏感度：{row.get('discount_sensitivity', 0):.2f}")

        seg_name = row.get("rfm_segment")
        peers = view[view["rfm_segment"] == seg_name]
        if len(peers) > 1:
            st.markdown(f"**同客群对比**（{seg_name}，{len(peers):,} 人）")
            cmp = pd.DataFrame({
                "指标": ["人均净消费", "人均频次", "平均 R 值"],
                "该会员": [row.get("monetary", 0), row.get("frequency", 0),
                           row.get("recency_days", 0)],
                "客群均值": [peers["monetary"].mean(), peers["frequency"].mean(),
                             peers["recency_days"].mean()],
            }).round(1)
            cmp["差异"] = (cmp["该会员"] / cmp["客群均值"].replace(0, pd.NA) - 1).map(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "-")
            st.dataframe(cmp, use_container_width=True)

with tab3:
    st.subheader("关联规则 Top 30（品类级）")
    rules = load("mart_basket_rules")
    st.dataframe(rules.head(30), use_container_width=True)

    sku_rules = load_optional("mart_basket_rules_sku")
    if sku_rules is not None:
        st.subheader("关联规则（SKU 级 · 单品级陈列建议）")
        st.caption(f"共 {len(sku_rules):,} 条，前 10 条按 lift 排序")
        st.dataframe(sku_rules.head(10), use_container_width=True)

    store_rules = load_optional("mart_basket_rules_store")
    if store_rules is not None:
        st.subheader("分门店关联规则（Top 20）")
        st.dataframe(store_rules.head(20), use_container_width=True)

    st.subheader("品类结构与时段")
    monthly = load("mart_monthly")
    st.dataframe(monthly.assign(month=monthly["month"].astype(str)),
                 use_container_width=True)

with tab4:
    st.subheader("流失预警名单")
    churn = load("mart_churn_score")
    st.caption(f"高危会员 {(churn['risk_level'] == '高').sum():,} 人")
    risk = st.select_slider("风险等级", ["低", "中", "高"], value="高")
    cols = ["member_id", "churn_prob", "risk_level", "recency_days", "frequency",
            "monetary", "top_shap_feature"]
    cols = [c for c in cols if c in churn.columns]
    view = churn[churn["risk_level"] == risk].sort_values("churn_prob", ascending=False)
    st.dataframe(view[cols].head(1000), use_container_width=True)
    st.download_button("导出高危名单 CSV",
                       churn[churn["risk_level"] == "高"][cols].to_csv(index=False),
                       "high_risk_members.csv", "text/csv")

    if "top_shap_feature" in churn.columns:
        st.subheader("归因特征分布（逐会员 Top 归因）")
        dist = churn["top_shap_feature"].value_counts().head(15).rename_axis("特征").reset_index(
            name="会员数")
        st.bar_chart(dist.set_index("特征")["会员数"])

# ---- 首屏耗时展示（计划 §11 W5：首屏 < 3 秒）----
_elapsed = time.perf_counter() - _T0
_st.info(f"⏱ 本次脚本执行耗时 **{_elapsed:.2f} 秒**（含缓存命中；验收线 3 秒）")
