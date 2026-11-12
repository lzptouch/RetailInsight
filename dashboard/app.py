"""Streamlit 看板：经营驾驶舱 / 客群分析 / 商品洞察 / 流失预警。

启动：.venv/bin/streamlit run dashboard/app.py
数据源只读 data/marts/，不在看板里重算指标（口径单一真相源在 retailinsight/features）。
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"

st.set_page_config(page_title="RetailInsight 商超会员画像看板", layout="wide")


@st.cache_data(ttl=3600)
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(MARTS / f"{name}.parquet")


st.title("RetailInsight · 商超会员用户画像看板")
tab1, tab2, tab3, tab4 = st.tabs(["经营驾驶舱", "客群分析", "商品洞察", "流失预警"])

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

    st.subheader("会员画像（抽样 500 条）")
    profile = load("mart_member_profile")
    seg_filter = st.multiselect("按客群筛选", sorted(profile["rfm_segment"].dropna().unique()))
    view = profile if not seg_filter else profile[profile["rfm_segment"].isin(seg_filter)]
    st.dataframe(view.sample(min(500, len(view)), random_state=42), use_container_width=True)

with tab3:
    st.subheader("关联规则 Top 30")
    rules = load("mart_basket_rules")
    st.dataframe(rules.head(30), use_container_width=True)

with tab4:
    st.subheader("流失预警名单")
    churn = load("mart_churn_score")
    st.caption(f"高危会员 {(churn['risk_level'] == '高').sum():,} 人")
    risk = st.select_slider("风险等级", ["低", "中", "高"], value="高")
    cols = ["member_id", "churn_prob", "risk_level", "recency_days", "frequency", "monetary"]
    view = churn[churn["risk_level"] == risk].sort_values("churn_prob", ascending=False)
    st.dataframe(view[cols].head(1000), use_container_width=True)
    st.download_button("导出高危名单 CSV",
                       churn[churn["risk_level"] == "高"][cols].to_csv(index=False),
                       "high_risk_members.csv", "text/csv")
