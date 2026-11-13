"""Excel 多 sheet 报表导出（甲方交付形态）。"""
from datetime import date
from pathlib import Path

import pandas as pd

from retailinsight.io.loader import read_parquet
from retailinsight.settings import MARTS_DIR, REPORTS_DIR


def export_excel(core: dict[str, pd.DataFrame]) -> Path:
    path = REPORTS_DIR / f"retailinsight_报表_{date.today():%Y%m%d}.xlsx"

    seg = read_parquet(MARTS_DIR / "mart_rfm.parquet")
    seg_summary = (seg.groupby("rfm_segment")
                   .agg(member_cnt=("member_id", "count"),
                        avg_monetary=("monetary", "mean"),
                        avg_recency=("recency_days", "mean"),
                        avg_frequency=("frequency", "mean"),
                        gmv=("monetary", "sum"))
                   .round(2).reset_index())
    seg_summary["gmv_share"] = (seg_summary["gmv"] / seg_summary["gmv"].sum()).round(4)

    profile = read_parquet(MARTS_DIR / "mart_member_profile.parquet")
    rules = read_parquet(MARTS_DIR / "mart_basket_rules.parquet")
    churn = read_parquet(MARTS_DIR / "mart_churn_score.parquet")
    high_risk = (churn[churn["risk_level"] == "高"]
                 .sort_values("churn_prob", ascending=False)
                 [["member_id", "churn_prob", "risk_level", "recency_days",
                   "frequency", "monetary"]]
                 .head(5000))
    high_risk["建议动作"] = "定向高面额券 + 客服回访"

    monthly = core["monthly"].copy()
    monthly["month"] = monthly["month"].astype(str)
    monthly = monthly.rename(columns={"month": "月份", "gmv": "GMV(元)", "orders": "订单量",
                                      "aov": "客单价(元)", "member_penetration": "会员渗透率",
                                      "mom": "环比", "yoy": "同比"})
    quarterly = core["quarterly"].copy()
    quarterly["quarter"] = quarterly["quarter"].astype(str)
    quarterly = quarterly.rename(columns={"quarter": "季度", "gmv": "GMV(元)", "orders": "订单量",
                                          "aov": "客单价(元)", "mom": "环比", "yoy": "同比"})

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        monthly.to_excel(w, sheet_name="1_经营总览", index=False)
        quarterly.to_excel(w, sheet_name="1b_季度分析", index=False)
        seg_summary.to_excel(w, sheet_name="2_客群结构", index=False)
        profile.head(20000).to_excel(w, sheet_name="3_会员画像(前2万)", index=False)
        top_cat = core["category"].head(30)
        top_cat.to_excel(w, sheet_name="4_商品洞察", index=False, startrow=1)
        rules.head(50).to_excel(w, sheet_name="4_商品洞察", index=False,
                                startrow=len(top_cat) + 4)
        high_risk.to_excel(w, sheet_name="5_流失预警", index=False)
    print(f"[reports] excel -> {path}")
    return path


if __name__ == "__main__":
    from retailinsight.reports.core_reports import build_core_reports
    export_excel(build_core_reports())
