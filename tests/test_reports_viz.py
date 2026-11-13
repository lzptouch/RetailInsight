"""报表、Excel 导出与图表的单测（基于临时目录的 fixture 数据）。"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.reports import core_reports
from retailinsight.reports.excel_report import export_excel
from retailinsight.viz import charts


def _seed_interim(tmp_path, monkeypatch):
    """在临时目录造出 ETL 之后的 5 张中间表，并让 core_reports 指向它。"""
    members = pd.DataFrame({"member_id": [1, 2, 3], "reg_date": pd.to_datetime(["2023-01-01"] * 3),
                            "gender": ["男", "女", "男"], "age_band": ["25-34"] * 3,
                            "city_tier": ["一线"] * 3, "store_id": [1, 1, 2]})
    rows = [
        (1, "2024-01-05", 100.0, 1), (1, "2024-02-08", 200.0, 1), (1, "2024-03-15", 50.0, 2),
        (2, "2024-02-10", 80.0, 1), (2, "2024-03-11", 120.0, 2),
        (3, "2024-01-20", 60.0, 1),
    ]
    txn = pd.DataFrame(rows, columns=["member_id", "ts", "amount", "store_id"])
    txn["txn_ts"] = pd.to_datetime(txn["ts"])
    txn["txn_id"] = range(1, len(txn) + 1)
    txn["member_type"] = "会员"
    txn["is_return"] = False
    txn["is_logic_error"] = False
    txn["net_amount"] = txn["amount"]
    txn["discount"] = 0.0
    items = pd.DataFrame({"txn_id": txn["txn_id"], "sku": [100001, 100002] * 3, "qty": 2,
                          "unit_price": 50.0})
    products = pd.DataFrame({"sku": [100001, 100002], "category": ["啤酒", "牛奶"],
                             "brand": ["青岛", "伊利"]})
    interim = tmp_path / "interim"
    interim.mkdir()
    for name, df in [("members", members), ("transactions", txn),
                     ("transaction_items", items), ("products", products)]:
        df.to_parquet(interim / f"{name}.parquet", index=False)
    monkeypatch.setattr(core_reports, "INTERIM_DIR", interim)
    monkeypatch.setattr(core_reports, "MARTS_DIR", tmp_path / "marts")
    return interim


def test_build_core_reports(tmp_path, monkeypatch):
    _seed_interim(tmp_path, monkeypatch)
    core = core_reports.build_core_reports()
    assert set(core) == {"monthly", "quarterly", "store", "category", "heatmap",
                         "funnel", "monthly_repurchase"}
    assert len(core["monthly"]) == 3                 # 2024-01/02/03
    assert {"gmv", "orders", "aov", "member_penetration", "mom", "yoy"} <= set(core["monthly"])
    assert len(core["store"]) == 2
    assert set(core["category"]["abc_class"]) <= {"A", "B", "C"}
    assert core["funnel"]["count"].iloc[0] == 3      # 注册会员数
    assert core["funnel"]["conversion"].iloc[0] == pytest.approx(1.0)


def test_export_excel_sheets(tmp_path, monkeypatch):
    _seed_interim(tmp_path, monkeypatch)
    core = core_reports.build_core_reports()
    # excel_report 从 MARTS_DIR 读 mart_rfm / profile / rules / churn_score
    marts = tmp_path / "marts"
    marts.mkdir(exist_ok=True)
    rfm = pd.DataFrame({"member_id": [1, 2, 3], "rfm_segment": ["重要价值客户"] * 3,
                        "monetary": [100.0, 200.0, 300.0], "recency_days": [10, 20, 30],
                        "frequency": [3, 2, 1]})
    rfm.to_parquet(marts / "mart_rfm.parquet", index=False)
    profile = pd.DataFrame({"member_id": [1, 2, 3], "gender": ["男", "女", "男"]})
    profile.to_parquet(marts / "mart_member_profile.parquet", index=False)
    rules = pd.DataFrame({"antecedents": ["啤酒"], "consequents": ["牛奶"], "lift": [2.0]})
    rules.to_parquet(marts / "mart_basket_rules.parquet", index=False)
    churn = pd.DataFrame({"member_id": [1, 2, 3], "churn_prob": [0.9, 0.5, 0.1],
                          "risk_level": ["高", "中", "低"], "recency_days": [10, 20, 30],
                          "frequency": [3, 2, 1], "monetary": [100.0, 200.0, 300.0]})
    churn.to_parquet(marts / "mart_churn_score.parquet", index=False)

    monkeypatch.setattr("retailinsight.reports.excel_report.MARTS_DIR", marts)
    monkeypatch.setattr("retailinsight.reports.excel_report.REPORTS_DIR", tmp_path)
    path = export_excel(core)
    assert path.exists()
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    names = [ws.title for ws in wb.worksheets]
    for expect in ["1_经营总览", "1b_季度分析", "2_客群结构", "3_会员画像(前2万)",
                   "4_商品洞察", "5_流失预警"]:
        assert expect in names, names


def test_charts_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "FIGURES_DIR", tmp_path)
    monthly = pd.DataFrame({"month": pd.period_range("2024-01", periods=3, freq="M"),
                            "gmv": [100.0, 200.0, 300.0]})
    assert charts.gmv_trend(monthly).endswith(".png")
    assert charts.monthly_yoy(monthly).endswith(".png")
    assert charts.quarterly_growth(pd.DataFrame(
        {"quarter": pd.period_range("2024Q1", periods=2, freq="Q"), "gmv": [1.0, 2.0],
         "yoy": [np.nan, 0.5]})).endswith(".png")
    store = pd.DataFrame({"store_id": [1, 2], "gmv": [10.0, 20.0]})
    assert charts.store_rank(store).endswith(".png")
    cat = pd.DataFrame({"category": ["啤酒", "牛奶"], "gmv": [10.0, 20.0]})
    assert charts.category_pareto(cat).endswith(".png")
    pivot = pd.DataFrame([[1, 2], [3, 4]], index=range(2), columns=range(2))
    assert charts.hour_heatmap(pivot).endswith(".png")
    seg = pd.DataFrame({"rfm_segment": ["A", "B"], "member_cnt": [10, 20], "gmv_share": [0.3, 0.7]})
    assert charts.segment_bar(seg).endswith(".png")
    assert charts.segment_value(seg).endswith(".png")
    assert charts.lift_curve(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])).endswith(".png")
    assert charts.churn_hist(np.array([0.1, 0.5, 0.9])).endswith(".png")
    assert charts.cluster_elbow({2: {"inertia": 100.0, "silhouette": 0.4},
                                  3: {"inertia": 80.0, "silhouette": 0.45}}, 0.45).endswith(".png")


def test_cluster_radar_and_segment_radar(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "FIGURES_DIR", tmp_path)
    desc = pd.DataFrame({
        "cluster_id": [0, 1], "cluster_name": ["高价值活跃", "沉睡流失"],
        "active_pct": [0.8, 0.2], "frequency_pct": [0.9, 0.1], "monetary_pct": [0.9, 0.1],
        "avg_basket": [200.0, 80.0], "avg_n_categories": [20.0, 5.0], "avg_monetary": [900.0, 50.0],
    })
    assert charts.cluster_radar(desc).endswith(".png")
    profile = pd.DataFrame({
        "rfm_segment": ["重要价值客户", "流失风险客户"],
        "avg_basket": [200.0, 80.0], "monthly_amount": [300.0, 50.0],
        "yearly_freq": [20.0, 2.0], "items_per_txn": [3.0, 1.0],
        "category_entropy": [2.0, 1.0], "return_rate": [0.01, 0.05],
    })
    assert charts.segment_radar(profile).endswith(".png")
