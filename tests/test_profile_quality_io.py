"""画像、ETL 质量、IO 的单测。"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.etl.quality import build_quality_report, missing_rates, write_quality_report
from retailinsight.features.tags import build_profile
from retailinsight.io.loader import optimize_dtypes, read_parquet, write_parquet


def _profile_fixture():
    members = pd.DataFrame({
        "member_id": [1, 2, 3],
        "reg_date": pd.to_datetime(["2023-01-01"] * 3),
        "gender": ["男", "女", "男"],
        "age_band": ["25-34"] * 3,
        "city_tier": ["一线", "二线", "三线"],
        "store_id": [1, 1, 2],
    })
    rows = [
        (1, "2025-03-01", 100.0, 0.0, 1, False),
        (1, "2025-03-08", 200.0, 10.0, 1, False),
        (1, "2025-03-15", 50.0, 0.0, 2, False),
        (2, "2025-02-10", 80.0, 0.0, 1, False),
        (3, "2025-01-05", 60.0, 0.0, 2, True),
    ]
    txn = pd.DataFrame(rows, columns=["member_id", "ts", "amount", "discount", "store_id",
                                      "is_return"])
    txn["txn_ts"] = pd.to_datetime(txn["ts"])
    txn["txn_id"] = range(1, len(txn) + 1)
    txn["member_type"] = "会员"
    txn["is_logic_error"] = False
    txn["net_amount"] = txn["amount"]
    items = pd.DataFrame({"txn_id": txn["txn_id"], "sku": [100001, 100002, 100001, 100002, 100001],
                          "qty": 1, "unit_price": 100.0})
    products = pd.DataFrame({"sku": [100001, 100002], "category": ["啤酒", "牛奶"],
                             "brand": ["青岛", "伊利"]})
    rfm = pd.DataFrame({
        "member_id": [1, 2, 3],
        "recency_days": [10, 60, 120],
        "frequency": [3, 1, 1],
        "monetary": [350.0, 80.0, 60.0],
        "r_score": [5, 3, 1], "f_score": [5, 2, 1], "m_score": [5, 2, 1],
        "rfm_segment": ["重要价值客户", "一般保持客户", "流失风险客户"],
    })
    return members, txn, items, products, rfm


def test_build_profile_20_labels():
    members, txn, items, products, rfm = _profile_fixture()
    prof = build_profile(members, txn, items, products, rfm)
    assert len(prof) == 3
    for col in ["gender", "age_band", "city_tier", "store_id", "tenure_days",
                "avg_basket", "monthly_amount", "yearly_freq", "discount_sensitivity",
                "return_rate", "active_band", "weekend_ratio", "top_categories",
                "top_brands", "category_entropy", "items_per_txn",
                "recency_days", "frequency", "monetary", "r_score", "f_score", "m_score",
                "rfm_segment", "value_tier"]:
        assert col in prof.columns, col


def test_build_profile_with_churn_adds_labels():
    """P0 第 20 标签：churn_prob + churn_covered，未覆盖会员保留 NaN。"""
    members, txn, items, products, rfm = _profile_fixture()
    churn = pd.DataFrame({"member_id": [1, 2], "churn_prob": [0.91, 0.13]})
    prof = build_profile(members, txn, items, products, rfm, churn=churn).set_index("member_id")
    assert "churn_prob" in prof.columns
    assert "churn_covered" in prof.columns
    assert prof.loc[1, "churn_prob"] == pytest.approx(0.91)
    assert bool(prof.loc[1, "churn_covered"]) is True
    assert bool(prof.loc[3, "churn_covered"]) is False
    assert np.isnan(prof.loc[3, "churn_prob"])


def test_missing_rates_only_reports_nan_columns():
    df = pd.DataFrame({"a": [1, 2, None], "b": [1, 2, 3]})
    rates = missing_rates({"t": df})
    assert rates["t.a"] == pytest.approx(1 / 3, abs=1e-6)
    assert "t.b" not in rates


def test_build_quality_report_structure():
    rep = build_quality_report({"members": 100}, {"members": 98}, {"dedup_removed": 2})
    assert set(rep) == {"before", "after", "details"}
    assert rep["before"]["members"] == 100


def test_write_quality_report(tmp_path):
    rep = build_quality_report({"members": 100}, {"members": 98},
                                {"dedup_removed": 2, "missing_rates": {"members.x": 0.01}})
    path = write_quality_report(rep, tmp_path / "q.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "数据质量报告" in text
    assert "members" in text


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = write_parquet(df, tmp_path / "t.parquet")
    assert p.exists()
    pd.testing.assert_frame_equal(read_parquet(p), df)


def test_optimize_dtypes():
    df = pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"),
                       "amt": pd.Series([1.5, 2.5], dtype="float64"),
                       "cat": pd.Series(["a", "b"], dtype=object)})
    out = optimize_dtypes(df)
    assert out["id"].dtype.itemsize <= 4
    assert out["amt"].dtype.itemsize <= 4
    assert str(out["cat"].dtype) == "category"
