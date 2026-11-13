"""特征工程单测：行为特征、通用算子、聚类特征、流失特征。"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.features.behavior import (
    behavior_features,
    discount_sensitivity,
    seasonality_strength,
    weekend_ratio,
)
from retailinsight.features.churn_features import CHURN_FEATURES, build_features, build_labels
from retailinsight.features.cluster_features import (
    ALL_FEATURES,
    CLUSTER_FEATURE_SPACES,
    CLUSTER_FEATURES,
    build_cluster_features,
)
from retailinsight.features.common import cv, entropy


def _toy_txn():
    """toy 交易：m1 周末买得多且用券，m2 工作日常买。"""
    rows = [
        (1, "2025-03-01", 100.0, 10.0, 1),   # 周六
        (1, "2025-03-02", 100.0, 0.0, 1),    # 周日
        (1, "2025-03-10", 50.0, 0.0, 2),     # 周一
        (2, "2025-03-11", 80.0, 0.0, 1),     # 周二
        (2, "2025-03-12", 80.0, 0.0, 2),     # 周三
    ]
    df = pd.DataFrame(rows, columns=["member_id", "ts", "net_amount", "discount", "store_id"])
    df["txn_ts"] = pd.to_datetime(df["ts"])
    df["txn_id"] = range(1, len(df) + 1)
    df["member_type"] = "会员"
    df["is_return"] = False
    df["is_logic_error"] = False
    return df.drop(columns=["ts"])


def test_entropy_and_cv():
    assert entropy(pd.Series([1, 1])) == pytest.approx(1.0)
    assert entropy(pd.Series([1])) == pytest.approx(0.0, abs=1e-6)
    assert entropy(pd.Series([0, 0])) == 0.0
    assert cv(pd.Series([10.0, 10.0, 10.0])) == pytest.approx(0.0)
    assert cv(pd.Series([0.0, 0.0])) == 0.0


def test_weekend_ratio():
    txn = _toy_txn()
    out = weekend_ratio(txn)
    assert out.loc[1] == pytest.approx(2 / 3, abs=1e-3)   # 3 单里 2 单在周末
    assert out.loc[2] == pytest.approx(0.0, abs=1e-3)


def test_discount_sensitivity():
    txn = _toy_txn()
    out = discount_sensitivity(txn)
    # m1: 折扣 10 / 净额 250
    assert out.loc[1] == pytest.approx(10 / 250, abs=1e-4)
    assert out.loc[2] == pytest.approx(0.0, abs=1e-4)


def test_seasonality_strength():
    txn = _toy_txn()
    out = seasonality_strength(txn)
    assert (out >= 0).all()
    assert out.loc[2] == pytest.approx(0.0, abs=1e-6)   # 同月两单 → 无月度波动


def test_behavior_features_bundle():
    out = behavior_features(_toy_txn())
    assert set(out.columns) == {"weekend_ratio", "discount_sensitivity", "seasonality_strength"}
    assert len(out) == 2


def _churn_fixture(with_brand=True):
    cutoff = pd.Timestamp("2025-04-01")
    members = pd.DataFrame({"member_id": [1, 2], "reg_date": [pd.Timestamp("2024-01-01")] * 2})
    txn = pd.DataFrame([
        (1, "2024-10-01", 100.0), (1, "2025-01-01", 100.0),
        (1, "2025-03-01", 100.0), (1, "2025-04-15", 100.0),
        (2, "2024-09-01", 100.0), (2, "2024-12-01", 100.0),
    ], columns=["member_id", "ts", "amount"])
    txn["txn_ts"] = pd.to_datetime(txn["ts"])
    txn["txn_id"] = range(1, len(txn) + 1)
    txn["member_type"] = "会员"
    txn["is_return"] = False
    txn["is_logic_error"] = False
    txn["net_amount"] = txn["amount"]
    txn["discount"] = 0.0
    txn["store_id"] = 1
    items = pd.DataFrame({"txn_id": txn["txn_id"], "sku": 100001, "qty": 1, "unit_price": 100.0})
    products = pd.DataFrame({"sku": [100001], "category": ["啤酒"]})
    if with_brand:
        products["brand"] = ["青岛"]
    return txn, items, products, members, cutoff


def test_churn_features_all_present():
    txn, items, products, members, cutoff = _churn_fixture()
    feat = build_features(txn, items, products, members, cutoff)
    for col in CHURN_FEATURES:
        assert col in feat.columns, col
    assert len(CHURN_FEATURES) == 31


def test_churn_features_brand_optional():
    """没有 brand 列的数据源应优雅降级，而不是报错。"""
    txn, items, products, members, cutoff = _churn_fixture(with_brand=False)
    feat = build_features(txn, items, products, members, cutoff)
    assert "brand_entropy" in feat.columns
    assert (feat["brand_entropy"] == 0).all()


def test_churn_features_no_label_leak():
    """特征只允许看到观察窗内数据：标签窗的交易不能影响任何特征。"""
    txn, items, products, members, cutoff = _churn_fixture()
    base = build_features(txn, items, products, members, cutoff).set_index("member_id")
    # 篡改标签窗（cutoff 之后）的交易金额，特征不应变化
    tampered = txn.copy()
    tampered.loc[tampered["txn_ts"] > cutoff, "net_amount"] = 99999
    after = build_features(tampered, items, products, members, cutoff).set_index("member_id")
    for col in CHURN_FEATURES:
        assert base.loc[1, col] == pytest.approx(after.loc[1, col], nan_ok=True), col


def test_build_labels():
    txn, items, products, members, cutoff = _churn_fixture()
    lab = build_labels(txn, cutoff)
    assert set(lab["member_id"]) == {1}   # 只有 m1 在标签窗有交易


def test_cluster_features_and_spaces():
    txn = _toy_txn()
    items = pd.DataFrame({"txn_id": txn["txn_id"], "sku": 100001, "qty": 1, "unit_price": 100.0})
    products = pd.DataFrame({"sku": [100001], "category": ["啤酒"]})
    rfm = pd.DataFrame({"member_id": [1, 2], "recency_days": [10, 5],
                        "frequency": [3, 2], "monetary": [250.0, 160.0]})
    feats = build_cluster_features(rfm, txn, items, products)
    assert len(feats) == 2
    for col in ALL_FEATURES:
        assert col in feats.columns, col
    assert set(CLUSTER_FEATURE_SPACES) == {"base6", "rfm3", "rfm3_behavior",
                                           "rfm3_behavior_p1"}
    assert CLUSTER_FEATURES == ["recency_days", "frequency", "monetary",
                                "avg_basket", "n_categories", "items_per_txn"]


def test_cluster_features_negative_monetary_kept():
    """退货大于消费的负值会员必须保留，不能被 clip 掉。"""
    txn = _toy_txn()
    txn.loc[txn["member_id"] == 2, "net_amount"] = -50.0
    items = pd.DataFrame({"txn_id": txn["txn_id"], "sku": 100001, "qty": 1, "unit_price": 100.0})
    products = pd.DataFrame({"sku": [100001], "category": ["啤酒"]})
    rfm = pd.DataFrame({"member_id": [1, 2], "recency_days": [10, 5],
                        "frequency": [3, 2], "monetary": [250.0, -100.0]})
    feats = build_cluster_features(rfm, txn, items, products).set_index("member_id")
    assert feats.loc[2, "monetary"] < 0
    assert np.isfinite(feats.loc[2, "avg_basket"])
