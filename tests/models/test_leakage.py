"""流失预测防泄漏单测：特征只用观察窗数据，标签只看标签窗。"""
import pandas as pd

from retailinsight.models.churn import FEATURES, build_cohort_samples


def _fixture():
    cutoff = pd.Timestamp("2025-04-01")
    members = pd.DataFrame({"member_id": [1, 2], "reg_date": [pd.Timestamp("2024-01-01")] * 2})
    rows = [
        # member 1：观察窗活跃，标签窗也有交易 → 不流失
        (1, "2024-10-01", 100.0), (1, "2025-01-01", 100.0),
        (1, "2025-03-01", 100.0), (1, "2025-04-15", 100.0),
        # member 2：观察窗活跃，标签窗 0 交易 → 流失
        (2, "2024-09-01", 100.0), (2, "2024-12-01", 100.0),
    ]
    txn = pd.DataFrame(rows, columns=["member_id", "ts", "amount"])
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
    return txn, items, products, members, cutoff


def test_label_correctness():
    txn, items, products, members, cutoff = _fixture()
    df = build_cohort_samples(txn, items, products, members, cutoff).set_index("member_id")
    assert df.loc[1, "churn"] == 0
    assert df.loc[2, "churn"] == 1


def test_no_leakage_recency():
    """recency 必须基于观察窗内的最后交易，不能看到标签窗数据。"""
    txn, items, products, members, cutoff = _fixture()
    df = build_cohort_samples(txn, items, products, members, cutoff).set_index("member_id")
    # member 1 观察窗内最后交易 2025-03-01，距 cutoff 31 天；
    # 若泄漏（看到 2025-04-15），recency 会是 -14 天
    assert df.loc[1, "recency_days"] == 31
    assert (df["recency_days"] >= 0).all()


def test_feature_columns_complete():
    txn, items, products, members, cutoff = _fixture()
    df = build_cohort_samples(txn, items, products, members, cutoff)
    for col in FEATURES:
        assert col in df.columns
