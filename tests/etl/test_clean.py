"""清洗规则单测。"""
import pandas as pd

from retailinsight.etl.clean import (
    clip_outliers,
    dedup_transactions,
    flag_logic_conflicts,
    flag_returns,
    handle_missing_members,
    normalize_category,
)
from retailinsight.io.schema import MISSING_MEMBER_ID


def _txn():
    return pd.DataFrame({
        "txn_id": [1, 1, 2, 3, 4],
        "member_id": [10, 10, 11, MISSING_MEMBER_ID, 12],
        "store_id": [1] * 5,
        "txn_ts": pd.to_datetime(["2025-06-01"] * 5),
        "amount": [100.0, 100.0, -20.0, 50.0, 80.0],
        "discount": [0.0] * 5,
    })


def test_dedup_transactions():
    df, removed = dedup_transactions(_txn())
    assert removed == 1
    assert df["txn_id"].is_unique


def test_flag_returns():
    df = flag_returns(_txn())
    assert df["is_return"].sum() == 1
    assert df.loc[df["txn_id"] == 2, "net_amount"].iloc[0] == -20.0


def test_missing_member_to_non_member():
    df = handle_missing_members(_txn())
    assert (df["member_type"] == "非会员").sum() == 1
    assert df.loc[df["txn_id"] == 3, "member_type"].iloc[0] == "非会员"


def test_normalize_category():
    products = pd.DataFrame({"sku": [1, 2, 3],
                             "category": ["清洁 用品", "啤酒", "乳 制品"]})
    df, fixed = normalize_category(products)
    assert fixed == 2
    assert set(df["category"]) == {"清洁用品", "啤酒", "乳制品"}


def test_logic_conflict():
    txn = _txn().head(2)
    txn.loc[0, "txn_ts"] = pd.Timestamp("2010-01-01")
    members = pd.DataFrame({"member_id": [10], "reg_date": [pd.Timestamp("2024-01-01")]})
    df = flag_logic_conflicts(txn, members)
    assert df["is_logic_error"].iloc[0]
    assert not df["is_logic_error"].iloc[1]


def test_clip_outliers():
    txn = pd.DataFrame({"txn_id": range(1000),
                        "amount": [10.0] * 999 + [1e6]})
    df, thr = clip_outliers(txn, q=0.99)
    assert df["amount"].max() <= thr + 1e-6
    assert df["amount"].max() < 1e6
