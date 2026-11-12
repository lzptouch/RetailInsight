"""分区明细读写与 schema 映射层单测。"""
from pathlib import Path

import pandas as pd
import pytest

from retailinsight.io.loader import (
    derive_txn_month,
    optimize_dtypes,
    read_items,
    read_parquet,
    read_partitioned,
    write_parquet,
    write_partitioned,
)
from retailinsight.io.schema import CANONICAL, SOURCE_MAPPINGS, SchemaError, apply_mapping

# ---------------------------------------------------------------- 分区读写

def test_write_and_read_partitioned(tmp_path: Path):
    df = pd.DataFrame({
        "txn_id": [1, 2, 3, 4],
        "sku": [10, 11, 12, 13],
        "qty": [1, 2, 3, 4],
        "txn_month": ["2021-01", "2021-01", "2021-02", "2021-02"],
    })
    out = write_partitioned(df, tmp_path / "items", "txn_month")
    assert out.is_dir()

    back = read_parquet(out)
    assert len(back) == 4
    assert set(back["txn_month"].astype(str)) == {"2021-01", "2021-02"}


def test_write_partitioned_clears_stale_partitions(tmp_path: Path):
    d = tmp_path / "items"
    old = pd.DataFrame({"txn_id": [1], "txn_month": ["2020-01"]})
    write_partitioned(old, d, "txn_month")
    new = pd.DataFrame({"txn_id": [2], "txn_month": ["2021-01"]})
    write_partitioned(new, d, "txn_month")

    back = read_parquet(d)
    assert len(back) == 1
    assert "2020-01" not in set(back["txn_month"].astype(str))


def test_read_items_filters_by_month(tmp_path: Path):
    df = pd.DataFrame({
        "txn_id": [1, 2, 3, 4, 5],
        "txn_month": ["2021-01", "2021-02", "2021-03", "2022-01", "2022-02"],
    })
    d = write_partitioned(df, tmp_path / "items", "txn_month")

    sub = read_items(d, months=["2021-02", "2022-01"])
    assert len(sub) == 2
    assert set(sub["txn_month"].astype(str)) == {"2021-02", "2022-01"}

    # 不传 months 则全量
    assert len(read_items(d)) == 5


def test_read_items_falls_back_to_flat_file(tmp_path: Path):
    """兼容旧产物：单文件路径也能读。"""
    f = tmp_path / "items.parquet"
    write_parquet(pd.DataFrame({"txn_id": [1, 2]}), f)
    assert len(read_items(f)) == 2


def test_read_partitioned_in_memory():
    df = pd.DataFrame({"txn_month": ["2021-01", "2021-02", "2022-01"]})
    assert len(read_partitioned(df, "txn_month", ["2021-02"])) == 1
    assert len(read_partitioned(df, "txn_month", None)) == 3


def test_derive_txn_month_keeps_unmatched_rows():
    """未能匹配到交易的明细不能丢行（脏数据场景）。"""
    items = pd.DataFrame({"txn_id": [1, 2, 999]})
    txn = pd.DataFrame({
        "txn_id": [1, 2],
        "txn_ts": pd.to_datetime(["2021-03-05", "2021-07-09"]),
    })
    out = derive_txn_month(items, txn)
    assert len(out) == 3
    assert out.loc[out["txn_id"] == 1, "txn_month"].iloc[0] == "2021-03"
    assert out.loc[out["txn_id"] == 999, "txn_month"].iloc[0] == "unknown"


def test_optimize_dtypes_shrinks():
    df = pd.DataFrame({
        "id": pd.Series([1, 2, 3], dtype="int64"),
        "amount": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
        "cat": ["a", "b", "a"],
    })
    out = optimize_dtypes(df)
    # 整数降位到最小可容纳类型（不一定恰是 int32），但必须 ≤4 字节
    assert out["id"].dtype.itemsize <= 4
    assert out["amount"].dtype == "float32"
    assert str(out["cat"].dtype) == "category"


# ---------------------------------------------------------------- schema 映射

def test_source_mappings_cover_all_declared_sources():
    for source in ("synthetic", "dunnhumby", "uci_retail"):
        assert source in SOURCE_MAPPINGS


def test_apply_mapping_renames_columns():
    df = pd.DataFrame({
        "CustomerID": ["10001", "10002"],
        "InvoiceNo": ["1", "2"],
        "Quantity": [2, 3],
        "UnitPrice": [1.5, 2.5],
        "Description": ["A", "B"],
    })
    out = apply_mapping(df, "uci_retail")
    assert "member_id" in out.columns
    assert "txn_id" in out.columns
    assert "CustomerID" not in out.columns


def test_apply_mapping_raises_on_missing_required_columns():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(SchemaError) as exc:
        apply_mapping(df, "uci_retail", table="transactions")
    assert "uci_retail" in str(exc.value)


def test_apply_mapping_dunnhumby_converts_epoch_day():
    """dunnhumby 的 DAY 是「距 1970-01-01 的天数」，必须转成真实时间戳。"""
    df = pd.DataFrame({
        "household_key": [1001, 1002],
        "BASKET_ID": [900, 901],
        "DAY": [18262, 18263],
        "PRODUCT_ID": [123, 124],
        "QUANTITY": [1, 2],
        "SALES_VALUE": [10.5, 20.0],
    })
    out = apply_mapping(df, "dunnhumby", table="transactions")
    assert str(out["txn_ts"].dtype).startswith("datetime64")
    assert out["txn_ts"].iloc[0].year == 2020
    assert out["txn_ts"].iloc[0].month == 1
    assert out["member_id"].iloc[0] == 1001
    assert out["amount"].iloc[1] == pytest.approx(20.0)


def test_apply_mapping_unknown_source_raises():
    df = pd.DataFrame({"member_id": [1]})
    with pytest.raises(SchemaError):
        apply_mapping(df, "not_a_source")


def test_canonical_keys_are_consistent_with_mappings():
    """每个 source 的映射目标字段必须落在 canonical 允许的字段集合内。"""
    allowed = set()
    for cols in CANONICAL.values():
        allowed.update(cols)
    for source, mapping in SOURCE_MAPPINGS.items():
        for target in mapping.values():
            assert target in allowed, f"{source}: {target} 不在 canonical 字段集合内"
