"""DuckDB 星型模型与 RFM 一致性校验的测试。

依赖 `data/interim` 产物已生成（CI 中由 `make etl` 保证）。
"""
from pathlib import Path

import pytest

from retailinsight.etl import warehouse
from retailinsight.io.loader import read_parquet
from retailinsight.settings import INTERIM_DIR

pytestmark = pytest.mark.skipif(
    not (INTERIM_DIR / "transactions.parquet").exists(),
    reason="需要先运行 ETL 生成 interim 产物",
)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("wh") / "test.duckdb"
    warehouse.build_warehouse(p, verbose=False)
    return p


def test_star_schema_tables_exist(db_path: Path):
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert {"dim_member", "dim_product", "dim_store", "dim_date",
            "fact_transactions", "fact_transaction_items"} <= tables


def test_dimension_row_counts(db_path: Path):
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_member = con.execute("SELECT COUNT(*) FROM dim_member").fetchall()[0][0]
        n_product = con.execute("SELECT COUNT(*) FROM dim_product").fetchall()[0][0]
        n_store = con.execute("SELECT COUNT(*) FROM dim_store").fetchall()[0][0]
    finally:
        con.close()
    src_members = read_parquet(INTERIM_DIR / "members.parquet")
    src_products = read_parquet(INTERIM_DIR / "products.parquet")
    src_stores = read_parquet(INTERIM_DIR / "stores.parquet")
    assert n_member == len(src_members)
    assert n_product == len(src_products)
    assert n_store == len(src_stores)


def test_no_orphan_foreign_keys(db_path: Path):
    """事实表的外键必须都能在维表里找到。"""
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        o1 = con.execute("""
            SELECT COUNT(*) FROM fact_transaction_items f
            LEFT JOIN dim_product p USING (sku) WHERE p.sku IS NULL
        """).fetchall()[0][0]
        o2 = con.execute("""
            SELECT COUNT(*) FROM fact_transactions f
            LEFT JOIN dim_date d USING (date_key) WHERE d.date_key IS NULL
        """).fetchall()[0][0]
    finally:
        con.close()
    assert o1 == 0
    assert o2 == 0


def test_dim_date_covers_full_range_and_marks_weekend(db_path: Path):
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("""
            SELECT MIN(date), MAX(date), COUNT(*),
                   SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END),
                   SUM(is_holiday)
            FROM dim_date
        """).fetchall()[0]
    finally:
        con.close()
    assert row[0] is not None and row[1] is not None
    assert row[1] > row[0]
    assert row[2] == (row[1] - row[0]).days + 1
    assert row[3] > 0          # 存在周末
    assert row[4] > 0          # 存在节假日标记


def test_sql_rfm_matches_pandas_rfm(db_path: Path):
    """核心验收：两套独立实现必须给出完全一致的 RFM。"""
    result = warehouse.verify_rfm_consistency(db_path)
    assert result["rows_match"], result
    assert result["consistent"], result
    assert result["pandas_rows"] > 0
    assert all(v == 0 for v in result["diffs"].values()), result


def test_sql_rfm_has_expected_columns(db_path: Path):
    df = warehouse.rfm_via_sql(db_path)
    assert list(df.columns) == ["member_id", "recency_days", "frequency",
                                 "monetary", "txn_count", "return_amount"]
    assert df["recency_days"].min() >= 0
    assert df["frequency"].min() >= 1
