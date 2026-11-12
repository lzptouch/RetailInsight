"""DuckDB 星型模型（计划 §5.2 / R-1）。

设计要点
--------
DuckDB 在这里**不是炫技**，而是用**第二实现来自证 pandas 结果正确**：
同一套 RFM 口径用 SQL 窗口函数再算一遍，与 pandas 版逐行一致性断言。
两套实现对得上，口径才可信；对不上，立刻能定位是哪一步漂移。

星型模型
--------
```
                    dim_member
                        │
    fact_transactions ──┤
       fact_transaction_items ── dim_product
                        │            │
                   dim_store     dim_date
```
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retailinsight.io.loader import read_parquet
from retailinsight.settings import DATA_DIR, INTERIM_DIR, ITEMS_DIR

WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"

# 数据窗口内的主要法定节假日（仅覆盖合成数据窗口 2020-01 ~ 2022-07）
# 说明：农历节日按年份查表，平日不做完整日历 —— 用途是给 dim_date 打标，
#       让「节假日效应」分析有依据，不追求完备。
HOLIDAYS: dict[str, list[str]] = {
    "2020": ["01-01", "01-24", "01-25", "01-26", "01-27", "01-28", "01-29", "01-30",
             "01-31", "04-04", "04-05", "04-06", "05-01", "05-02", "05-03", "05-04",
             "05-05", "06-25", "06-26", "10-01", "10-02", "10-03", "10-04", "10-05",
             "10-06", "10-07", "10-08"],
    "2021": ["01-01", "02-11", "02-12", "02-13", "02-14", "02-15", "02-16", "02-17",
             "04-03", "04-04", "04-05", "05-01", "05-02", "05-03", "05-04", "05-05",
             "06-12", "06-13", "06-14", "09-19", "09-20", "09-21",
             "10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07"],
    "2022": ["01-01", "01-02", "01-03", "01-31", "02-01", "02-02", "02-03", "02-04",
             "02-05", "02-06", "04-03", "04-04", "04-05", "05-01", "05-02", "05-03",
             "05-04", "06-03", "06-04", "06-05", "09-10", "09-11", "09-12",
             "10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07"],
}


def _holiday_set() -> set[dt.date]:
    out: set[dt.date] = set()
    for year, mds in HOLIDAYS.items():
        for md in mds:
            m, d = md.split("-")
            out.add(dt.date(int(year), int(m), int(d)))
    return out


def build_dim_date(con: duckdb.DuckDBPyConnection, start: str, end: str) -> int:
    """建 dim_date：日期主键 + 年/月/季度/星期/是否周末/是否节假日。"""
    holidays = _holiday_set()
    dates = pd.date_range(start, end, freq="D")
    dim = pd.DataFrame({
        "date_key": dates.strftime("%Y%m%d").astype("int32"),
        "date": dates.date,
        "year": dates.year.astype("int16"),
        "quarter": dates.quarter.astype("int8"),
        "month": dates.month.astype("int8"),
        "day": dates.day.astype("int8"),
        "day_of_week": dates.day_of_week.astype("int8"),
        "is_weekend": (dates.day_of_week >= 5).astype("int8"),
        "is_holiday": [int(d.date() in holidays) for d in dates],
        "month_name": dates.month_name(),
    })
    con.register("_dim_date", dim)
    con.execute("CREATE OR REPLACE TABLE dim_date AS SELECT * FROM _dim_date")
    con.unregister("_dim_date")
    return len(dim)


def build_warehouse(db_path: Path = WAREHOUSE_PATH, verbose: bool = True) -> dict:
    """建 DuckDB 星型模型。返回各表行数。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    txn_pq = (INTERIM_DIR / "transactions.parquet").as_posix()
    items_pq = (ITEMS_DIR / "**" / "*.parquet").as_posix()
    members_pq = (INTERIM_DIR / "members.parquet").as_posix()
    products_pq = (INTERIM_DIR / "products.parquet").as_posix()
    stores_pq = (INTERIM_DIR / "stores.parquet").as_posix()

    con = duckdb.connect(str(db_path))
    stats: dict[str, int] = {}
    try:
        # ---- 维表 ----
        con.execute(f"""
            CREATE TABLE dim_member AS
            SELECT member_id, reg_date, gender, age_band, city_tier, store_id
            FROM read_parquet('{members_pq}')
        """)
        con.execute(f"""
            CREATE TABLE dim_product AS
            SELECT sku, category, sub_category, brand, base_price
            FROM read_parquet('{products_pq}')
        """)
        con.execute(f"""
            CREATE TABLE dim_store AS
            SELECT store_id, city, city_tier, area_m2, open_date
            FROM read_parquet('{stores_pq}')
        """)

        min_ts = con.execute(
            f"SELECT MIN(txn_ts) FROM read_parquet('{txn_pq}')").fetchall()[0][0]
        max_ts = con.execute(
            f"SELECT MAX(txn_ts) FROM read_parquet('{txn_pq}')").fetchall()[0][0]
        start = (min_ts - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        end = (max_ts + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        stats["dim_date"] = build_dim_date(con, start, end)

        # ---- 事实表（交易粒度）----
        con.execute(f"""
            CREATE TABLE fact_transactions AS
            SELECT
                txn_id, member_id, store_id, txn_ts,
                CAST(txn_ts AS DATE)                    AS txn_date,
                CAST(strftime(txn_ts, '%Y%m%d') AS INTEGER) AS date_key,
                amount, discount, net_amount,
                is_return, is_logic_error, member_type
            FROM read_parquet('{txn_pq}')
        """)

        # ---- 事实表（明细粒度）：明细 × 交易补齐会员/门店/时间 ----
        con.execute(f"""
            CREATE TABLE fact_transaction_items AS
            SELECT
                i.txn_id, i.sku, i.qty, i.unit_price,
                t.member_id, t.store_id, t.txn_ts,
                CAST(t.txn_ts AS DATE)                    AS txn_date,
                CAST(strftime(t.txn_ts, '%Y%m%d') AS INTEGER) AS date_key,
                i.qty * i.unit_price                      AS line_amount,
                t.member_type
            FROM read_parquet('{items_pq}') i
            JOIN fact_transactions t USING (txn_id)
        """)

        for table in ("dim_member", "dim_product", "dim_store", "dim_date",
                      "fact_transactions", "fact_transaction_items"):
            stats[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]

        # ---- 外键完整性自检 ----
        fk_issues = {}
        fk_issues["orphan_items_member"] = con.execute("""
            SELECT COUNT(*) FROM fact_transaction_items f
            LEFT JOIN dim_member d USING (member_id)
            WHERE f.member_id <> -1 AND d.member_id IS NULL
        """).fetchall()[0][0]
        fk_issues["orphan_items_sku"] = con.execute("""
            SELECT COUNT(*) FROM fact_transaction_items f
            LEFT JOIN dim_product p USING (sku)
            WHERE p.sku IS NULL
        """).fetchall()[0][0]
        fk_issues["orphan_txn_date"] = con.execute("""
            SELECT COUNT(*) FROM fact_transactions f
            LEFT JOIN dim_date d USING (date_key)
            WHERE d.date_key IS NULL
        """).fetchall()[0][0]
        stats["fk_issues"] = fk_issues
    finally:
        con.close()

    if verbose:
        print(f"[warehouse] {db_path}")
        for k, v in stats.items():
            print(f"[warehouse]   {k}: {v}")
    return stats


def rfm_via_sql(db_path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    """用 SQL 窗口函数/聚合重算 RFM，口径对齐 ``features/rfm.py::compute_rfm``。

    口径：
    - 仅统计会员（member_type='会员'）与非逻辑异常交易
    - F = 去重后的交易**日期**数（同日多单合并计 1 次）
    - M = net_amount 之和（退货为负）
    - R = 基准日 - 最近消费日，基准日 = 数据最大交易日 + 1 天
    """
    con = duckdb.connect(str(Path(db_path)), read_only=True)
    try:
        out = con.execute("""
            WITH base AS (
                SELECT member_id,
                       CAST(txn_ts AS DATE) AS txn_date,
                       net_amount, amount, is_return
                FROM fact_transactions
                WHERE member_type = '会员' AND NOT is_logic_error
            ),
            agg AS (
                SELECT member_id,
                       MAX(txn_date)                              AS last_date,
                       COUNT(DISTINCT txn_date)                  AS frequency,
                       SUM(net_amount)                            AS monetary,
                       COUNT(*)                                  AS txn_count,
                       SUM(CASE WHEN is_return THEN amount ELSE 0 END) AS return_amount
                FROM base
                GROUP BY member_id
            ),
            anchor AS (
                SELECT CAST(MAX(txn_ts) AS DATE) + INTERVAL 1 DAY AS end_date FROM fact_transactions
            )
            SELECT a.member_id,
                   DATE_DIFF('day', a.last_date, k.end_date) AS recency_days,
                   a.frequency,
                   ROUND(a.monetary, 2)                        AS monetary,
                   a.txn_count,
                   a.return_amount
            FROM agg a CROSS JOIN anchor k
            ORDER BY a.member_id
        """).fetchdf()
    finally:
        con.close()
    return out


def verify_rfm_consistency(db_path: Path = WAREHOUSE_PATH) -> dict:
    """pandas 版 RFM 与 SQL 版逐行比对（计划 §5.2 的核心验收）。"""
    from retailinsight.features.rfm import compute_rfm

    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    pandas_rfm = compute_rfm(txn).sort_values("member_id").reset_index(drop=True)
    sql_rfm = rfm_via_sql(db_path).sort_values("member_id").reset_index(drop=True)

    result = {
        "pandas_rows": len(pandas_rfm),
        "sql_rows": len(sql_rfm),
        "rows_match": len(pandas_rfm) == len(sql_rfm),
    }
    if not result["rows_match"]:
        result["matched"] = 0
        result["consistent"] = False
        return result

    diffs = {}
    for col in ("recency_days", "frequency", "monetary", "txn_count", "return_amount"):
        a = pandas_rfm[col].to_numpy(dtype="float64")
        b = sql_rfm[col].to_numpy(dtype="float64")
        if np.issubdtype(pandas_rfm[col].dtype, np.integer):
            # 整数字段必须完全一致（口径差异会直接暴露）
            n_bad = int((a != b).sum())
        else:
            # 浮点字段用相对容差：parquet 里金额是 float32，
            # pandas 在 float32 上累加、DuckDB 按 float64 累加，
            # 实测最大相对误差 ~1e-7（float32 的 eps 量级），属精度差而非口径差。
            n_bad = int((~np.isclose(a, b, rtol=1e-5, atol=1e-4)).sum())
        diffs[col] = n_bad
    result["diffs"] = diffs
    result["matched"] = len(pandas_rfm) - max(diffs.values()) if diffs else 0
    result["consistent"] = all(v == 0 for v in diffs.values())
    return result


def main() -> None:
    build_warehouse()
    check = verify_rfm_consistency()
    print(f"[warehouse] RFM 一致性校验: {check}")
    if not check["consistent"]:
        raise SystemExit("[warehouse] ❌ pandas 与 SQL 的 RFM 结果不一致，星型模型不可信")
    print("[warehouse] ✅ pandas 与 SQL 的 RFM 完全一致")


if __name__ == "__main__":
    main()
