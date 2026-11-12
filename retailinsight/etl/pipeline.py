"""ETL 流水线：raw → 清洗 → interim（dtype 优化后的 parquet，明细按月分区）→ 数据质量报告。"""

from retailinsight.etl import clean
from retailinsight.etl.quality import build_quality_report, missing_rates, write_quality_report
from retailinsight.io.loader import (
    derive_txn_month,
    optimize_dtypes,
    read_items,
    read_parquet,
    write_parquet,
    write_partitioned,
)
from retailinsight.settings import INTERIM_DIR, ITEMS_DIR, ITEMS_FILE, RAW_DIR, RAW_ITEMS_DIR


def run_etl() -> dict:
    raw = {name: read_parquet(RAW_DIR / f"{name}.parquet")
           for name in ["members", "stores", "products", "transactions"]}
    raw_items = read_items(RAW_ITEMS_DIR)
    raw["transaction_items"] = raw_items
    before = {k: len(v) for k, v in raw.items()}

    products, alias_fixed = clean.normalize_category(raw["products"])
    txn, dedup_removed = clean.dedup_transactions(raw["transactions"])
    txn = clean.flag_returns(txn)
    txn = clean.handle_missing_members(txn)
    txn = clean.flag_logic_conflicts(txn, raw["members"])
    txn, outlier_thr = clean.clip_outliers(txn)

    items = raw["transaction_items"]
    # 明细只保留去重后的交易（剔除被去重掉的重复 txn 的明细）
    items = items[items["txn_id"].isin(txn["txn_id"])]
    # 补分区列：明细表本身无时间戳，需通过 txn_id 关联交易表取月份
    items = derive_txn_month(items, txn)

    members = raw["members"]
    cleaned = {"members": members, "stores": raw["stores"], "products": products,
               "transactions": txn}
    after = {k: len(v) for k, v in cleaned.items()}
    after["transaction_items"] = len(items)

    for name, df in cleaned.items():
        write_parquet(optimize_dtypes(df), INTERIM_DIR / f"{name}.parquet")
    # 明细按 txn_month 分区落盘（计划 §4.4）
    write_partitioned(optimize_dtypes(items), ITEMS_DIR, "txn_month")
    # 清掉旧的单文件产物，避免与分区目录形成两个真相源
    for legacy in (ITEMS_FILE,):
        if legacy.exists():
            legacy.unlink()

    details = {
        "dedup_removed": dedup_removed,
        "missing_member": int((txn["member_type"] == "非会员").sum()),
        "returns": int(txn["is_return"].sum()),
        "alias_fixed": alias_fixed,
        "logic_errors": int(txn["is_logic_error"].sum()),
        "outlier_threshold": outlier_thr,
        "missing_rates": missing_rates(cleaned),
    }
    report = build_quality_report(before, after, details)
    path = write_quality_report(report)
    print(f"[etl] before={before}")
    print(f"[etl] after={after}")
    print(f"[etl] 明细分区目录 -> {ITEMS_DIR}")
    print(f"[etl] quality report -> {path}")
    return report


if __name__ == "__main__":
    run_etl()
