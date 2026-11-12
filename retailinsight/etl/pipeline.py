"""ETL 流水线：raw → 清洗 → interim（dtype 优化后的 parquet）→ 数据质量报告。"""

from retailinsight.etl import clean
from retailinsight.etl.quality import build_quality_report, missing_rates, write_quality_report
from retailinsight.io.loader import optimize_dtypes, read_parquet, write_parquet
from retailinsight.settings import INTERIM_DIR, RAW_DIR


def run_etl() -> dict:
    raw = {name: read_parquet(RAW_DIR / f"{name}.parquet")
           for name in ["members", "stores", "products", "transactions", "transaction_items"]}
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

    members = raw["members"]
    cleaned = {"members": members, "stores": raw["stores"], "products": products,
               "transactions": txn, "transaction_items": items}
    after = {k: len(v) for k, v in cleaned.items()}

    for name, df in cleaned.items():
        write_parquet(optimize_dtypes(df), INTERIM_DIR / f"{name}.parquet")

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
    print(f"[etl] quality report -> {path}")
    return report


if __name__ == "__main__":
    run_etl()
