"""统一 CLI：python -m retailinsight.cli run --stage all --scale 0.3

Stage 依赖顺序（`all` 按此顺序执行）：
    gen → etl → rfm → cluster → rules → churn → profile → reports

- `rfm`     只产出 RFM 分群表（mart_rfm）
- `cluster` 依赖 mart_rfm，产出聚类表（mart_cluster）
- `churn`   产出流失得分表（mart_churn_score）
- `profile` 依赖 mart_rfm + mart_churn_score，产出画像宽表（含 churn_prob）
- `warehouse` 建 DuckDB 星型模型，并用 SQL 重算 RFM 与 pandas 版做一致性断言
"""
import argparse
import time

STAGES = ["gen", "etl", "rfm", "cluster", "rules", "rules_sku", "rules_store",
          "churn", "profile", "warehouse", "reports", "report"]


def stage_gen(scale: float, seed: int) -> None:
    import subprocess
    import sys

    from retailinsight.settings import ROOT

    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"),
                    "--scale", str(scale), "--seed", str(seed)], check=True)


def stage_rfm() -> None:
    from retailinsight.features.rfm import assign_segment, compute_rfm, score_rfm
    from retailinsight.io.loader import read_parquet, write_parquet
    from retailinsight.settings import INTERIM_DIR, MART_RFM

    txn = read_parquet(INTERIM_DIR / "transactions.parquet")

    rfm = compute_rfm(txn)
    scored = assign_segment(score_rfm(rfm))
    write_parquet(scored, MART_RFM)
    print(f"[rfm] members={len(scored)}")
    print(scored.groupby("rfm_segment")["member_id"].count().sort_values(ascending=False))


def stage_profile() -> None:
    """画像宽表：依赖 mart_rfm 与 mart_churn_score。"""
    from retailinsight.features.tags import build_profile
    from retailinsight.io.loader import read_items, read_parquet, write_parquet
    from retailinsight.settings import INTERIM_DIR, ITEMS_DIR, MART_CHURN, MART_PROFILE, MART_RFM

    members = read_parquet(INTERIM_DIR / "members.parquet")
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_items(ITEMS_DIR)
    products = read_parquet(INTERIM_DIR / "products.parquet")
    rfm = read_parquet(MART_RFM)
    churn = read_parquet(MART_CHURN)

    profile = build_profile(members, txn, items, products, rfm, churn=churn)
    # CLV 简化法（计划 §8.4 / R-6）：月均消费 × 预期留存月数 × 毛利率
    from retailinsight.models.clv import compute_clv
    profile = compute_clv(profile, churn=churn)
    write_parquet(profile, MART_PROFILE)
    covered = int(profile["churn_covered"].sum())
    print(f"[profile] rows={len(profile)} cols={len(profile.columns)} "
          f"churn_covered={covered} ({covered / len(profile):.1%})")
    print(profile.groupby("clv_tier", observed=True)["clv"].agg(["count", "mean"]).round(1))


def stage_reports() -> None:
    from retailinsight.io.loader import read_parquet
    from retailinsight.reports.core_reports import build_core_reports
    from retailinsight.reports.excel_report import export_excel
    from retailinsight.settings import MART_CHURN, MART_RFM
    from retailinsight.viz import charts

    core = build_core_reports()

    seg = read_parquet(MART_RFM)
    seg_summary = (seg.groupby("rfm_segment")
                   .agg(member_cnt=("member_id", "count"), gmv=("monetary", "sum"))
                   .reset_index())
    seg_summary["gmv_share"] = seg_summary["gmv"] / seg_summary["gmv"].sum()
    print("[reports] segment figures:",
          charts.segment_bar(seg_summary), charts.segment_value(seg_summary))

    churn = read_parquet(MART_CHURN)
    print("[reports] churn figures:",
          charts.lift_curve(churn["churn"].to_numpy(), churn["churn_prob"].to_numpy()),
          charts.churn_hist(churn["churn_prob"].to_numpy()))

    export_excel(core)


def main() -> None:
    parser = argparse.ArgumentParser(prog="retailinsight")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", required=True, choices=STAGES + ["all"])
    run.add_argument("--scale", type=float, default=0.3)
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        if s == "gen":
            stage_gen(args.scale, args.seed)
        elif s == "etl":
            from retailinsight.etl.pipeline import run_etl
            run_etl()
        elif s == "rfm":
            stage_rfm()
        elif s == "cluster":
            from retailinsight.models.cluster import run_cluster
            run_cluster()
        elif s == "rules":
            from retailinsight.models.rules import run_rules
            run_rules()
        elif s == "rules_sku":
            from retailinsight.models.rules import run_sku_rules
            run_sku_rules()
        elif s == "rules_store":
            from retailinsight.models.rules import run_store_rules
            run_store_rules(level="category")
        elif s == "churn":
            from retailinsight.models.churn import run_churn
            run_churn()
        elif s == "profile":
            stage_profile()
        elif s == "warehouse":
            from retailinsight.etl.warehouse import main as build_wh
            build_wh()
        elif s == "reports":
            stage_reports()
        elif s == "report":
            from retailinsight.reports.render_decision_report import main as render_report
            render_report()
        print(f"[cli] stage={s} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
