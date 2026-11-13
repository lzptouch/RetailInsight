"""统一 CLI：python -m retailinsight.cli run --stage all --scale 0.3

Stage 依赖顺序（`all` 按此顺序执行）：
    gen → etl → rfm → cluster → rules → rules_sku → rules_store → churn →
    profile → warehouse → reports → report → cards → segment_compare

- `rfm`     只产出 RFM 分群表（mart_rfm）
- `cluster` 依赖 mart_rfm，产出聚类表（mart_cluster）
- `churn`   产出流失得分表（mart_churn_score）
- `profile` 依赖 mart_rfm + mart_churn_score，产出画像宽表（含 churn_prob）
- `warehouse` 建 DuckDB 星型模型，并用 SQL 重算 RFM 与 pandas 版做一致性断言
- `cards`   产出客群画像卡片（docs/客群画像.md）
- `segment_compare` 产出规则法 vs 聚类法差异论证（docs/客群方法差异论证.md）
- `dashboard` 起 Streamlit 看板（**交互式、会阻塞**，因此不进 `all`）
"""
import argparse
import time

STAGES = ["gen", "etl", "rfm", "cluster", "rules", "rules_sku", "rules_store",
          "churn", "profile", "warehouse", "reports", "report", "cards",
          "segment_compare", "dashboard"]

#: 交互式 stage：会占住终端直到 Ctrl+C，因此 `--stage all` 跳过它们
INTERACTIVE_STAGES = ["dashboard"]
#: `--stage all` 实际执行的批次
ALL_STAGES = [s for s in STAGES if s not in INTERACTIVE_STAGES]


def stage_gen(scale: float, seed: int, members: int | None = None,
              chunk_size: int = 0) -> None:
    """生成合成数据。``chunk_size`` > 0 时分块流式生成（大规模数据必需）。"""
    import subprocess
    import sys

    from retailinsight.settings import ROOT

    cmd = [sys.executable, str(ROOT / "scripts" / "generate_synthetic.py"),
           "--scale", str(scale), "--seed", str(seed)]
    if members:
        cmd += ["--members", str(members)]
    if chunk_size:
        cmd += ["--chunk-size", str(chunk_size)]
    subprocess.run(cmd, check=True)


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
    from retailinsight.settings import MART_CHURN, MART_RFM, MART_RULES
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

    # 关联规则网络图（计划 §0.8.1 / R-9）：确定性圆周布局，不引 networkx
    rules = read_parquet(MART_RULES)
    print("[reports] rule network:", charts.rule_network(rules, top_n=18))

    export_excel(core)


def stage_dashboard(port: int = 8501) -> None:
    """起 Streamlit 看板（阻塞直到 Ctrl+C）。计划 §3.1 的 `dashboard` stage。"""
    import subprocess
    import sys

    from retailinsight.settings import ROOT

    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
           "--server.port", str(port), "--server.headless", "true"]
    print("[dashboard] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="retailinsight")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", required=True, choices=STAGES + ["all"])
    run.add_argument("--scale", type=float, default=0.3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--port", type=int, default=8501,
                     help="dashboard stage 的本地端口")
    run.add_argument("--members", type=int, default=None,
                     help="直接指定生成的会员数（覆盖 --scale 的计算结果）")
    run.add_argument("--chunk-size", type=int, default=0,
                     help="gen stage 的分块大小（会员数/块）。大规模生成时用，峰值内存 ≈ 单块")
    args = parser.parse_args()

    stages = ALL_STAGES if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        if s == "gen":
            stage_gen(args.scale, args.seed, args.members, args.chunk_size)
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
        elif s == "cards":
            from retailinsight.reports.profile_cards import main as render_cards
            render_cards()
        elif s == "segment_compare":
            from retailinsight.models.segment_compare import run_segment_compare
            run_segment_compare()
        elif s == "dashboard":
            stage_dashboard(args.port)
        print(f"[cli] stage={s} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
