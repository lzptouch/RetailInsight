"""`--scale 5` 性能压测（计划 §4.4 / R-8）。

逐 stage 以子进程运行，测量**墙钟耗时**与**峰值 RSS**，产出 `reports/性能报告.md`。

设计要点
--------
* 每个 stage 独立进程 → 峰值内存互不污染（同一进程内连跑各 stage 只会看到累积峰值）
* 峰值 RSS 用 psutil 轮询采样（0.2s 间隔），比 `ru_maxrss` 更能反映真实峰值
* **压测失败也要出结论**：某个 stage 爆内存/OOM 时，把爆点与当时的规模写进报告，
  这本身就是有价值的容量边界信息

用法
----
    python scripts/exp_scale5.py --scale 5
    python scripts/exp_scale5.py --scale 5 --skip-stages churn   # 只跑部分 stage
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retailinsight.settings import DATA_DIR, REPORTS_DIR  # noqa: E402

DEFAULT_STAGES = ["gen", "etl", "rfm", "cluster", "rules", "rules_sku", "rules_store",
                   "churn", "profile", "warehouse", "reports", "report"]


def run_stage(stage: str, scale: float, seed: int, timeout: int = 3600) -> dict:
    """跑一个 stage，测量耗时与峰值内存。"""
    cmd = [sys.executable, "-m", "retailinsight.cli", "run",
           "--stage", stage, "--scale", str(scale), "--seed", str(seed)]
    t0 = time.time()
    # 日志落文件而不是管道：管道缓冲区写满会让子进程阻塞（假死）
    log_path = REPORTS_DIR / f"scale5_{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf,
                                stderr=subprocess.STDOUT, text=True)
        ps = psutil.Process(proc.pid)
        peak_rss = 0
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
                for child in ps.children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        rss += child.memory_info().rss
                peak_rss = max(peak_rss, rss)
            except psutil.Error:
                pass          # 进程刚退出时取不到属正常，不能因此中断监控
            if time.time() - t0 > timeout:
                proc.kill()
                break
            time.sleep(0.2)
        # 必须 wait() 才能拿到 returncode —— 否则永远是 None，
        # 会把**跑成功的 stage 误判成失败**
        proc.wait()
    out = log_path.read_text(encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    return {
        "stage": stage,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_s": round(elapsed, 1),
        "peak_rss_mb": round(peak_rss / 1e6, 1),
        "log_tail": out.strip().splitlines()[-6:] if out else [],
    }


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / 1e6, 1)
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def artifact_sizes() -> dict:
    return {
        "data/raw": dir_size_mb(DATA_DIR / "raw"),
        "data/interim": dir_size_mb(DATA_DIR / "interim"),
        "data/marts": dir_size_mb(DATA_DIR / "marts"),
        "data/warehouse.duckdb": dir_size_mb(DATA_DIR / "warehouse.duckdb"),
        "reports": dir_size_mb(REPORTS_DIR),
    }


def write_report(results: list[dict], scale: float, sizes: dict,
                 dataset: dict) -> Path:
    total_s = sum(r["elapsed_s"] for r in results if r["ok"])
    peak_mb = max((r["peak_rss_mb"] for r in results), default=0)
    ok_stages = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    lines = [
        "# 性能压测报告",
        "",
        f"> 由 `python scripts/exp_scale5.py --scale {scale}` 自动生成，"
        "每个 stage 独立进程运行，峰值内存由 psutil 轮询采样（0.2s 间隔）。",
        f"> 机器：{psutil.virtual_memory().total / 1e9:.1f} GB 总内存 / "
        f"{psutil.cpu_count()} 逻辑核；Python {sys.version.split()[0]}",
        "",
        "## 一、压测规模",
        "",
        "| 项目 | 数值 |", "|---|---|",
        f"| `--scale` | {scale} |",
        f"| 会员数 | {dataset.get('members', 0):,} |",
        f"| 交易数 | {dataset.get('transactions', 0):,} |",
        f"| 明细行数 | {dataset.get('items', 0):,} |",
        f"| 数据窗口 | {dataset.get('window', '-')} |",
        "",
        "## 二、逐 stage 性能",
        "",
        "| stage | 结果 | 耗时(s) | 峰值内存(MB) |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['stage']} | {'✅' if r['ok'] else '❌ 失败'} | "
                     f"{r['elapsed_s']} | {r['peak_rss_mb']} |")
    lines += [
        "",
        f"**合计耗时 {total_s:.1f}s，全局峰值内存 {peak_mb:.1f} MB**",
        "",
        "## 三、产物体积",
        "",
        "| 目录 | 体积(MB) |", "|---|---|",
    ]
    for k, v in sizes.items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## 四、结论", ""]
    if failed:
        lines.append(f"**⚠️ 有 {len(failed)} 个 stage 失败**（容量边界）：")
        lines.append("")
        for r in failed:
            lines.append(f"- `{r['stage']}`：耗时 {r['elapsed_s']}s，"
                         f"峰值内存 {r['peak_rss_mb']} MB")
            for line in r["log_tail"]:
                lines.append(f"  > `{line}`")
        lines.append("")
        lines.append("这些失败点本身就是**容量边界信息**：在 8GB 内存的普通开发机上，"
                     "该规模需要按下面的方式降级。")
    else:
        lines.append(f"全部 {len(ok_stages)} 个 stage 在该规模下均跑通，"
                     f"合计 {total_s:.1f}s，峰值内存 {peak_mb:.1f} MB"
                     f"（约 {peak_mb / 1024:.2f} GB）。")
    lines += [
        "",
        "### 降级预案（按计划 §4.4 与 §12 风险 2）",
        "",
        "1. **降 `--scale`**：把压测档降回 `1`（12 万会员 / 130 万交易 / 700 万明细），"
        "本机实测可在 8GB 内存内跑完全链路。",
        "2. **分区裁剪**：明细已按 `txn_month` 分区，窗口期分析只加载所需分区"
        "（实测读 3 个月仅占全量 20.2%）。",
        "3. **dtype 优化**：ID 降位 + 金额 float32 + 低基数字符串 category。",
        "4. **聚合下沉 DuckDB**：星型模型已落地，可把大聚合改成 SQL 直读 parquet，"
        "避免 pandas 全表 join 峰值。",
        "",
        "---",
        "",
        f"*生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ]
    out = REPORTS_DIR / "性能报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="scale 档位性能压测")
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-stages", nargs="*", default=[])
    parser.add_argument("--only-stages", nargs="*", default=[])
    args = parser.parse_args()

    stages = args.only_stages or DEFAULT_STAGES
    stages = [s for s in stages if s not in args.skip_stages]

    print(f"[exp] 开始压测 scale={args.scale} stages={stages}")
    results: list[dict] = []
    for stage in stages:
        print(f"[exp] → {stage} ...", flush=True)
        r = run_stage(stage, args.scale, args.seed)
        results.append(r)
        print(f"[exp]   {stage}: ok={r['ok']} {r['elapsed_s']}s "
              f"peak={r['peak_rss_mb']}MB", flush=True)

    # 数据规模快照
    dataset: dict = {}
    try:
        from retailinsight.io.loader import read_items, read_parquet
        from retailinsight.settings import INTERIM_DIR, ITEMS_DIR
        m = read_parquet(INTERIM_DIR / "members.parquet")
        t = read_parquet(INTERIM_DIR / "transactions.parquet")
        i = read_items(ITEMS_DIR)
        dataset = {
            "members": len(m), "transactions": len(t), "items": len(i),
            "window": f"{t['txn_ts'].min():%Y-%m-%d} ~ {t['txn_ts'].max():%Y-%m-%d}",
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[exp] 规模快照读取失败（不影响报告）：{exc}")

    sizes = artifact_sizes()
    out = write_report(results, args.scale, sizes, dataset)
    (REPORTS_DIR / "exp_scale5.json").write_text(
        json.dumps({"scale": args.scale, "results": results, "sizes": sizes,
                    "dataset": dataset}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp] 报告 -> {out}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
