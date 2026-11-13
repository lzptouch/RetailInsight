"""数据质量报告：清洗前后对比，落盘 docs/数据质量报告.md。"""
import json
from pathlib import Path

import pandas as pd

from retailinsight.settings import DOCS_DIR


def build_quality_report(before: dict, after: dict, details: dict) -> dict:
    report = {"before": before, "after": after, "details": details}
    return report


def write_quality_report(report: dict, path: Path | None = None) -> Path:
    path = path or DOCS_DIR / "数据质量报告.md"
    b, a, d = report["before"], report["after"], report["details"]
    lines = [
        "# 数据质量报告",
        "",
        "## 一、清洗前后对比",
        "",
        "| 表 | 清洗前行数 | 清洗后行数 | 变化 |",
        "|---|---|---|---|",
    ]
    for table in b:
        delta = a[table] - b[table]
        lines.append(f"| {table} | {b[table]:,} | {a[table]:,} | {delta:+,} |")
    lines += [
        "",
        "## 二、清洗动作明细",
        "",
        "| 规则 | 处理量 | 策略 |",
        "|---|---|---|",
        f"| 重复交易去重 | {d.get('dedup_removed', 0):,} 行 | 按 txn_id 保留首条 |",
        f"| 会员ID缺失 | {d.get('missing_member', 0):,} 行 | 归入「非会员」客群单独分析，不删除 |",
        f"| 退货负金额 | {d.get('returns', 0):,} 行 | 打 is_return 标记，单列分析 |",
        f"| 品类别名归一 | {d.get('alias_fixed', 0):,} 行 | 别名映射表归一 |",
        f"| 逻辑冲突 | {d.get('logic_errors', 0):,} 行 | 打 is_logic_error 标记，分析时剔除 |",
        f"| 金额极端值 | 阈值 {d.get('outlier_threshold', 0):.2f} | P99.9 分位截断 |",
        "",
        "## 三、字段缺失率（清洗后）",
        "",
        "| 表.字段 | 缺失率 |",
        "|---|---|",
    ]
    for k, v in d.get("missing_rates", {}).items():
        lines.append(f"| {k} | {v:.4%} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    (path.parent / "数据质量报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # HTML 版（自包含，便于直接发给业务方）
    from retailinsight.etl.quality_html import render_quality_html
    html_path = render_quality_html(report)
    print(f"[quality] HTML 报告 -> {html_path}")
    return path


def missing_rates(dfs: dict[str, pd.DataFrame]) -> dict[str, float]:
    rates = {}
    for name, df in dfs.items():
        for col in df.columns:
            r = float(df[col].isna().mean())
            if r > 0:
                rates[f"{name}.{col}"] = r
    return rates
