"""数据质量报告：清洗前后对比，落盘 docs/数据质量报告.md。"""
import json
from pathlib import Path

import pandas as pd

from retailinsight.settings import DOCS_DIR

#: ID 列的「缺失哨兵值」。生成器把「会员ID缺失」写成 ``member_id = -1`` 而不是 NaN
#: （NaN 在 dtype 优化与 groupby 里会被静默吞掉），所以缺失率**不能只判 isna**，
#: 否则这一节永远是空表 —— 见 v3.4 缺陷记录（计划 §0.12）。
_ID_MISSING = -1
#: 字符串列的缺失哨兵值（大小写/空白在比较前统一处理）
_STR_MISSING = frozenset({"", "-", "nan", "none", "null", "unknown", "未知", "未填"})


def _missing_mask(s: pd.Series) -> pd.Series:
    """按**业务口径**判定缺失，而不是只判 NaN。"""
    name = "" if s.name is None else str(s.name)
    if name.endswith("_id") and s.dtype.kind in "iu":
        return s.isna() | (s == _ID_MISSING)
    if s.dtype.kind in "OUS":
        txt = s.astype("string").str.strip().str.lower()
        return txt.isna() | txt.isin(_STR_MISSING)
    return s.isna()


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
        "## 三、字段缺失率（清洗前 → 清洗后）",
        "",
        "| 表.字段 | 清洗前缺失率 | 清洗后缺失率 | 处理方式 |",
        "|---|---|---|---|",
    ]
    mb = d.get("missing_rates", {})
    ma = d.get("missing_rates_after", {})
    if not mb and not ma:
        lines.append("| — | — | — | 无缺失字段 |")
    for k in sorted(set(mb) | set(ma)):
        how = "转 member_type='非会员' 标记列" if k.endswith("member_id") else "兜底填充 / 标记"
        lines.append(f"| {k} | {mb.get(k, 0.0):.4%} | {ma.get(k, 0.0):.4%} | {how} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    (path.parent / "数据质量报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # HTML 版（自包含，便于直接发给业务方）
    # 路径**必须跟着 md 走**：早期写死默认路径，导致「只想验证 md」的单测
    # 顺手把仓库里真实的 HTML 产物覆盖成假数据（v3.4 缺陷 16）。
    from retailinsight.etl.quality_html import render_quality_html

    html_path = render_quality_html(report, path=path.with_suffix(".html"))
    print(f"[quality] HTML 报告 -> {html_path}")
    return path


def missing_rates(dfs: dict[str, pd.DataFrame]) -> dict[str, float]:
    """各字段缺失率（**哨兵值感知**：``*_id == -1``、空串 / unknown / 未知 都算缺失）。"""
    rates = {}
    for name, df in dfs.items():
        for col in df.columns:
            r = float(_missing_mask(df[col]).mean())
            if r > 0:
                rates[f"{name}.{col}"] = r
    return rates
