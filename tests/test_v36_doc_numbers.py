"""文档数字与真实产物的对账守卫（v3.6 #24）。

背景：v3.1 说「品类规则 373 条 / 分店规则 1,398 条」，v3.5 改了生成器的品类乘子后
实测变成 177 / 1,092，但 13 处文档仍写旧值 —— 因为**没有任何测试比对文档与产物**。

这里做最小的那种守卫：只解析 `docs/数据字典.md` 里 mart 表的行数声明，和真实
parquet 行数比对。规模不大，但能挡住"改了生成器、忘了改文档"这一类漂移。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DICT_MD = ROOT / "docs" / "数据字典.md"
MARTS = ROOT / "data" / "marts"

#: 只对账这几张"条数会随数据变化"的表（其它 mart 的规模是结构性常量）。
CHECKED = [
    "mart_basket_rules",
    "mart_basket_rules_sku",
    "mart_basket_rules_store",
]


def _doc_counts() -> dict[str, int]:
    """从数据字典的表格里抽 `| mart_xxx | 字段 | **N** | 说明 |` 的 N。

    只认「整格就是一个数字」的第三列 —— 数据字典里还有「挖掘口径」表，
    那些行是 `| mart_basket_rules | 交易（品类集合） | 1% | 30% | ... |`，
    第三列是 `1%`，用宽松正则会被误读成 1。
    """
    text = DICT_MD.read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("| mart_"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0].strip("`").strip()
        m = re.fullmatch(r"\**(\d[\d,]*)\**", cells[2])
        if name and m:
            out[name] = int(m.group(1).replace(",", ""))
    return out


@pytest.mark.skipif(not DICT_MD.exists(), reason="数据字典缺失")
@pytest.mark.skipif(not MARTS.exists(), reason="未跑流水线，无 marts")
def test_mart_row_counts_in_dictionary_match_artifacts():
    doc = _doc_counts()
    assert doc, "没能从数据字典解析出任何 mart 行数（格式变了？）"

    mismatched = []
    for name in CHECKED:
        path = MARTS / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"{name}.parquet 缺失，跳过对账")
        actual = len(pd.read_parquet(path))
        declared = doc.get(name)
        if declared is None:
            mismatched.append(f"{name}: 数据字典没有登记行数")
        elif declared != actual:
            mismatched.append(f"{name}: 文档 {declared} ≠ 实测 {actual}")

    assert not mismatched, "文档数字与产物不一致（跑完流水线后回填数据字典）：\n" + "\n".join(mismatched)
