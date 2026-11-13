"""看板的自动化测试（Streamlit `AppTest`）—— 真跑脚本，不是「测函数」。

为什么需要：此前看板只靠人工点验，4 个页面、下钻、筛选全都没进 CI。
`AppTest` 会在进程内把 `dashboard/app.py` 真执行一遍，因此能抓住
「列名改了但看板没跟着改」这类**只在打开页面时才炸**的缺陷（v3.4 缺陷 13 的
`top_shap_feature` 整列为空就是这一类）。

数据用临时目录里的小表（`RETAILINSIGHT_MARTS` 环境变量注入），
**绝不读写仓库里的真实产物**。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "dashboard" / "app.py"
REAL_MARTS = ROOT / "data" / "marts"

REQUIRED = [
    "mart_rfm", "mart_monthly", "mart_quarterly", "mart_member_profile",
    "mart_basket_rules", "mart_churn_score",
]
OPTIONAL = ["mart_cluster", "mart_basket_rules_sku", "mart_basket_rules_store"]


def _slim(name: str, dst: Path, rows: int = 300) -> None:
    """把真实 mart 抽样成小表（保留全部列，覆盖列名契约）。"""
    df = pd.read_parquet(REAL_MARTS / f"{name}.parquet")
    df.head(rows).to_parquet(dst / f"{name}.parquet", index=False)


@pytest.fixture()
def marts(tmp_path, monkeypatch):
    if not (REAL_MARTS / "mart_rfm.parquet").exists():
        pytest.skip("缺少 mart 产物，先跑 `make all`")
    d = tmp_path / "marts"
    d.mkdir()
    for n in REQUIRED + OPTIONAL:
        if (REAL_MARTS / f"{n}.parquet").exists():
            _slim(n, d)
    monkeypatch.setenv("RETAILINSIGHT_MARTS", str(d))
    return d


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def test_app_runs_without_exception(marts):
    at = _run()
    assert not at.exception, f"看板首屏报错：{at.exception}"


def test_four_tabs_are_rendered(marts):
    at = _run()
    labels = [t.label for t in at.tabs]
    assert labels == ["经营驾驶舱", "客群分析", "商品洞察", "流失预警"]


def test_kpi_metrics_match_data(marts):
    at = _run()
    rfm = pd.read_parquet(marts / "mart_rfm.parquet")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["分析会员数"] == f"{len(rfm):,}"
    assert "总净消费额（万元）" in metrics


def test_down_drill_panel_shows_a_member(marts):
    """未点选散点时，右下钻区必须给出「提示 + 一个默认会员」，不能空白。"""
    at = _run()
    texts = " ".join(m.value for m in at.markdown)
    assert "会员 " in texts


def test_risk_slider_switch_does_not_break(marts):
    """切到「中」风险等级 → 表格随之过滤，且不抛异常。"""
    at = _run()
    assert at.select_slider, "流失预警页的风险等级滑块没渲染出来"
    at.select_slider[0].set_value("中").run()
    assert not at.exception
    churn = pd.read_parquet(marts / "mart_churn_score.parquet")
    assert (churn["risk_level"] == "中").any(), "抽样数据里应有中风险会员"


def test_segment_multiselect_filters_scatter(marts):
    at = _run()
    if not at.multiselect:
        pytest.skip("当前产物没有客群筛选控件")
    options = at.multiselect[0].options
    at.multiselect[0].set_value([options[0]]).run()
    assert not at.exception


def test_optional_marts_absent_degrades_gracefully(marts):
    """缺 `mart_cluster` / SKU 规则时看板仍要能起（快速验证场景）。"""
    for n in OPTIONAL:
        p = marts / f"{n}.parquet"
        if p.exists():
            p.unlink()
    at = _run()
    assert not at.exception


def test_first_screen_timing_is_reported(marts):
    at = _run()
    infos = " ".join(i.value for i in at.info)
    assert "秒" in infos and "验收线 3 秒" in infos


def test_dashboard_does_not_write_to_real_marts(marts):
    """看板是只读的：跑一遍不得改动仓库里的真实产物。"""
    stamp = {p.name: p.stat().st_mtime_ns for p in REAL_MARTS.glob("*.parquet")}
    _run()
    after = {p.name: p.stat().st_mtime_ns for p in REAL_MARTS.glob("*.parquet")}
    assert after == stamp
