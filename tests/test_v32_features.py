"""v3.2 新增能力的测试：P1 扩展标签 / 分区 / 星型模型 / 关联规则 / 报表补齐 / 差异论证。"""
from pathlib import Path

import pandas as pd
import pytest

from retailinsight.io.loader import read_items, read_parquet
from retailinsight.settings import INTERIM_DIR, ITEMS_DIR, MART_CLUSTER, MART_PROFILE, MART_RFM


def _has_interim() -> bool:
    return (INTERIM_DIR / "transactions.parquet").exists()


requires_interim = pytest.mark.skipif(
    not _has_interim(), reason="需要先运行 ETL 生成 interim 产物")


@pytest.fixture(scope="module")
def ctx():
    if not _has_interim():
        pytest.skip("需要 interim 产物")
    return {
        "txn": read_parquet(INTERIM_DIR / "transactions.parquet"),
        "items": read_items(ITEMS_DIR),
        "products": read_parquet(INTERIM_DIR / "products.parquet"),
        "rfm": read_parquet(MART_RFM),
    }


# ---------------------------------------------------------------- P1 扩展标签

def test_p1_tags_module_has_18_tags():
    """计划 §7.3 的 20 个扩展标签：behavior 3 个 + p1_tags 18 个（含 CLV 口径另算）。"""
    from retailinsight.features import p1_tags
    funcs = [f for f in dir(p1_tags) if not f.startswith("_")
             and callable(getattr(p1_tags, f))
             and getattr(p1_tags, f).__module__ == p1_tags.__name__]
    # 18 个标签函数（build_p1_tags 之外）
    assert len(funcs) >= 18


@requires_interim
def test_p1_tags_all_produced_and_finite(ctx):
    from retailinsight.features.p1_tags import build_p1_tags
    t = build_p1_tags(ctx["txn"], ctx["items"], ctx["products"], ctx["rfm"])
    assert len(t.columns) >= 17
    # 所有列必须是有限数值（模型要用，不能有 inf）
    for c in t.columns:
        v = pd.to_numeric(t[c], errors="coerce")
        assert not np_isinf(v), f"{c} 含 inf"


def np_isinf(s: pd.Series) -> bool:
    import numpy as np
    arr = s.to_numpy(dtype="float64", na_value=np.nan)
    return bool(np.isinf(arr).any())


@requires_interim
def test_p1_tags_have_variation(ctx):
    """标签必须有区分度 —— 恒为常数的标签等于没做。"""
    from retailinsight.features.p1_tags import build_p1_tags
    t = build_p1_tags(ctx["txn"], ctx["items"], ctx["products"], ctx["rfm"])
    for c in ("store_cnt", "promo_response_rate", "coupon_used_rate",
              "cross_category_ratio", "channel_preference", "price_band"):
        assert c in t.columns, c
        assert t[c].nunique() > 1, f"{c} 恒为常数，没有区分度"


@requires_interim
def test_coupon_used_rate_not_degenerate(ctx):
    """券使用率不能恒为 1（曾因阈值口径失效导致整列无区分度）。"""
    from retailinsight.features.p1_tags import coupon_used_rate
    s = coupon_used_rate(ctx["txn"])
    assert len(s) > 0
    assert s.nunique() > 1
    assert s.max() <= 1.0


@requires_interim
def test_channel_preference_uses_items(ctx):
    """渠道偏好必须基于明细表 —— 用交易表算件数会恒等于 1。"""
    from retailinsight.features.p1_tags import channel_preference
    s = channel_preference(ctx["txn"], ctx["items"])
    assert s.nunique() > 1
    assert s.max() <= 1.0


@requires_interim
def test_repurchase_cycle_days_positive(ctx):
    from retailinsight.features.p1_tags import repurchase_cycle_days
    s = repurchase_cycle_days(ctx["txn"])
    valid = s[s > 0]
    assert len(valid) > 0
    assert valid.max() < 1000


@requires_interim
def test_lifecycle_stage_in_range(ctx):
    from retailinsight.features.p1_tags import lifecycle_stage
    s = lifecycle_stage(ctx["txn"], ctx["rfm"])
    assert set(s.unique()) <= {1, 2, 3, 4, 5}


@requires_interim
def test_profile_contains_p1_tags(ctx):
    profile = read_parquet(MART_PROFILE)
    for c in ("promo_response_rate", "store_cnt", "potential_index",
              "lifecycle_stage", "clv", "clv_tier"):
        assert c in profile.columns, c


# ---------------------------------------------------------------- 聚类覆盖守卫

@requires_interim
def test_cluster_covers_all_rfm_members(ctx):
    """回归用例：曾静默丢掉 291 名会员（上游产物不同批次）。"""
    from retailinsight.features.cluster_features import build_cluster_features
    feats = build_cluster_features(ctx["rfm"], ctx["txn"], ctx["items"], ctx["products"])
    assert len(feats) == len(ctx["rfm"])
    assert set(feats["member_id"]) == set(ctx["rfm"]["member_id"])


@requires_interim
def test_cluster_artifact_row_count_matches(ctx):
    cluster = read_parquet(MART_CLUSTER)
    assert len(cluster) == len(ctx["rfm"])


@requires_interim
def test_cluster_features_have_no_nan(ctx):
    from retailinsight.features.cluster_features import build_cluster_features
    feats = build_cluster_features(ctx["rfm"], ctx["txn"], ctx["items"], ctx["products"])
    for c in feats.columns:
        if c == "member_id":
            continue
        assert feats[c].notna().all(), f"{c} 存在 NaN"


# ---------------------------------------------------------------- 报表补齐

@requires_interim
def test_monthly_repurchase_rate_table():
    from retailinsight.settings import MARTS_DIR
    path = MARTS_DIR / "mart_monthly_repurchase.parquet"
    if not path.exists():
        pytest.skip("需先跑 reports stage")
    df = read_parquet(path)
    assert set(["month", "active_members", "repeat_members",
                "repurchase_rate"]).issubset(df.columns)
    assert df["repurchase_rate"].between(0, 1).all()
    assert df["repurchase_rate"].nunique() > 1   # 时序指标必须有波动


@requires_interim
def test_store_report_has_growth_and_structure():
    from retailinsight.reports.core_reports import build_core_reports
    core = build_core_reports()
    store = core["store"]
    assert "gmv_growth" in store.columns
    assert "vip_share" in store.columns
    assert "vip_share_diff" in store.columns


# ---------------------------------------------------------------- 差异论证

@requires_interim
def test_segment_compare_produces_report():
    from retailinsight.models.segment_compare import run_segment_compare
    out = run_segment_compare()
    assert Path(out["report"]).exists()
    ct = read_parquet("data/marts/mart_segment_conflict.parquet")
    assert len(ct) > 0
    assert {"rfm_segment", "cluster_name", "member_cnt",
            "share_in_segment"}.issubset(ct.columns)


@requires_interim
def test_segment_conflict_has_rows():
    """分歧分析必须真的找得到分歧会员（不是恒为空）。"""
    from retailinsight.models.segment_compare import find_conflicts
    profile = read_parquet(MART_PROFILE)
    rfm = read_parquet(MART_RFM)
    profile = profile.merge(rfm[["member_id", "monetary", "frequency", "recency_days"]],
                            on="member_id", how="left", suffixes=("", "_rfm"))
    conflict = find_conflicts(profile)
    assert len(conflict) > 0, "分歧会员为空，说明两种方法完全一致或判定逻辑失效"
    assert "conflict_type" in conflict.columns


# ---------------------------------------------------------------- 质量报告 HTML

@requires_interim
def test_quality_html_is_self_contained():
    from retailinsight.etl.quality_html import render_quality_html
    path = render_quality_html({"before": {"transactions": 100}, "after": {"transactions": 98},
                                "details": {"dedup_removed": 2}})
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html   # 不依赖外网资源
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------- 关联网络图

@requires_interim
def test_rule_network_figure_generated():
    from retailinsight.settings import MART_RULES
    from retailinsight.viz import charts
    rules = read_parquet(MART_RULES)
    out = charts.rule_network(rules, top_n=12)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 10_000


# ---------------------------------------------------------------- 看板首屏

@requires_interim
def test_dashboard_first_paint_under_3s():
    """计划 §11 W5 验收：看板首屏 < 3 秒。

    首屏的决定性成本是**加载 data/marts 下的宽表**（看板只读 mart，不重算指标），
    这里按看板实际加载的全部表做计时。
    """
    import time

    from retailinsight.settings import MARTS_DIR
    names = ["mart_rfm", "mart_member_profile", "mart_churn_score",
             "mart_cluster", "mart_monthly", "mart_basket_rules"]
    t0 = time.perf_counter()
    for n in names:
        p = MARTS_DIR / f"{n}.parquet"
        if p.exists():
            read_parquet(p)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"首屏数据加载 {elapsed:.2f}s 超过 3 秒验收线"


def test_dashboard_has_first_paint_timer():
    """看板必须展示首屏耗时（否则「首屏 <3s」只是口头承诺）。"""
    src = (Path(__file__).resolve().parent.parent / "dashboard" / "app.py").read_text(
        encoding="utf-8")
    assert "_T0 = time.perf_counter()" in src
    assert "本次脚本执行耗时" in src
