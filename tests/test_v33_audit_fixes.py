"""v3.3 审计批次的回归测试。

对应 2026-11-13 全量核查发现的 3 个真实缺陷：

1. **P1 标签索引错位一格**：`lifecycle_stage` / `potential_index` 直接沿用
   传入的 RFM 表索引（RangeIndex），而其它标签的索引是 member_id
   （从 1 开始）→ 汇总时 pandas 按 member_id 对齐，两个标签整体错位一格，
   会员拿到的潜力指数其实属于**下一个**会员。
2. **决策建议报告主模型口径错配**：AUC/KS/Lift 取的是对照模型（GBDT）的，
   增益取的是主模型（RandomForest）的 → 报告出现「已达标」与
   「无一达到 +0.03」并排的自相矛盾。
3. **客群画像卡片偏好品类零区分度**：8 个客群的 TOP3 品类完全相同。

另含：v3.2 的 4 项能力（画像卡片 / 差异论证 / 规则网络图 / 质量报告 HTML）
必须真正接进 CLI 流水线，而不是只能靠跑测试时顺带产出。
"""
from pathlib import Path

import pytest

from retailinsight.io.loader import read_items, read_parquet
from retailinsight.settings import (
    INTERIM_DIR,
    ITEMS_DIR,
    MART_PROFILE,
    MART_RFM,
)

ROOT = Path(__file__).resolve().parent.parent


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


# ---------------------------------------------------------------- 缺陷 1：索引错位

@requires_interim
def test_p1_tags_index_is_member_id(ctx):
    """build_p1_tags 的返回表索引必须是 member_id（否则 join 时静默错位）。"""
    from retailinsight.features.p1_tags import build_p1_tags
    t = build_p1_tags(ctx["txn"], ctx["items"], ctx["products"], ctx["rfm"])
    assert t.index.name == "member_id"
    assert t.index.is_unique


@requires_interim
def test_potential_index_not_shifted_by_one(ctx):
    """回归：potential_index 曾整体错位一格（会员拿到的是下一个会员的值）。

    断言方式：直接按 member_id 建索引重算一遍，与 build_p1_tags 的结果逐行比对。
    """
    from retailinsight.features.p1_tags import build_p1_tags, potential_index
    t = build_p1_tags(ctx["txn"], ctx["items"], ctx["products"], ctx["rfm"])
    rfm_idx = ctx["rfm"].set_index("member_id")
    expected = potential_index(rfm_idx)
    got = t["potential_index"].reindex(expected.index)
    common = expected.notna() & got.notna()
    assert common.sum() > 1000
    mismatch = (got[common].round(4) != expected[common].round(4)).sum()
    assert mismatch == 0, f"{mismatch} 个会员的潜力指数与按 member_id 口径不一致"


@requires_interim
def test_lifecycle_stage_not_shifted_by_one(ctx):
    from retailinsight.features.p1_tags import build_p1_tags, lifecycle_stage
    t = build_p1_tags(ctx["txn"], ctx["items"], ctx["products"], ctx["rfm"])
    rfm_idx = ctx["rfm"].set_index("member_id")
    expected = lifecycle_stage(None, rfm_idx)
    got = t["lifecycle_stage"].reindex(expected.index)
    common = expected.notna() & got.notna()
    assert common.sum() > 1000
    mismatch = (got[common] != expected[common]).sum()
    assert mismatch == 0, f"{mismatch} 个会员的生命周期阶段错位"


@requires_interim
def test_normalize_rfm_index_guards_misalignment(ctx):
    """守卫：RFM 表无论以哪种形态传入，都必须归一成「索引 = member_id」。"""
    from retailinsight.features.p1_tags import _normalize_rfm_index
    rfm = ctx["rfm"]
    # 形态 1：RangeIndex + member_id 列（mart_rfm 的真实形态）
    out = _normalize_rfm_index(rfm)
    assert out.index.name == "member_id"
    assert out.index.is_unique
    # 形态 2：已经是 member_id 索引（幂等）
    out2 = _normalize_rfm_index(out)
    assert out2.index.name == "member_id"
    assert len(out2) == len(out)


# ---------------------------------------------------------------- 缺陷 2：报告主模型口径

def test_report_uses_main_model_metrics():
    """决策建议报告的模型指标必须取自 `main_model_metrics`（主模型），不是对照模型。"""
    src = (ROOT / "retailinsight" / "reports" / "render_decision_report.py").read_text(
        encoding="utf-8")
    assert 'm.get("main_model_metrics")' in src
    # 不允许再从 gbdt 里直接取对外披露的指标
    assert 'k["model_auc"] = m.get("gbdt"' not in src
    assert 'k["model_ks"] = m.get("gbdt"' not in src


def test_report_template_has_no_contradictory_prose():
    """模板里不能同时出现「已达标」与「无一达到 +0.03」这种自相矛盾的固定文字。"""
    tpl = (ROOT / "retailinsight" / "reports" / "templates" / "决策建议报告.md.tpl").read_text(
        encoding="utf-8")
    assert "无一达到 +0.03" not in tpl
    assert "该增益未达验收线的原因" not in tpl
    assert "{{model_verdict_paragraph}}" in tpl


@requires_interim
def test_report_kpis_are_self_consistent():
    """回填后的 KPI：主模型 AUC − LR AUC 必须等于披露的增益。"""
    from retailinsight.reports.render_decision_report import collect_kpis
    k = collect_kpis()
    auc = float(k["model_auc"])
    lr = float(k["model_auc_lr"])
    gain = float(k["model_auc_gain"])
    assert abs((auc - lr) - gain) < 1e-6, "报告里 AUC/基线/增益三者不自洽"
    assert k["model_name"] != "-"
    assert k["model_verdict_paragraph"] != "-"


@requires_interim
def test_store_gmv_uses_same_basis_as_report_header():
    """S8 的门店 GMV 必须与**报告头 GMV**（全量销售净额）逐口径相等。

    历史缺陷三段式：
    ① 最初用明细 `qty × unit_price`（商品目录价）→ 13996.5 万，比报告头皮 3.2 倍；
    ② 修成「净额」但仍只取会员 → 4320.7 万，比报告头少 2.98%（非会员部分），
       S8 年化金额系统性偏低；
    ③ 现口径：全量销售净额，与 `gmv_wan` 同源。
    """
    from retailinsight.reports.render_decision_report import collect_kpis
    k = collect_kpis()
    total = float(k["gmv_wan"])
    store_total = float(k["store_gmv_total_wan"])
    assert store_total <= total + 1e-6, "门店 GMV 超过总 GMV：口径必然错了"
    assert abs(store_total - total) / max(total, 1e-9) < 0.005, (
        f"门店 GMV {store_total} 与报告头 GMV {total} 不同口径")


@requires_interim
def test_s1_suggestion_has_its_own_calculation():
    """S1 曾直接复用 S2 的测算占位符，导致公式与金额对不上。"""
    from retailinsight.reports.render_decision_report import collect_kpis
    k = collect_kpis()
    assert "s1_gmv_lo" in k and "s1_gmv_hi" in k
    assert k["s1_gmv_lo"] != k["s2_gmv_lo"] or k["s1_gmv_hi"] != k["s2_gmv_hi"]
    tpl = (ROOT / "retailinsight" / "reports" / "templates" / "决策建议报告.md.tpl").read_text(
        encoding="utf-8")
    s1_block = tpl.split("**S1 ")[1].split("**S2 ")[0]
    assert "{{s1_gmv_lo}}" in s1_block
    assert "{{s2_gmv_lo}}" not in s1_block


# ---------------------------------------------------------------- 缺陷 3：画像卡片区分度

@requires_interim
def test_segment_cards_report_category_difference():
    """回归：客群画像卡片曾取「个人 top3 众数」，8 个客群结果完全相同且无任何提示。

    修复后：
    1. 口径改为「客群内品类消费额」；
    2. **同时输出品类结构差异度**（1 − cos(客群占比, 全司占比)）与品类占比，
       让读者看到这一行的真实区分度；v3.5 起生成器已注入门店档位 × 人群的
       品类乘子，差异度不再恒为 ≈0，但受 ARI≈0.13 约束在客群聚合后被稀释，
       因此卡片仍必须如实披露口径与局限。
    """
    from retailinsight.reports.profile_cards import (
        build_segment_cards,
        compute_segment_category_top,
    )
    profile = read_parquet(MART_PROFILE)
    top = compute_segment_category_top(
        profile, ctx_txn(), ctx_items(), ctx_products())
    assert top, "客群级偏好品类为空"
    for seg, info in top.items():
        assert "top" in info and "diff" in info, seg
        assert info["diff"] is not None, f"{seg} 缺少品类结构差异度"

    md = build_segment_cards(profile, read_parquet(MART_RFM), category_top=top)
    assert "品类结构差异度" in md
    assert "数据局限自曝" in md


def ctx_txn():
    return read_parquet(INTERIM_DIR / "transactions.parquet")


def ctx_items():
    return read_items(ITEMS_DIR)


def ctx_products():
    return read_parquet(INTERIM_DIR / "products.parquet")


# ---------------------------------------------------------------- 缺陷 4：门店报表错位

@requires_interim
def test_store_report_rows_are_not_shifted():
    """回归：门店报表的 gmv_growth / vip_share 曾整体错位一格。

    根因：``store["col"] = series`` 时 pandas 按**索引**对齐，
    而 reindex 后的 Series 索引是 store_id（1~15）、store 的索引是 0~14
    → 门店 1 变 NaN、门店 k 拿到门店 k-1 的值。
    """
    from retailinsight.reports.core_reports import build_core_reports
    store = build_core_reports()["store"]
    # 每个门店都必须有值（错位时门店 1 是 NaN）
    assert store["gmv_growth"].notna().all(), "门店增长率存在 NaN（疑似错位）"
    assert store["vip_share"].notna().all(), "门店 VIP 占比存在 NaN（疑似错位）"
    # 独立复算门店 1 的增长率
    txn = ctx_txn()
    ok = txn[(~txn["is_logic_error"]) & (~txn["is_return"])].copy()
    ok["month"] = ok["txn_ts"].dt.to_period("M")
    months = sorted(ok["month"].unique())
    late, early = months[-3:], months[-6:-3]
    g_late = ok[ok["month"].isin(late)].groupby("store_id")["net_amount"].sum()
    g_early = ok[ok["month"].isin(early)].groupby("store_id")["net_amount"].sum()
    expect = float(g_late[1] / g_early[1] - 1)
    got = float(store.loc[store["store_id"] == 1, "gmv_growth"].iloc[0])
    assert got == pytest.approx(round(expect, 4), abs=1e-4), "门店 1 的增长率错位"


# ---------------------------------------------------------------- 缺陷 5：仓库不可复现

def test_source_package_not_gitignored():
    """回归：`.gitignore` 的 `reports/` 未锚定仓库根，把**源码包**一起忽略了。

    后果：`git clone` 出来的仓库整个 `retailinsight/reports/` 包不存在，
    `--stage all` 直接 ModuleNotFoundError —— 交付物完全不可用。
    """
    import subprocess

    targets = [
        "retailinsight/reports/core_reports.py",
        "retailinsight/reports/excel_report.py",
        "retailinsight/reports/profile_cards.py",
        "retailinsight/reports/render_decision_report.py",
        "retailinsight/reports/templates/决策建议报告.md.tpl",
    ]
    for t in targets:
        p = ROOT / t
        assert p.exists(), f"源码文件不存在：{t}"
        r = subprocess.run(["git", "check-ignore", "-q", t],
                           cwd=ROOT, capture_output=True)
        assert r.returncode != 0, f"{t} 被 .gitignore 忽略了 —— clone 出来会缺文件"


def test_gitignore_paths_are_root_anchored():
    """目录型忽略规则必须锚定仓库根 —— 否则会误伤**同名子目录**。

    真正危险的是那些与源码目录同名的规则（如 `reports/` 曾把源码包
    `retailinsight/reports/` 一起忽略掉）。`.venv/`、`__pycache__/` 这类
    特殊名不受影响，因此只检查「会与仓库目录撞名」的规则。
    """
    dangerous = {"reports", "data", "src", "docs", "tests", "scripts",
                 "retailinsight", "dashboard", "notebooks"}
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "*" in line:
            continue
        if line.endswith("/") and not line.startswith("/"):
            name = line[:-1]
            if name in dangerous:
                pytest.fail(
                    f".gitignore 的 `{line}` 未锚定仓库根 —— 会误伤同名子目录 "
                    f"（历史事故：它把源码包 `retailinsight/{name}/` 一起忽略了）"
                )


def test_gitignore_still_ignores_data_products():
    """修 bug 不能矫枉过正：数据产物仍然必须被忽略。

    无 `.git` 的快照/导出包（v3.6 #25 的取证场景）跳过 —— 没有版本库时
    `git check-ignore` 必然报「not a git repository」，那不是 .gitignore 的问题。
    """
    import subprocess
    if not (ROOT / ".git").exists():
        pytest.skip("快照目录没有 .git，check-ignore 无从谈起")
    for t in ["reports/x.xlsx", "data/marts/mart_rfm.parquet", "data/warehouse.duckdb"]:
        r = subprocess.run(["git", "check-ignore", "-q", t],
                           cwd=ROOT, capture_output=True)
        assert r.returncode == 0, f"{t} 应被忽略（数据产物不进 git）"


# ---------------------------------------------------------------- v3.2 能力接线

def test_cli_exposes_cards_and_segment_compare_stages():
    """v3.2 的画像卡片与差异论证必须接进 CLI，而不是只能手动跑模块。"""
    from retailinsight.cli import STAGES
    assert "cards" in STAGES
    assert "segment_compare" in STAGES
    assert "warehouse" in STAGES
    assert "rules_sku" in STAGES
    assert "rules_store" in STAGES


def test_reports_stage_produces_rule_network():
    """关联规则网络图必须由 reports stage 产出（此前只有跑测试才会生成）。"""
    src = (ROOT / "retailinsight" / "cli.py").read_text(encoding="utf-8")
    assert "charts.rule_network(" in src


@requires_interim
def test_quality_html_artifact_exists():
    """质量报告 HTML 由 etl stage 产出，产物必须真实存在于 docs/。"""
    p = ROOT / "docs" / "数据质量报告.html"
    assert p.exists(), "docs/数据质量报告.html 缺失：etl stage 没跑或没接线"
    html = p.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
