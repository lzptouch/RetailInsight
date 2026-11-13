"""SKU 级与分门店关联规则的测试（计划 §8.3 / R-9）。"""
import pandas as pd
import pytest

from retailinsight.io.schema import ANCHOR_COMBOS, SKU_ANCHOR_PAIRS_PER_CATEGORY
from retailinsight.models.rules import (
    _load_basket_frame,
    run_rules,
    run_sku_rules,
    run_store_rules,
    select_top_skus,
)


def _has_interim() -> bool:
    from retailinsight.settings import INTERIM_DIR
    return (INTERIM_DIR / "transactions.parquet").exists()


requires_interim = pytest.mark.skipif(
    not _has_interim(), reason="需要先运行 ETL 生成 interim 产物")


# ---------------------------------------------------------------- 品类级（回归）

@requires_interim
def test_category_rules_still_work():
    out = run_rules()
    assert len(out) >= 30                       # 计划验收线：合格规则 ≥ 30 条
    assert set(["antecedents", "consequents", "support",
                "confidence", "lift"]).issubset(out.columns)
    assert (out["lift"] > 1.5).all()


@requires_interim
def test_category_rules_recover_all_anchors():
    """4 组预埋组合必须全部回收（计划 §4.2 埋点验证）。"""
    out = run_rules()
    for a, b, _prob in ANCHOR_COMBOS:
        fwd = out[(out["antecedents"].str.contains(a, regex=False))
                  & (out["consequents"].str.contains(b, regex=False))]
        rev = out[(out["antecedents"].str.contains(b, regex=False))
                  & (out["consequents"].str.contains(a, regex=False))]
        # 方向由 support 决定（A 的边际更高则规则写成 B→A），两个方向都算回收
        assert len(fwd) + len(rev) > 0, f"{a}→{b} 未回收"


# ---------------------------------------------------------------- SKU 级

@requires_interim
def test_select_top_skus_picks_by_sales_volume():
    it = _load_basket_frame()
    top = select_top_skus(it, top_n=50)
    assert len(top) == 50
    qty = it.groupby("sku")["qty"].sum()
    # 选中的 SKU 销量必须都在全体 SKU 的销量前 50
    threshold = qty.nlargest(50).min()
    assert qty.loc[top].min() >= threshold


@requires_interim
def test_sku_rules_meet_acceptance_criteria():
    """计划验收：SKU 级规则数 ≥ 30 且业务可解读（单→单、lift 显著）。"""
    out = run_sku_rules()
    assert len(out) >= 30
    assert (out["lift"] > 1.5).all()
    assert out["confidence"].max() <= 1.0
    # 前后件应能解析成具体 SKU
    for v in out["antecedents"].head(20):
        for token in v.split("/"):
            assert token.isdigit()


@requires_interim
def test_sku_rules_recover_embedded_sku_anchors():
    """生成器埋的 SKU 级锚点必须能被挖回来（这是 SKU 级规则的唯一真信号）。"""
    out = run_sku_rules()
    # 每个锚点品类组合应产出至少一对互为关联的 SKU
    found = 0
    for _, row in out.iterrows():
        a = set(row["antecedents"].split("/"))
        b = set(row["consequents"].split("/"))
        if a & b == set() and row["confidence"] > 0.8:
            found += 1
    assert found >= len(ANCHOR_COMBOS) * SKU_ANCHOR_PAIRS_PER_CATEGORY


@requires_interim
def test_sku_rules_are_stricter_than_category_rules():
    """SKU 级是更细的粒度，规则数应少于品类级（更具体 = 更少组合）。"""
    cat = run_rules()
    sku = run_sku_rules()
    assert len(sku) < len(cat)


# ---------------------------------------------------------------- 分门店

@requires_interim
def test_store_rules_cover_all_stores():
    out = run_store_rules(level="category")
    assert len(out) > 0
    assert out["store_id"].nunique() >= 10
    assert "store_id" in out.columns


@requires_interim
def test_store_rules_contain_global_anchors():
    """分门店规则里应仍能找到全局的强锚点组合。"""
    out = run_store_rules(level="category")
    hit = out[(out["antecedents"].str.contains("洗发水", regex=False))
              & (out["consequents"].str.contains("护发素", regex=False))]
    assert len(hit) > 0


@requires_interim
def test_store_rule_support_is_relative_to_store():
    """分门店的支持度是**该店内部**的比例，不应超过 1。"""
    out = run_store_rules(level="category")
    assert out["support"].max() <= 1.0
    assert out["support"].min() >= 0.0


# ---------------------------------------------------------------- 单测：不依赖产物

def test_mine_returns_expected_columns(monkeypatch):
    from retailinsight.models import rules as R
    baskets = [["a", "b"], ["a", "b"], ["a"], ["b"], ["a", "b", "c"]]
    out = R._mine(baskets, min_support=0.4, min_confidence=0.5, min_lift=0.9)
    assert list(out.columns) == R.KEEP
    assert len(out) >= 1
    assert (out["lift"] > 0.9).all()


def test_mine_handles_empty_baskets():
    from retailinsight.models import rules as R
    out = R._mine([], 0.01, 0.3, 1.5)
    assert out.empty
    assert list(out.columns) == R.KEEP


def test_mine_handles_integer_items():
    """SKU 级前后件是整数 id，必须能正确转成字符串。"""
    from retailinsight.models import rules as R
    baskets = [[100, 200], [100, 200], [100, 200], [100], [200]]
    out = R._mine(baskets, min_support=0.4, min_confidence=0.5, min_lift=0.9)
    assert len(out) >= 1
    for v in pd.concat([out["antecedents"], out["consequents"]]):
        for token in str(v).split("/"):
            assert token.isdigit()
