"""CLV 简化法单测：月流失换算、留存封顶、兜底来源与四分位分层。

简化法的每个输入都必须可追溯（见 `models/clv.py` docstring），
所以这里逐条钉住「公式 → 输出列」的对应关系，而不是只断言「有值」。
"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.models.clv import (
    CHURN_LABEL_MONTHS,
    CLV_GROSS_MARGIN,
    clv_summary,
    compute_clv,
    expected_lifetime_months,
    monthly_churn_rate,
)


def test_monthly_churn_follows_geometric_assumption():
    """月流失率必须满足 (1-m)^3 = 1-p，且越界概率被裁剪。"""
    m = monthly_churn_rate(pd.Series([0.5, 0.0]))
    assert m.iloc[0] == pytest.approx(1.0 - 0.5 ** (1.0 / CHURN_LABEL_MONTHS))
    assert m.iloc[1] == pytest.approx(0.0)

    # p=1 会让月流失率变成 1（留存月数除零），必须被裁剪到 (0, 1)
    clipped = monthly_churn_rate(pd.Series([1.5]))
    assert 0.0 < clipped.iloc[0] < 1.0


def test_expected_lifetime_matches_inverse_and_is_capped():
    lt = expected_lifetime_months(pd.Series([0.0, 0.5, 1.0]))
    assert lt.iloc[0] == pytest.approx(120.0)      # 零流失封顶 120 个月
    assert lt.iloc[1] == pytest.approx(2.0)
    assert lt.iloc[2] == pytest.approx(1.0)


def test_compute_clv_merges_churn_table_and_uses_formula():
    profile = pd.DataFrame({
        "member_id": [1, 2, 3, 4],
        "monthly_amount": [100.0, 200.0, 300.0, 400.0],
        "rfm_segment": ["A", "A", "B", "B"],
    })
    churn = pd.DataFrame({"member_id": [1, 2, 3, 4],
                          "churn_prob": [0.0, 0.5, 0.9, 0.999999]})
    out = compute_clv(profile, churn)

    assert set(out["clv_source"]) == {"model"}
    assert out["clv"].gt(0).all()

    # 第 2 行：留存月数 = 1 / (1 - (1-0.5)^(1/3))，CLV = 月均 × 留存 × 毛利率
    lt = 1.0 / (1.0 - (1.0 - 0.5) ** (1.0 / CHURN_LABEL_MONTHS))
    row = out.loc[out["member_id"] == 2].iloc[0]
    # clv 列本身保留 2 位小数，容差按分计
    assert row["clv"] == pytest.approx(200.0 * lt * CLV_GROSS_MARGIN, abs=0.01)
    assert row["expected_lifetime_months"] == pytest.approx(lt)


def test_compute_clv_requires_churn_prob():
    profile = pd.DataFrame({"member_id": [1], "monthly_amount": [10.0]})
    with pytest.raises(ValueError, match="churn_prob"):
        compute_clv(profile)


def test_missing_score_falls_back_to_segment_median():
    """未被流失模型覆盖的会员用同客群中位留存月数兜底，并留痕来源。"""
    profile = pd.DataFrame({
        "member_id": [1, 2, 3, 4],
        "monthly_amount": [100.0, 100.0, 100.0, 100.0],
        "churn_prob": [0.1, 0.2, 0.1, np.nan],
        "rfm_segment": ["A", "A", "B", "B"],
    })
    out = compute_clv(profile)

    assert out.loc[out["member_id"] == 4, "clv_source"].iloc[0] == "segment_median"
    assert out["clv"].notna().all()
    # 兜底值取自 B 客群的模型会员（member_id=3）的留存月数
    b_median = out.loc[out["member_id"] == 3, "expected_lifetime_months"].iloc[0]
    assert out.loc[out["member_id"] == 4, "expected_lifetime_months"].iloc[0] == pytest.approx(b_median)


def test_compute_clv_falls_back_to_monetary_column():
    """没有 monthly_amount 时退化为累计 monetary（仍要算出 CLV）。"""
    profile = pd.DataFrame({"member_id": [1, 2], "monetary": [50.0, 80.0],
                            "churn_prob": [0.1, 0.3]})
    out = compute_clv(profile)
    assert (out["clv"] > 0).all()


def test_clv_tier_is_monotone_quartile_split():
    profile = pd.DataFrame({
        "member_id": list(range(8)),
        "monthly_amount": [float(i + 1) * 10 for i in range(8)],
        "churn_prob": [0.05] * 8,
    })
    out = compute_clv(profile)
    assert list(out["clv_tier"].cat.categories) == ["低CLV", "中低CLV", "中高CLV", "高CLV"]
    assert out["clv_tier"].value_counts().to_dict() == {
        "低CLV": 2, "中低CLV": 2, "中高CLV": 2, "高CLV": 2}
    # 分层必须与 CLV 单调一致（低层不该出现更高的 CLV）
    ordered = out.sort_values("clv")["clv_tier"].astype(str).tolist()
    assert ordered == sorted(ordered, key=["低CLV", "中低CLV", "中高CLV", "高CLV"].index)


def test_clv_summary_reports_four_tiers_with_totals():
    profile = pd.DataFrame({
        "member_id": list(range(8)),
        "monthly_amount": [float(i + 1) * 10 for i in range(8)],
        "churn_prob": [0.05] * 8,
    })
    summary = clv_summary(compute_clv(profile))
    assert list(summary.index.astype(str)) == ["低CLV", "中低CLV", "中高CLV", "高CLV"]
    assert {"members", "avg_clv", "total_clv", "avg_lifetime_months"} <= set(summary.columns)
    assert summary["members"].sum() == 8
