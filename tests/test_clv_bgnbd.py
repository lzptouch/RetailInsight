"""CLV 主口径（BG/NBD + Gamma-Gamma）单测。

重点不是「能拟合」，而是三件容易被悄悄做错的事：

1. `build_summary` 的 frequency 是 **重复购买次数**（交易数 − 1），
   与画像里「去重购买日数」不是同一口径，混用会让模型整体偏移；
2. 只买过一次的客户**不硬估** —— 返回 NaN 并让上层退简化法，
   而不是用一条不相干的解析式凑一个数；
3. `apply_bgnbd_clv` 必须逐行标注口径来源，混用口径要可追溯。
"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.models import clv_bgnbd as cb
from retailinsight.models.clv import apply_bgnbd_clv, compute_clv

pytestmark = pytest.mark.skipif(not cb.available(), reason="未安装 lifetimes，BG/NBD 不可用")

CUTOFF = pd.Timestamp("2021-12-31")


def _txn(n_members: int = 60, seed: int = 7) -> pd.DataFrame:
    """80 天~1 年不等的购买流水，保证有「只买一次」和「多次购买」两类客户。"""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-01-01")
    rows = []
    for mid in range(1, n_members + 1):
        n = 1 if mid % 7 == 0 else int(rng.integers(3, 10))
        for off in np.sort(rng.integers(0, 300, n)):
            rows.append({
                "member_id": mid,
                "txn_ts": base + pd.Timedelta(days=int(off)),
                "net_amount": float(rng.integers(50, 400)),
                "member_type": "会员" if mid % 11 else "非会员",
                "is_logic_error": mid % 13 == 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    txn = _txn()
    summary = cb.build_summary(txn, CUTOFF)
    bgf, ggf = cb.fit_models(summary)
    return txn, summary, bgf, ggf


def test_build_summary_uses_repeat_purchase_convention():
    """frequency = 交易次数 − 1，T/recency 以首购为原点，monetary 为单次均额。"""
    txn = pd.DataFrame({
        "member_id": [1, 1, 1],
        "txn_ts": [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-31"), CUTOFF],
        "net_amount": [100.0, 200.0, 300.0],
        "member_type": ["会员"] * 3,
        "is_logic_error": [False] * 3,
    })
    s = cb.build_summary(txn, CUTOFF)
    row = s.loc[1]
    assert row["frequency"] == 2                       # 3 次交易 → 2 次重复购买
    assert row["recency"] == 364                       # 首购到今天
    assert row["T"] == 364
    assert row["monetary_value"] == pytest.approx(200.0)


def test_build_summary_drops_non_member_and_logic_error():
    """非会员与逻辑差错的流水不进模型 —— 与画像口径保持一致。"""
    txn = pd.DataFrame({
        "member_id": [1, 2, 3],
        "txn_ts": [CUTOFF - pd.Timedelta(days=10)] * 3,
        "net_amount": [100.0, 100.0, 100.0],
        "member_type": ["会员", "非会员", "会员"],
        "is_logic_error": [False, False, True],
    })
    s = cb.build_summary(txn, CUTOFF)
    assert list(s.index) == [1]


def test_predict_clv_refuses_to_guess_single_purchase_customers(fitted):
    """只买过一次的客户返回 NaN，由上层退简化法，而不是硬凑一个值。"""
    _, summary, bgf, ggf = fitted
    single = summary[summary["frequency"] == 0]
    assert not single.empty, "合成数据必须包含只买一次的客户"

    clv = cb.predict_clv(bgf, ggf, single)
    assert clv.isna().all()

    repeat = summary[summary["frequency"] > 0]
    assert cb.predict_clv(bgf, ggf, repeat).notna().all()


def test_predict_clv_is_positive_and_ordered_by_frequency(fitted):
    """预测值必须非负，且高频客户整体高于低频客户（模型没有反着学）。"""
    _, summary, bgf, ggf = fitted
    repeat = summary[summary["frequency"] > 0]
    clv = cb.predict_clv(bgf, ggf, repeat)

    assert (clv >= 0).all()
    lo = clv[repeat["frequency"] <= 3].median()
    hi = clv[repeat["frequency"] >= 6].median()
    assert hi > lo


def test_apply_bgnbd_clv_overrides_main_column_and_marks_source():
    """clv 取 BG/NBD 值，缺失处回退 clv_simple，clv_method 逐行标注。"""
    profile = pd.DataFrame({
        "member_id": [1, 2, 3, 4],
        "monthly_amount": [100.0, 100.0, 100.0, 100.0],
        "rfm_segment": ["A", "A", "B", "B"],
    })
    churn = pd.DataFrame({"member_id": [1, 2, 3, 4], "churn_prob": [0.1, 0.2, 0.3, 0.4]})
    profile = compute_clv(profile, churn=churn)

    clv_bgnbd = pd.Series([500.0, 300.0, np.nan, np.nan],
                          index=pd.Index([1, 2, 3, 4], name="member_id"))
    out = apply_bgnbd_clv(profile, clv_bgnbd)

    assert out.loc[out.member_id == 1, "clv"].iloc[0] == pytest.approx(500.0)
    assert out.loc[out.member_id == 2, "clv"].iloc[0] == pytest.approx(300.0)
    # 没有 BG/NBD 估计的人保留简化法原值，且来源可查
    for mid in (3, 4):
        row = out.loc[out.member_id == mid].iloc[0]
        assert row["clv"] == pytest.approx(row["clv_simple"])
        assert row["clv_method"] == "simple_fallback"
    assert (out.loc[out.member_id.isin([1, 2]), "clv_method"] == "bgnbd").all()
    assert set(out["clv_tier"].astype(str)) == {"低CLV", "中低CLV", "中高CLV", "高CLV"}


def test_apply_bgnbd_clv_requires_simple_column():
    with pytest.raises(ValueError, match="clv_simple"):
        apply_bgnbd_clv(pd.DataFrame({"member_id": [1]}), pd.Series([1.0], index=[1]))


def test_apply_bgnbd_clv_marks_registered_non_buyers_as_no_purchase():
    """注册未购（CLV 恒 0）必须单独标 `no_purchase`，不能混进 simple_fallback ——
    否则「简化法兜底客户平均价值」会被一群 0 稀释，看不出兜底到底兜了什么。"""
    profile = pd.DataFrame({
        "member_id": [1, 2, 3, 4],
        "monthly_amount": [100.0, 50.0, 0.0, np.nan],
        "rfm_segment": ["A", "A", "B", "B"],
    })
    churn = pd.DataFrame({"member_id": [1, 2, 3, 4], "churn_prob": [0.1, 0.2, 0.3, 0.4]})
    profile = compute_clv(profile, churn=churn)
    out = apply_bgnbd_clv(profile, pd.Series([500.0, np.nan, np.nan, np.nan],
                                             index=pd.Index([1, 2, 3, 4], name="member_id")))

    methods = dict(zip(out["member_id"], out["clv_method"], strict=True))
    assert methods == {1: "bgnbd", 2: "simple_fallback", 3: "no_purchase", 4: "no_purchase"}
    # 未购客户的 CLV 是 0 而不是 NaN（否则下游 groupby 直接丢行）
    assert (out["clv"] >= 0).all()
    assert out.loc[out["member_id"].isin([3, 4]), "clv"].eq(0).all()


def test_backtest_compares_both_methods_on_same_cohort():
    """回测必须同人群比较，且两种方法的样本量一致（否则是「比谁的兜底更准」）。"""
    txn = _txn(n_members=40)
    out = cb.backtest(txn, pd.Timestamp("2021-09-30"), CUTOFF)

    assert out["method"] == "bgnbd_gamma_gamma"
    assert out["bgnbd"]["n"] == out["simple"]["n"] == out["n_members"]
    assert out["winner"] in {"bgnbd", "simple"}
    assert out["bgnbd"]["amount_mae"] > 0
    # 这里的合成流水是纯随机的，未来购买与历史节奏无关，所以只要求
    # 相关系数是有限值（[-1, 1]）—— 「BG/NBD 更准」由真实数据回测证明，
    # 不能拿随机数据反过来要求模型
    assert -1 <= out["bgnbd"]["amount_spearman"] <= 1
    # 参数必须收敛到有限值，不能是 NaN（NaN 会被 json.dumps 写成非法 JSON）
    assert all(np.isfinite(list(out["bgf_params"].values())))
