"""聚类模块单测：特征矩阵、定 K、簇命名、先验对照。"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.models.cluster import (
    _name_cluster,
    build_cluster_matrix,
    choose_best_k,
    choose_k,
    describe_clusters,
    evaluate_space,
    fit_kmeans,
    oracle_silhouette,
    select_space,
    validate_against_persona,
)


def _toy_features(n=300, seed=42):
    """造 3 个明显可分的团：高频高值 / 低频低值 / 沉睡。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        grp = i % 3
        if grp == 0:
            rec, freq, mon = rng.normal(10, 3), rng.normal(30, 5), rng.normal(5000, 800)
        elif grp == 1:
            rec, freq, mon = rng.normal(20, 5), rng.normal(8, 2), rng.normal(600, 150)
        else:
            rec, freq, mon = rng.normal(400, 60), rng.normal(3, 1), rng.normal(200, 60)
        rows.append((i, rec, freq, mon, mon / max(freq, 1), 10, 2.0))
    df = pd.DataFrame(rows, columns=["member_id", "recency_days", "frequency", "monetary",
                                      "avg_basket", "n_categories", "items_per_txn"])
    return df


def test_signed_log1p_keeps_negative():
    """对称 log1p：负值保留符号与量级，不被 clip 成 0。"""
    from retailinsight.models.cluster import _signed_log1p
    x = pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = _signed_log1p(x)
    assert out[0] < 0 and out[3] > 0
    assert out[2] == pytest.approx(0.0)
    # 反对称：f(-x) == -f(x)
    assert out[0] == pytest.approx(-out[4])
    assert out[1] == pytest.approx(-out[3])


def test_build_cluster_matrix_keeps_negative_distinguishable():
    """负值会员必须与零消费会员区分开（clip 方案会把两者压成同一个值）。"""
    df = _toy_features(n=60)
    df.loc[0, "monetary"] = -500.0
    df.loc[1, "monetary"] = 0.0
    df.loc[2, "monetary"] = 100.0
    X, _ = build_cluster_matrix(df, ["recency_days", "frequency", "monetary"])
    assert np.isfinite(X).all()
    # 负值 < 零值 < 正值，顺序未被破坏
    assert X[0, 2] < X[1, 2] < X[2, 2]


def test_choose_k_returns_metrics():
    df = _toy_features()
    X, _ = build_cluster_matrix(df)
    metrics = choose_k(X, k_range=range(2, 5))
    assert set(metrics) == {2, 3, 4}
    for m in metrics.values():
        assert m["inertia"] > 0
        assert -1 <= m["silhouette"] <= 1


def test_choose_best_k_respects_guard():
    """最小群占比 <2% 的 K 即使轮廓系数更高也不应被选中。"""
    rows = [
        {"k": 2, "silhouette": 0.90, "min_cluster_share": 0.005, "ari": 0.9},
        {"k": 3, "silhouette": 0.50, "min_cluster_share": 0.20, "ari": 0.5},
    ]
    best, reason = choose_best_k(rows, has_persona=True)
    assert best["k"] == 3
    assert reason == "max_ari_under_silhouette_and_share_guard"


def test_choose_best_k_fallback_below_target():
    rows = [{"k": 2, "silhouette": 0.10, "min_cluster_share": 0.50, "ari": 0.1},
            {"k": 3, "silhouette": 0.20, "min_cluster_share": 0.30, "ari": 0.2}]
    best, reason = choose_best_k(rows, has_persona=True)
    assert best["k"] == 3          # 都没过护栏 → 取轮廓系数最高
    assert reason == "no_k_satisfies_guard_fallback_max_silhouette"


def test_choose_best_k_without_persona_uses_silhouette():
    rows = [{"k": 2, "silhouette": 0.40, "min_cluster_share": 0.30, "ari": None},
            {"k": 3, "silhouette": 0.60, "min_cluster_share": 0.10, "ari": None}]
    best, reason = choose_best_k(rows, has_persona=False)
    assert best["k"] == 3
    assert reason == "max_silhouette_under_guard"


def test_name_cluster_rules():
    import pandas as pd
    base = {"active_pct": 0.5, "monetary_pct": 0.5, "frequency_pct": 0.5}
    assert _name_cluster(pd.Series({**base, "monetary_pct": 0.9, "active_pct": 0.8})) == "高价值活跃"
    assert _name_cluster(pd.Series({**base, "monetary_pct": 0.9, "active_pct": 0.2})) == "高价值流失预警"
    assert _name_cluster(pd.Series({**base, "frequency_pct": 0.9, "monetary_pct": 0.2})) == "高频低额"
    assert _name_cluster(pd.Series({**base, "active_pct": 0.1})) == "沉睡流失"
    assert _name_cluster(pd.Series({**base, "frequency_pct": 0.2, "monetary_pct": 0.2})) == "低频低额"


def test_describe_clusters_unique_names():
    df = _toy_features()
    X, _ = build_cluster_matrix(df)
    _, labels = fit_kmeans(X, 4)
    desc = describe_clusters(df, labels)
    assert len(desc) == 4
    assert desc["member_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert desc["gmv_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert desc["cluster_name"].is_unique    # 同名簇必须加序号后缀


def test_validate_against_persona():
    df = _toy_features()
    persona = pd.Series(np.repeat(["a", "b"], len(df) // 2), index=df["member_id"])
    labels = np.repeat([0, 1], len(df) // 2)
    res = validate_against_persona(labels, df["member_id"].values, persona)
    assert res["ari"] == pytest.approx(1.0, abs=1e-6)   # 完全一致 → ARI=1
    assert res["n_compared"] == len(df)


def test_oracle_silhouette_detects_unreachable_target():
    """真实人群在特征空间里不可分时，神谕上界会很低甚至为负。"""
    df = _toy_features()
    X, _ = build_cluster_matrix(df)
    persona = np.random.default_rng(0).choice([0, 1, 2], size=len(X))
    assert oracle_silhouette(X, persona) is not None
    assert oracle_silhouette(X, None) is None


def test_evaluate_space_and_select():
    df = _toy_features()
    persona = pd.Series(np.repeat(["a", "b", "c"], len(df) // 3), index=df["member_id"])
    evs = [evaluate_space(df, name, cols, persona)
           for name, cols in [("base6", ["recency_days", "frequency", "monetary",
                                         "avg_basket", "n_categories", "items_per_txn"]),
                              ("rfm3", ["recency_days", "frequency", "monetary"])]]
    choice = select_space(evs, has_persona=True)
    assert choice["selected_space"] in {"base6", "rfm3"}
    assert choice["criterion"] == "ari_vs_persona_then_silhouette"
    assert len(evs[0]["per_k"]) == 7      # K=2..8


def test_select_space_without_persona():
    df = _toy_features()
    evs = [evaluate_space(df, "rfm3", ["recency_days", "frequency", "monetary"], None)]
    choice = select_space(evs, has_persona=False)
    assert choice["criterion"] == "silhouette_only"
    assert choice["selected_space"] == "rfm3"
