"""聚类选型对比单测：三模型同空间对比、退化情形、规则化选型建议。

`compare_models` 的产出会被写进 `reports/metrics_cluster.json` 并驱动
《决策建议报告》里「为什么选 K-Means」的论证，所以这里钉的是
「对比口径一致 + 退化输入不炸 + 建议由规则给出」三件事。
"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.models.cluster_compare import (
    _silhouette,
    compare_models,
    model_comparison_report,
)


def _blobs(n: int = 120, seed: int = 7) -> np.ndarray:
    """三个明显分离的球状簇 —— 任何合理聚类都该给出正轮廓系数。"""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(0.0, 0.5, (n, 3)),
        rng.normal(6.0, 0.5, (n, 3)),
        rng.normal(-6.0, 0.5, (n, 3)),
    ])


# ------------------------------------------------------------ 轮廓系数退化

def test_silhouette_returns_minus_one_when_not_evaluable():
    X = np.zeros((10, 2))
    assert _silhouette(X, np.zeros(10, dtype=int)) == -1.0        # 单簇
    assert _silhouette(X, np.arange(10)) == -1.0                 # 每点自成一簇


def test_silhouette_is_positive_for_separated_clusters():
    X = _blobs()
    labels = np.repeat([0, 1, 2], 120)
    assert _silhouette(X, labels) > 0.5


# ------------------------------------------------------------ 三模型对比

def test_compare_models_returns_three_models_with_consistent_basis():
    X = _blobs()
    res = compare_models(X, k=3)

    assert res["k"] == 3
    assert res["n_samples"] == len(X)
    assert [r["model"] for r in res["results"]] == ["KMeans", "GMM", "DBSCAN"]
    assert res["note"]

    km = res["results"][0]
    assert km["n_clusters"] == 3
    assert km["silhouette"] > 0.5, "分离良好的球状簇上 K-Means 轮廓系数应明显为正"
    assert km["inertia"] > 0
    assert km["min_cluster_share"] == pytest.approx(1 / 3, rel=0.2)
    assert km["batch_scorable"] is True
    assert km["ari_vs_persona"] is None          # 未给先验人群

    db = res["results"][2]
    assert db["batch_scorable"] is False
    assert 0.0 <= db["noise_share"] <= 1.0
    assert db["dbscan_eps"] > 0
    assert db["inertia"] is None

    assert res["recommendation"] in {"KMeans", "GMM", None}
    assert any("批量打分" in r for r in res["reasons"])


def test_compare_models_reports_ari_against_persona():
    X = _blobs()
    persona = np.repeat([0, 1, 2], 120)          # 与真实簇结构一致
    res = compare_models(X, k=3, persona=persona)

    km = next(r for r in res["results"] if r["model"] == "KMeans")
    assert km["ari_vs_persona"] == pytest.approx(1.0, abs=0.05)
    assert res["best_silhouette_model"] in {"KMeans", "GMM", "DBSCAN"}


def test_compare_models_with_degenerate_dbscan_eps_does_not_raise():
    """eps 小到全部判噪声时，DBSCAN 各项指标必须优雅退化而不是崩。"""
    X = _blobs()
    res = compare_models(X, k=3, dbscan_eps=1e-6)
    db = next(r for r in res["results"] if r["model"] == "DBSCAN")
    assert db["noise_share"] == pytest.approx(1.0)
    assert db["n_clusters"] == 0
    assert db["silhouette"] == -1.0
    assert db["ari_vs_persona"] is None
    # 全噪声时不能用 DBSCAN 做推荐
    assert res["recommendation"] != "DBSCAN"


# ------------------------------------------------------------ 特征宽表入口

def test_model_comparison_report_maps_persona_from_features():
    rng = np.random.default_rng(3)
    n = 90
    features = pd.DataFrame({
        "member_id": [f"m{i}" for i in range(n * 2)],
        "recency_days": np.concatenate([rng.normal(10, 2, n), rng.normal(200, 5, n)]),
        "frequency": np.concatenate([rng.normal(20, 2, n), rng.normal(2, 1, n)]),
        "monetary": np.concatenate([rng.normal(5000, 200, n), rng.normal(100, 30, n)]),
    })
    persona = pd.Series({f"m{i}": "high" if i < n else "low" for i in range(n * 2)})

    cols = ["recency_days", "frequency", "monetary"]
    res = model_comparison_report(features, cols, k=2, persona=persona)

    assert res["n_samples"] == n * 2
    km = next(r for r in res["results"] if r["model"] == "KMeans")
    assert km["ari_vs_persona"] == pytest.approx(1.0, abs=0.1)
