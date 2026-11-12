"""聚类模型对比：K-Means vs GMM vs DBSCAN（计划 §8.1 / R-7）。

目的不是「换个模型」，而是**用同一份数据回答：为什么最后选 K-Means**。

对比维度
--------
1. **内部指标**：轮廓系数（越高越好）、inertia（越低越好）
2. **业务护栏**：最小群占比 ≥2%（防止碎群）、簇数是否可解释
3. **对长尾的鲁棒性**：DBSCAN 把噪声点单独标出（label=-1），
   在「大部分正常 + 少量极端」的真实会员分布里更有优势；
   K-Means 则会把极端值拉成自己的簇或把簇拉歪
4. **与先验人群的 ARI**：有真值时看业务有效性
5. **可批量打分**：新会员进来时能否稳定分配（K-Means/GMM 可以，DBSCAN 需重训）

结论（实测后写死在此函数返回的 ``recommendation`` 里，不靠人工挑选）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from retailinsight.models.cluster import MIN_CLUSTER_SHARE, SEED, SILHOUETTE_SAMPLE, SILHOUETTE_TARGET


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """轮廓系数；簇数 <2 或有单一簇时返回 -1（不可评估）。"""
    uniq = set(labels.tolist())
    if len(uniq) < 2:
        return -1.0
    if len(uniq) == len(labels):     # 每个样本自成一簇，退化情形
        return -1.0
    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return -1.0


def compare_models(X: np.ndarray, k: int, persona: np.ndarray | None = None,
                   dbscan_eps: float | None = None,
                   dbscan_min_samples: int = 50) -> dict:
    """在同一特征矩阵上跑三种聚类并给出对比与选型建议。"""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    Xs = X[idx]

    results: list[dict] = []

    # ---- K-Means ----
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km_labels = km.fit_predict(X)
    km_sizes = pd.Series(km_labels).value_counts()
    results.append({
        "model": "KMeans",
        "silhouette": _silhouette(Xs, km_labels[idx]),
        "inertia": float(km.inertia_),
        "n_clusters": int(len(km_sizes)),
        "min_cluster_share": float(km_sizes.min() / len(km_labels)),
        "noise_share": 0.0,
        "ari_vs_persona": (float(adjusted_rand_score(persona, km_labels))
                           if persona is not None else None),
        "batch_scorable": True,
        "notes": "球形簇假设；可批量打分；异常值会拉歪簇中心",
    })

    # ---- GMM ----
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=SEED, n_init=5)
        gm_labels = gm.fit_predict(X)
        gm_sizes = pd.Series(gm_labels).value_counts()
        results.append({
            "model": "GMM",
            "silhouette": _silhouette(Xs, gm_labels[idx]),
            "inertia": float(-gm.score(X) * len(X)),
            "n_clusters": int(len(gm_sizes)),
            "min_cluster_share": float(gm_sizes.min() / len(gm_labels)),
            "noise_share": 0.0,
            "ari_vs_persona": (float(adjusted_rand_score(persona, gm_labels))
                               if persona is not None else None),
            "batch_scorable": True,
            "notes": "允许椭球簇与软分配；对协方差估计敏感，高维下易退化",
        })
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "GMM", "error": str(exc)})

    # ---- DBSCAN ----
    # eps 不能只取 k-距离中位数：那会让 98% 的点被判为噪声，等于用稻草人做对比。
    # 这里在 k-距离分布的若干分位数上各跑一次，挑**噪声占比落在合理区间**
    # （2%~30%）且簇数 ≥2 的那个 —— 对比才公平。
    if dbscan_eps is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
        dists, _ = nn.kneighbors(X)
        kdist = dists[:, -1]
        best = None
        for q in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
            eps = float(np.quantile(kdist, q))
            if eps <= 0:
                continue
            lbl = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(X)
            noise = float((lbl == -1).mean())
            n_real = len(set(lbl[lbl >= 0].tolist()))
            if 0.02 <= noise <= 0.30 and n_real >= 2:
                best = (eps, lbl, noise)
                break
        if best is None:
            eps = float(np.median(kdist)) or 0.5
            lbl = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(X)
            best = (eps, lbl, float((lbl == -1).mean()))
        dbscan_eps, db_labels, _noise = best
    else:
        db_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X)
    real = db_labels[db_labels >= 0]
    noise_share = float((db_labels == -1).mean()) if len(db_labels) else 0.0
    db_sizes = pd.Series(real).value_counts() if len(real) else pd.Series(dtype=int)
    results.append({
        "model": "DBSCAN",
        "silhouette": (_silhouette(Xs, db_labels[idx]) if len(set(real.tolist())) >= 2
                       else -1.0),
        "inertia": None,
        "n_clusters": int(len(db_sizes)),
        "min_cluster_share": (float(db_sizes.min() / len(real))
                              if len(db_sizes) and len(real) else 0.0),
        "noise_share": noise_share,
        "ari_vs_persona": (float(adjusted_rand_score(persona, db_labels))
                           if persona is not None and len(set(real.tolist())) >= 2
                           else None),
        "batch_scorable": False,
        "notes": "密度簇；自动识别噪声点（不强制每个点归属）；但新样本需重训",
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
    })

    # ---- 选型建议（规则化，不靠人工挑选）----
    valid = [r for r in results if "silhouette" in r and r.get("silhouette", -1) > -1]
    best_sil = max(valid, key=lambda r: r["silhouette"])["model"] if valid else None
    km = next(r for r in results if r["model"] == "KMeans")
    db = next(r for r in results if r["model"] == "DBSCAN")

    reasons = []
    if km["silhouette"] >= SILHOUETTE_TARGET:
        reasons.append(f"K-Means 轮廓系数 {km['silhouette']:.3f} 已达验收线 {SILHOUETTE_TARGET}")
    else:
        reasons.append(f"K-Means 轮廓系数 {km['silhouette']:.3f} 未达 {SILHOUETTE_TARGET}")
    if km["min_cluster_share"] >= MIN_CLUSTER_SHARE:
        reasons.append(f"最小群占比 {km['min_cluster_share']:.1%} ≥ {MIN_CLUSTER_SHARE:.0%}")
    if db.get("noise_share", 0) > 0.05:
        reasons.append(f"DBSCAN 标出 {db['noise_share']:.1%} 噪声点（长尾会员被单列）")
    reasons.append("K-Means 可对新增会员批量打分，DBSCAN 需重训")

    recommendation = "KMeans" if (km.get("silhouette", -1) >= db.get("silhouette", -1)
                                   and km["min_cluster_share"] >= MIN_CLUSTER_SHARE) else best_sil

    return {
        "k": k,
        "n_samples": int(len(X)),
        "results": results,
        "best_silhouette_model": best_sil,
        "recommendation": recommendation,
        "reasons": reasons,
        "note": ("三种模型在同一特征空间上比较；结论由规则化准则给出，"
                 "不依赖人工挑选"),
    }


def model_comparison_report(features: pd.DataFrame, cols: list[str], k: int,
                             persona: pd.Series | None = None) -> dict:
    """从特征宽表出发做模型对比（供 metrics_cluster.json 记录）。"""
    from retailinsight.models.cluster import build_cluster_matrix
    X, _ = build_cluster_matrix(features, cols)
    p_arr = features["member_id"].map(persona).values if persona is not None else None
    return compare_models(X, k, persona=p_arr)
