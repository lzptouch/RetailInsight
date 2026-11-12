"""K-Means 客户聚类（计划 §8.1）。

**三重定 K 逻辑**（对应计划「肘部法 + 轮廓系数 + 业务可解释性」）：
1. 肘部法看 inertia 拐点
2. 轮廓系数看聚类质量（验收线 ≥0.35，最小群占比 ≥2%）
3. 业务可解释性：簇必须能映射到「高价值 / 潜力 / 流失风险 / 沉睡」等语义

**特征空间选择**：先验人群埋点（``_persona``）是唯一可用的业务真值，
因此在多个候选特征空间里按「与先验人群的 ARI（业务有效性）」择优，
轮廓系数作为护栏；无先验可用时退回按轮廓系数择优。
每个空间的评估结果全部落盘，保证可审计。

**神谕上界检验**：直接用生成器埋的真实 8 类人群当标签算轮廓系数，
用于回答「轮廓系数 0.35 到底可达不可达」——若神谕值也低，
说明该特征空间本身无法分离真实结构，再调参也无济于事。
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from retailinsight.features.cluster_features import (
    CLUSTER_FEATURE_SPACES,
    CLUSTER_FEATURES,
    LOG1P_FEATURES,
    build_cluster_features,
)
from retailinsight.io.loader import read_parquet, write_parquet
from retailinsight.settings import INTERIM_DIR, MART_CLUSTER, MART_RFM, REPORTS_DIR, SEED

#: 轮廓系数验收线（计划 §8.1）
SILHOUETTE_TARGET = 0.35
#: 最小簇占比验收线（计划 §8.1）
MIN_CLUSTER_SHARE = 0.02
#: 轮廓系数计算抽样上限（全量 n² 距离矩阵在大样本上不可接受，抽样是标准做法）
SILHOUETTE_SAMPLE = 5000
#: 参与 log1p 压缩的维度
_LOG1P = LOG1P_FEATURES


def _signed_log1p(x: pd.Series | np.ndarray) -> np.ndarray:
    """对称 log1p 压缩：保留负值语义，不在 0 处制造人为尖峰。

    直接 ``clip(lower=0)`` 再 log1p 会把「退货大于消费」的负值会员全部压到 0，
    形成一个远离均值的尖峰（标准化后可达 -9σ），把该维度的尺度彻底拉坏，
    进而压低整体轮廓系数。此处改用 sign(x)·log1p(|x|)。
    """
    v = np.asarray(x, dtype=float)
    return np.sign(v) * np.log1p(np.abs(v))


def build_cluster_matrix(features: pd.DataFrame,
                         cols: list[str] | None = None) -> tuple[np.ndarray, StandardScaler]:
    """log1p 压缩长尾 → StandardScaler 标准化。

    参数
    ----
    features: 特征表
    cols: 参与聚类的列名，默认用计划 §8.1 的 6 维

    返回 (标准化后的 X, scaler)。X 的行顺序与 ``features`` 一致。
    """
    cols = cols or CLUSTER_FEATURES
    X = features[cols].copy()
    for col in cols:
        X[col] = _signed_log1p(X[col]) if col in _LOG1P else X[col].astype(float)
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def choose_k(X: np.ndarray, k_range: range = range(2, 9),
             seed: int = SEED) -> dict[int, dict]:
    """对每个 K 计算 inertia 与 silhouette（独立于选 K 的逻辑，供调用方复用）。

    silhouette 在大样本上为 O(n²)，因此按固定种子抽样 5000 条计算。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    out: dict[int, dict] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        out[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Xs, labels[idx])),
            "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
        }
    return out


def fit_kmeans(X: np.ndarray, k: int, seed: int = SEED) -> tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    return km, labels


def oracle_silhouette(X: np.ndarray, persona: np.ndarray, seed: int = SEED) -> float | None:
    """神谕上界：直接用真实先验人群当标签算轮廓系数。

    若该值也低于验收线，说明「用这些特征聚类」在数学上就到不了 0.35，
    再调 K 或调参也无济于事 —— 这是判定「目标不可达」最硬的证据。
    """
    if persona is None or len(persona) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    try:
        return float(silhouette_score(X[idx], persona[idx]))
    except ValueError:
        return None


def describe_clusters(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """描述每个簇：规模、占比、GMV 贡献、特征分位、可读命名。"""
    df = features.copy()
    df["cluster_id"] = labels
    total_gmv = df["monetary"].sum()

    agg = df.groupby("cluster_id").agg(
        member_cnt=("member_id", "count"),
        gmv=("monetary", "sum"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_basket=("avg_basket", "mean"),
        avg_n_categories=("n_categories", "mean"),
    )
    agg["member_share"] = agg["member_cnt"] / len(df)
    agg["gmv_share"] = agg["gmv"] / total_gmv if total_gmv else 0.0

    # 相对全体会员的分位（0~1），用于业务语义判定
    for col, out_col in [("recency_days", "recency_pct"), ("frequency", "frequency_pct"),
                         ("monetary", "monetary_pct")]:
        ranks = df[col].rank(pct=True)
        agg[out_col] = ranks.groupby(df["cluster_id"]).mean()
    agg["active_pct"] = 1 - agg["recency_pct"]  # recency 越小越活跃

    agg["cluster_name"] = [_name_cluster(r) for _, r in agg.iterrows()]
    # 同名簇加序号后缀，避免 K 较大时出现多个「高价值活跃」无法区分
    seen: dict[str, int] = {}
    names = []
    for n in agg["cluster_name"]:
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}·{seen[n]}")
    agg["cluster_name"] = names
    return agg.reset_index()


def _name_cluster(row: pd.Series) -> str:
    """业务语义命名（计划 §8.1「业务可解释性」一票否决项）。

    阈值以全体会员分位 0.5 为基准线，取 ±0.05~0.15 的容差带，
    避免因分位贴边导致所有簇都落到「一般维持」。
    """
    active = row["active_pct"]
    value = row["monetary_pct"]
    freq = row["frequency_pct"]

    if value >= 0.60 and active >= 0.55:
        return "高价值活跃"
    if value >= 0.60 and active < 0.40:
        return "高价值流失预警"
    if freq >= 0.65 and value < 0.45:
        return "高频低额"
    if active <= 0.30:
        return "沉睡流失"
    if freq <= 0.30 and value <= 0.30:
        return "低频低额"
    if active >= 0.55:
        return "潜力成长"
    return "一般维持"


def validate_against_persona(labels: np.ndarray, member_ids: np.ndarray,
                             persona: pd.Series) -> dict:
    """与生成器埋的 8 类先验人群做列联表 + ARI/NMI 对照。

    这是计划 §4.2 埋点闭环的关键动作：先验人群若与聚类结果显著相关，
    说明聚类确实找回了数据里的真实业务结构。
    """
    df = pd.DataFrame({"member_id": member_ids, "cluster_id": labels})
    df["persona"] = df["member_id"].map(persona)
    ct = pd.crosstab(df["persona"], df["cluster_id"])
    return {
        "ari": float(adjusted_rand_score(df["persona"].astype(str), df["cluster_id"].astype(str))),
        "nmi": float(normalized_mutual_info_score(df["persona"].astype(str),
                                                  df["cluster_id"].astype(str))),
        "crosstab": ct.to_dict(),
        "n_compared": int(len(df)),
    }


def choose_best_k(rows: list[dict], has_persona: bool) -> tuple[dict, str]:
    """在逐 K 评估结果中选 K。

    **护栏**（计划 §8.1 的两条验收线，硬约束）：
    - 轮廓系数 ≥ ``SILHOUETTE_TARGET``（0.35）
    - 最小簇占比 ≥ ``MIN_CLUSTER_SHARE``（2%）

    **择优**：在满足护栏的 K 中，有先验真值时按 ARI（业务有效性）最大、
    轮廓系数次之；无先验真值时按轮廓系数最大。
    若没有任何 K 满足护栏 → 取轮廓系数最高的 K 并说明原因。

    返回 (选中的行, 选择理由)。
    """
    feasible = [r for r in rows
                if r["silhouette"] >= SILHOUETTE_TARGET
                and r["min_cluster_share"] >= MIN_CLUSTER_SHARE]
    if feasible:
        if has_persona:
            best = max(feasible, key=lambda r: (round(r["ari"] or 0.0, 4), r["silhouette"]))
            return best, "max_ari_under_silhouette_and_share_guard"
        best = max(feasible, key=lambda r: r["silhouette"])
        return best, "max_silhouette_under_guard"
    best = max(rows, key=lambda r: r["silhouette"])
    return best, "no_k_satisfies_guard_fallback_max_silhouette"


def evaluate_space(features: pd.DataFrame, space_name: str, cols: list[str],
                    persona: pd.Series | None) -> dict:
    """在给定特征空间上完整跑一遍：逐 K 评估 → 选 K → 描述 → 先验对照。

    **选 K 的准则**：把计划 §8.1 的两条验收线当硬护栏
    （轮廓系数 ≥0.35、最小群占比 ≥2%），在满足护栏的 K 中
    按「与先验人群的 ARI（业务有效性）」择优；轮廓系数作为次级排序。
    无先验真值时退回按轮廓系数择优。
    若没有任何 K 满足护栏 → 取轮廓系数最高的 K，并标记未达标。
    """
    X, _ = build_cluster_matrix(features, cols)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    Xs = X[idx]
    has_persona = persona is not None and len(persona) > 0
    p_arr = features["member_id"].map(persona).values if has_persona else None

    rows: list[dict] = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = km.fit_predict(X)
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Xs, labels[idx])),
            "min_cluster_share": float(sizes.min() / len(labels)),
            "ari": float(adjusted_rand_score(p_arr, labels)) if has_persona else None,
            "labels": labels,
        })

    feasible = [r for r in rows
                if r["silhouette"] >= SILHOUETTE_TARGET
                and r["min_cluster_share"] >= MIN_CLUSTER_SHARE]
    best, reason = choose_best_k(rows, has_persona)
    reached = bool(feasible)

    labels = best["labels"]
    desc = describe_clusters(features, labels)
    persona_check: dict = {"ari": None, "nmi": None, "crosstab": {}, "n_compared": 0}
    if has_persona:
        persona_check = validate_against_persona(labels, features["member_id"].values, persona)

    return {
        "space": space_name,
        "features": cols,
        "k": best["k"],
        "silhouette": best["silhouette"],
        "inertia": best["inertia"],
        "ari": best["ari"],
        "min_cluster_share": best["min_cluster_share"],
        "reached_target": reached,
        "selection_reason": reason,
        "per_k": [{kk: vv for kk, vv in r.items() if kk != "labels"} for r in rows],
        "oracle_silhouette": oracle_silhouette(X, p_arr),
        "labels": labels,
        "desc": desc,
        "persona_check": persona_check,
    }


def select_space(evaluations: list[dict], has_persona: bool) -> dict:
    """在候选特征空间之间择优。

    有先验真值时：按 ARI（能找回多少真实业务结构）优先，轮廓系数次之；
    无先验真值时：退回按轮廓系数择优。
    """
    if not evaluations:
        raise ValueError("没有可评估的特征空间")

    def key(ev: dict) -> tuple:
        if has_persona and ev["ari"] is not None:
            return (round(ev["ari"], 4), ev["silhouette"])
        return (0.0, ev["silhouette"])

    best = max(evaluations, key=key)
    return {
        "selected_space": best["space"],
        "selected_features": best["features"],
        "criterion": "ari_vs_persona_then_silhouette" if has_persona else "silhouette_only",
        "ranking": [{"space": e["space"], "ari": e["ari"], "silhouette": e["silhouette"],
                     "k": e["k"], "reached_target": e["reached_target"]}
                    for e in sorted(evaluations, key=key, reverse=True)],
    }


def run_cluster() -> dict:
    """执行聚类全流程：特征 → 多空间评估 → 选空间 → 描述 → 先验对照 → 落盘。"""
    from retailinsight.viz import charts

    rfm = read_parquet(MART_RFM)
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_parquet(INTERIM_DIR / "transaction_items.parquet")
    products = read_parquet(INTERIM_DIR / "products.parquet")

    feats = build_cluster_features(rfm, txn, items, products)

    # 先验人群（计划 §4.2 埋点闭环的对照基准）
    members = read_parquet(INTERIM_DIR / "members.parquet")
    persona = (members.set_index("member_id")["_persona"]
               if "_persona" in members.columns else pd.Series(dtype=object))

    evaluations = [evaluate_space(feats, name, cols, persona)
                   for name, cols in CLUSTER_FEATURE_SPACES.items()]
    choice = select_space(evaluations, has_persona=bool(len(persona)))
    chosen = next(e for e in evaluations if e["space"] == choice["selected_space"])

    labels = chosen["labels"]
    desc = chosen["desc"]
    out = pd.DataFrame({"member_id": feats["member_id"].values, "cluster_id": labels})
    name_map = dict(zip(desc["cluster_id"], desc["cluster_name"], strict=False))
    out["cluster_name"] = out["cluster_id"].map(name_map)
    write_parquet(out, MART_CLUSTER)

    charts.cluster_elbow({int(r["k"]): {"inertia": r["inertia"], "silhouette": r["silhouette"],
                                       "cluster_sizes": {}}
                          for r in chosen["per_k"]},
                         chosen["silhouette"])
    charts.cluster_radar(desc)

    result = {
        "k": chosen["k"],
        "feature_space": choice,
        "n_members": int(len(feats)),
        "silhouette": chosen["silhouette"],
        "silhouette_target": SILHOUETTE_TARGET,
        "reached_target": chosen["reached_target"],
        "min_cluster_share": chosen["min_cluster_share"],
        "ari_vs_persona": chosen["ari"],
        "oracle_silhouette": chosen["oracle_silhouette"],
        "space_evaluations": [{k: v for k, v in ev.items()
                               if k not in ("labels", "desc", "persona_check")}
                              for ev in evaluations],
        "clusters": desc.to_dict("records"),
        "persona_validation": chosen["persona_check"],
    }
    with open(REPORTS_DIR / "metrics_cluster.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"[cluster] space={choice['selected_space']} k={chosen['k']} "
          f"silhouette={chosen['silhouette']:.4f} "
          f"reached_target={chosen['reached_target']} "
          f"ARI={chosen['ari']}")
    print(f"[cluster] 神谕上界 silhouette={chosen['oracle_silhouette']} "
          f"（用真实先验人群当标签）")
    print(desc[["cluster_id", "cluster_name", "member_cnt", "member_share", "gmv_share"]])
    return result
