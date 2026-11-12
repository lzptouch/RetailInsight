"""会员流失预测（P0 主打模型）。

方法论（对应计划 §8.2，防泄漏是核心）：
- 观察窗 270 天 + 标签窗 90 天，滚动 4 期 cohort
- 特征只用观察窗内数据，标签只看标签窗，物理隔离
- **训练/验证/测试三向时间切分**：前 2 期训练、第 3 期验证、末期测试，禁止随机切分
- LR 可解释基线 + GBDT 主模型 + RandomForest 对照
  （本机无 libomp 时自动从 LightGBM 降级 sklearn HistGB）
- 类别不平衡：GBDT 显式加权（LightGBM 用 scale_pos_weight，HistGB 用 class_weight）
- 阈值调优：在验证集上按**业务成本最优**选风险分层切点，
  成本系数与《决策建议报告》S2 同口径（漏判 1 人 = 误触 8 人）
- 评估：AUC / KS / Top20% 名单 Lift（业务导向）
- 可解释性：SHAP 全局重要性 + 单会员 Top 归因（无 shap 时降级 permutation importance 并告警）
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from retailinsight.features.churn_features import CHURN_FEATURES, build_features, build_labels
from retailinsight.settings import (
    COHORT_CUTOFFS,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    ITEMS_DIR,
    MART_CHURN,
    REPORTS_DIR,
    SEED,
    TRAIN_COHORTS,
    VALID_COHORTS,
)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_LGBM = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

#: 建模特征列（28 维，单一来源见 features/churn_features.py）
FEATURES = CHURN_FEATURES

#: SHAP 计算抽样上限（计划 §12 风险 6：控制解释性计算的耗时）
SHAP_SAMPLE = 5000


def _gbdt(pos_weight: float = 1.0, **params):
    """GBDT 主模型。pos_weight 用于处理类别不平衡。"""
    if _HAS_LGBM:
        p = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
             "scale_pos_weight": pos_weight, "random_state": SEED, "verbose": -1}
        p.update(params)
        return lgb.LGBMClassifier(**p)
    p = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
         "class_weight": {0: 1.0, 1: pos_weight}, "random_state": SEED}
    p.update({k: v for k, v in params.items()
              if k in ("max_iter", "learning_rate", "max_leaf_nodes")})
    return HistGradientBoostingClassifier(**p)


#: 验证集超参搜索网格（计划 §8.2「验证集」的真正用途）
GBDT_GRID = [
    {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 60},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 60},
    {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30},
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 30},
    {"n_estimators": 200, "learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 30},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 60},
]


def tune_gbdt_params(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_valid: pd.DataFrame, y_valid: np.ndarray,
                     pos_weight: float) -> tuple[dict, list[dict]]:
    """在验证集上搜索 LightGBM 超参，按验证集 AUC 选最优。

    返回 (最优超参, 全部候选的评估记录)。
    测试集全程不参与选择，避免用测试集调参造成乐观偏差。
    """
    records = []
    best = (None, -np.inf)
    for params in GBDT_GRID:
        model = _gbdt(pos_weight=pos_weight, **params)
        model.fit(X_train, y_train)
        auc = float(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))
        records.append({**params, "valid_auc": round(auc, 4)})
        if auc > best[1]:
            best = (params, auc)
    records.sort(key=lambda r: -r["valid_auc"])
    return best[0], records


def _random_forest(pos_weight: float = 1.0) -> RandomForestClassifier:
    """RandomForest 非线性对照（计划 §8.2）。"""
    return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=50,
                                  class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def build_cohort_samples(txn: pd.DataFrame, items: pd.DataFrame,
                         products: pd.DataFrame, members: pd.DataFrame,
                         cutoff: pd.Timestamp) -> pd.DataFrame:
    """构造单期样本：观察窗内活跃会员，标签 = 标签窗内 0 交易。"""
    feat = build_features(txn, items, products, members, cutoff)
    active_in_label = build_labels(txn, cutoff)
    feat = feat.merge(active_in_label.assign(churn=0), on="member_id", how="left")
    feat["churn"] = feat["churn"].fillna(1).astype(int)
    feat["cohort"] = str(cutoff.date())
    return feat


def _feature_importance(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """HistGB 无 feature_importances_，统一用 permutation importance（测试集）。"""
    if hasattr(model, "feature_importances_"):
        return (pd.Series(model.feature_importances_, index=FEATURES)
                .sort_values(ascending=False).round(4).to_dict())
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X_test, y_test, n_repeats=5,
                               random_state=SEED, scoring="roc_auc")
    return (pd.Series(r.importances_mean, index=FEATURES)
            .sort_values(ascending=False).round(4).to_dict())


def shap_explain(model, X: pd.DataFrame, seed: int = SEED,
                sample: bool = True) -> tuple[dict, pd.DataFrame | None]:
    """SHAP 全局重要性 + 单会员归因。

    参数
    ----
    sample:
        True 时按固定种子抽样 ``SHAP_SAMPLE`` 条计算（控制耗时，用于**全局重要性**）；
        False 时对全量计算（用于**逐会员归因**，否则抽样外的会员拿不到归因）。

    返回 (全局重要性 dict, 单会员 shap 值 DataFrame 或 None)。
    无 shap 依赖或模型不支持 TreeExplainer 时降级为 permutation importance，
    并在返回值中标注 ``method``，不静默失败。
    """
    if not _HAS_SHAP:
        print("[churn] 警告：未安装 shap，降级为 permutation importance")
        return {"method": "permutation_importance"}, None
    try:
        if sample and len(X) > SHAP_SAMPLE:
            X = X.sample(SHAP_SAMPLE, random_state=seed)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # 二分类在不同 shap 版本下有三种返回形态：
        #   1) list[ndarray]（旧版，每个输出一项）
        #   2) ndarray 形状 (n, n_features)（单输出）
        #   3) ndarray 形状 (n, n_features, 2)（新版，每类一项）→ 取正类
        if isinstance(sv, list):
            sv = sv[-1]
        elif getattr(sv, "ndim", 1) == 3:
            sv = sv[:, :, -1]
        sv = pd.DataFrame(sv, columns=list(X.columns), index=X.index)
        global_imp = (sv.abs().mean().sort_values(ascending=False).round(4).to_dict())
        global_imp["method"] = "shap.TreeExplainer"
        return global_imp, sv
    except Exception as exc:  # noqa: BLE001
        print(f"[churn] 警告：SHAP 计算失败（{exc}），降级为 permutation importance")
        return {"method": "permutation_importance"}, None


def tune_thresholds(y_valid: np.ndarray, p_valid: np.ndarray,
                    cost_fn: float = COST_FALSE_NEGATIVE,
                    cost_fp: float = COST_FALSE_POSITIVE) -> dict:
    """在验证集上按业务成本最优选三个风险切点。

    对切点 t，名单 = {p >= t}，总成本 =
        cost_fn × (漏掉的流失会员数) + cost_fp × (误触的正常会员数)

    漏判成本越高 → 名单应越大 → 切点越低。因此：

    - ``action``：用完整成本比 8:1 求得，是**建议的触达名单**切点（最激进）
    - ``mid``   ：用 2:1 求得，区分「中危」与「低危」
    - ``high``  ：用 1:1 求得，区分「高危」与「中危」

    三个切点保证 action ≤ mid ≤ high，风险分层单调可解释。
    成本比 8:1 与《决策建议报告》S2「流失 1 人 ≈ 挽留 8 名流失风险客户」一致。
    """
    grid = np.arange(0.05, 0.96, 0.01)
    n_pos = max(int((y_valid == 1).sum()), 1)

    def _cost(t: float, fn_cost: float) -> float:
        flagged = p_valid >= t
        missed = n_pos - int((y_valid[flagged] == 1).sum())
        false_alarm = int((flagged & (y_valid == 0)).sum())
        return fn_cost * missed + cost_fp * false_alarm

    t_action = float(min(grid, key=lambda t: _cost(t, cost_fn)))
    t_mid = float(min(grid, key=lambda t: _cost(t, max(cost_fn / 4.0, 1.0))))
    t_high = float(min(grid, key=lambda t: _cost(t, 1.0)))

    # 保证单调递增，避免出现非法分箱
    t_mid = min(max(t_mid, t_action), t_high)
    t_high = max(t_high, t_mid)

    return {
        "action": t_action,
        "mid": t_mid,
        "high": t_high,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "n_valid": int(len(y_valid)),
        "base_rate": round(float(y_valid.mean()), 4),
    }


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, top_pct: float = 0.2) -> dict:
    """评估指标：AUC / 标准 KS / Top20% 名单 Lift。

    KS 采用标准定义 ``max|TPR - FPR|``（按预测概率降序遍历切点），
    与业界口径一致，可直接与「KS ≥ 0.30」的验收线比较。
    """
    auc = roc_auc_score(y_true, y_prob)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = max(int(y_sorted.sum()), 1)
    n_neg = max(len(y_sorted) - int(y_sorted.sum()), 1)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    ks = float(np.max(np.abs(tp / n_pos - fp / n_neg)))
    k = max(int(len(y_true) * top_pct), 1)
    precision_top = float(y_sorted[:k].mean())
    base_rate = float(y_true.mean())
    return {"auc": round(float(auc), 4), "ks": round(ks, 4),
            "lift_top20": round(precision_top / max(base_rate, 1e-9), 2),
            "capture_top20": round(float(y_sorted[:k].sum() / max(y_sorted.sum(), 1)), 4),
            "base_rate": round(base_rate, 4)}


def _risk_level(prob: pd.Series, thresholds: dict) -> pd.Series:
    """按调优后的切点做风险分层（替代硬编码 0.4/0.7）。"""
    return pd.cut(prob, [-0.001, thresholds["mid"], thresholds["high"], 1.001],
                  labels=["低", "中", "高"])


def run_churn() -> dict:
    from retailinsight.io.loader import read_items, read_parquet, write_parquet
    from retailinsight.settings import INTERIM_DIR

    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_items(ITEMS_DIR)
    products = read_parquet(INTERIM_DIR / "products.parquet")
    members = read_parquet(INTERIM_DIR / "members.parquet")

    cohorts = []
    for c in COHORT_CUTOFFS:
        df = build_cohort_samples(txn, items, products, members, pd.Timestamp(c))
        cohorts.append(df)
        print(f"[churn] cohort {c}: n={len(df)} churn_rate={df['churn'].mean():.3f}")
    data = pd.concat(cohorts, ignore_index=True)

    # ---- 三向时间切分（严格按时间顺序，禁止随机切分）----
    train_cohorts = COHORT_CUTOFFS[:TRAIN_COHORTS]
    valid_cohorts = COHORT_CUTOFFS[TRAIN_COHORTS:TRAIN_COHORTS + VALID_COHORTS]
    test_cohorts = COHORT_CUTOFFS[TRAIN_COHORTS + VALID_COHORTS:]
    train = data[data["cohort"].isin(train_cohorts)]
    valid = data[data["cohort"].isin(valid_cohorts)]
    test = data[data["cohort"].isin(test_cohorts)]
    assert train["cohort"].max() < valid["cohort"].min() < test["cohort"].min(), \
        "时间切分必须严格按时间先后"

    X_train, y_train = train[FEATURES].fillna(0), train["churn"].to_numpy()
    X_valid, y_valid = valid[FEATURES].fillna(0), valid["churn"].to_numpy()
    X_test, y_test = test[FEATURES].fillna(0), test["churn"].to_numpy()

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # LR 可解释基线
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(scaler.transform(X_train), y_train)
    p_lr = lr.predict_proba(scaler.transform(X_test))[:, 1]
    m_lr = evaluate(y_test, p_lr)

    # GBDT 主模型（LightGBM 不可用时自动降级 sklearn HistGB）
    # 超参在验证集上选择，测试集不参与
    best_params, tuning_log = tune_gbdt_params(X_train, y_train, X_valid, y_valid, pos_weight)
    gbdt = _gbdt(pos_weight=pos_weight, **best_params)
    gbdt.fit(X_train, y_train)
    p_gb = gbdt.predict_proba(X_test)[:, 1]
    m_gb = evaluate(y_test, p_gb)

    # RandomForest 对照
    rf = _random_forest()
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    m_rf = evaluate(y_test, p_rf)

    # 阈值调优（只用验证集）
    p_valid = gbdt.predict_proba(X_valid)[:, 1]
    thresholds = tune_thresholds(y_valid, p_valid)

    importance, _ = shap_explain(gbdt, X_valid)
    if "method" not in importance:
        importance = _feature_importance(gbdt, X_test, y_test)

    # 对最新窗口全量打分，产出 mart_churn_score
    latest = build_cohort_samples(txn, items, products, members, pd.Timestamp(COHORT_CUTOFFS[-1]))
    latest["churn_prob"] = gbdt.predict_proba(latest[FEATURES].fillna(0))[:, 1].round(4)
    latest["risk_level"] = _risk_level(latest["churn_prob"], thresholds)

    # 单会员归因：Top 贡献特征。
    # 注意必须在**末期 cohort 自身**上算 SHAP —— 验证集的 shap 值与末期会员对不上，
    # 合并后会得到全空列。
    out_cols = ["member_id", "churn_prob", "risk_level", "churn"] + FEATURES
    # 用 member_id 作为索引，SHAP 的行才能和会员一一对应（位置索引会导致错位）
    latest_feat = latest.set_index("member_id")[FEATURES].fillna(0)
    _, shap_latest = shap_explain(gbdt, latest_feat, sample=False)
    if shap_latest is not None:
        top = (shap_latest.abs().idxmax(axis=1).rename("top_shap_feature")
               .reset_index())   # index 即 member_id
        latest = latest.merge(top, on="member_id", how="left")
        out_cols.append("top_shap_feature")
    write_parquet(latest[out_cols], MART_CHURN)

    result = {
        "backend": "lightgbm" if _HAS_LGBM else "sklearn_histgb",
        "lr_baseline": m_lr,
        "gbdt": m_gb,
        "random_forest": m_rf,
        "auc_gain_vs_lr": round(m_gb["auc"] - m_lr["auc"], 4),
        "auc_gain_target": 0.03,
        "auc_gain_reached": bool(m_gb["auc"] - m_lr["auc"] >= 0.03),
        "feature_importance": importance,
        "thresholds": thresholds,
        "gbdt_params": best_params,
        "gbdt_tuning_log": tuning_log,
        "n_features": len(FEATURES),
        "pos_weight": round(pos_weight, 3),
        "split": {"train": train_cohorts, "valid": valid_cohorts, "test": test_cohorts},
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "shap_available": _HAS_SHAP,
    }
    (REPORTS_DIR / "metrics_churn.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[churn] backend={result['backend']}")
    print(f"[churn] LR 基线: {m_lr}")
    print(f"[churn] GBDT 主模型: {m_gb}")
    print(f"[churn] RandomForest 对照: {m_rf}")
    print(f"[churn] AUC 增益 vs LR = {result['auc_gain_vs_lr']:+.4f} "
          f"(验收线 +0.03，{'达标' if result['auc_gain_reached'] else '未达标'})")
    print(f"[churn] 阈值（验证集成本最优）: {thresholds}")
    return result


if __name__ == "__main__":
    run_churn()
