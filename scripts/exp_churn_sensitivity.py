"""流失模型 AUC 增益的敏感性实验：cohort 切分方式 × 特征数量。

用途
----
支撑《项目编写计划.md》§0.5 与《决策建议报告》中的结论：
「加入独立验证集后 AUC 增益由 +0.031 降至 +0.027，这不是调参问题，
  而是时间外推的真实难度」。

做法
----
复用 `retailinsight.models.churn` 的样本构造与评估函数，
按不同 (训练期数, 是否有验证集, 特征数) 组合重跑，
只比较 **测试集 AUC 相对 LR 基线的增益**。

注意：为控制耗时，本实验使用默认 GBDT 超参（不跑超参搜索），
      因此绝对值略低于正式产出，**关注的是相对趋势**。

用法
----
    python scripts/exp_churn_sensitivity.py
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from retailinsight.io.loader import read_parquet
from retailinsight.models.churn import (
    _gbdt,
    build_cohort_samples,
    evaluate,
)
from retailinsight.settings import INTERIM_DIR, REPORTS_DIR, SEED

ALL_CUTOFFS = ["2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01"]

# 早期口径的 12 维特征（体量 + 基础行为），用于对照
FEATURES_12 = [
    "recency_days", "frequency", "monetary", "avg_basket", "tenure_days",
    "total_amount", "avg_txn_amount", "n_txn", "n_categories", "items_per_txn",
    "discount_ratio", "weekend_ratio",
]
FEATURES_31 = [
    "recency_days", "frequency", "monetary", "avg_basket", "tenure_days",
    "total_amount", "avg_txn_amount", "n_txn", "gap_mean", "gap_std",
    "interval_cv", "trend_90d", "freq_slope", "amt_slope", "freq_30d",
    "freq_90d", "amt_30d", "amt_90d", "days_since_2nd_last", "n_categories",
    "items_per_txn", "category_entropy", "brand_entropy", "top1_category_share",
    "refund_cnt", "refund_amt_ratio", "discount_ratio", "coupon_used_rate",
    "store_cnt", "weekend_ratio", "seasonality_strength",
]


def build_cohorts(txn, items, products, members) -> dict[str, pd.DataFrame]:
    return {c: build_cohort_samples(txn, items, products, members, pd.Timestamp(c))
            for c in ALL_CUTOFFS}


def run_one(cohorts: dict, cutoffs: list[str], n_train: int, n_valid: int,
            features: list[str]) -> dict:
    data = pd.concat([cohorts[c] for c in cutoffs], ignore_index=True)
    train = data[data["cohort"].isin(cutoffs[:n_train])]
    # valid 段不参与本实验的训练/评估（该实验只比较切分方式对增益的影响）
    test = data[data["cohort"].isin(cutoffs[n_train + n_valid:])]

    Xtr, ytr = train[features].fillna(0), train["churn"].to_numpy()
    Xte, yte = test[features].fillna(0), test["churn"].to_numpy()

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(sc.transform(Xtr), ytr)
    m_lr = evaluate(yte, lr.predict_proba(sc.transform(Xte))[:, 1])

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    gb = _gbdt(pos_weight=pos_weight)
    gb.fit(Xtr, ytr)
    m_gb = evaluate(yte, gb.predict_proba(Xte)[:, 1])

    return {
        "n_train_cohorts": n_train,
        "n_valid_cohorts": n_valid,
        "n_features": len(features),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "auc_lr": round(m_lr["auc"], 4),
        "auc_gbdt": round(m_gb["auc"], 4),
        "auc_gain": round(m_gb["auc"] - m_lr["auc"], 4),
        "ks_gbdt": round(m_gb["ks"], 4),
        "lift_top20": round(m_gb["lift_top20"], 3),
    }


def main() -> None:
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_parquet(INTERIM_DIR / "transaction_items.parquet")
    products = read_parquet(INTERIM_DIR / "products.parquet")
    members = read_parquet(INTERIM_DIR / "members.parquet")

    cohorts = build_cohorts(txn, items, products, members)

    configs = [
        # 早期口径：4 期，前 2 期训练 + 末期测试（**无独立验证集**）
        ("无验证集 · 12 维", ALL_CUTOFFS[:4], 3, 0, FEATURES_12),
        ("无验证集 · 31 维", ALL_CUTOFFS[:4], 3, 0, FEATURES_31),
        # 加验证集：前 2 期训练 + 1 期验证 + 末期测试
        ("有验证集(2+1) · 12 维", ALL_CUTOFFS[:4], 2, 1, FEATURES_12),
        ("有验证集(2+1) · 31 维", ALL_CUTOFFS[:4], 2, 1, FEATURES_31),
        # 当前口径：5 期，前 3 期训练 + 1 期验证 + 末期测试
        ("有验证集(3+1) · 12 维", ALL_CUTOFFS, 3, 1, FEATURES_12),
        ("有验证集(3+1) · 31 维（当前口径）", ALL_CUTOFFS, 3, 1, FEATURES_31),
    ]

    rows = []
    for label, cutoffs, n_train, n_valid, feats in configs:
        res = run_one(cohorts, cutoffs, n_train, n_valid, feats)
        res["config"] = label
        rows.append(res)
        print(f"[exp] {label:32s} AUC LR={res['auc_lr']:.4f} GBDT={res['auc_gbdt']:.4f} "
              f"gain={res['auc_gain']:+.4f} KS={res['ks_gbdt']:.4f}")

    df = pd.DataFrame(rows)[["config", "n_train_cohorts", "n_valid_cohorts", "n_features",
                             "auc_lr", "auc_gbdt", "auc_gain", "ks_gbdt", "lift_top20"]]
    out = REPORTS_DIR / "exp_churn_sensitivity.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "exp_churn_sensitivity.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp] 结果已写入 {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
