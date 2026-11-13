"""聚类特征矩阵（计划 §8.1：log1p → StandardScaler）。

与画像 20 标签是两条独立的特征线：
- 画像标签（``features/tags.py``）面向业务解释与报告
- 聚类特征（本模块）面向无监督分组

**候选特征空间**（实测结论见 ``reports/metrics_cluster.json``）：

======================  ==========================================  ===============
空间名                  维度                                          用途
======================  ==========================================  ===============
``base6``              计划 §8.1 规定的 6 维（消费体量 + 广度）        基线，合规交付
``rfm3``               recency / frequency / monetary              只含有区分度的核心维度
``rfm3_behavior``      rfm3 + 周末占比/折扣敏感度/季节性强弱          追加行为维度，尝试找回行为型人群
======================  ==========================================  ===============

为什么需要多个空间：生成器埋的 8 类先验人群里，「促销敏感 / 周末囤货 / 季节性」
三类在消费体量上几乎重合，而 ``n_categories`` / ``items_per_txn`` / ``avg_basket``
三维在人群之间几乎没有差异 —— 放进聚类空间只会稀释距离度量、压低轮廓系数。
"""
import pandas as pd

from retailinsight.features.behavior import behavior_features

#: 计划 §8.1 规定的 6 维聚类特征
CLUSTER_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket",
    "n_categories",
    "items_per_txn",
]

#: 行为维度特征（来自 features/behavior.py，画像层共用）
BEHAVIOR_FEATURES = [
    "weekend_ratio",
    "discount_sensitivity",
    "seasonality_strength",
]

#: P1 扩展标签中**对人群有区分度**的若干维（来自 features/p1_tags.py）
#:
#: 挑选依据（对应生成器埋的 8 类先验人群）：
#: - promo_response_rate / coupon_used_rate -> 区分「促销敏感」（折扣概率 0.55）
#: - weekend_ratio -> 区分「周末囤货」（周末权重 2.5）
#: - seasonality_strength -> 区分「季节性」
#: - store_concentration -> 区分「高频低额」（跨店概率 0.30）
#: - channel_preference -> 区分「周末囤货」（单次件数 14）
#: - repurchase_cycle_days -> 区分「高频低额」（年频 16 次）
#: - cross_category_ratio -> 搭配型 vs 刚需型
#: - silence_ratio -> 区分「沉默流失」
P1_CLUSTER_FEATURES = [
    "promo_response_rate",
    "coupon_used_rate",
    "store_concentration",
    "channel_preference",
    "repurchase_cycle_days",
    "cross_category_ratio",
    "silence_ratio",
]

#: 全部可用特征
ALL_FEATURES = CLUSTER_FEATURES + BEHAVIOR_FEATURES + P1_CLUSTER_FEATURES

#: 候选特征空间（按顺序评估）
CLUSTER_FEATURE_SPACES: dict[str, list[str]] = {
    "base6": CLUSTER_FEATURES,
    "rfm3": ["recency_days", "frequency", "monetary"],
    "rfm3_behavior": ["recency_days", "frequency", "monetary"] + BEHAVIOR_FEATURES,
    # v3.2 新增：在 rfm3_behavior 之上再叠 P1 行为时序维度，
    # 目标是找回「促销敏感 / 周末囤货 / 季节性」这三类在体量上重合的人群
    "rfm3_behavior_p1": (["recency_days", "frequency", "monetary"]
                         + BEHAVIOR_FEATURES + P1_CLUSTER_FEATURES),
}

#: 需要做 log1p 压缩的维度（recency_days 为天数，接近均匀分布，不做压缩）
LOG1P_FEATURES = {
    "frequency", "monetary", "avg_basket", "items_per_txn", "n_categories",
    "discount_sensitivity",
}


def build_cluster_features(rfm: pd.DataFrame, txn: pd.DataFrame, items: pd.DataFrame,
                           products: pd.DataFrame) -> pd.DataFrame:
    """构造聚类特征矩阵（6 维基线 + 3 维行为特征）。

    参数
    ----
    rfm:  RFM 分群结果，需含 member_id / recency_days / frequency / monetary
    txn:  交易明细（清洗后），需含 txn_id / member_id
    items: 交易行明细，需含 txn_id / sku / qty
    products: 商品维度，需含 sku / category

    返回
    ----
    member_id + ALL_FEATURES 全部列（各候选空间按列名取用）。
    """
    # 必须以 member_id 为索引，否则与 member_id 索引的 Series 做 join 会静默错位丢行
    base = rfm[["member_id", "recency_days", "frequency", "monetary"]].copy().set_index("member_id")
    base["avg_basket"] = (base["monetary"] / base["frequency"].replace(0, pd.NA)).round(2)

    m_txn = txn[txn["member_type"] == "会员"].copy()
    n_txn = m_txn.groupby("member_id").size().rename("n_txn")

    it = items.merge(m_txn[["txn_id", "member_id"]], on="txn_id")
    it = it.merge(products[["sku", "category"]], on="sku")
    n_cat = it.groupby("member_id")["category"].nunique().rename("n_categories")
    n_items = it.groupby("member_id")["qty"].sum().rename("n_items")

    feats = base.join(n_txn).join(n_cat).join(n_items)
    feats["items_per_txn"] = (feats["n_items"] / feats["n_txn"].replace(0, pd.NA)).round(2)
    feats = feats.drop(columns=["n_txn", "n_items"])

    # 行为特征（画像层同一份实现，见 features/behavior.py）
    beh = behavior_features(txn)
    feats = feats.join(beh)

    # P1 扩展标签（同样一份实现，见 features/p1_tags.py）
    from retailinsight.features.p1_tags import build_p1_tags
    p1 = build_p1_tags(txn, items, products, rfm)
    if len(p1.columns):
        feats = feats.join(p1[[c for c in P1_CLUSTER_FEATURES if c in p1.columns]])

    for col in ALL_FEATURES:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")

    feats = feats.replace([float("inf"), float("-inf")], pd.NA)

    # ---- 覆盖守卫：特征矩阵必须覆盖全部 RFM 会员 ----
    # 历史教训：这里曾用 ``dropna(subset=CLUSTER_FEATURES)`` 静默丢掉 291 名会员
    # （根因是上游 mart_rfm 与 interim 明细来自不同批次的数据）。
    # 现在改成：缺失一律**填补**而不是丢弃，并且**断言覆盖率 100%** ——
    # 覆盖不住就立刻报错，绝不让"少了一批人"这件事悄无声息地发生。
    if len(feats) != len(rfm):
        missing = len(rfm) - len(feats)
        raise ValueError(
            f"聚类特征矩阵覆盖不全：rfm {len(rfm)} 人 vs 特征 {len(feats)} 人，"
            f"缺 {missing} 人。通常是上游产物来自不同批次的数据，"
            f"请先重跑 `python -m retailinsight.cli run --stage all`。"
        )
    for col in CLUSTER_FEATURES:
        feats[col] = feats[col].fillna(0.0)
    for col in BEHAVIOR_FEATURES + P1_CLUSTER_FEATURES:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0.0)
    # reset_index 而非 drop=True：member_id 是索引名，需要还原成列
    return feats.reset_index()
