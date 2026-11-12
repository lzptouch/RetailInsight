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

#: 全部可用特征
ALL_FEATURES = CLUSTER_FEATURES + BEHAVIOR_FEATURES

#: 候选特征空间（按顺序评估）
CLUSTER_FEATURE_SPACES: dict[str, list[str]] = {
    "base6": CLUSTER_FEATURES,
    "rfm3": ["recency_days", "frequency", "monetary"],
    "rfm3_behavior": ["recency_days", "frequency", "monetary"] + BEHAVIOR_FEATURES,
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

    for col in ALL_FEATURES:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")

    feats = feats.replace([float("inf"), float("-inf")], pd.NA)
    feats = feats.dropna(subset=CLUSTER_FEATURES)
    for col in BEHAVIOR_FEATURES:
        feats[col] = feats[col].fillna(0.0)
    # reset_index 而非 drop=True：member_id 是索引名，需要还原成列
    return feats.reset_index()
