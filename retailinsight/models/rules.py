"""购物篮关联规则（FP-Growth）。

三个层级（对应计划 §8.3 / R-9）：

| 层级 | 入口 | 数据范围 | 说明 |
|---|---|---|---|
| 品类级 | :func:`run_rules` | 全部交易 × 60 品类 | 业务最可读，作为主输出 |
| SKU 级 | :func:`run_sku_rules` | 销量 Top300 SKU，篮子长度 ≤30 | 单品级组合建议 |
| 分门店 | :func:`run_store_rules` | 按门店分批跑后合并 | 支撑门店差异化陈列 |

内存控制
--------
1. 品类级：60 维 one-hot，全量安全。
2. SKU 级：**先裁 SKU 再挖规则**（Top300），并限制篮子长度 —— 5000 维直接 one-hot 必爆内存。
3. 分门店：按 ``store_id`` 分批，每批篮子数约为全量的 1/15，峰值内存显著下降。
"""
from __future__ import annotations

import gc
import time

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from retailinsight.io.loader import read_items, read_parquet, write_parquet
from retailinsight.settings import INTERIM_DIR, ITEMS_DIR, MARTS_DIR

SKU_TOP_N = 300       # SKU 级参与挖掘的 SKU 数（按销量）
MAX_BASKET_LEN = 30   # 篮子内 SKU 数上限（控制 one-hot 维度与组合爆炸）
KEEP = ["antecedents", "consequents", "support", "confidence", "lift"]


def _mine(baskets: list[list[str]], min_support: float, min_confidence: float,
          min_lift: float) -> pd.DataFrame:
    """在给定篮子上跑 FP-Growth 并返回规则表（列名与品类级一致）。"""
    if not baskets:
        return pd.DataFrame(columns=KEEP)
    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] > min_lift].sort_values("lift", ascending=False)
    # SKU 级的前后件是整数 id，品类级是字符串 —— 统一转字符串再拼接
    rules["antecedents"] = rules["antecedents"].apply(
        lambda s: "/".join(sorted(str(x) for x in s)))
    rules["consequents"] = rules["consequents"].apply(
        lambda s: "/".join(sorted(str(x) for x in s)))
    out = rules[KEEP].round(4).reset_index(drop=True)
    del onehot, freq, rules
    gc.collect()
    return out


def _load_basket_frame() -> pd.DataFrame:
    """明细 × 交易（取门店），返回 txn_id / sku / category / qty / store_id。"""
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_items(ITEMS_DIR)
    products = read_parquet(INTERIM_DIR / "products.parquet")
    t = txn[~txn["is_logic_error"]][["txn_id", "store_id"]]
    return (items.merge(t, on="txn_id")
            .merge(products[["sku", "category"]], on="sku"))


def run_rules(min_support: float = 0.01, min_confidence: float = 0.3,
              min_lift: float = 1.5) -> pd.DataFrame:
    """品类级关联规则（全量交易 × 60 品类）—— 主输出。"""
    it = _load_basket_frame()
    baskets = it.groupby("txn_id")["category"].agg(lambda s: sorted(set(s))).tolist()
    print(f"[rules] 品类级 baskets={len(baskets)}")

    out = _mine(baskets, min_support, min_confidence, min_lift)
    write_parquet(out, MARTS_DIR / "mart_basket_rules.parquet")
    print(f"[rules] 品类级 rules(lift>{min_lift})={len(out)}")
    return out


def select_top_skus(it: pd.DataFrame, top_n: int = SKU_TOP_N) -> pd.Series:
    """按销量选 Top-N SKU（销量 = 明细数量之和）。"""
    qty = it.groupby("sku")["qty"].sum().sort_values(ascending=False)
    return qty.head(top_n).index


def run_sku_rules(min_support: float = 0.005, min_confidence: float = 0.2,
                  min_lift: float = 1.5, top_n: int = SKU_TOP_N,
                  max_basket_len: int = MAX_BASKET_LEN) -> pd.DataFrame:
    """SKU 级关联规则（R-9）。

    参数
    ----
    top_n:          参与挖掘的 SKU 数（按销量）
    max_basket_len: 每个篮子最多保留多少个 SKU —— 抑制组合爆炸的关键
    min_support:    支持度阈值（SKU 级基数大，默认放宽到 0.5%）
    """
    it = _load_basket_frame()
    top_skus = set(select_top_skus(it, top_n).tolist())
    it = it[it["sku"].isin(top_skus)]
    print(f"[rules] SKU 级：入选 {len(top_skus)} 个 SKU，明细 {len(it):,} 行")

    def _basket(s: pd.Series) -> list[str]:
        skus = sorted(set(s.tolist()))
        # 篮子过长时按销量降序截断（销量已隐含在 SKU 选取中）
        return skus[:max_basket_len]

    baskets = it.groupby("txn_id")["sku"].agg(_basket).tolist()
    print(f"[rules] SKU 级 baskets={len(baskets)}")

    t0 = time.time()
    out = _mine(baskets, min_support, min_confidence, min_lift)
    out = out.assign(level="sku")
    write_parquet(out, MARTS_DIR / "mart_basket_rules_sku.parquet")
    print(f"[rules] SKU 级 rules(lift>{min_lift})={len(out)} "
          f"耗时={time.time() - t0:.1f}s")
    return out


def run_store_rules(min_support: float = 0.02, min_confidence: float = 0.3,
                    min_lift: float = 1.5, level: str = "category") -> pd.DataFrame:
    """分门店关联规则（R-9）：按 ``store_id`` 分批跑后合并。

    ``level`` 可选 ``category`` / ``sku``；分批跑能把峰值内存降到全量的 1/门店数。
    """
    it = _load_basket_frame()
    col = "category" if level == "category" else "sku"

    if level == "sku":
        top_skus = set(select_top_skus(it, SKU_TOP_N).tolist())
        it = it[it["sku"].isin(top_skus)]

    frames = []
    for store_id, sub in it.groupby("store_id"):
        baskets = sub.groupby("txn_id")[col].agg(lambda s: sorted(set(s))).tolist()
        if len(baskets) < 50:
            continue
        r = _mine(baskets, min_support, min_confidence, min_lift)
        if len(r):
            r["store_id"] = store_id
            r["level"] = level
            frames.append(r)
        print(f"[rules]   门店 {store_id}: baskets={len(baskets):,} rules={len(r)}")

    if not frames:
        print("[rules] 分门店规则为空（各门店样本不足以挖掘）")
        return pd.DataFrame(columns=KEEP + ["store_id", "level"])

    out = pd.concat(frames, ignore_index=True).sort_values("lift", ascending=False)
    out = out.round(4).reset_index(drop=True)
    write_parquet(out, MARTS_DIR / "mart_basket_rules_store.parquet")
    print(f"[rules] 分门店（{level}）总规则数={len(out)}，门店数={out['store_id'].nunique()}")
    return out


if __name__ == "__main__":
    run_rules()
