"""购物篮关联规则（品类级，FP-Growth）。

内存控制（对应计划 §8.3）：只在品类级（60 类）做全量 one-hot；
SKU 级在 5000 维度上会爆内存，留作 P1 分店分批方案。
"""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from retailinsight.io.loader import read_parquet, write_parquet
from retailinsight.settings import INTERIM_DIR, MARTS_DIR


def run_rules(min_support: float = 0.01, min_confidence: float = 0.3,
              min_lift: float = 1.5) -> pd.DataFrame:
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_parquet(INTERIM_DIR / "transaction_items.parquet")
    products = read_parquet(INTERIM_DIR / "products.parquet")

    t = txn[~txn["is_logic_error"]][["txn_id"]]
    it = (items.merge(t, on="txn_id")
          .merge(products[["sku", "category"]], on="sku"))
    baskets = it.groupby("txn_id")["category"].agg(lambda s: sorted(set(s))).tolist()
    print(f"[rules] baskets={len(baskets)}")

    te = TransactionEncoder()
    onehot = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] > min_lift].sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: "/".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: "/".join(sorted(s)))
    keep = ["antecedents", "consequents", "support", "confidence", "lift"]
    out = rules[keep].round(4).reset_index(drop=True)
    write_parquet(out, MARTS_DIR / "mart_basket_rules.parquet")
    print(f"[rules] itemsets={len(freq)} rules(lift>{min_lift})={len(out)}")
    return out


if __name__ == "__main__":
    run_rules()
