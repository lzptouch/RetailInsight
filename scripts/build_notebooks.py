"""生成 5 个分析 Notebook（计划 §11 W2~W5 的交付物）。

**纪律**：notebook 只允许 import ``retailinsight`` 的函数，
不得在 notebook 内复制任何指标计算逻辑 —— 否则会破坏单一真相源
（计划 §3.1）。本脚本把 notebook 的骨架与代码单元统一维护在此，
保证 5 个 notebook 与主代码同步。

用法
----
    python scripts/build_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def md(text: str, cid: str = "m") -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str, cid: str = "c") -> dict:
    return {"cell_type": "code", "id": cid, "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


BOOTSTRAP = """# 让 notebook 无论从哪个目录启动都能 import retailinsight
import sys
from pathlib import Path

_candidates = [Path.cwd(), *Path.cwd().parents]
_root = next(p for p in _candidates if (p / "retailinsight" / "__init__.py").exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print("项目根目录:", _root)
"""


def build(name: str, cells: list[dict]) -> dict:
    # 第一个单元固定为路径自举，保证在任何目录下都能跑
    cells = [code(BOOTSTRAP, "bootstrap")] + list(cells)
    # nbformat 4.5 要求每个 cell 有唯一 id
    for i, c in enumerate(cells):
        c["id"] = f"{name[:2]}-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


EDA = [
    md("""# 01 · EDA 数据探索
> 目标：摸清数据规模、质量与业务形态，并**回收计划 §4.2 预埋的关联锚点**。
> 纪律：所有指标调用 `retailinsight`，不重算。"""),
    code("""import pandas as pd
from retailinsight.io.loader import read_parquet
from retailinsight.settings import INTERIM_DIR

pd.set_option("display.max_columns", 50)
members = read_parquet(INTERIM_DIR / "members.parquet")
txn = read_parquet(INTERIM_DIR / "transactions.parquet")
items = read_parquet(INTERIM_DIR / "transaction_items.parquet")
products = read_parquet(INTERIM_DIR / "products.parquet")
stores = read_parquet(INTERIM_DIR / "stores.parquet")

for name, df in [("members", members), ("transactions", txn),
                 ("transaction_items", items), ("products", products), ("stores", stores)]:
    print(f"{name:20s} rows={len(df):>9,}  cols={df.shape[1]:>3}  缺失率={df.isna().mean().mean():.2%}")
"""),
    md("""## 1.1 时间覆盖与业务规模"""),
    code("""print("时间范围:", txn["txn_ts"].min(), "~", txn["txn_ts"].max())
print("会员数:", members["member_id"].nunique())
print("门店数:", stores["store_id"].nunique(), "| SKU 数:", products["sku"].nunique())
print("品类数:", products["category"].nunique())

sales = txn[(~txn["is_logic_error"]) & (~txn["is_return"])]
print("\\n有效 GMV: %.1f 万元" % (sales["net_amount"].sum() / 1e4))
print("会员订单占比: %.1f%%" % (100 * (sales["member_type"] == "会员").mean()))
"""),
    md("""## 1.2 清洗前后的质量状况（质量报告同源）"""),
    code("""from retailinsight.etl.quality import missing_rates

dfs = {"members": members, "transactions": txn, "items": items}
rates = missing_rates(dfs)
print("清洗后仍存在缺失的字段：", rates if rates else "无（缺失已被转为哨兵值/标记列）")

print("\\n逻辑冲突行:", int(txn["is_logic_error"].sum()))
print("退货行:", int(txn["is_return"].sum()))
print("非会员交易行:", int((txn["member_type"] != "会员").sum()))
"""),
    md("""## 1.3 埋点回收验证：4 组强关联商品组合
> 生成器预埋的 4 组组合（计划 §4.2）必须在真实挖掘结果里被找回，
> 这是「数据里确实有可被发现的故事」的证据。"""),
    code("""from retailinsight.settings import MART_RULES
from retailinsight.io.schema import ANCHOR_COMBOS

rules = read_parquet(MART_RULES)
print(f"共 {len(rules)} 条规则（lift>1.5）\\n")
for a, b, _target in ANCHOR_COMBOS:
    hit = rules[(rules["antecedents"].astype(str).str.contains(a, regex=False))
                & (rules["consequents"].astype(str).str.contains(b, regex=False))]
    if len(hit):
        print(f"  ✅ {a} → {b}: lift={hit['lift'].max():.2f}  confidence={hit['confidence'].max():.2f}")
    else:
        print(f"  ❌ {a} → {b}: 未回收")
"""),
    md("""## 1.4 季节性与时段形态"""),
    code("""monthly = (sales.assign(m=sales["txn_ts"].dt.to_period("M"))
                 .groupby("m").agg(gmv=("net_amount", "sum"), orders=("txn_id", "nunique")))
monthly["aov"] = (monthly["gmv"] / monthly["orders"]).round(2)
ax = monthly["gmv"].plot(figsize=(11, 3.5), title="月度 GMV")
monthly.tail(12)
"""),
]

RFM = [
    md("""# 02 · RFM 与会员画像
> 目标：把会员切成 8 个可运营的客群，并给出每个客群的画像特征。"""),
    code("""import pandas as pd
from retailinsight.io.loader import read_parquet
from retailinsight.settings import MART_PROFILE, MART_RFM

rfm = read_parquet(MART_RFM)
profile = read_parquet(MART_PROFILE)
print(f"mart_rfm: {rfm.shape}  mart_profile: {profile.shape}")
"""),
    md("""## 2.1 RFM 打分口径
> 五分位打分 1~5，≥4 记「高」，再按 2×2×2 映射成 8 类客群。"""),
    code("""from retailinsight.features.rfm import compute_rfm, score_rfm, assign_segment

print(rfm[["recency_days", "frequency", "monetary", "r_score", "f_score", "m_score",
           "rfm_segment"]].describe().round(2))
"""),
    md("""## 2.2 八类客群分布与 GMV 贡献"""),
    code("""summary = (rfm.groupby("rfm_segment")
             .agg(members=("member_id", "count"), gmv=("monetary", "sum"),
                  avg_recency=("recency_days", "mean"), avg_frequency=("frequency", "mean"),
                  avg_monetary=("monetary", "mean"))
             .round(2))
summary["member_share"] = (summary["members"] / len(rfm)).round(4)
summary["gmv_share"] = (summary["gmv"] / summary["gmv"].sum()).round(4)
summary.sort_values("gmv", ascending=False)
"""),
    md("""## 2.3 客群特征对比（画像宽表）
> 画像宽表已 join 流失概率（第 20 个 P0 标签），`churn_covered` 标注是否被覆盖。"""),
    code("""cols = ["avg_basket", "monthly_amount", "yearly_freq", "items_per_txn",
        "category_entropy", "discount_sensitivity", "weekend_ratio"]
cmp = profile.groupby("rfm_segment")[cols].mean().round(3)
cmp.join(summary[["members", "gmv_share"]])
"""),
    md("""## 2.4 流失概率覆盖率说明
> 流失得分只覆盖末期 cohort 观察窗内的活跃会员，其余为 NaN（保留而非填均值）。"""),
    code("""if "churn_prob" in profile.columns:
    cov = profile["churn_covered"].mean()
    print(f"churn_prob 覆盖率: {cov:.1%}（未覆盖 {int((~profile['churn_covered']).sum()):,} 人保留 NaN）")
    print(profile.groupby("rfm_segment")["churn_prob"].mean().round(4).sort_values(ascending=False))
else:
    print("未生成 churn_prob，请先运行 python -m retailinsight.cli run --stage profile")
"""),
    md("""## 2.5 分群雷达图"""),
    code("""from retailinsight.viz import charts
print(charts.segment_radar(profile, by="rfm_segment"))
"""),
]

CLUSTER = [
    md("""# 03 · K-Means 客户聚类
> 目标：回答「会员除了 RFM 规则分群，还能不能被无监督地分成结构不同的群体」。
> 关键结论见文末：**轮廓系数达标与否取决于特征空间，不取决于调参**。"""),
    code("""import pandas as pd
from retailinsight.io.loader import read_parquet
from retailinsight.settings import INTERIM_DIR, MART_CLUSTER, MART_RFM

rfm = read_parquet(MART_RFM)
txn = read_parquet(INTERIM_DIR / "transactions.parquet")
items = read_parquet(INTERIM_DIR / "transaction_items.parquet")
products = read_parquet(INTERIM_DIR / "products.parquet")
"""),
    md("""## 3.1 候选特征空间
> `base6` = 计划 §8.1 规定的 6 维；`rfm3` = 只含对人群有区分度的核心维度；
> `rfm3_behavior` = 再叠加行为维度（周末占比 / 折扣敏感度 / 季节性强弱）。"""),
    code("""from retailinsight.features.cluster_features import (CLUSTER_FEATURE_SPACES, ALL_FEATURES,
                                                     build_cluster_features)
from retailinsight.models.cluster import build_cluster_matrix, choose_k

feats = build_cluster_features(rfm, txn, items, products)
print(f"参与聚类的会员: {len(feats):,}（mart_rfm 共 {len(rfm):,}）")
for name, cols in CLUSTER_FEATURE_SPACES.items():
    print(f"  {name:16s} -> {cols}")
"""),
    md("""## 3.2 逐 K 评估：肘部法 + 轮廓系数
> 轮廓系数在 3.5 万样本上是 O(n²)，因此按固定种子抽样 5000 条计算。"""),
    code("""X, _ = build_cluster_matrix(feats, ["recency_days", "frequency", "monetary"])
metrics = choose_k(X)
pd.DataFrame(metrics).T.assign(
    min_share=lambda d: d["cluster_sizes"].apply(lambda s: min(s.values()) / sum(s.values()))
).round(4)
"""),
    md("""## 3.3 神谕上界检验：0.35 到底可达不可达？
> 直接用生成器埋的**真实 8 类人群**当标签算轮廓系数。
> 若神谕值也很低，说明该特征空间本身无法分离真实结构，再调参也无济于事。"""),
    code("""from retailinsight.models.cluster import oracle_silhouette

members = read_parquet(INTERIM_DIR / "members.parquet")
persona = members.set_index("member_id")["_persona"]
p = feats["member_id"].map(persona).values

for name, cols in CLUSTER_FEATURE_SPACES.items():
    Xs, _ = build_cluster_matrix(feats, cols)
    print(f"  {name:16s} 神谕上界 silhouette = {oracle_silhouette(Xs, p):.4f}")
"""),
    md("""## 3.4 各特征空间的完整评估与择优"""),
    code("""from retailinsight.models.cluster import evaluate_space, select_space

evals = [evaluate_space(feats, name, cols, persona) for name, cols in CLUSTER_FEATURE_SPACES.items()]
for e in evals:
    print(f"  {e['space']:16s} K={e['k']}  silhouette={e['silhouette']:.4f}  "
          f"ARI={e['ari']:.4f}  达标={e['reached_target']}  神谕={e['oracle_silhouette']:.4f}")

choice = select_space(evals, has_persona=True)
print("\\n选中:", choice["selected_space"], "（择优准则:", choice["criterion"], "）")
"""),
    md("""## 3.5 簇画像与业务命名"""),
    code("""chosen = next(e for e in evals if e["space"] == choice["selected_space"])
chosen["desc"][["cluster_id", "cluster_name", "member_cnt", "member_share", "gmv_share",
                 "active_pct", "frequency_pct", "monetary_pct"]].round(3)
"""),
    md("""## 3.6 与先验人群的列联表（埋点闭环）
> 计划 §4.2 预埋的 8 类人群在这里终于有了消费方。"""),
    code("""cluster = read_parquet(MART_CLUSTER)
ct = pd.crosstab(cluster["cluster_name"],
                 cluster["member_id"].map(persona))
ct
"""),
]

RULES = [
    md("""# 04 · 关联规则挖掘
> 目标：把「一起买」的商品组合变成可执行的陈列与捆绑策略。"""),
    code("""import pandas as pd
from retailinsight.io.loader import read_parquet
from retailinsight.settings import MART_RULES

rules = read_parquet(MART_RULES)
print(f"共 {len(rules)} 条规则（min_support 1% / min_confidence 30% / lift > 1.5）")
rules.head(10)
"""),
    md("""## 4.1 规则强度分布"""),
    code("""print("lift 分布:", rules["lift"].describe().round(2).to_dict())
print("\\nconfidence 分布:", rules["confidence"].describe().round(2).to_dict())
rules["lift_bin"] = pd.cut(rules["lift"], [1.5, 2, 3, 5, 10], right=False)
rules.groupby("lift_bin", observed=True)["lift"].size()
"""),
    md("""## 4.2 Top 20 强关联规则"""),
    code("""rules.nlargest(20, "lift")[["antecedents", "consequents", "support",
                                  "confidence", "lift"]].round(4)
"""),
    md("""## 4.3 埋点回收验证：4 组预埋组合"""),
    code("""from retailinsight.io.schema import ANCHOR_COMBOS

for a, b, _target in ANCHOR_COMBOS:
    hit = rules[(rules["antecedents"].astype(str).str.contains(a, regex=False))
                & (rules["consequents"].astype(str).str.contains(b, regex=False))]
    print(f"  {a} → {b}: {'✅ lift=' + format(hit['lift'].max(), '.2f') if len(hit) else '❌ 未回收'}")
"""),
    md("""## 4.4 高频共同出现品类对"""),
    code("""# 前后件都可能是多品类组合，逐行展开成「品类对」
_rows = []
for _, r in rules.iterrows():
    for a in str(r["antecedents"]).split("/"):
        for b in str(r["consequents"]).split("/"):
            if a != b:
                _rows.append({"a": a, "b": b, "lift": float(r["lift"]),
                              "confidence": float(r["confidence"])})
_pair_df = pd.DataFrame(_rows)

pairs = (_pair_df.groupby(["a", "b"])
         .agg(n=("lift", "size"), lift=("lift", "max"), confidence=("confidence", "max"))
         .sort_values(["n", "lift"], ascending=False))
pairs.head(15)
"""),
]

CHURN = [
    md("""# 05 · 流失预测模型
> 目标：给每个会员一个流失概率，并给出可运营的名单与阈值依据。
> **防泄漏是底线**：特征只用观察窗，标签只看标签窗。"""),
    code("""import json
import pandas as pd
from retailinsight.io.loader import read_parquet
from retailinsight.settings import MART_CHURN, REPORTS_DIR

metrics = json.loads((REPORTS_DIR / "metrics_churn.json").read_text(encoding="utf-8"))
print("backend:", metrics["backend"])
print("时间切分:", metrics["split"])
print("特征数:", metrics["n_features"])
pd.DataFrame({"LR 基线": metrics["lr_baseline"], "GBDT 主模型": metrics["gbdt"],
              "RandomForest": metrics["random_forest"]})
"""),
    md("""## 5.1 验收线核对"""),
    code("""gb, lr = metrics["gbdt"], metrics["lr_baseline"]
upper = 1 / gb["base_rate"]
print(f"AUC 增益: {gb['auc'] - lr['auc']:+.4f}（验收线 +0.03）→ "
      f"{'达标' if gb['auc'] - lr['auc'] >= 0.03 else '未达标'}")
print(f"KS: {gb['ks']:.4f}（验收线 ≥0.30）→ {'达标' if gb['ks'] >= 0.30 else '未达标'}")
print(f"Lift@20%: {gb['lift_top20']:.2f} / 理论上限 {upper:.2f} = "
      f"{gb['lift_top20'] / upper:.1%}（验收线 ≥85%）")
"""),
    md("""## 5.2 特征重要性（SHAP 或 permutation importance 降级）"""),
    code("""imp = pd.Series({k: v for k, v in metrics["feature_importance"].items() if k != "method"},
                 dtype="float64").sort_values(ascending=False)
imp.head(15)
"""),
    md("""## 5.3 验证集上的阈值调优
> 按业务成本最优（漏判 : 误触 = 8 : 1），与《决策建议报告》S2 同口径。"""),
    code("""th = metrics["thresholds"]
print(json.dumps(th, ensure_ascii=False, indent=2))
"""),
    md("""## 5.4 名单产出与分层"""),
    code("""churn = read_parquet(MART_CHURN)
print(f"打分覆盖 {len(churn):,} 人")
print(churn["risk_level"].value_counts())
print(f"\\nTop20 高危名单平均流失概率: {churn.nlargest(int(len(churn)*0.2), 'churn_prob')['churn_prob'].mean():.4f}")
print("\\n单会员归因字段:", [c for c in churn.columns if 'shap' in c.lower()])
"""),
    md("""## 5.5 概率分布"""),
    code("""from retailinsight.viz import charts
print(charts.churn_hist(churn["churn_prob"].to_numpy()))
print(charts.lift_curve(churn["churn"].to_numpy(), churn["churn_prob"].to_numpy()))
"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_EDA.ipynb": EDA,
        "02_RFM.ipynb": RFM,
        "03_cluster.ipynb": CLUSTER,
        "04_rules.ipynb": RULES,
        "05_churn.ipynb": CHURN,
    }
    for name, cells in notebooks.items():
        path = NB_DIR / name
        path.write_text(json.dumps(build(name, cells), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"[notebooks] 生成 {path.relative_to(ROOT)}  ({len(cells)} 个单元)")


if __name__ == "__main__":
    main()
