"""决策建议报告渲染器。

**为什么需要它**：报告里的每一个数字都手写，会随数据重跑而漂移
（实测曾出现「会员 35,576 / 实际 35,612」这类失配）。
本模块把 KPI 与建议测算全部改为**从 data/marts 产物回填**，
报告只作为模板，数字唯一来源是代码（呼应计划 §3.1 单一真相源纪律）。

用法
----
    python -m retailinsight.reports.render_decision_report

输出
----
    docs/决策建议报告.md
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retailinsight.io.loader import read_items, read_parquet
from retailinsight.settings import (
    DATA_END,
    DATA_START,
    DOCS_DIR,
    INTERIM_DIR,
    ITEMS_DIR,
    MART_CHURN,
    MART_MONTHLY,
    MART_PROFILE,
    MART_QUARTERLY,
    MART_RFM,
    MART_RULES,
    REPORTS_DIR,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "决策建议报告.md.tpl"
OUTPUT_PATH = DOCS_DIR / "决策建议报告.md"


# ---------------------------------------------------------------- 基础 KPI

def _w(x: float) -> float:
    """万元。"""
    return round(float(x) / 1e4, 1)


def _pct(x: float) -> str:
    """百分比字符串（保留 1 位小数）。"""
    return f"{float(x):.1%}"


def _round4(x) -> str:
    """指标保留 4 位小数；非数值（`-` / None）原样透传，避免报告出现浮点尾数。"""
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x) if x is not None else "-"


#: 模型代码名 → 中文名（报告里对外只说人话）
MODEL_CN = {
    "random_forest": "随机森林（RandomForest）",
    "gbdt": "GBDT（LightGBM）",
    "logistic_regression": "逻辑回归（LR 基线）",
}


def _model_cn(name: str) -> str:
    """模型代码名 → 中文展示名。"""
    return MODEL_CN.get(name, name)


def _verdict_paragraph(k: dict, m: dict) -> str:
    """生成「模型验收结论」段落。

    **为什么整段在代码里拼**：这段话要同时陈述「主模型是谁 / 增益多少 /
    达没达标 / 没达标的话原因是什么」。如果写成模板里的固定文字，
    主模型换人时就会出现「已达标」与「无一达到 +0.03」并排的自相矛盾 ——
    这正是 v3.2 切主模型后遗留的缺陷。所以整段由数据驱动生成。
    """
    reached = m.get("auc_gain_reached")
    head = (f"流失模型用「训练 {k.get('model_cohorts')} / 验证 {k.get('model_valid_cohort')}"
            f" / 测试 {k.get('model_test_cohort')}」三段式时间切分，主模型由验收线自动选出："
            f"**{k.get('model_name')}**（AUC {k.get('model_auc')}，较 LR 基线 "
            f"{k.get('model_auc_lr')} 提升 {k.get('model_auc_gain')}，验收线 +0.03，"
            f"**{k.get('model_auc_gain_verdict')}**）。")
    if reached:
        tail = ("选型规则写死在 `models/churn.py::select_main_model`"
                "（先筛「AUC ≥ 基线+0.03」的达标组 → 取其中 KS 最高者），"
                "因此结论可复现、可向甲方解释。对照模型 GBDT AUC "
                f"{k.get('model_gbdt_auc')}"
                f"（增益 {k.get('model_gbdt_gain')}）未达标，"
                "经 6 组「切分 × 特征」敏感性实验（`scripts/exp_churn_sensitivity.py`）"
                "确认 **GBDT 口径下 +0.03 是当前数据窗口的天花板**："
                "增益与训练期数正相关、与特征维度负相关。")
    else:
        tail = ("选型规则写死在 `models/churn.py::select_main_model`"
                "（达标组 → KS 最高者 → 全不达标才取 AUC 最高者），"
                "当前**无模型达到 +0.03**。经 6 组「切分 × 特征」敏感性实验"
                "（`scripts/exp_churn_sensitivity.py`）确认这是当前数据窗口下的天花板"
                "（增益与训练期数正相关、与特征维度负相关），而非调参问题。")
    return head + tail


def collect_kpis() -> dict:
    """从 mart / interim 产物回填全部 KPI。"""
    rfm = read_parquet(MART_RFM)
    profile = read_parquet(MART_PROFILE)
    churn = read_parquet(MART_CHURN)
    rules = read_parquet(MART_RULES)
    monthly = read_parquet(MART_MONTHLY)
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_items(ITEMS_DIR)
    products = read_parquet(INTERIM_DIR / "products.parquet")

    ok = txn[(~txn["is_logic_error"]) & (txn["member_type"] == "会员") & (~txn["is_return"])]
    sales = txn[(~txn["is_logic_error"]) & (~txn["is_return"])]

    k: dict = {}
    k["n_members"] = len(rfm)
    k["n_profile"] = len(profile)
    k["n_orders"] = int(sales["txn_id"].nunique())
    k["gmv_wan"] = _w(sales["net_amount"].sum())
    k["gmv_member_wan"] = _w(ok["net_amount"].sum())
    k["data_window"] = f"{DATA_START} ~ {DATA_END}"
    k["n_months"] = int(monthly["month"].nunique())

    # ---- 客群 ----
    seg = rfm.groupby("rfm_segment").agg(n=("member_id", "count"), gmv=("monetary", "sum"))
    seg["gmv_share"] = seg["gmv"] / seg["gmv"].sum()
    k["n_vip"] = int(seg.loc["重要价值客户", "n"])
    k["vip_share"] = _pct(seg.loc["重要价值客户", "n"] / len(rfm))
    k["vip_gmv_share"] = _pct(seg.loc["重要价值客户", "gmv_share"])
    k["vip_arpu"] = round(float(rfm.loc[rfm["rfm_segment"] == "重要价值客户", "monetary"].mean()), 1)
    k["vip_arpu_month"] = round(k["vip_arpu"] / 24, 1)
    k["n_churn_risk"] = int(seg.loc["流失风险客户", "n"])
    k["churn_risk_share"] = _pct(seg.loc["流失风险客户", "n"] / len(rfm))
    k["churn_risk_gmv_share"] = _pct(seg.loc["流失风险客户", "gmv_share"])
    k["churn_risk_avg_recency"] = int(rfm.loc[rfm["rfm_segment"] == "流失风险客户",
                                          "recency_days"].mean())
    # 实测口径：会员人均订单金额（客单价）与人均月频 —— 供 S3 的测算使用，
    # 不写死常数（早期模板里的「客单价 144 元」就是手写常数，与实测脱节）
    freq_all = float(rfm["frequency"].mean())
    monetary_all = float(rfm["monetary"].mean())
    k["aov_member"] = round(monetary_all / max(freq_all, 1e-9), 2)
    k["monthly_freq_member"] = round(freq_all / 24.0, 3)
    top2 = seg.nlargest(2, "gmv")
    k["top2_share"] = _pct(float(top2["n"].sum() / len(rfm)))
    k["top2_gmv_share"] = _pct(float(top2["gmv_share"].sum()))

    # ---- 流失模型 ----
    k["n_high_risk"] = int((churn["risk_level"] == "高").sum())
    k["high_risk_arpu"] = round(float(churn.loc[churn["risk_level"] == "高", "monetary"].mean()), 1)
    # 高危会员的历史人均月消费（S1 的假设值要拿它做参照，说明「恢复到 100 元」意味着什么）
    k["high_risk_arpu_month"] = round(k["high_risk_arpu"] / 24.0, 1)
    metrics_path = REPORTS_DIR / "metrics_churn.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        # **主模型口径**：所有对外披露的指标一律取 `main_model_metrics`
        # （即 `models/churn.py::select_main_model` 按验收线选中的那一个），
        # 不能拿对照模型的指标去配主模型的增益 —— 那是两套模型的数字，
        # 混用会让报告出现「AUC 是 A 的、增益是 B 的」这种自相矛盾。
        main_name = m.get("main_model", "-")
        main_m = m.get("main_model_metrics") or m.get("gbdt", {})
        gbdt_m = m.get("gbdt", {})
        rf_m = m.get("random_forest", {})
        k["model_backend"] = m.get("backend", "-")
        k["model_name"] = _model_cn(main_name)
        k["model_name_raw"] = main_name
        k["model_auc"] = main_m.get("auc", "-")
        k["model_auc_lr"] = m.get("lr_baseline", {}).get("auc", "-")
        k["model_auc_gain"] = m.get("auc_gain_vs_lr", "-")
        k["model_auc_gain_reached"] = m.get("auc_gain_reached", "-")
        k["model_auc_gain_verdict"] = ("已达标" if m.get("auc_gain_reached")
                                       else ("未达标" if m.get("auc_gain_reached") is False
                                             else "-"))
        k["model_ks"] = main_m.get("ks", "-")
        k["model_lift"] = main_m.get("lift_top20", "-")
        k["model_capture"] = main_m.get("capture_top20", "-")
        k["model_base_rate"] = main_m.get("base_rate", "-")
        # 对照模型（用于「为什么选它」的说明）
        k["model_gbdt_auc"] = gbdt_m.get("auc", "-")
        k["model_gbdt_gain"] = m.get("gbdt_auc_gain_vs_lr", "-")
        k["model_gbdt_ks"] = gbdt_m.get("ks", "-")
        k["model_rf_auc"] = rf_m.get("auc", "-")
        k["model_rf_gain"] = m.get("rf_auc_gain_vs_lr", "-")
        k["model_selection_reason"] = m.get("main_model_selection", {}).get("reason", "-")
        try:
            k["model_lift_upper"] = round(1 / float(k["model_base_rate"]), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            k["model_lift_upper"] = "-"
        k["model_n_features"] = m.get("n_features", "-")
        k["model_cohorts"] = " / ".join(m.get("split", {}).get("train", [])) + "（训练）"
        k["model_valid_cohort"] = " / ".join(m.get("split", {}).get("valid", []))
        k["model_test_cohort"] = " / ".join(m.get("split", {}).get("test", []))
        th = m.get("thresholds", {})
        k["threshold_action"] = _round4(th.get("action"))
        k["threshold_high"] = _round4(th.get("high"))
        k["model_verdict_paragraph"] = _verdict_paragraph(k, m)
    else:
        for key in ["model_backend", "model_name", "model_name_raw", "model_auc",
                    "model_auc_lr", "model_auc_gain",
                    "model_auc_gain_reached", "model_auc_gain_verdict", "model_ks",
                    "model_lift", "model_capture",
                    "model_base_rate", "model_lift_upper", "model_n_features",
                    "model_cohorts", "model_valid_cohort", "model_test_cohort",
                    "threshold_action", "threshold_high",
                    "model_gbdt_auc", "model_gbdt_gain", "model_gbdt_ks",
                    "model_rf_auc", "model_rf_gain", "model_selection_reason",
                    "model_verdict_paragraph"]:
            k[key] = "-"

    # ---- 聚类 ----
    cluster_path = REPORTS_DIR / "metrics_cluster.json"
    if cluster_path.exists():
        cm = json.loads(cluster_path.read_text(encoding="utf-8"))
        k["cluster_space"] = cm.get("feature_space", {}).get("selected_space", "-")
        k["cluster_k"] = cm.get("k", "-")
        # 指标进交付文档一律保留 4 位小数：原始 JSON 里的浮点尾数
        # （0.4164508774349119 这类）出现在报告里既不专业也没意义。
        k["cluster_silhouette"] = _round4(cm.get("silhouette"))
        k["cluster_ari"] = _round4(cm.get("ari_vs_persona"))
        k["cluster_oracle"] = _round4(cm.get("oracle_silhouette"))
        k["cluster_reached"] = cm.get("reached_target", "-")
    else:
        for key in ["cluster_space", "cluster_k", "cluster_silhouette", "cluster_ari",
                    "cluster_oracle", "cluster_reached"]:
            k[key] = "-"

    # ---- 漏斗 ----
    m_txn = txn[(~txn["is_logic_error"]) & (txn["member_type"] == "会员")]
    per = m_txn.groupby("member_id")["txn_ts"].agg(["nunique", "count"])
    k["n_registered"] = len(read_parquet(INTERIM_DIR / "members.parquet"))
    k["n_first_buy"] = len(per)
    k["n_repeat"] = int((per["nunique"] >= 2).sum())
    k["n_high_freq"] = int((per["nunique"] >= 12).sum())
    k["conv_first"] = _pct(k["n_first_buy"] / k["n_registered"])
    k["conv_repeat"] = _pct(k["n_repeat"] / k["n_first_buy"])
    k["conv_high_freq"] = _pct(k["n_high_freq"] / k["n_repeat"])
    k["n_repeat_not_high"] = k["n_repeat"] - k["n_high_freq"]

    # ---- 季节性 ----
    monthly = monthly.sort_values("month")
    peak = monthly.loc[monthly["gmv"].idxmax()]
    trough = monthly.loc[monthly["gmv"].idxmin()]
    k["peak_month"] = str(peak["month"])
    k["peak_gmv_wan"] = _w(peak["gmv"])
    k["trough_month"] = str(trough["month"])
    k["trough_gmv_wan"] = _w(trough["gmv"])
    k["season_ratio"] = round(float(peak["gmv"] / trough["gmv"]), 1)
    nov = monthly[monthly["month"].astype(str).str.endswith("-11")]
    k["nov_gmv_wan"] = _w(nov["gmv"].max()) if len(nov) else 0.0

    # ---- 时间特征 ----
    is_weekend = ok["txn_ts"].dt.day_of_week >= 5
    k["weekend_ratio"] = _pct(float(ok[is_weekend].shape[0] / len(ok)))
    hour_top = ok["txn_ts"].dt.hour.value_counts().head(3).index.tolist()
    k["peak_hours"] = "/".join(str(h) for h in hour_top)

    # ---- 品类结构 ----
    it = (items.merge(ok[["txn_id"]], on="txn_id")
          .merge(products[["sku", "category"]], on="sku"))
    it["line_amount"] = it["qty"] * it["unit_price"]
    cat = it.groupby("category").agg(gmv=("line_amount", "sum"),
                                      sku_cnt=("sku", "nunique")).reset_index()
    cat["share"] = cat["gmv"] / cat["gmv"].sum()
    cat = cat.sort_values("gmv", ascending=False).reset_index(drop=True)
    cat["cum"] = cat["share"].cumsum()
    k["top5_categories"] = "/".join(cat.head(5)["category"].tolist())
    k["top5_share"] = _pct(float(cat.head(5)["share"].sum()))
    k["n_categories"] = len(cat)
    k["n_abc_a"] = int((cat["cum"] <= 0.7 + 1e-9).sum())
    # 品类金额按**份额**折算到交易净额口径（见下方 S5 的说明）：
    # 合成数据的明细是目录价（`qty × unit_price`），与交易实付金额不同口径，
    # 直接把明细金额当 GMV 会与报告头的总 GMV 差约 3.2 倍。
    net_ratio = float(sales["net_amount"].sum() / max(it["line_amount"].sum(), 1e-9))
    cat_gmv_monthly = (cat.set_index("category")["gmv"] * net_ratio / k["n_months"])

    # ---- 关联规则 ----
    k["n_rules"] = len(rules)
    for a, b in [("洗发水", "护发素"), ("火锅底料", "牛羊肉"), ("面包糕点", "乳制品"), ("纸尿裤", "啤酒")]:
        hit = rules[(rules["antecedents"].astype(str).str.contains(a, regex=False))
                    & (rules["consequents"].astype(str).str.contains(b, regex=False))]
        key = f"rule_{a}_{b}"
        k[key] = f"{a}→{b} lift {hit['lift'].max():.2f}" if len(hit) else f"{a}→{b} 未回收"

    # ---- 门店品类结构差异（S8）----
    ok_it = (items.merge(ok[["txn_id", "store_id"]], on="txn_id")
             .merge(products[["sku", "category"]], on="sku"))
    ok_it["line_amount"] = ok_it["qty"] * ok_it["unit_price"]
    store_cat = ok_it.pivot_table(index="store_id", columns="category",
                                  values="line_amount", aggfunc="sum", fill_value=0)
    overall = store_cat.sum() / store_cat.sum().sum()
    # 门店 GMV 口径必须与报告头的「总 GMV」（`gmv_wan` = 全量销售净额）一致：
    # 既不能用明细 `qty × 目录价`（合成数据里是商品目录价，与实付差约 3.2 倍），
    # 也不能只用会员净额（少 2.98% 的非会员，会把 S8 结论系统性压低）。
    store_gmv = (sales.groupby("store_id")["net_amount"].sum()
                 .reindex(store_cat.index).fillna(0.0))
    diffs = {}
    for st in store_cat.index:
        v = store_cat.loc[st] / max(store_gmv[st], 1e-9)
        cos = float((v * overall).sum() / (np.linalg.norm(v) * np.linalg.norm(overall) + 1e-9))
        diffs[st] = 1 - cos
    k["store_count"] = len(diffs)
    k["store_diff_max"] = round(float(max(diffs.values())), 4)
    k["store_diff_mean"] = round(float(np.mean(list(diffs.values()))), 4)
    k["store_gmv_total_wan"] = _w(store_gmv.sum())

    k.update(_calc_s1(k))
    k.update(_calc_s2(k))
    k.update(_calc_s3(k))
    k.update(_calc_s4(k, profile))
    k.update(_calc_s5(rules, cat_gmv_monthly))
    k.update(_calc_s7(monthly, k))
    k.update(_calc_s8(store_gmv, diffs))

    k["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 数据规模告警：`--scale 0.05` 之类的小规模跑批会覆盖 docs/ 产物，
    # 报告首行必须显式标出来，否则读者会把小样本结论当正式结论（v3.6 #29）。
    n_members = int(k.get("n_members", 0) or 0)
    k["scale_line"] = (
        f"> ⚠ **本报告基于小规模验证数据生成**（会员 {n_members:,} 人，作品集口径 "
        f"35,650 人）。数字与结论不可与正式文档混用，正式口径请重跑 `make all`。"
        if 0 < n_members < 32_000 else "")
    k["excel_name"] = sorted(p.name for p in REPORTS_DIR.glob("*.xlsx"))[-1] if list(
        REPORTS_DIR.glob("*.xlsx")) else "-"

    # ---- 季度同比（末期不完整季度不参与解读）----
    quarterly = read_parquet(MART_QUARTERLY).sort_values("quarter")
    # 季度末月超过数据末月的季度属于不完整季度，其同比是口径假象
    q_end_month = quarterly["quarter"].dt.to_timestamp(how="end").dt.to_period("M")
    q_complete = q_end_month <= pd.Period(DATA_END, freq="M")
    q_valid = quarterly[q_complete & quarterly["yoy"].notna()]
    if len(q_valid):
        k["q_last"] = str(q_valid.iloc[-1]["quarter"])
        k["q_last_yoy"] = f"{q_valid.iloc[-1]['yoy']:+.1%}"
        k["q_yoy_max"] = f"{q_valid['yoy'].max():+.1%}"
        k["q_yoy_max_q"] = str(q_valid.loc[q_valid["yoy"].idxmax(), "quarter"])
    else:
        k["q_last"] = k["q_yoy_max"] = k["q_yoy_max_q"] = "-"
    incomplete = quarterly[~q_complete]
    k["q_last_incomplete"] = str(incomplete.iloc[-1]["quarter"]) if len(incomplete) else "-"
    m_valid = monthly[monthly["yoy"].notna()]
    if len(m_valid):
        k["m_last"] = str(m_valid.iloc[-1]["month"])
        k["m_last_yoy"] = f"{m_valid.iloc[-1]['yoy']:+.1%}"
    else:
        k["m_last"] = k["m_last_yoy"] = "-"
    return k


# ---------------------------------------------------------------- 建议测算

def _calc_s1(k: dict) -> dict:
    """S1 高危会员挽留：ΔGMV = N_高危 × 触达率 × 挽回率 × 月均消费 × 6 个月。

    **为什么必须单独算**：早期版本的模板直接复用了 S2 的 VIP 测算占位符，
    导致「高危挽留」这条建议印出来的金额与它自己写的公式完全对不上。
    每条建议的测算必须由自己的公式独立算出，不允许借用别人的数。
    """
    n = k["n_high_risk"]
    touch = 0.80          # 假设：80% 高危会员进入触达名单
    save = (0.10, 0.20)   # 假设：触达后 10%~20% 的挽回率
    arpu_m = 100.0        # 假设：挽留成功后月均消费恢复到 100 元
    months = 6
    lo = n * touch * save[0] * arpu_m * months
    hi = n * touch * save[1] * arpu_m * months
    return {
        "s1_gmv_lo": f"{_w(lo):.1f}", "s1_gmv_hi": f"{_w(hi):.1f}",
        "s1_touch": f"{touch:.0%}",
        "s1_save_lo": f"{save[0]:.0%}", "s1_save_hi": f"{save[1]:.0%}",
        "s1_arpu_m": f"{arpu_m:.0f}", "s1_months": f"{months}",
    }


def _calc_s3(k: dict) -> dict:
    """S3 复购→高频转化：ΔGMV = N × 转化率 × 人均月频提升 × 客单价 × 12。

    人均月频与客单价取自 `mart_rfm` 实测（不是手写常数）；
    「转化率 5%」与「转化者月频翻倍」是假设值。
    """
    n = k["n_repeat_not_high"]
    conv = 0.05                        # 假设：任务体系转化率
    monthly_freq = k["monthly_freq_member"]   # 实测：人均月频（次/月）
    freq_gain = monthly_freq           # 假设：转化者月频翻倍（+100%）
    aov = k["aov_member"]              # 实测：会员客单价
    gmv = n * conv * freq_gain * aov * 12
    return {
        "s3_gmv_wan": f"{_w(gmv):.1f}",
        "s3_conv": f"{conv:.0%}",
        "s3_freq_gain": f"{freq_gain:.2f}",
        "s3_aov": f"{aov:.0f}",
    }


def _calc_s4(k: dict, profile: pd.DataFrame) -> dict:
    """S4 促销精准化：券成本节省 = 全域发券人数×人均券成本 − 精准投放人数×人均券成本。

    **圈选口径用分位数而非绝对阈值**：`discount_sensitivity` 是「让掉的利占消费额的比例」，
    分布极度右偏（实测中位 0.026、均值 0.049、最大 120.8）——
    用固定阈值 0.3 只会圈出 45 个人（0.1%），既不现实也没有业务意义。
    改用**折扣敏感度 Top 20%** 圈选，与流失模型「运营只发得动 20% 的人」的口径一致；
    实际阈值由分位数算出（并写进报告，便于复核）。
    """
    coupon_cost = 10.0          # 假设：人均券成本 10 元
    pct = 0.80                   # 圈选折扣敏感度 Top 20%
    n_all = k["n_profile"]
    n_target = 0
    threshold = float("nan")
    if "discount_sensitivity" in profile.columns:
        ds = profile["discount_sensitivity"].dropna()
        if len(ds):
            threshold = float(ds.quantile(pct))
            n_target = int((ds >= threshold).sum())
    cost_all = n_all * coupon_cost
    cost_target = n_target * coupon_cost
    save = cost_all - cost_target
    rate = save / cost_all if cost_all else 0.0
    return {
        "s4_cost_all_wan": f"{_w(cost_all):.1f}",
        "s4_cost_target_wan": f"{_w(cost_target):.1f}",
        "s4_save_wan": f"{_w(save):.1f}",
        "s4_save_rate": f"{rate:.0%}",
        "s4_n_all": f"{n_all:,}",
        "s4_n_target": f"{n_target:,}",
        "s4_coupon_cost": f"{coupon_cost:.0f}",
        "s4_top_pct": f"{1 - pct:.0%}",
        "s4_threshold": f"{threshold:.4f}",
    }


def calc_s2(k: dict) -> dict:
    """S2 重要价值客户防流失：ΔGMV = N × 自然流失率 × 挽回率 × 月均消费 × 12。"""
    n_vip = k["n_vip"]
    arpu_m = k["vip_arpu_month"]
    r_attr = (0.15, 0.25)   # 假设：VIP 群年自然流失率
    r_save = (0.15, 0.30)   # 假设：干预挽回率
    lo = n_vip * r_attr[0] * r_save[0] * arpu_m * 12
    hi = n_vip * r_attr[1] * r_save[1] * arpu_m * 12
    # 成本只发生在「会流失且被触达」的那部分 VIP 身上，
    # 否则收益与成本口径不一致，ROI 会被系统性低估
    cost_lo = n_vip * r_attr[0] * 30 * 6
    cost_hi = n_vip * r_attr[1] * 30 * 6
    return {
        "s2_attr_lo": f"{r_attr[0]:.0%}", "s2_attr_hi": f"{r_attr[1]:.0%}",
        "s2_save_lo": f"{r_save[0]:.0%}", "s2_save_hi": f"{r_save[1]:.0%}",
        "s2_gmv_lo": f"{_w(lo):.1f}", "s2_gmv_hi": f"{_w(hi):.1f}",
        "s2_cost_wan": f"{_w(cost_lo):.1f}~{_w(cost_hi):.1f}",
        "s2_roi_lo": f"{lo / max(cost_hi, 1):.1f}", "s2_roi_hi": f"{hi / max(cost_lo, 1):.1f}",
    }


def _calc_s2(k: dict) -> dict:
    return calc_s2(k)


def _calc_s5(rules: pd.DataFrame, cat_gmv_monthly: pd.Series) -> dict:
    """S5 关联陈列：ΔGMV = Σ(后件品类月GMV) × (confidence × 转化系数) × support × 12。

    转化系数 30%~60% 表示「相邻陈列后，后件购买率能兑现关联强度的三到六成」。
    """
    top = rules.nlargest(10, "lift")
    monthly_sum = 0.0
    for _, r in top.iterrows():
        cons = [c for c in str(r["consequents"]).split("/") if c in cat_gmv_monthly.index]
        if not cons:
            continue
        gmv_per = float(cat_gmv_monthly[cons].mean())
        monthly_sum += gmv_per * (float(r["confidence"]) * 0.45) * float(r["support"])
    lo = monthly_sum * (0.30 / 0.45) * 12
    hi = monthly_sum * (0.60 / 0.45) * 12
    return {
        "s5_gmv_lo": f"{_w(lo):.1f}", "s5_gmv_hi": f"{_w(hi):.1f}",
        "s5_lift_lo": "30%", "s5_lift_hi": "60%",
        "s5_rules_used": str(len(top)),
    }


def _calc_s7(monthly: pd.DataFrame, k: dict) -> dict:
    """S7 季节性采购：备货倍数 / 滞销资金占用 / 缺货损失。

    - 淡季基准取**最低 3 个月的均值**而非绝对最小值：首月往往是不完整月，
      用绝对最小值会把备货倍数放大到不合理的量级。
    - 缺货损失 = 峰值月 GMV × 缺货率（峰值 GMV 本身已是金额，不再乘客单价）。
    """
    complete = monthly.iloc[1:-1] if len(monthly) > 3 else monthly
    peak_gmv = float(complete["gmv"].max())
    trough_gmv = float(complete.nsmallest(3, "gmv")["gmv"].mean())
    ratio = round(peak_gmv / max(trough_gmv, 1e-9), 1)
    slow_rate = (0.10, 0.20)
    short_rate = (0.03, 0.08)
    slow_lo = trough_gmv * ratio * slow_rate[0]
    slow_hi = trough_gmv * ratio * slow_rate[1]
    loss_lo = peak_gmv * short_rate[0]
    loss_hi = peak_gmv * short_rate[1]
    return {
        "s7_ratio": f"{ratio}",
        "s7_trough_gmv_wan": f"{_w(trough_gmv):.1f}",
        "s7_peak_gmv_wan": f"{_w(peak_gmv):.1f}",
        "s7_slow_lo": f"{_w(slow_lo):.1f}", "s7_slow_hi": f"{_w(slow_hi):.1f}",
        "s7_loss_lo": f"{_w(loss_lo):.1f}", "s7_loss_hi": f"{_w(loss_hi):.1f}",
        "s7_slow_rate_lo": f"{slow_rate[0]:.0%}", "s7_slow_rate_hi": f"{slow_rate[1]:.0%}",
        "s7_short_rate_lo": f"{short_rate[0]:.0%}", "s7_short_rate_hi": f"{short_rate[1]:.0%}",
    }


def _calc_s8(store_gmv: pd.Series, diffs: dict) -> dict:
    """S8 门店差异化：ΔGMV = Σ(门店GMV × 品类结构差异度 × 优化系数)。"""
    coef = (0.03, 0.08)
    base = float(sum(store_gmv[st] * diffs[st] for st in diffs))
    return {
        "s8_gmv_lo": f"{_w(base * coef[0]):.1f}", "s8_gmv_hi": f"{_w(base * coef[1]):.1f}",
        "s8_coef_lo": f"{coef[0]:.0%}", "s8_coef_hi": f"{coef[1]:.0%}",
    }


# ---------------------------------------------------------------- 渲染

def render(kpis: dict) -> str:
    """把 KPI 填入模板。

    「整行只有一个占位符、且值为空」的行会**整行消失**（`{{scale_line}}` 在
    正则口径下就是空串），否则交付文档里会多出一串空行。
    """
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    tpl = re.sub(r"^[ \t]*\{\{\s*(\w+)\s*\}\}[ \t]*\r?\n",
                 lambda m: "" if str(kpis.get(m.group(1), "«missing»")) == "" else m.group(0),
                 tpl, flags=re.MULTILINE)
    missing = []

    def _sub(m: re.Match) -> str:
        key = m.group(1).strip()
        if key not in kpis:
            missing.append(key)
            return f"«{key}»"
        return str(kpis[key])

    out = re.sub(r"\{\{\s*(\w+)\s*\}\}", _sub, tpl)
    if missing:
        raise KeyError(f"模板占位符缺少 KPI：{sorted(set(missing))}")
    return out


def main() -> Path:
    kpis = collect_kpis()
    text = render(kpis)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"[report] 决策建议报告 -> {OUTPUT_PATH}")
    print(f"[report] 会员 {kpis['n_members']} 人 / GMV {kpis['gmv_wan']} 万元 / "
          f"规则 {kpis['n_rules']} 条 / 高危 {kpis['n_high_risk']} 人")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
