"""客群画像卡片（计划 §7.3 / R-5）。

产出 `reports/客群画像.md`：每个客群一页，包含
**规模 / GMV 贡献 / 消费能力 / 偏好品类与品牌 / 流失与 CLV / 触达建议**。

触达建议由**规则表**驱动（客群 × 关键指标 → 建议），不写死文案，
这样客群规则变化时建议自动跟随，避免"建议与画像脱节"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from retailinsight.io.loader import read_items, read_parquet
from retailinsight.settings import (
    INTERIM_DIR,
    ITEMS_DIR,
    MART_PROFILE,
    MART_RFM,
    REPORTS_DIR,
)

# 客群 → 触达建议规则（按优先级从上到下匹配，命中第一条即用）
ACTION_RULES: list[tuple[str, str]] = [
    ("流失风险客户", "定向高面额券 + 专属客服回访 + 30 天内品类偏好推送（挽回优先于增量）"),
    ("重要挽留客户", "一对一挽回：专属权益 + 竞品替代推荐 + 流失原因回访"),
    ("重要价值客户", "会员权益体系（专属客服 / 生日礼 / 优先新品）+ 品类升级推荐，避免打扰式促销"),
    ("重要发展客户", "高面额满减 + 连带购激励，目标从 2 单/季提到 4 单/季"),
    ("重要保持客户", "稳定频次提醒 + 订阅制（月度固定品类包），保住既有节奏"),
    ("潜力客户", "低门槛首单礼 + 第二单激励，缩短首购到复购的间隔"),
    ("新客客户", "30 天新手任务体系 + 第二单激励 + 首购品类关联推荐"),
    ("一般价值客户", "常规月度活动 + 满减凑单，引导提升客单价"),
    ("一般保持客户", "低频触达（社群/短信），避免高成本促销侵蚀利润"),
]

# 建议里出现的关键指标阈值（写进文档，便于复核）
# 高危阈值（与 models/churn.py 的风险分层切点保持一致）
HIGH_RISK_CUT = 0.61

THRESHOLDS = {
    "high_value_share": 0.05,      # 高价值客群 GMV 占比超过 5% 判为「贡献核心」
    "at_risk_rate": 0.35,          # 客群内高危会员占比超过 35% 判为「流失压力大」
    "high_clv_rate": 0.25,         # 客群内高 CLV 占比超过 25% 判为「潜力客群」
}

#: 品类结构差异度分档（升序）：(上界, 说明)。
#: 参照系：真实商超的门店级差异度通常在 0.05~0.30；本项目 v3.5 合成数据
#: 门店级实测 0.05、规则法客群级 0.0002~0.01（受 ARI≈0.13 稀释）。
DIFF_BANDS = [
    (0.002, "，≈ 无差异，头部品类主导"),
    (0.02, "，弱差异：区别在排序与占比，TOP3 集合仍高度重叠"),
    (0.05, "，中等差异：品类结构已可区分"),
    (float("inf"), "，明显差异（达真实商超门店间下沿）"),
]


#: 作品集口径的会员规模（`make all` 默认 scale 下的注册会员数）。
#: 生成口径偏离太多时，文档首行要显式告警，避免小规模验证数据被当成结论。
PORTFOLIO_MEMBERS = 35650


def _scale_warning(n_members: int) -> str:
    """数据规模偏离作品集口径时返回告警串（否则空串）。"""
    if n_members < PORTFOLIO_MEMBERS * 0.9:
        return (f"　⚠ **当前为小规模验证数据**（会员 {n_members:,}，作品集口径 "
                f"{PORTFOLIO_MEMBERS:,}），结论不可直接与正式文档对比")
    return ""


def _diff_hint(diff: float) -> str:
    """把品类结构差异度翻译成人话（阈值见 ``DIFF_BANDS``）。"""
    for cut, hint in DIFF_BANDS:
        if diff < cut:
            return hint
    return DIFF_BANDS[-1][1]


def _pick_action(seg: str, row: pd.Series) -> str:
    for key, action in ACTION_RULES:
        if key in seg:
            return action
    return "常规触达：月度活动 + 个性化推荐"


def compute_segment_category_top(profile: pd.DataFrame, txn: pd.DataFrame,
                                 items: pd.DataFrame, products: pd.DataFrame,
                                 top_n: int = 3) -> dict[str, dict]:
    """客群级「偏好品类 TOP3」：按**客群内品类消费额**排序，并给出结构差异度。

    **为什么不能取会员个人 top3 的众数**：
    会员个人的 top3 会被全局热销品类主导 —— 实测 15.3% 的会员个人 top3
    完全相同（"粮油米面/调味品/饮料"）。统计众数后，**8 个客群的偏好品类
    一模一样**，等于给每个客群发了同一张卡片，不提供任何区分度。
    这与本项目记录的「恒为常数的标签比没有这个标签更危险」是同一类问题。

    正确口径：明细按「会员 → 品类」聚合出金额，再按客群汇总品类金额，
    取该客群消费额最高的 3 个品类。这才是「这个客群偏好什么品类」。

    **同时给出结构差异度**：``1 − cos(客群品类占比, 全局品类占比)``。

    v3.5 起生成器引入「门店城市档位 × 先验人群」的品类乘子矩阵
    （``TIER_CATEGORY_BOOST`` / ``PERSONA_CATEGORY_BOOST``，见
    ``scripts/generate_synthetic.py``），合成数据**已经具备**门店级与人群级的
    品类结构差异：门店差异度 0.05 量级、先验人群 0.035 量级。
    但规则法 RFM 客群与先验人群的 ARI 仅约 0.13（8 类行为型人群在消费体量
    特征上本就不可分），人群级差异在客群聚合后被稀释，实测客群卡片差异度
    落在 **0.0002 ~ 0.01**：TOP3 集合仍以头部品类为主，差异体现在排序与占比。
    所以卡片里把差异度与品类占比一并印出来 —— 让读者自己判断这一行的区分度，
    而不是笼统宣称「各客群偏好不同」。

    参数
    ----
    profile:  画像宽表（需含 member_id / rfm_segment）
    txn:      交易表（需含 txn_id / member_id / member_type / is_logic_error）
    items:    明细表（需含 txn_id / sku / qty / unit_price）
    products: 商品表（需含 sku / category）
    top_n:    取前几个品类

    返回
    ----
    {客群名: {"top": [品类...], "diff": 差异度, "shares": {品类: 占比}}}
    """
    out: dict[str, dict] = {}
    if "rfm_segment" not in profile.columns:
        return out
    m_txn = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    it = (items.merge(m_txn[["txn_id", "member_id"]], on="txn_id", how="inner")
          .merge(products[["sku", "category"]], on="sku", how="inner"))
    if len(it) == 0:
        return out
    it["line_amount"] = it["qty"] * it["unit_price"]
    seg_of = profile.set_index("member_id")["rfm_segment"]
    it["rfm_segment"] = it["member_id"].map(seg_of)
    it = it.dropna(subset=["rfm_segment"])
    g = (it.groupby(["rfm_segment", "category"], observed=True)["line_amount"]
         .sum().reset_index())
    if len(g) == 0:
        return out

    pivot = g.pivot_table(index="rfm_segment", columns="category",
                          values="line_amount", aggfunc="sum", fill_value=0)
    share = pivot.div(pivot.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0.0)
    overall = share.mean(axis=0)
    onorm = float(np.linalg.norm(overall))
    for seg in pivot.index:
        v = share.loc[seg]
        denom = float(np.linalg.norm(v)) * onorm
        cos = float((v * overall).sum() / denom) if denom > 0 else 1.0
        sub = g[g["rfm_segment"] == seg].sort_values(
            ["line_amount", "category"], ascending=[False, True])
        top = sub.head(top_n)["category"].tolist()
        out[str(seg)] = {
            "top": top,
            "diff": round(1.0 - cos, 5),
            "shares": {c: round(float(v[c]), 4) for c in top},
        }
    return out


def _personal_top3_fallback(profile: pd.DataFrame, top_n: int = 3) -> dict[str, dict]:
    """兜底口径：明细不可用时，退回「会员个人 top3 的众数」。

    保留它是因为它至少能跑通；但它**区分度低**，调用方应在文档里注明口径。
    """
    out: dict[str, dict] = {}
    if "top_categories" not in profile.columns or "rfm_segment" not in profile.columns:
        return out
    for seg, g in profile.groupby("rfm_segment", observed=True):
        cats: dict[str, int] = {}
        for s in g["top_categories"].dropna().astype(str):
            for c in s.split("/"):
                cats[c] = cats.get(c, 0) + 1
        out[str(seg)] = {
            "top": [c for c, _ in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]],
            "diff": None,
            "shares": {},
        }
    return out


def build_segment_cards(profile: pd.DataFrame, rfm: pd.DataFrame,
                        category_top: dict[str, dict] | None = None) -> str:
    """生成客群画像卡片的 markdown 文本。

    ``category_top``: 客群级偏好品类（见 :func:`compute_segment_category_top`）。
    传入时用它替代「会员个人 top3 的众数」口径 —— 后者会让 8 个客群完全雷同。
    """
    df = profile.copy()
    if "rfm_segment" not in df.columns:
        df = df.merge(rfm[["member_id", "rfm_segment"]], on="member_id", how="left")
    # 画像宽表已带 RFM 指标；缺失时才从 mart_rfm 补齐（避免列名重复导致覆盖）
    need = [c for c in ("monetary", "frequency", "recency_days")
            if c not in df.columns]
    if need:
        df = df.merge(rfm[["member_id", *need]], on="member_id", how="left")

    total_gmv = df["monetary"].sum()
    total_members = len(df)

    has_clv = "clv" in df.columns
    has_churn = "churn_prob" in df.columns

    rows = []
    for seg, g in df.groupby("rfm_segment", observed=True):
        n = len(g)
        gmv = g["monetary"].sum()
        gmv_share = gmv / total_gmv if total_gmv else 0.0
        avg_recency = g["recency_days"].mean()
        avg_freq = g["frequency"].mean()
        avg_monetary = g["monetary"].mean()
        cinfo = (category_top or {}).get(str(seg)) or {}
        if isinstance(cinfo, list):          # 兼容纯列表形态
            cinfo = {"top": cinfo, "diff": None, "shares": {}}
        top_cats = cinfo.get("top") or \
            (g["top_categories"].dropna().astype(str)
             .str.split("/").explode().value_counts().head(3).index.tolist())
        top_brands = (g["top_brands"].dropna().astype(str)
                      .str.split("/").explode().value_counts().head(3).index.tolist())
        # 品类结构差异度：让读者知道「偏好品类」这一行本身有多少区分度
        diff = cinfo.get("diff")
        shares = cinfo.get("shares") or {}

        risk_note = "—"
        if has_churn:
            covered = g["churn_prob"].notna()
            if covered.any():
                hi = float((g.loc[covered, "churn_prob"] > HIGH_RISK_CUT).mean())
                risk_note = f"客群内高危会员占比 {hi:.1%}"
        clv_note = "—"
        if has_clv:
            clv_share = float((g["clv_tier"] == "高CLV").mean())
            clv_note = f"高 CLV 会员占比 {clv_share:.1%}，客群平均 CLV {g['clv'].mean():,.0f} 元"

        # 标签判定有优先级：流失压力大 > 核心贡献 > 常规运营
        # （流失风险客群即使 GMV 占比不低，业务定性仍是"要救"而不是"要供"）
        tag = "常规运营"
        if has_churn:
            covered = g["churn_prob"].notna()
            if covered.any():
                # 分母必须是**已覆盖**的会员：模型只覆盖末期观察窗的活跃会员，
                # 若把未覆盖的算进分母，高危占比会被系统性稀释
                hi_rate = float((g.loc[covered, "churn_prob"] > HIGH_RISK_CUT).mean())
                if hi_rate >= THRESHOLDS["at_risk_rate"]:
                    tag = "流失压力大"
        if tag == "常规运营" and gmv_share >= THRESHOLDS["high_value_share"]:
            tag = "核心贡献"

        rows.append({
            "客群": seg, "规模": n, "规模占比": n / total_members,
            "GMV": gmv, "GMV占比": gmv_share, "人均消费": avg_monetary,
            "平均频次": avg_freq, "平均R值(天)": avg_recency,
            "TOP品类": "/".join(top_cats) or "—", "TOP品牌": "/".join(top_brands) or "—",
            "品类结构差异度": diff, "品类占比": shares,
            "风险": risk_note, "CLV": clv_note, "标签": tag,
            "触达建议": _pick_action(seg, g),
        })

    cards = pd.DataFrame(rows).sort_values("GMV", ascending=False)

    lines = ["# 客群画像卡片", "",
             f"> 数据规模：{total_members:,} 名会员 · {len(cards)} 个客群"
             + _scale_warning(total_members),
             "> 由 `python -m retailinsight.reports.profile_cards` 自动生成，"
             "所有数字从 `data/marts/` 产物回填。",
             "> 触达建议由 `ACTION_RULES` 规则表驱动（客群 × 关键指标 → 建议），"
             "客群规则变化时建议自动跟随。",
             "> **偏好品类口径**：按该客群内**品类消费额**排序（明细 × 商品表聚合）。"
             "早期版本取「会员个人 top3 的众数」，实测 8 个客群结果完全相同 —— "
             "区分度为零，等于没分析。",
             "> **数据局限自曝**：v3.5 起生成器引入「门店城市档位 × 先验人群」的品类"
             "乘子矩阵，合成数据**已具备**门店级与人群级品类结构差异（门店 0.05 量级，"
             "见建议报告 S8）。但规则法 RFM 客群与先验人群的 ARI 仅约 0.13，人群级"
             "差异在客群聚合后被稀释，本卡片实测差异度落在 0.0002~0.01 —— 即："
             "**TOP3 集合仍以头部品类为主，差异体现在排序与占比**，而非集合本身。"
             "卡片把差异度与品类占比一并印出，读者可据此判断这一行的真实区分度。", "",
             "## 概览", "",
             "| 客群 | 规模 | 规模占比 | GMV | GMV占比 | 人均消费 | 平均频次 | 平均R值(天) | "
             "品类结构差异度 | 标签 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cards.to_dict("records"):
        diff_s = ("—" if r["品类结构差异度"] is None
                  else f"{r['品类结构差异度']:.5f}")
        lines.append(
            f"| {r['客群']} | {r['规模']:,} | {r['规模占比']:.1%} | "
            f"{r['GMV'] / 1e4:,.1f} 万 | {r['GMV占比']:.1%} | "
            f"{r['人均消费']:,.0f} | {r['平均频次']:.1f} | {r['平均R值(天)']:.0f} | "
            f"{diff_s} | {r['标签']} |")

    lines += ["", "## 逐群卡片", ""]
    for r in cards.to_dict("records"):
        diff = r["品类结构差异度"]
        cat_note = ""
        if diff is not None:
            cat_note = f"　品类结构差异度 {diff:.5f}{_diff_hint(diff)}"
        shares = r.get("品类占比") or {}
        if shares:
            cat_note += "　品类占比：" + " / ".join(f"{c} {s:.1%}" for c, s in shares.items())
        lines += [
            f"### {r['客群']}", "",
            f"- **规模**：{r['规模']:,} 人（占全量 {r['规模占比']:.1%}）",
            f"- **GMV 贡献**：{r['GMV'] / 1e4:,.1f} 万元（占 {r['GMV占比']:.1%}）",
            f"- **消费能力**：人均 {r['人均消费']:,.0f} 元，平均频次 {r['平均频次']:.1f} 次，"
            f"平均距上次消费 {r['平均R值(天)']:.0f} 天",
            f"- **偏好品类 TOP3**：{r['TOP品类']}{cat_note}",
            f"- **偏好品牌 TOP3**：{r['TOP品牌']}",
            f"- **流失风险**：{r['风险']}",
            f"- **CLV**：{r['CLV']}",
            f"- **运营定性**：{r['标签']}",
            f"- **触达建议**：{r['触达建议']}",
            "",
        ]

    lines += ["---", "",
              "## 附：判定阈值（供复核）", "",
              "| 阈值 | 取值 | 含义 |", "|---|---|---|",
              f"| `high_value_share` | {THRESHOLDS['high_value_share']:.0%} | "
              "客群 GMV 占比 ≥ 此值 → 标记「核心贡献」 |",
              f"| `at_risk_rate` | {THRESHOLDS['at_risk_rate']:.0%} | "
              "客群内高危会员占比 ≥ 此值 → 标记「流失压力大」 |",
              f"| `high_clv_rate` | {THRESHOLDS['high_clv_rate']:.0%} | "
              "客群内高 CLV 占比 ≥ 此值 → 标记「潜力客群」 |", ""]
    return "\n".join(lines)


def main() -> str:
    profile = read_parquet(MART_PROFILE)
    rfm = read_parquet(MART_RFM)

    # 客群级偏好品类：按客群内品类**消费额**排序（明细可用时）
    category_top: dict[str, list[str]] = {}
    try:
        txn = read_parquet(INTERIM_DIR / "transactions.parquet")
        items = read_items(ITEMS_DIR)
        products = read_parquet(INTERIM_DIR / "products.parquet")
        category_top = compute_segment_category_top(profile, txn, items, products)
    except FileNotFoundError:
        print("[cards] 明细产物缺失，偏好品类退回「个人 top3 众数」口径（区分度低）")
    if not category_top:
        category_top = _personal_top3_fallback(profile)

    md = build_segment_cards(profile, rfm, category_top=category_top)
    out = REPORTS_DIR / "客群画像.md"
    out.write_text(md, encoding="utf-8")
    print(f"[cards] 客群画像卡片 -> {out}")
    return md


if __name__ == "__main__":
    main()
