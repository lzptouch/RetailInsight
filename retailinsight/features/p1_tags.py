"""P1 扩展标签（计划 §7.3 的 20 个扩展标签）。

**为什么要单独一个模块**：`features/behavior.py` 管的是「行为时序」维度
（周末占比 / 折扣敏感度 / 季节性强弱），本模块管的是「商品结构」与「生命周期」
维度，两者口径互不重叠，合起来把 P1 的 20 个扩展标签补齐。

口径纪律
--------
1. 全部基于**会员且无逻辑错误**的交易（与 RFM / 画像层一致）。
2. 每个标签只在这里实现一次，画像宽表、聚类特征、看板都从这里取。
3. 比率类分母加 `1e-9` 防零除；空值填 0 而不是 NaN（画像宽表要能直接进模型）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from retailinsight.io.schema import FRESH_CATEGORIES


def _member_txn(txn: pd.DataFrame) -> pd.DataFrame:
    """会员且无逻辑错误的交易。"""
    return txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])].copy()


def _amount(txn: pd.DataFrame) -> pd.Series:
    """订单金额列：优先用 ``amount``，缺失时退回 ``net_amount``。

    canonical schema 里两者都有（``amount`` 是原价、``net_amount`` 是净额）；
    退回逻辑让本模块对只给净额的数据源也能工作，而不是直接 KeyError。
    """
    if "amount" in txn.columns:
        return txn["amount"]
    return txn["net_amount"]


def _prod_cols(products: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    """取商品表里**实际存在**的列；缺列时由调用方显式降级。

    这样本模块对「只有 sku+category 的精简商品表」也能跑（第三方数据源常这样），
    而不是直接 KeyError。
    """
    return products[[c for c in needed if c in products.columns]].copy()


def _member_items(items: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    """会员交易对应的明细（带 member_id / txn_ts / store_id / 金额）。"""
    m = _member_txn(txn)
    cols = ["txn_id", "member_id", "txn_ts", "store_id", "discount", "net_amount"]
    if "amount" in m.columns:
        cols.append("amount")
    keep = m[cols]
    out = items.merge(keep, on="txn_id", how="inner")
    if "amount" not in out.columns:
        out["amount"] = out["net_amount"]
    return out


# ---------------------------------------------------------------- 促销相关

def promo_response_rate(txn: pd.DataFrame) -> pd.Series:
    """促销响应率：有折扣的订单数 / 全部订单数。

    与 `discount_sensitivity`（折扣额占比）互补：前者看「动不动就上钩」，
    后者看「让掉的利有多少」。
    """
    m = _member_txn(txn)
    total = m.groupby("member_id").size()
    hit = m[m["discount"] > 0].groupby("member_id").size()
    return (hit / total).rename("promo_response_rate").fillna(0.0).round(4)


def coupon_used_rate(txn: pd.DataFrame, small_coupon: float = 10.0) -> pd.Series:
    """券使用率：**小额券**订单（折扣额 ≤ 10 元）占**有折扣订单**的比例。

    区分「用券凑单」与「整单打折」：前者是价格敏感但仍要买，
    后者是真促销驱动 —— 两类会员的运营策略完全不同。

    口径用**绝对金额**而不是折扣占比：实测折扣额/订单额的中位数是 0、
    90 分位也只有 20.5%，按占比切会退化成恒等于 1（没有区分度）。
    """
    m = _member_txn(txn)
    d = m[m["discount"] > 0].copy()
    if len(d) == 0:
        return pd.Series(dtype="float64", name="coupon_used_rate")
    small = d["discount"] <= small_coupon
    used = d[small].groupby("member_id").size()
    total = d.groupby("member_id").size()
    return (used / total).rename("coupon_used_rate").fillna(0.0).round(4)


def price_elasticity(txn: pd.DataFrame) -> pd.Series:
    """价格弹性（折扣依赖度）：有折扣订单的平均折扣额 / 该会员平均订单金额。

    越接近 1，说明这个会员的购买越是「靠打折驱动」；
    越接近 0，说明是刚需/习惯性消费，对价格不敏感。

    分母用**中位订单额**而不是均值：少数极小额订单会把均值拉低，
    使比值爆到几百，失去意义。
    """
    m = _member_txn(txn)
    g = m.groupby("member_id")
    med_amt = g.apply(lambda x: _amount(x).median(), include_groups=False).abs()
    disc = m[m["discount"] > 0].groupby("member_id")["discount"].mean()
    out = (disc / (med_amt + 1e-9)).rename("price_elasticity")
    return out.clip(0.0, 1.0).fillna(0.0).round(4)


# ---------------------------------------------------------------- 门店相关

def store_count(txn: pd.DataFrame) -> pd.Series:
    """跨店消费：消费过的门店数。"""
    m = _member_txn(txn)
    return m.groupby("member_id")["store_id"].nunique().rename("store_cnt").astype("int16")


def store_concentration(txn: pd.DataFrame) -> pd.Series:
    """门店集中度：最大门店的消费额占比（1 = 只去一家店，0 = 很分散）。"""
    m = _member_txn(txn)
    if len(m) == 0:
        return pd.Series(dtype="float64", name="store_concentration")
    g = m.groupby(["member_id", "store_id"])["net_amount"].sum().abs().reset_index()
    tot = g.groupby("member_id")["net_amount"].sum().replace(0, np.nan)
    top = g.groupby("member_id")["net_amount"].max()
    return (top / tot).rename("store_concentration").fillna(0.0).round(4)


# ---------------------------------------------------------------- 商品结构

def fresh_ratio(mi: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """生鲜占比：生鲜品类消费额占比。"""
    fresh = set(FRESH_CATEGORIES)
    prod = _prod_cols(products, ["sku", "category"])
    if "category" not in prod.columns:
        out = pd.Series(0.0, index=mi["member_id"].unique(), name="fresh_ratio")
        out.attrs["degraded"] = "商品表缺少 category 字段，生鲜占比按 0 处理"
        return out
    prod["is_fresh"] = prod["category"].isin(fresh).astype("float32")
    d = mi.merge(prod, on="sku", how="left")
    d["fresh_amt"] = d["qty"] * d["unit_price"] * d["is_fresh"].fillna(0)
    g = d.groupby("member_id")
    return (g["fresh_amt"].sum() / (g["qty"].sum() * g["unit_price"].sum() + 1e-9)) \
        .rename("fresh_ratio").fillna(0.0).round(4)


def import_preference(mi: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """进口偏好：进口商品消费额占比。

    商品表若没有 ``is_import`` 字段（老产物 / 第三方数据源），显式降级为全 0，
    并在返回值的 attrs 里标注 —— **不静默当成"没有进口商品"**。
    """
    if "is_import" not in products.columns:
        out = pd.Series(0.0, index=mi["member_id"].unique(), name="import_preference")
        out.attrs["degraded"] = "商品表缺少 is_import 字段，进口偏好按 0 处理"
        return out
    prod = products[["sku", "is_import"]].copy()
    prod["is_import"] = prod["is_import"].astype("float32")
    d = mi.merge(prod, on="sku", how="left")
    d["imp_amt"] = d["qty"] * d["unit_price"] * d["is_import"].fillna(0)
    g = d.groupby("member_id")
    return (g["imp_amt"].sum() / (g["qty"].sum() * g["unit_price"].sum() + 1e-9)) \
        .rename("import_preference").fillna(0.0).round(4)


def brand_loyalty(mi: pd.DataFrame, products: pd.DataFrame) -> pd.Series:
    """品牌忠诚度：购买金额最大的品牌所占份额（越高越"只认一个牌子"）。"""
    prod = _prod_cols(products, ["sku", "brand"])
    if "brand" not in prod.columns:
        out = pd.Series(0.0, index=mi["member_id"].unique(), name="brand_loyalty")
        out.attrs["degraded"] = "商品表缺少 brand 字段，品牌忠诚度按 0 处理"
        return out
    d = mi.merge(prod, on="sku", how="left")
    if len(d) == 0:
        return pd.Series(dtype="float64", name="brand_loyalty")
    d["line_amt"] = d["qty"] * d["unit_price"]
    g = d.groupby(["member_id", "brand"])["line_amt"].sum().reset_index()
    tot = g.groupby("member_id")["line_amt"].sum().replace(0, np.nan)
    top = g.groupby("member_id")["line_amt"].max()
    return (top / tot).rename("brand_loyalty").fillna(0.0).round(4)


def new_product_rate(mi: pd.DataFrame) -> pd.Series:
    """新品尝鲜率：所购 SKU 中「全局首次成交日晚于该会员首单日」的件数占比。

    口径：SKU 的全局首次成交日 vs 会员自己的首单日 —— 后者晚于前者说明
    这个会员开始消费之后，商品才在门店出现，他买的是当季新品。
    """
    if len(mi) == 0:
        return pd.Series(dtype="float64", name="new_product_rate")
    d = mi.copy()
    d["txn_day"] = d["txn_ts"].dt.normalize()
    sku_first = d.groupby("sku")["txn_day"].min()
    mem_first = d.groupby("member_id")["txn_day"].min()
    d["sku_first"] = d["sku"].map(sku_first)
    d["mem_first"] = d["member_id"].map(mem_first)
    d["is_new"] = (d["sku_first"] > d["mem_first"]).astype("float32")
    g = d.groupby("member_id")
    return (g["is_new"].sum() / (g["qty"].sum() + 1e-9)) \
        .rename("new_product_rate").fillna(0.0).round(4)


# ---------------------------------------------------------------- 购买节奏

def _intervals(m: pd.DataFrame) -> pd.Series:
    """每个会员相邻两次购买日期间隔（天），index=member_id。

    同一天的多单合并为一次购买（与 RFM 的 frequency 口径一致）。
    """
    s = m.sort_values(["member_id", "txn_ts"]).copy()
    s["txn_day"] = s["txn_ts"].dt.normalize()
    s = s.drop_duplicates(["member_id", "txn_day"])
    s["gap"] = s.groupby("member_id")["txn_day"].diff().dt.days
    out = s.dropna(subset=["gap"]).set_index("member_id")["gap"]
    return out.astype("float64")


def repurchase_cycle_days(txn: pd.DataFrame) -> pd.Series:
    """复购周期中位数：相邻两次购买间隔（天）的中位数。"""
    m = _member_txn(txn)
    iv = _intervals(m)
    return iv.groupby("member_id").median().rename("repurchase_cycle_days").fillna(0.0)


def interval_std(txn: pd.DataFrame) -> pd.Series:
    """购买间隔方差：间隔的标准差（越大说明购买越没规律）。"""
    m = _member_txn(txn)
    iv = _intervals(m)
    return iv.groupby("member_id").std().rename("interval_std").fillna(0.0).round(4)


def silence_ratio(txn: pd.DataFrame) -> pd.Series:
    """沉默占比：距基准日的天数 / 观察窗长度（0 = 刚买过，1 = 整窗没买）。"""
    m = _member_txn(txn)
    if len(m) == 0:
        return pd.Series(dtype="float64", name="silence_ratio")
    end = m["txn_ts"].max().normalize() + pd.Timedelta(days=1)
    start = m["txn_ts"].min().normalize()
    span = max((end - start).days, 1)
    last = m.groupby("member_id")["txn_ts"].max().dt.normalize()
    recency = (end - last).dt.days
    return (recency / span).rename("silence_ratio").fillna(0.0).round(4)


# ---------------------------------------------------------------- 结构与偏好

def cross_category_ratio(txn: pd.DataFrame, items: pd.DataFrame,
                         products: pd.DataFrame) -> pd.Series:
    """关联品倾向：单次订单的平均品类数 / 该会员买过的品类总数（0~1）。

    高 = 每次下单都很杂（爱一次买多品类，有「搭配习惯」）；
    低 = 每次只买一个品类（刚需补货型）。
    """
    prod = _prod_cols(products, ["sku", "category"])
    m_txn = _member_txn(txn)[["txn_id", "member_id"]]
    member_ids = m_txn["member_id"].unique()
    if "category" not in prod.columns:
        out = pd.Series(0.0, index=member_ids, name="cross_category_ratio")
        out.attrs["degraded"] = "商品表缺少 category 字段，关联品倾向按 0 处理"
        return out
    d = items.merge(prod, on="sku", how="left").merge(m_txn, on="txn_id", how="inner")
    if len(d) == 0:
        return pd.Series(0.0, index=member_ids, name="cross_category_ratio")
    per_order = d.groupby(["member_id", "txn_id"])["category"].nunique()
    avg_cats = per_order.groupby("member_id").mean()
    total_cats = d.groupby("member_id")["category"].nunique().clip(lower=1)
    return (avg_cats / total_cats).rename("cross_category_ratio").clip(0, 1).round(4)


def basket_structure(txn: pd.DataFrame) -> pd.Series:
    """客单结构：订单金额的变异系数（std / mean），反映大额单与小额单的混合度。

    用**中位金额**做分母，避免个别极小额订单把变异系数放大到几百。
    """
    m = _member_txn(txn)
    g = m.groupby("member_id").apply(lambda x: _amount(x).std(), include_groups=False)
    med = m.groupby("member_id").apply(
        lambda x: _amount(x).median(), include_groups=False).abs().replace(0, np.nan)
    cv = (g / med).fillna(0.0)
    return cv.rename("basket_structure").clip(0.0, 10.0).round(4)


def channel_preference(txn: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """渠道偏好：平均单次件数 / 该会员最大单次件数（1 = 每次都买很多，偏囤货/计划采购）。

    **必须基于明细表**：交易表每笔只有一行，用它算件数会恒等于 1。
    """
    m = _member_txn(txn)[["txn_id", "member_id"]]
    d = items.merge(m, on="txn_id", how="inner")
    if len(d) == 0:
        return pd.Series(dtype="float64", name="channel_preference")
    sizes = d.groupby(["member_id", "txn_id"]).size()
    avg = sizes.groupby("member_id").mean()
    mx = sizes.groupby("member_id").max()
    return (avg / mx).rename("channel_preference").round(4)


def price_band(txn: pd.DataFrame) -> pd.Series:
    """价格带：客单价在全体会员中的分位（0~1，越高越偏贵）。"""
    m = _member_txn(txn)
    amt = m.groupby("member_id").apply(lambda x: _amount(x).sum(), include_groups=False).abs()
    aov = amt / m.groupby("member_id").size()
    if len(aov) == 0:
        return pd.Series(dtype="float64", name="price_band")
    return aov.rank(pct=True).rename("price_band").round(4)


# ---------------------------------------------------------------- 生命周期

def lifecycle_stage(txn: pd.DataFrame, rfm: pd.DataFrame) -> pd.Series:
    """生命周期阶段（序数编码 1~5）。

    | 1 新客 | 2 成长 | 3 成熟 | 4 衰退 | 5 沉睡 |
    规则：按 recency 排优先级，再用 frequency 区分新客/成长与成熟/衰退。
    """
    if rfm is None or len(rfm) == 0:
        return pd.Series(dtype="float64", name="lifecycle_stage")
    df = rfm[["member_id", "recency_days", "frequency"]].copy()
    span = max(float(df["recency_days"].max()), 1.0)
    df["r_pct"] = df["recency_days"] / span
    f_max = max(float(df["frequency"].max()), 1.0)
    df["f_pct"] = df["frequency"] / f_max
    stage = pd.Series(3, index=df.index, dtype="int8")   # 成熟
    stage[df["f_pct"] < 0.25] = 1                       # 新客
    stage[(df["f_pct"] >= 0.25) & (df["f_pct"] < 0.6) & (df["r_pct"] < 0.5)] = 2
    stage[(df["f_pct"] >= 0.6) & (df["r_pct"] >= 0.6)] = 4   # 衰退
    stage[df["r_pct"] >= 0.8] = 5                        # 沉睡
    return stage.rename("lifecycle_stage").astype("int8")


def potential_index(rfm: pd.DataFrame) -> pd.Series:
    """潜力指数：当前贡献一般、但消费频次不低的会员（0~1，越高越有潜力）。

    公式：`0.5 × f 分位 + 0.3 × m 分位 + 0.2 × (1 - r 分位)`，
    再按「m 分位低于 0.6」加权 —— **贡献还没起来**的会员才是"潜力"，
    否则高价值客户会把自己的潜力指数刷满。
    """
    if rfm is None or len(rfm) == 0:
        return pd.Series(dtype="float64", name="potential_index")
    df = rfm[["recency_days", "frequency", "monetary"]].copy()
    f_pct = df["frequency"].rank(pct=True)
    m_pct = df["monetary"].rank(pct=True)
    r_pct = df["recency_days"].rank(pct=True, ascending=False)  # 越小越好
    base = 0.5 * f_pct + 0.3 * m_pct + 0.2 * r_pct
    boost = np.where(m_pct < 0.6, 1.15, 1.0)   # 贡献还没起来的加权
    return (base * boost).rename("potential_index").round(4)


# ---------------------------------------------------------------- 汇总

def build_p1_tags(txn: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame,
                   rfm: pd.DataFrame | None = None) -> pd.DataFrame:
    """一次性产出 P1 扩展标签（index=member_id）。

    已实现的 20 个 P1 扩展标签里，本模块负责 **17 个**；
    另外 3 个（周末比 / 折扣敏感度 / 季节性强弱）在 `features/behavior.py`，
    CLV 在 `models/clv.py`。
    """
    mi = _member_items(items, txn)
    tags = pd.DataFrame(index=pd.Index([], name="member_id"))
    series_list = [
        promo_response_rate(txn),
        coupon_used_rate(txn),
        price_elasticity(txn),
        store_count(txn),
        store_concentration(txn),
        fresh_ratio(mi, products),
        import_preference(mi, products),
        brand_loyalty(mi, products),
        new_product_rate(mi),
        repurchase_cycle_days(txn),
        interval_std(txn),
        silence_ratio(txn),
        cross_category_ratio(txn, items, products),
        basket_structure(txn),
        channel_preference(txn, items),
        price_band(txn),
        lifecycle_stage(txn, rfm),
        potential_index(rfm),
    ]
    for s in series_list:
        if s is not None and len(s):
            tags[s.name] = s
    return tags
