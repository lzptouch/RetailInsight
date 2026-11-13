"""悦鲜生活超市 合成数据生成器（P0 主数据源）。

设计要点（对应计划 §4.2）：
- 8 类隐式人群埋点：行为差异可验证聚类有效性
- 4 组购物篮强关联埋点：可验证关联规则挖掘
- 故意注入脏数据：重复行 0.5% / 会员ID缺失 3% / 退货负金额 1% / 品类别名 2% / 逻辑冲突 0.5%
- 全向量化的 numpy 生成，scale=1（12万会员/约130万交易）在本机秒级完成
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retailinsight.io.loader import (
    append_partitioned,
    concat_parquet_parts,
    derive_txn_month,
    write_parquet,
    write_partitioned,
)
from retailinsight.io.schema import (
    AGE_BAND_WEIGHTS,
    AGE_BANDS,
    ANCHOR_COMBOS,
    CATEGORIES,
    CATEGORY_ALIASES,
    CITY_TIER_WEIGHTS,
    CITY_TIERS,
    IMPORT_SKU_RATIO,
    PERSONAS,
    SAME_CATEGORY_COMBOS,
    SKU_ANCHOR_PAIRS_PER_CATEGORY,
)
from retailinsight.settings import DATA_END, DATA_START, RAW_DIR, RAW_ITEMS_DIR, RAW_ITEMS_FILE, SEED

START = pd.Timestamp(DATA_START)
END = pd.Timestamp(DATA_END)
N_DAYS = (END - START).days + 1

CITIES = [("上海", "一线"), ("北京", "一线"), ("广州", "一线"), ("深圳", "一线"), ("成都", "一线"),
          ("杭州", "二线"), ("武汉", "二线"), ("南京", "二线"), ("西安", "二线"), ("苏州", "二线"),
          ("绵阳", "三线"), ("洛阳", "三线"), ("宜昌", "三线"), ("柳州", "三线"), ("保定", "三线")]


#: 区位驱动的**门店品类结构差异**（v3.5）。
#: 此前 15 家店共用同一套品类权重 → 建议报告 S8 实测「门店品类结构差异度 < 0.001」，
#: 门店级差异化策略**无法被量化验证**。这里按城市档位给品类加乘子：
#: 一线偏即食/进口/现制，三四线偏粮油/调味/腌腊（符合真实商超规律）。
TIER_CATEGORY_BOOST: dict[str, dict[str, float]] = {
    "一线": {
        "进口食品": 2.2, "咖啡茶饮": 1.9, "现制饮品": 1.9, "面包糕点": 1.6,
        "熟食卤味": 1.6, "方便速食": 1.5, "乳制品": 1.4, "化妆品": 1.3,
        "水果": 1.25, "饮料": 1.2,
        "粮油米面": 0.45, "腌腊制品": 0.5, "调味品": 0.6, "纸尿裤": 0.7,
    },
    "二线": {
        "乳制品": 1.15, "面包糕点": 1.1, "休闲零食": 1.1, "方便速食": 1.05,
        "粮油米面": 0.9, "调味品": 0.95, "进口食品": 0.85, "咖啡茶饮": 0.9,
    },
    "三线": {
        "粮油米面": 1.9, "腌腊制品": 1.8, "调味品": 1.5, "南北干货": 1.5,
        "白酒红酒": 1.4, "纸品": 1.35, "冷冻食品": 1.3, "休闲零食": 1.25,
        "进口食品": 0.4, "咖啡茶饮": 0.5, "现制饮品": 0.5, "化妆品": 0.55,
        "熟食卤味": 0.7,
    },
}


#: **人群级**品类偏好（v3.5）。此前各客群品类结构差异度仅 0.00002~0.00158，
#: 画像卡片的「偏好品类 TOP3」8 个客群写的都是「饮料/乳制品/调味品」——
#: 这一行永远相同，等于没有信息量。这里按人群行为给品类加乘子。
PERSONA_CATEGORY_BOOST: dict[str, dict[str, float]] = {
    "高频高值": {"进口食品": 1.8, "牛羊肉": 1.6, "水产海鲜": 1.6, "水果": 1.4,
                 "白酒红酒": 1.4, "化妆品": 1.2, "粮油米面": 0.7},
    "周末囤货": {"纸品": 1.6, "粮油米面": 1.5, "清洁用品": 1.5, "冷冻食品": 1.4,
                 "纸尿裤": 1.3, "熟食卤味": 0.7},
    "促销敏感": {"休闲零食": 1.4, "饮料": 1.3, "纸品": 1.3, "糖果巧克力": 1.3,
                 "进口食品": 0.6, "牛羊肉": 0.7},
    "新客尝鲜": {"进口食品": 1.5, "现制饮品": 1.5, "烘焙原料": 1.4, "休闲零食": 1.2,
                 "粮油米面": 0.7},
    "高频低额": {"方便速食": 1.6, "面包糕点": 1.5, "饮料": 1.4, "熟食卤味": 1.3,
                 "粮油米面": 0.6, "纸品": 0.75},
    "季节性": {"水果": 1.6, "饮料": 1.5, "冷冻食品": 1.3, "啤酒": 1.3,
               "纸品": 0.8},
    "沉默流失": {},
    "一次性": {},
}

#: 乘子的极化指数（采样时取幂）。
#: k=1 时全量实测门店品类结构差异度仅 **0.015** —— 头部共性品类（粮油/纸品）
#: 把余弦距离稀释掉了；k=2 时实测 **平均 0.052 / 最高 0.116**，
#: 落在真实商超 0.05~0.30 区间的下沿，S8 因此从「无法验证」变为「可量化验证」。
TIER_BOOST_POWER = 2.5
#: 人群乘子的极化指数。幂 1 时先验人群的品类差异度仅 0.003~0.009（被 Zipf 头部品类稀释），
#: 幂 3 时达 **0.035**，与门店级（0.050）同量级，画像的「偏好品类 TOP3」才真正分化。
#: 注意：客群还原受 ARI 上限（≈0.13）约束，规则法客群看到的差异度会**明显小于**这里的先验值。
PERSONA_BOOST_POWER = 3.0


TIER_NAMES = ["一线", "二线", "三线"]
TIER_CODE = {name: i for i, name in enumerate(TIER_NAMES)}


def _build_boost_matrix(persona_names: np.ndarray) -> np.ndarray:
    """形状 (档位数, 人群数, 品类数) 的品类权重乘子矩阵。"""
    n_tier, n_persona, n_cat = len(TIER_NAMES), len(persona_names), len(CATEGORIES)
    mat = np.ones((n_tier, n_persona, n_cat), dtype=np.float64)
    for ti, tier in enumerate(TIER_NAMES):
        tier_boost = TIER_CATEGORY_BOOST.get(tier, {})
        for pi, persona in enumerate(persona_names):
            persona_boost = PERSONA_CATEGORY_BOOST.get(str(persona), {})
            for ci, cat in enumerate(CATEGORIES):
                mat[ti, pi, ci] = (tier_boost.get(cat, 1.0) ** TIER_BOOST_POWER) * \
                    (persona_boost.get(cat, 1.0) ** PERSONA_BOOST_POWER)
    return mat


def _sample_with_cdf(cdf: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.searchsorted(cdf, rng.random(n), side="right")


# 季节性月度权重：春节(1-2月)、618(6月)、双11/年货节(11-12月)为高峰，3-4月为低谷
MONTH_WEIGHTS = np.array([1.30, 1.35, 0.85, 0.85, 1.00, 1.10, 0.95, 0.95, 1.00, 1.05, 1.30, 1.35])


def _hour_cdf(kind: str) -> np.ndarray:
    """小时分布：peak=白天+晚高峰；day=白天；uniform=均匀。"""
    w = np.ones(24, dtype=float)
    if kind == "peak":
        w[[9, 10, 11, 17, 18, 19, 20]] = 4.0
    elif kind == "day":
        w[8:18] = 3.0
    cdf = np.cumsum(w)
    return cdf / cdf[-1]


def generate_members(n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(PERSONAS)
    weights = np.array([PERSONAS[p]["weight"] for p in names])
    persona = rng.choice(names, size=n, p=weights / weights.sum())
    life = np.array([PERSONAS[p]["life"] for p in persona])

    # 注册日期：新客尝鲜集中在最近 6 个月，其余全期均匀
    reg_day = rng.integers(0, N_DAYS, size=n)
    is_new = life == "new"
    reg_day[is_new] = rng.integers(N_DAYS - 183, N_DAYS, size=int(is_new.sum()))

    # 生命周期终点（stop_day）：stable/new 基本不停；churn 人群中途流失；once 一两次后停
    stop_day = np.full(n, N_DAYS - 1, dtype=np.int64)
    churn_mask = life == "churn"
    stop_day[churn_mask] = rng.integers(int(N_DAYS * 0.35), int(N_DAYS * 0.97), size=int(churn_mask.sum()))
    once_mask = life == "once"
    stop_day[once_mask] = reg_day[once_mask] + rng.integers(0, 30, size=int(once_mask.sum()))
    # stable/new 也有 5% 自然流失
    quit_mask = (life == "stable") & (rng.random(n) < 0.05)
    stop_day[quit_mask] = rng.integers(int(N_DAYS * 0.5), N_DAYS, size=int(quit_mask.sum()))
    stop_day = np.clip(np.maximum(stop_day, reg_day), 0, N_DAYS - 1)

    return pd.DataFrame({
        "member_id": np.arange(1, n + 1, dtype=np.int64),
        "reg_date": START + pd.to_timedelta(reg_day, unit="D"),
        "gender": rng.choice(["男", "女", "未知"], size=n, p=[0.44, 0.52, 0.04]),
        "age_band": rng.choice(AGE_BANDS, size=n, p=AGE_BAND_WEIGHTS),
        "city_tier": rng.choice(CITY_TIERS, size=n, p=CITY_TIER_WEIGHTS),
        "store_id": rng.integers(1, 16, size=n),
        # 先验人群仅用于聚类效果对照，不参与任何特征工程
        "_persona": persona,
        "_reg_day": reg_day,
        "_stop_day": stop_day,
    })


def generate_stores() -> pd.DataFrame:
    rows = []
    for i, (city, tier) in enumerate(CITIES, start=1):
        rows.append(dict(store_id=i, city=city, city_tier=tier,
                         area_m2=int(3000 + 500 * (i % 5)),
                         open_date=START - pd.Timedelta(days=365 + i * 30)))
    return pd.DataFrame(rows)


def generate_products(n_sku: int, rng: np.random.Generator) -> pd.DataFrame:
    """5000 SKU：品类(60)→子类→品牌 三级；2% 品类名注入别名脏数据。"""
    cat_ids = np.arange(len(CATEGORIES))
    sku_cat = np.resize(cat_ids, n_sku)
    rng.shuffle(sku_cat)
    # 品类基础价格（lognormal，量级区分）
    cat_base = np.round(np.exp(rng.normal(np.log(25), 0.9, size=len(CATEGORIES))), 1)
    base_price = np.round(cat_base[sku_cat] * np.exp(rng.normal(0, 0.25, size=n_sku)), 2)
    brands = np.array([f"品牌{chr(65 + i % 26)}{i % 100:02d}" for i in range(300)])
    df = pd.DataFrame({
        "sku": np.arange(100001, 100001 + n_sku),
        "category": np.array(CATEGORIES, dtype=object)[sku_cat],
        "sub_category": [f"{CATEGORIES[c]}-{i % 4 + 1}"
                         for c, i in zip(sku_cat, range(n_sku), strict=False)],
        "brand": rng.choice(brands, size=n_sku),
        "base_price": base_price,
        # 进口标记（计划 §7.3 的「进口偏好」标签需要这个商品属性）
        "is_import": (rng.random(n_sku) < IMPORT_SKU_RATIO).astype("int8"),
    })
    # 脏数据：2% 的 SKU 品类名写成别名（如「清洁 用品」）
    dirty_idx = rng.random(n_sku) < 0.02
    alias_cycle = list(CATEGORY_ALIASES.items())
    for j, idx in enumerate(np.where(dirty_idx)[0]):
        alias, canonical = alias_cycle[j % len(alias_cycle)]
        if df.at[idx, "category"] == canonical:
            df.at[idx, "category"] = alias
    return df, cat_base


def generate_transactions(members: pd.DataFrame, rng: np.random.Generator,
                          txn_id_offset: int = 0) -> pd.DataFrame:
    """按人群行为生成交易：频次/时段/周末偏好/生命周期衰减。

    ``txn_id_offset``：分块生成时保证全局 ``txn_id`` 唯一（块内从 1 开始）。
    """
    m = members
    n = len(m)
    persona = m["_persona"].to_numpy()
    freq = np.array([PERSONAS[p]["freq"] for p in persona], dtype=float)
    reg_day = m["_reg_day"].to_numpy()
    stop_day = m["_stop_day"].to_numpy()

    active_days = np.maximum(stop_day - reg_day, 1)
    lam = freq * (active_days / 365.0)
    n_txn = rng.poisson(lam) + 1

    member_idx = np.repeat(np.arange(n), n_txn)
    t = len(member_idx)
    reg = reg_day[member_idx]
    stop = stop_day[member_idx]

    # 交易日期：活跃窗内采样；churn 人群临近流失点时频次衰减（制造可学习信号）
    u = rng.random(t)
    life = np.array([PERSONAS[p]["life"] for p in persona])[member_idx]
    is_churn = life == "churn"
    u[is_churn] = u[is_churn] ** 1.6  # 偏向活跃窗前段，后段变稀
    day = reg + (u * np.maximum(stop - reg, 1)).astype(np.int64)
    day = np.minimum(day, stop)

    # 周末偏好：按人群加权重采样
    weekend_w = np.array([PERSONAS[p]["weekend"] for p in persona])[member_idx]
    is_wknd = (START + pd.to_timedelta(day, unit="D")).day_of_week.to_numpy() >= 5
    heavy = weekend_w > 1.5
    accept = np.where(is_wknd, 1.0, 1.0 / np.maximum(weekend_w, 1e-9))
    reject = heavy & (rng.random(t) > accept)
    day[reject] = np.minimum(day[reject] + rng.choice([5, 6], size=int(reject.sum())), stop[reject])

    # 季节性：按月度权重拒绝采样（3 轮近似），使月度/季度分析有真实季节结构
    max_w = MONTH_WEIGHTS.max()
    for _ in range(3):
        month = (START + pd.to_timedelta(day, unit="D")).month.to_numpy()
        rej = rng.random(t) > (MONTH_WEIGHTS[month - 1] / max_w)
        if not rej.any():
            break
        u2 = rng.random(t)
        day[rej] = np.minimum(
            reg[rej] + (u2[rej] * np.maximum(stop[rej] - reg[rej], 1)).astype(np.int64),
            stop[rej])

    # 时段：peak / day 两类小时分布
    cdf_peak, cdf_day = _hour_cdf("peak"), _hour_cdf("day")
    hour = _sample_with_cdf(cdf_peak, t, rng)
    day_persona = np.isin(persona[member_idx], ["周末囤货"])
    hour[day_persona] = _sample_with_cdf(cdf_day, int(day_persona.sum()), rng)

    txn_ts = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") \
        + pd.to_timedelta(rng.integers(0, 60, size=t), unit="m")

    pm = np.array([PERSONAS[p]["price_mult"] for p in persona])[member_idx]
    dc = np.array([PERSONAS[p]["discount"] for p in persona])[member_idx]
    basket_value = np.round(pm * np.exp(rng.normal(np.log(120), 0.55, size=t)), 2)
    discount = np.where(rng.random(t) < dc, np.round(basket_value * rng.uniform(0.05, 0.3, size=t), 2), 0.0)

    # ---- 跨店购物埋点 ----
    # 原实现里每笔交易都发生在会员的「主门店」，导致「跨店消费」「门店集中度」
    # 两个标签恒为 1 —— 有标签但没有区分度。现实里会员确实会去就近的其它门店，
    # 这里给一部分会员加「跨店购物」习惯（比例按人群区分）。
    home_store = m["store_id"].to_numpy()[member_idx]
    n_stores = 15
    cross_prob = np.array([PERSONAS[p]["cross_store"] for p in persona])[member_idx]
    cross_hit = rng.random(t) < cross_prob
    # 去邻近门店：相邻门店 id（1..15 环形相邻），保证仍在同一批门店里
    offset = rng.choice([-1, 1], size=t)
    alt_store = ((home_store - 1 + offset) % n_stores + 1).astype(np.int64)
    store_id = np.where(cross_hit, alt_store, home_store)

    txn = pd.DataFrame({
        "txn_id": np.arange(txn_id_offset + 1, txn_id_offset + t + 1, dtype=np.int64),
        "member_id": m["member_id"].to_numpy()[member_idx],
        "store_id": store_id,
        "txn_ts": txn_ts,
        "amount": basket_value,
        "discount": discount,
    })
    return txn, member_idx, persona


def generate_items(txn: pd.DataFrame, persona_of_member_txn: np.ndarray,
                   persona_names: np.ndarray, products: pd.DataFrame,
                   cat_base: np.ndarray, rng: np.random.Generator,
                   store_tier: np.ndarray | None = None) -> pd.DataFrame:
    """生成交易明细：件数按人群，品类按 Zipf 热度 × **门店区位乘子**，4 组锚点组合注入。

    ``store_tier``：长度 = 门店数，第 i 个元素是 ``store_id = i+1`` 的城市档位。
    给了它，品类分布就按门店所在档位偏移 —— 这是「门店级品类策略可量化验证」的前提。
    """
    t = len(txn)
    basket_mu = np.array([PERSONAS[p]["basket"] for p in persona_names[persona_of_member_txn]])
    k = rng.poisson(np.maximum(basket_mu - 1, 1)) + 1  # 至少 1 件
    k = np.minimum(k, 30)

    # 锚点组合：被选中的交易强制前两件为锚点品类
    n_cat = len(CATEGORIES)
    cat_of = np.empty(int(k.sum()), dtype=np.int64)
    txn_pos = np.repeat(np.arange(t), k)
    starts = np.zeros(t, dtype=np.int64)
    np.cumsum(k[:-1], out=starts[1:])

    # 默认：Zipf 热度采样品类
    zipf_w = 1.0 / np.arange(1, n_cat + 1) ** 1.1
    # 双维乘子：门店档位（区位）× 人群（行为）。两者都缺席时退化为全局 Zipf。
    boost_mat = _build_boost_matrix(persona_names)
    persona_of_item = persona_of_member_txn[txn_pos]
    if store_tier is None:
        tier_code = np.zeros(len(cat_of), dtype=np.int8)
    else:
        # 门店档位 → 整数编码（只对 15 家店做一次转换，再向量化索引到每件商品）
        tier_of_store = np.array([TIER_CODE.get(str(t), 0) for t in np.asarray(store_tier)],
                                 dtype=np.int8)
        store_of_item = txn["store_id"].to_numpy()[txn_pos]
        tier_code = tier_of_store[store_of_item - 1]
    group_key = tier_code * len(persona_names) + persona_of_item
    for g in pd.unique(group_key):
        m = group_key == g
        ti, pi = divmod(int(g), len(persona_names))
        w = zipf_w * boost_mat[ti, pi]
        cdf = np.cumsum(w / w.sum())
        cat_of[m] = _sample_with_cdf(cdf, int(m.sum()), rng)

    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    # 记录每组锚点的命中掩码 —— 后面 SKU 级锚点要用同一批命中交易，
    # 保证「品类锚点」与「SKU 锚点」落在同一批交易上（口径一致）
    anchor_hits: dict[tuple[str, str], np.ndarray] = {}
    for cat_a, cat_b, prob in ANCHOR_COMBOS:
        hit = (rng.random(t) < prob) & (k >= 2)
        anchor_hits[(cat_a, cat_b)] = hit
        pos = starts[hit]
        cat_of[pos] = cat_index[cat_a]
        cat_of[pos + 1] = cat_index[cat_b]

    # 品类 → SKU：按品类文本分组建索引（products 里 2% 是别名脏数据，先按别名表归一再分组）
    sku_arr = products["sku"].to_numpy()
    prod_cat_codes = pd.Categorical(
        products["category"].replace(CATEGORY_ALIASES), categories=CATEGORIES
    ).codes
    cat_to_skus = {c: sku_arr[prod_cat_codes == c] for c in range(n_cat)}
    sku_of = np.empty(len(cat_of), dtype=np.int64)
    for c in range(n_cat):
        mask = cat_of == c
        pool = cat_to_skus[c]
        if len(pool) == 0:
            pool = sku_arr[:1]
        sku_of[mask] = pool[rng.integers(0, len(pool), size=int(mask.sum()))]

    # ---- SKU 级强关联埋点（计划 §8.3 / R-9）----
    # 品类级锚点只保证「这两个品类会一起出现」，但每个品类有 80+ 个 SKU，
    # 同品类内的具体商品是随机采样的 → SKU 级信号被稀释到
    # lift 上限仅 1.8、confidence <4%（实测），挖不出有业务价值的单品级规则。
    # 因此对每组锚点，再指定**固定的 SKU 组合**（品类内销量前 2 的 SKU），
    # 让单品级关联规则有可验证的埋点。
    for cat_a, cat_b in anchor_hits:
        pool_a = np.sort(cat_to_skus[cat_index[cat_a]])
        pool_b = np.sort(cat_to_skus[cat_index[cat_b]])
        if len(pool_a) == 0 or len(pool_b) == 0:
            continue
        hit = anchor_hits[(cat_a, cat_b)]
        n_pairs = min(SKU_ANCHOR_PAIRS_PER_CATEGORY, len(pool_a), len(pool_b))
        # 每笔命中交易**只分配给一个** SKU 对（按顺序交错），且固定放在篮子前两位。
        # 若让同一批交易承载多组 SKU，挖出来的全是 2→2 组合规则；
        # 而业务要的是清晰的「单品 A → 单品 B」货架建议。
        idx = np.flatnonzero(hit & (k >= 2))
        for j in range(n_pairs):
            sel = idx[j::n_pairs]
            pos = starts[sel]
            sku_of[pos] = pool_a[j]
            sku_of[pos + 1] = pool_b[j]

    # ---- 同品类共购埋点 ----
    # §0.7 补的 SKU 级锚点是**跨品类**的（啤酒→纸尿裤），只覆盖 4 组品类。
    # 同品类内的搭配（可乐+雪碧、牛奶+酸奶）是商超里最常见的关联形态，
    # 若不单独埋点，SKU 级规则就只有跨品类那几对。这里再埋一组
    # **同品类**固定 SKU 搭配，放在篮子第 3、4 位（与品类锚点的第 1、2 位不冲突）。
    any_anchor_hit = np.zeros(t, dtype=bool)
    for hit_mask in anchor_hits.values():
        any_anchor_hit |= hit_mask
    for cat, prob in SAME_CATEGORY_COMBOS:
        pool = np.sort(cat_to_skus[cat_index[cat]])
        if len(pool) < 2:
            continue
        hit = (~any_anchor_hit) & (rng.random(t) < prob) & (k >= 4)
        pos = starts[hit]
        sku_of[pos + 2] = pool[0]
        sku_of[pos + 3] = pool[1]

    qty = rng.poisson(1.2, size=len(cat_of)) + 1
    price_map = pd.Series(products["base_price"].to_numpy(), index=products["sku"].to_numpy())
    unit_price = np.round(price_map.loc[sku_of].to_numpy() * np.exp(rng.normal(0, 0.08, size=len(cat_of))), 2)

    return pd.DataFrame({
        "txn_id": txn["txn_id"].to_numpy()[txn_pos],
        "sku": sku_of,
        "qty": qty.astype(np.int32),
        "unit_price": unit_price,
    })


def inject_dirty(txn: pd.DataFrame, items: pd.DataFrame,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """注入脏数据并返回统计（供数据质量报告核对）。"""
    stats = {}
    n = len(txn)

    # 1) 重复行 0.5%
    dup = txn.sample(frac=0.005, random_state=SEED)
    stats["dup_injected"] = len(dup)
    txn = pd.concat([txn, dup], ignore_index=True)

    # 2) 会员ID 缺失 3%（哨兵 -1）
    miss = rng.random(n) < 0.03
    txn.loc[np.where(miss)[0], "member_id"] = -1
    stats["missing_member_injected"] = int(miss.sum())

    # 3) 退货负金额 1%
    ret = rng.random(n) < 0.01
    txn.loc[np.where(ret)[0], "amount"] = -txn.loc[np.where(ret)[0], "amount"]
    stats["returns_injected"] = int(ret.sum())

    # 4) 逻辑冲突 0.5%：交易时间戳倒拨到 2010 年（早于任何注册时间）
    conflict = rng.random(n) < 0.005
    txn.loc[np.where(conflict)[0], "txn_ts"] = pd.Timestamp("2010-01-01")
    stats["logic_conflict_injected"] = int(conflict.sum())

    return txn, items, stats


def _generate_block(members_block: pd.DataFrame, products: pd.DataFrame,
                    cat_base: np.ndarray, persona_names: np.ndarray,
                    rng: np.random.Generator, txn_id_offset: int,
                    store_tier: np.ndarray | None = None):
    """生成一个会员块的交易与明细（供分块流式生成复用）。"""
    txn, member_idx, _persona = generate_transactions(
        members_block, rng, txn_id_offset=txn_id_offset)
    n_raw = len(txn)   # 注入重复行**之前**的交易数 —— 下一块的 txn_id 偏移必须用它
    persona_code = pd.Categorical(members_block["_persona"], categories=list(PERSONAS)).codes
    items = generate_items(txn, persona_code[member_idx], persona_names, products, cat_base, rng,
                           store_tier=store_tier)
    txn, items, dirty = inject_dirty(txn, items, rng)
    return txn, items, dirty, n_raw


def generate(n_members: int, seed: int = SEED, chunk_size: int = 0,
             raw_dir: Path = RAW_DIR, raw_items_dir: Path = RAW_ITEMS_DIR,
             raw_items_file: Path | None = RAW_ITEMS_FILE) -> dict:
    """生成全套合成数据并落盘，返回统计。

    ``chunk_size``：>0 且小于总会员数时**分块流式生成**——每块生成完立刻落盘，
    峰值内存 ≈ 单块，因此可以生成千万级明细而不爆内存（`--scale 5` 及以上必需）。
    分块模式下 ``txn_id`` 通过偏移保持全局唯一，明细按块追加 ``part-{i}`` 文件，
    最后把交易 part 流式合并成单个 ``transactions.parquet``。

    注意：``chunk_size=0``（默认）走一次性生成路径，**随机数调用顺序与历史产物完全一致**，
    因此既有产物、报告与测试基线都不受影响。
    """
    rng = np.random.default_rng(seed)
    n_sku = 5000
    stores = generate_stores()   # 不消耗 rng
    store_tier = stores.sort_values("store_id")["city_tier"].to_numpy(dtype=object)
    persona_names = np.array(list(PERSONAS), dtype=object)
    totals = {"transactions": 0, "items": 0, "dup_injected": 0,
              "missing_member_injected": 0, "returns_injected": 0,
              "logic_conflict_injected": 0}
    raw_dir = Path(raw_dir)
    raw_items_dir = Path(raw_items_dir)

    chunked = 0 < chunk_size < n_members
    if not chunked:
        # ---- 一次性路径（保持历史随机序列不变）----
        members = generate_members(n_members, rng)
        products, cat_base = generate_products(n_sku, rng)
        txn, items, dirty, _n_raw = _generate_block(
            members, products, cat_base, persona_names, rng, 0, store_tier)
        totals.update({k: int(v) for k, v in dirty.items()})
        totals["transactions"], totals["items"] = len(txn), len(items)
        members_out = members.drop(columns=["_reg_day", "_stop_day"])
        write_parquet(members_out, raw_dir / "members.parquet")
        write_parquet(stores, raw_dir / "stores.parquet")
        write_parquet(products, raw_dir / "products.parquet")
        write_parquet(txn, raw_dir / "transactions.parquet")
        items = derive_txn_month(items, txn)
        write_partitioned(items, raw_items_dir, "txn_month")
        if raw_items_file is not None and Path(raw_items_file).exists():
            Path(raw_items_file).unlink()
        return totals

    # ---- 分块流式路径 ----
    products, cat_base = generate_products(n_sku, rng)
    write_parquet(stores, raw_dir / "stores.parquet")
    write_parquet(products, raw_dir / "products.parquet")

    # 清空上一次的产物（分块是「整体重建」语义）
    if raw_items_dir.exists():
        for stale in raw_items_dir.glob("txn_month=*"):
            for f in stale.rglob("*"):
                if f.is_file():
                    f.unlink()
    txn_parts_dir = raw_dir / "_txn_parts"
    if txn_parts_dir.exists():
        for f in txn_parts_dir.glob("*.parquet"):
            f.unlink()
    txn_parts_dir.mkdir(parents=True, exist_ok=True)

    member_frames: list[pd.DataFrame] = []
    txn_parts: list[Path] = []
    n_chunks = (n_members + chunk_size - 1) // chunk_size
    txn_offset = 0
    for i in range(n_chunks):
        nb = min(chunk_size, n_members - i * chunk_size)
        block = generate_members(nb, rng)
        block["member_id"] = block["member_id"] + i * chunk_size   # 全局唯一 ID
        member_frames.append(block)
        txn, items, dirty, n_raw = _generate_block(
            block, products, cat_base, persona_names, rng, txn_offset, store_tier)
        # 偏移用「注入重复前」的交易数：重复行沿用原 txn_id，不占用新号段
        txn_offset += n_raw
        for k in ("dup_injected", "missing_member_injected", "returns_injected",
                  "logic_conflict_injected"):
            totals[k] += int(dirty.get(k, 0))
        totals["transactions"] += len(txn)
        totals["items"] += len(items)

        txn_part = txn_parts_dir / f"part-{i}.parquet"
        write_parquet(txn, txn_part)
        txn_parts.append(txn_part)
        items = derive_txn_month(items, txn)
        append_partitioned(items, raw_items_dir, "txn_month", f"part-{i}")
        print(f"[gen] chunk {i + 1}/{n_chunks}: members={nb} txn={len(txn)} items={len(items)}")

    members_all = pd.concat(member_frames, ignore_index=True)
    write_parquet(members_all.drop(columns=["_reg_day", "_stop_day"]), raw_dir / "members.parquet")
    # 交易 part 流式合并成单文件（峰值 ≈ 单 part）
    concat_parquet_parts(txn_parts, raw_dir / "transactions.parquet")
    for f in txn_parts:
        f.unlink()
    txn_parts_dir.rmdir()
    if raw_items_file is not None and Path(raw_items_file).exists():
        Path(raw_items_file).unlink()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="生成悦鲜生活超市合成数据")
    parser.add_argument("--scale", type=float, default=0.3,
                        help="规模档位：0.3 调试 / 1 正式 / 5 压测（会员数 = 120000 × scale）")
    parser.add_argument("--members", type=int, default=None,
                        help="直接指定会员数（优先级高于 --scale；用于「精确要多少」的场景）")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="分块大小（会员数/块）。0=不分块；大规模生成建议 50000~200000，"
                             "峰值内存 ≈ 单块")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    n_members = args.members if args.members else int(120_000 * args.scale)
    print(f"[gen] scale={args.scale} members={n_members} chunk_size={args.chunk_size} "
          f"seed={args.seed}")
    stats = generate(n_members, seed=args.seed, chunk_size=args.chunk_size)
    print(f"[gen] transactions={stats['transactions']} items={stats['items']}")
    print(f"[gen] dirty_stats={ {k: v for k, v in stats.items() if k.endswith('_injected')} }")
    print(f"[gen] raw -> {RAW_DIR}")


if __name__ == "__main__":
    main()
