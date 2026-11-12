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

from retailinsight.io.loader import write_parquet
from retailinsight.io.schema import (
    AGE_BAND_WEIGHTS,
    AGE_BANDS,
    ANCHOR_COMBOS,
    CATEGORIES,
    CATEGORY_ALIASES,
    CITY_TIER_WEIGHTS,
    CITY_TIERS,
    PERSONAS,
)
from retailinsight.settings import DATA_END, DATA_START, RAW_DIR, SEED

START = pd.Timestamp(DATA_START)
END = pd.Timestamp(DATA_END)
N_DAYS = (END - START).days + 1

CITIES = [("上海", "一线"), ("北京", "一线"), ("广州", "一线"), ("深圳", "一线"), ("成都", "一线"),
          ("杭州", "二线"), ("武汉", "二线"), ("南京", "二线"), ("西安", "二线"), ("苏州", "二线"),
          ("绵阳", "三线"), ("洛阳", "三线"), ("宜昌", "三线"), ("柳州", "三线"), ("保定", "三线")]


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
    })
    # 脏数据：2% 的 SKU 品类名写成别名（如「清洁 用品」）
    dirty_idx = rng.random(n_sku) < 0.02
    alias_cycle = list(CATEGORY_ALIASES.items())
    for j, idx in enumerate(np.where(dirty_idx)[0]):
        alias, canonical = alias_cycle[j % len(alias_cycle)]
        if df.at[idx, "category"] == canonical:
            df.at[idx, "category"] = alias
    return df, cat_base


def generate_transactions(members: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """按人群行为生成交易：频次/时段/周末偏好/生命周期衰减。"""
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

    txn = pd.DataFrame({
        "txn_id": np.arange(1, t + 1, dtype=np.int64),
        "member_id": m["member_id"].to_numpy()[member_idx],
        "store_id": m["store_id"].to_numpy()[member_idx],
        "txn_ts": txn_ts,
        "amount": basket_value,
        "discount": discount,
    })
    return txn, member_idx, persona


def generate_items(txn: pd.DataFrame, persona_of_member_txn: np.ndarray,
                   persona_names: np.ndarray, products: pd.DataFrame,
                   cat_base: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """生成交易明细：件数按人群，品类按 Zipf 热度，4 组锚点组合注入。"""
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
    cat_cdf = np.cumsum(zipf_w / zipf_w.sum())
    cat_of[:] = _sample_with_cdf(cat_cdf, len(cat_of), rng)

    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    for cat_a, cat_b, prob in ANCHOR_COMBOS:
        hit = (rng.random(t) < prob) & (k >= 2)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="生成悦鲜生活超市合成数据")
    parser.add_argument("--scale", type=float, default=0.3, help="规模档位：0.3 调试 / 1 正式 / 5 压测")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    n_members = int(120_000 * args.scale)
    n_sku = 5000
    print(f"[gen] scale={args.scale} seed={args.seed} members={n_members}")

    members = generate_members(n_members, rng)
    stores = generate_stores()
    products, cat_base = generate_products(n_sku, rng)
    txn, member_idx, persona = generate_transactions(members, rng)
    persona_names = np.array(list(PERSONAS), dtype=object)
    # persona_of_member_txn: 每笔交易对应会员的 persona 下标
    persona_code = pd.Categorical(members["_persona"], categories=list(PERSONAS)).codes
    items = generate_items(txn, persona_code[member_idx], persona_names, products, cat_base, rng)
    txn, items, dirty_stats = inject_dirty(txn, items, rng)

    # 内部埋点列不入库（仅成员表保留 _persona 供对照，另存一份供测试）
    members_out = members.drop(columns=["_reg_day", "_stop_day"])
    write_parquet(members_out, RAW_DIR / "members.parquet")
    write_parquet(stores, RAW_DIR / "stores.parquet")
    write_parquet(products, RAW_DIR / "products.parquet")
    write_parquet(txn, RAW_DIR / "transactions.parquet")
    write_parquet(items, RAW_DIR / "transaction_items.parquet")

    print(f"[gen] transactions={len(txn)} items={len(items)}")
    print(f"[gen] dirty_stats={dirty_stats}")
    print(f"[gen] raw -> {RAW_DIR}")


if __name__ == "__main__":
    main()
