"""Canonical schema 与领域常量。

多套数据源（synthetic / dunnhumby / uci_retail）统一映射到 canonical 列名，
ETL 下游只认 canonical schema，实现数据源可插拔。

**v3.0 起 `SOURCE_MAPPINGS` 有了真实消费方**：`apply_mapping()` 由
`scripts/download_data.py` 调用，把外部数据源的原始列名改写成 canonical 列名，
下游 ETL 代码无需改动即可吃下第三方数据。
"""
from __future__ import annotations

import pandas as pd

# canonical 列定义
CANONICAL = {
    "members": ["member_id", "reg_date", "gender", "age_band", "city_tier", "store_id"],
    "stores": ["store_id", "city", "city_tier", "area_m2", "open_date"],
    "products": ["sku", "category", "sub_category", "brand", "base_price"],
    "transactions": ["txn_id", "member_id", "store_id", "txn_ts", "amount", "discount"],
    "items": ["txn_id", "sku", "qty", "unit_price"],
}

# 各表在 canonical 下的**必需列**（缺任一列即视为该源不兼容）
REQUIRED_COLUMNS = {
    "transactions": ["txn_id", "member_id", "txn_ts", "amount"],
    "items": ["txn_id", "sku", "qty", "unit_price"],
}

# 外部数据源 → canonical 的字段映射
SOURCE_MAPPINGS = {
    "synthetic": {},  # 合成数据直接产出 canonical 列
    "dunnhumby": {
        "household_key": "member_id",
        "BASKET_ID": "txn_id",
        "DAY": "txn_ts",
        "PRODUCT_ID": "sku",
        "QUANTITY": "qty",
        "SALES_VALUE": "amount",
    },
    "uci_retail": {
        "CustomerID": "member_id",
        "InvoiceNo": "txn_id",
        "InvoiceDate": "txn_ts",
        "StockCode": "sku",
        "Quantity": "qty",
        "Price": "unit_price",
        "UnitPrice": "unit_price",   # items 表用 UnitPrice，transactions 用 Price
    },
}

# 外部数据源 → canonical 的**类型转换规则**
SOURCE_DTYPES = {
    "synthetic": {},
    "dunnhumby": {
        "member_id": "int32_member", "txn_id": "int32", "sku": "int32_sku",
        "qty": "int32", "amount": "float32", "txn_ts": "datetime_epoch_day",
    },
    "uci_retail": {
        "member_id": "int32_member", "txn_id": "int32", "sku": "int32_sku",
        "qty": "int32", "unit_price": "float32", "txn_ts": "datetime64[s]",
    },
}


class SchemaError(ValueError):
    """数据源与 canonical schema 不兼容。"""


# 非数值编码的稳定代理 ID 起始值（UCI 的 StockCode 含 "POST"/"BANK CHARGES" 等非数字码）
SURROGATE_ID_BASE = 10_000_000


def _to_int32(series: pd.Series, source: str, col: str,
              na_sentinel: int | None = None) -> pd.Series:
    """转 int32：数值直接转；缺失填哨兵；**非数值编码映射为稳定代理 ID**。

    为什么需要代理 ID：UCI Online Retail 的 ``StockCode`` 不全是数字
    （"POST"、"M"、"D"、"BANK CHARGES" 等是服务/费用码），
    直接 ``astype(int32)`` 会整列失败。丢弃这些行会丢真实交易，
    映射到稳定代理 ID 则既保住了行、又保住了跨表引用一致性。
    """
    num = pd.to_numeric(series, errors="coerce")
    non_numeric = num.isna() & series.notna()
    if na_sentinel is not None:
        num = num.fillna(na_sentinel)
    if non_numeric.any():
        codes = series[non_numeric].astype(str)
        uniq = sorted(codes.unique())
        mapping = {c: SURROGATE_ID_BASE + i for i, c in enumerate(uniq)}
        num.loc[non_numeric] = codes.map(mapping).astype("float64")
    try:
        return num.astype("int32")
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{source}.{col} 转 int32 失败：{exc}") from exc


def apply_mapping(df: pd.DataFrame, source: str, table: str | None = None) -> pd.DataFrame:
    """把外部数据源的原始列名映射为 canonical 列名，并做类型转换与必需列校验。

    参数
    ----
    df:     外部数据源的原始表
    source: 数据源名（``synthetic`` / ``dunnhumby`` / ``uci_retail``）
    table:  canonical 表名（``transactions`` / ``items``）。为 None 时跳过必需列校验。

    行为约定
    --------
    * 映射不存在的源列会被丢弃（外部源多出的列对下游无意义）
    * 缺必需列 → 抛 :class:`SchemaError`（**不静默补空列**）
    * 会员ID 缺失 → 填 ``MISSING_MEMBER_ID``（与 ETL 的「非会员」口径一致）
    * SKU 为非数字编码 → 稳定代理 ID（不丢行）
    """
    if source not in SOURCE_MAPPINGS:
        raise SchemaError(f"未知数据源：{source!r}，可选：{sorted(SOURCE_MAPPINGS)}")

    # 列名映射：多个源列可能指向同一 canonical 列（如 UCI 的 Price / UnitPrice
    # 都是 unit_price），此时做**合并**而不是重命名（重命名会产生重复列名）。
    rename_map: dict[str, str] = {}
    coalesce: list[tuple[str, str]] = []
    for src_col, tgt_col in SOURCE_MAPPINGS[source].items():
        if src_col not in df.columns:
            continue
        if tgt_col in df.columns or tgt_col in rename_map.values():
            coalesce.append((tgt_col, src_col))
            continue
        rename_map[src_col] = tgt_col
    out = df.rename(columns=rename_map).copy()
    for tgt_col, src_col in coalesce:
        if tgt_col in out.columns and src_col in out.columns:
            out[tgt_col] = out[tgt_col].fillna(out[src_col])
        out = out.drop(columns=[c for c in (src_col,) if c in out.columns])

    for col, dtype in SOURCE_DTYPES.get(source, {}).items():
        if col not in out.columns:
            continue  # 该源没有这个字段
        try:
            if dtype == "datetime_epoch_day":
                # dunnhumby 的 DAY 是「距 1970-01-01 的天数」
                out[col] = pd.to_datetime(out[col], unit="D", origin="1970-01-01")
            elif dtype.startswith("datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce")
            elif dtype == "int32_sku":
                out[col] = _to_int32(out[col], source, col, na_sentinel=-1)
            elif dtype == "int32_member":
                out[col] = _to_int32(out[col], source, col, na_sentinel=MISSING_MEMBER_ID)
            elif dtype.startswith("int") or dtype.startswith("float"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{source}.{col} 类型转换失败（目标 {dtype}）：{exc}") from exc

    # ---- 源特有的派生列（canonical 需要、但源里没有的字段）----
    if source == "uci_retail" and table == "transactions":
        # Online Retail II 没有金额列，金额 = 数量 × 单价；也没有折扣列
        if "amount" not in out.columns and {"qty", "unit_price"} <= set(out.columns):
            out["amount"] = (out["qty"] * out["unit_price"]).astype("float32")
        if "discount" not in out.columns:
            out["discount"] = 0.0
    if source == "dunnhumby" and table == "transactions" and "discount" not in out.columns:
        out["discount"] = 0.0

    if table:
        required = REQUIRED_COLUMNS.get(table, [])
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise SchemaError(
                f"数据源 {source!r} 的 {table} 表缺少必需列：{missing}；"
                f"实际列：{list(out.columns)}"
            )
    return out

# 60 个品类（含 4 组关联埋点所需品类）
CATEGORIES = [
    "粮油米面", "调味品", "饮料", "啤酒", "乳制品", "面包糕点", "休闲零食", "糖果巧克力",
    "冷冻食品", "速冻面点", "鲜肉", "牛羊肉", "禽肉", "水产海鲜", "蔬菜", "水果",
    "鸡蛋", "豆制品", "火锅底料", "方便速食", "罐头食品", "咖啡茶饮", "白酒红酒", "个人护理",
    "洗发水", "护发素", "口腔护理", "纸品", "清洁用品", "纸尿裤", "婴儿奶粉", "婴儿辅食",
    "宠物食品", "家居用品", "厨房用品", "文具用品", "小家电", "服饰内衣", "床上用品", "玩具",
    "图书音像", "五金工具", "汽车用品", "鲜花绿植", "烘焙原料", "进口食品", "保健品", "药品",
    "化妆品", "母婴用品", "运动户外", "数码配件", "日用百货", "节日礼品", "熟食卤味", "现制饮品",
    "早餐谷物", "果酱蜂蜜", "南北干货", "腌腊制品",
]

# 脏数据埋点：品类别名 → 规范名（ETL 用别名映射表归一）
CATEGORY_ALIASES = {
    "清洁 用品": "清洁用品",
    "个人 护理": "个人护理",
    "休闲 零食": "休闲零食",
    "乳 制品": "乳制品",
}

# 购物篮强关联埋点：(品类A, 品类B, 命中概率)
ANCHOR_COMBOS = [
    ("啤酒", "纸尿裤", 0.06),
    ("火锅底料", "牛羊肉", 0.05),
    ("面包糕点", "乳制品", 0.08),
    ("洗发水", "护发素", 0.05),
]

# 每组品类锚点对应的**单品级**锚点对数量（计划 §8.3 / R-9）
#
# 为什么需要：品类级锚点只保证「这两个品类会一起出现」，但每个品类有 80+ 个 SKU，
# 同品类内的具体商品是随机采样的 → SKU 级信号被稀释（实测 lift 上限仅 1.8、
# confidence <4%，挖不出有业务价值的单品级规则）。
# 指定固定的 SKU 组合后，单品级关联规则才有可验证的埋点。
SKU_ANCHOR_PAIRS_PER_CATEGORY = 4

# 进口商品占比（计划 §7.3「进口偏好」标签的商品属性）
IMPORT_SKU_RATIO = 0.15

# 生鲜品类（计划 §7.3「生鲜占比」标签）
FRESH_CATEGORIES = [
    "蔬菜", "水果", "鲜肉", "牛羊肉", "禽肉", "水产海鲜", "鸡蛋", "乳制品",
]

# 同品类共购埋点：(品类, 命中概率)
#
# 为什么需要：§0.7 里补的 SKU 级锚点是**跨品类**的（如啤酒→纸尿裤），
# 只覆盖了 4 组品类。若只有跨品类锚点，**同品类内**的搭配（如「可乐 + 雪碧」）
# 在规则挖掘里完全没有信号 —— 而这恰恰是商超里最常见的关联形态
#（同品类的替代品/互补品一起买）。
# 这里再埋一组**同品类**的固定 SKU 搭配，让 SKU 级规则不再只靠跨品类锚点。
SAME_CATEGORY_COMBOS = [
    ("饮料", 0.08), ("休闲零食", 0.07), ("乳制品", 0.06),
    ("面包糕点", 0.06), ("调味品", 0.05), ("个人护理", 0.06),
    ("冷冻食品", 0.05), ("粮油米面", 0.05),
]

# 8 类隐式人群（合成数据先验，用于聚类有效性对照）
# freq_per_year: 年消费次数期望；basket: 单次件数期望；price_mult: 客单价系数
# weekend: 周末消费权重；discount: 折扣敏感概率；life: 生命周期形态
PERSONAS = {
    "高频高值": dict(weight=0.10, freq=12, basket=10, price_mult=1.6, weekend=0.5, discount=0.15, life="stable", cross_store=0.08),
    "周末囤货": dict(weight=0.12, freq=6, basket=14, price_mult=1.2, weekend=2.5, discount=0.25, life="stable", cross_store=0.05),
    "促销敏感": dict(weight=0.15, freq=8, basket=7, price_mult=0.9, weekend=1.0, discount=0.55, life="stable", cross_store=0.22),
    "新客尝鲜": dict(weight=0.10, freq=4, basket=5, price_mult=1.0, weekend=1.0, discount=0.20, life="new", cross_store=0.18),
    "高频低额": dict(weight=0.12, freq=16, basket=3, price_mult=0.7, weekend=1.0, discount=0.20, life="stable", cross_store=0.30),
    "季节性": dict(weight=0.08, freq=5, basket=6, price_mult=1.1, weekend=1.0, discount=0.20, life="seasonal", cross_store=0.10),
    "沉默流失": dict(weight=0.18, freq=7, basket=6, price_mult=1.0, weekend=1.0, discount=0.20, life="churn", cross_store=0.06),
    "一次性": dict(weight=0.15, freq=1, basket=4, price_mult=1.0, weekend=1.0, discount=0.20, life="once", cross_store=0.12),
}

AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]
AGE_BAND_WEIGHTS = [0.15, 0.30, 0.25, 0.18, 0.12]
CITY_TIERS = ["一线", "二线", "三线"]
CITY_TIER_WEIGHTS = [0.35, 0.40, 0.25]

MISSING_MEMBER_ID = -1  # 会员ID 缺失哨兵值（ETL 将其归入「非会员」客群）
