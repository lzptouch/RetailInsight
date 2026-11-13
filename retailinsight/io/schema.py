"""Canonical schema 与领域常量。

本模块只定义**数据契约**（canonical 列名）与**领域常量**（品类 / 人群 / 埋点）。
数据由 `scripts/generate_synthetic.py` **直接产出 canonical 列名**，因此不需要
第三方字段映射层。

> v3.5 变更：原先的 `SOURCE_MAPPINGS` / `SOURCE_DTYPES` / `REQUIRED_COLUMNS` /
> `apply_mapping()` / `SchemaError` 已随开源数据集下载器一并删除。两条理由：
> ① 它们唯一的消费方就是那个下载器，**主链路从不经过**（三源 sha256 全为 None、
> 两源无 url，校验与控制流都形同虚设）；
> ② 项目的数据故事（中国商超 + 8 类可验证的先验人群）只能由合成数据承载，
> 接英美的第三方数据反而破坏聚类 ARI 的可验证性。
> 详见 `项目编写计划.md` §0.13 #17 与 §4.1。
"""
from __future__ import annotations

# canonical 列定义
CANONICAL = {
    "members": ["member_id", "reg_date", "gender", "age_band", "city_tier", "store_id"],
    "stores": ["store_id", "city", "city_tier", "area_m2", "open_date"],
    "products": ["sku", "category", "sub_category", "brand", "base_price"],
    "transactions": ["txn_id", "member_id", "store_id", "txn_ts", "amount", "discount"],
    "items": ["txn_id", "sku", "qty", "unit_price"],
}

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
