"""Canonical schema 与领域常量。

多套数据源（synthetic / dunnhumby / uci_retail）统一映射到 canonical 列名，
ETL 下游只认 canonical schema，实现数据源可插拔。
"""

# canonical 列定义
CANONICAL = {
    "members": ["member_id", "reg_date", "gender", "age_band", "city_tier", "store_id"],
    "stores": ["store_id", "city", "city_tier", "area_m2", "open_date"],
    "products": ["sku", "category", "sub_category", "brand", "base_price"],
    "transactions": ["txn_id", "member_id", "store_id", "txn_ts", "amount", "discount"],
    "items": ["txn_id", "sku", "qty", "unit_price"],
}

# 外部数据源 → canonical 的字段映射（预留，P1 接入开源数据集时使用）
SOURCE_MAPPINGS = {
    "synthetic": {},  # 合成数据直接产出 canonical 列
    "dunnhumby": {
        "household_key": "member_id",
        "BASKET_ID": "txn_id",
        "DAY": "txn_day",
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
    },
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

# 8 类隐式人群（合成数据先验，用于聚类有效性对照）
# freq_per_year: 年消费次数期望；basket: 单次件数期望；price_mult: 客单价系数
# weekend: 周末消费权重；discount: 折扣敏感概率；life: 生命周期形态
PERSONAS = {
    "高频高值": dict(weight=0.10, freq=12, basket=10, price_mult=1.6, weekend=0.5, discount=0.15, life="stable"),
    "周末囤货": dict(weight=0.12, freq=6, basket=14, price_mult=1.2, weekend=2.5, discount=0.25, life="stable"),
    "促销敏感": dict(weight=0.15, freq=8, basket=7, price_mult=0.9, weekend=1.0, discount=0.55, life="stable"),
    "新客尝鲜": dict(weight=0.10, freq=4, basket=5, price_mult=1.0, weekend=1.0, discount=0.20, life="new"),
    "高频低额": dict(weight=0.12, freq=16, basket=3, price_mult=0.7, weekend=1.0, discount=0.20, life="stable"),
    "季节性": dict(weight=0.08, freq=5, basket=6, price_mult=1.1, weekend=1.0, discount=0.20, life="seasonal"),
    "沉默流失": dict(weight=0.18, freq=7, basket=6, price_mult=1.0, weekend=1.0, discount=0.20, life="churn"),
    "一次性": dict(weight=0.15, freq=1, basket=4, price_mult=1.0, weekend=1.0, discount=0.20, life="once"),
}

AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]
AGE_BAND_WEIGHTS = [0.15, 0.30, 0.25, 0.18, 0.12]
CITY_TIERS = ["一线", "二线", "三线"]
CITY_TIER_WEIGHTS = [0.35, 0.40, 0.25]

MISSING_MEMBER_ID = -1  # 会员ID 缺失哨兵值（ETL 将其归入「非会员」客群）
