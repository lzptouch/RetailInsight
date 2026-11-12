"""全局配置：路径、随机种子、分析基准。

统一使用 pathlib（本机 macOS 环境下规避 os.path 兼容性问题）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
MARTS_DIR = DATA_DIR / "marts"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"

for _d in (RAW_DIR, INTERIM_DIR, MARTS_DIR, FIGURES_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 集市层产物路径（单一真相源：其他地方一律引用这些常量，不手写文件名）----
MART_RFM = MARTS_DIR / "mart_rfm.parquet"
MART_PROFILE = MARTS_DIR / "mart_member_profile.parquet"
MART_CHURN = MARTS_DIR / "mart_churn_score.parquet"
MART_RULES = MARTS_DIR / "mart_basket_rules.parquet"
MART_CLUSTER = MARTS_DIR / "mart_cluster.parquet"

# ---- 分区明细路径（计划 §4.4：明细按 txn_month 分区，窗口期只加载所需分区）----
ITEMS_FILE = INTERIM_DIR / "transaction_items.parquet"   # 兼容旧产物（单文件）
ITEMS_DIR = INTERIM_DIR / "transaction_items"            # 分区目录（hive 风格）
RAW_ITEMS_FILE = RAW_DIR / "transaction_items.parquet"
RAW_ITEMS_DIR = RAW_DIR / "transaction_items"
MART_MONTHLY = MARTS_DIR / "mart_monthly.parquet"
MART_QUARTERLY = MARTS_DIR / "mart_quarterly.parquet"

SEED = 42

# 数据时间跨度（合成数据）：2020-01 ~ 2022-07，共 31 个月（非完整年度）
# 同比可比性：1~7 月有三年可比（2020/2021/2022），8~12 月有两年可比（2020 vs 2021）
DATA_START = "2020-01-01"
DATA_END = "2022-07-31"

# 流失建模：观察窗 / 标签窗
OBS_WINDOW_DAYS = 270   # 9 个月观察窗
LABEL_WINDOW_DAYS = 90  # 90 天标签窗
# 滚动多期 cohort 的截止日（特征只用截止日之前的数据；须落在 2020-01~2022-07 内）
# 末期 cohort 固定为 2022-04-01：其标签窗 (2022-04-01, 2022-07-01] 必须完全落在数据范围内
COHORT_CUTOFFS = ["2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01", "2022-04-01"]
# 训练/验证/测试按时间切分：前 N 期训练 + 1 期验证，末期测试
# 注：计划 §8.2 原文为「前 2 期训练」，此处用 3 期训练 —— 因为加入独立验证集后
# 训练窗口被迫前移，多留 1 期训练数据可缓解时间外推的精度损失（实测 AUC +0.002）
TRAIN_COHORTS = 3
VALID_COHORTS = 1

# 风险分层的业务成本假设：漏掉一个流失会员的代价 = 误触一个正常会员的 8 倍
# 该系数与《决策建议报告》S2「流失 1 人 ≈ 挽留 8 名流失风险客户」保持同一口径
COST_FALSE_NEGATIVE = 8.0
COST_FALSE_POSITIVE = 1.0
