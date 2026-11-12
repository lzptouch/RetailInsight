"""清洗规则（每条规则独立函数，可单测）。"""
import pandas as pd

from retailinsight.io.schema import CATEGORY_ALIASES, MISSING_MEMBER_ID


def normalize_category(products: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """品类别名归一。返回 (清洗后, 修正条数)。"""
    df = products.copy()
    fixed = int(df["category"].isin(CATEGORY_ALIASES).sum())
    df["category"] = df["category"].replace(CATEGORY_ALIASES)
    return df, fixed


def dedup_transactions(txn: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """按 txn_id 去重，保留首条。"""
    before = len(txn)
    df = txn.drop_duplicates(subset=["txn_id"], keep="first")
    return df.reset_index(drop=True), before - len(df)


def flag_returns(txn: pd.DataFrame) -> pd.DataFrame:
    """负金额标记为退货单（不删除，单列分析）。"""
    df = txn.copy()
    df["is_return"] = df["amount"] < 0
    df["net_amount"] = df["amount"]  # 净额：退货自然为负
    return df


def handle_missing_members(txn: pd.DataFrame) -> pd.DataFrame:
    """会员ID 缺失（哨兵 -1）→ 归入「非会员」客群，不删除。"""
    df = txn.copy()
    df["member_type"] = "会员"
    df.loc[df["member_id"] == MISSING_MEMBER_ID, "member_type"] = "非会员"
    return df


def flag_logic_conflicts(txn: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """逻辑冲突：交易时间早于注册时间，或时间戳超出数据范围 → 打异常标记。"""
    df = txn.copy()
    reg = members.set_index("member_id")["reg_date"]
    df["_reg"] = df["member_id"].map(reg)
    in_range = df["txn_ts"].between(pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01"))
    df["is_logic_error"] = (~in_range) | ((df["_reg"].notna()) & (df["txn_ts"] < df["_reg"]))
    return df.drop(columns=["_reg"])


def clip_outliers(txn: pd.DataFrame, q: float = 0.999) -> tuple[pd.DataFrame, float]:
    """金额极端值按分位截断（仅正向，退货负值保留）。返回 (结果, 阈值)。"""
    df = txn.copy()
    pos = df.loc[df["amount"] > 0, "amount"]
    thr = float(pos.quantile(q))
    df.loc[df["amount"] > thr, "amount"] = thr
    return df, thr
