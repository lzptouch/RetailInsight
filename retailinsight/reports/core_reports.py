"""核心统计报表：经营总览（月度环比/同比、季度环比/同比）/ 门店对比 / 品类结构 / 时段热力 / 会员漏斗 + 出图。"""
import numpy as np
import pandas as pd

from retailinsight.io.loader import read_items, read_parquet, write_parquet
from retailinsight.settings import INTERIM_DIR, ITEMS_DIR, MARTS_DIR
from retailinsight.viz import charts


def add_growth_columns(df: pd.DataFrame, period_col: str, value_col: str) -> pd.DataFrame:
    """为月度/季度汇总表增加环比与同比列。

    - 环比：与上一期相比（pct_change）
    - 同比：与上年同期相比（按月份/季度序号分组后 pct_change）
    要求 df 按 period_col 升序排列。
    """
    df = df.sort_values(period_col).reset_index(drop=True).copy()
    df["mom"] = df[value_col].pct_change().round(4)
    key = df[period_col].dt.month if period_col == "month" else df[period_col].dt.quarter
    df["yoy"] = df.groupby(key)[value_col].pct_change().round(4)
    return df


def build_core_reports() -> dict[str, pd.DataFrame]:
    txn = read_parquet(INTERIM_DIR / "transactions.parquet")
    items = read_items(ITEMS_DIR)
    products = read_parquet(INTERIM_DIR / "products.parquet")
    members = read_parquet(INTERIM_DIR / "members.parquet")

    ok = txn[~txn["is_logic_error"]].copy()
    ok["month"] = ok["txn_ts"].dt.to_period("M")

    # 1) 经营总览（月度）：GMV / 订单量 / 客单价 / 会员渗透率 + 环比 + 同比
    valid = ok[~ok["is_return"]]
    monthly = valid.groupby("month").agg(
        gmv=("net_amount", "sum"), orders=("txn_id", "nunique"))
    monthly["aov"] = (monthly["gmv"] / monthly["orders"]).round(2)
    member_orders = valid[valid["member_type"] == "会员"].groupby("month")["txn_id"].nunique()
    monthly["member_penetration"] = (member_orders / monthly["orders"]).round(4)
    monthly = monthly.round(2).reset_index()
    monthly = add_growth_columns(monthly, "month", "gmv")

    # 1b) 季度：GMV / 订单量 / 客单价 + 环比 + 同比
    ok["quarter"] = ok["txn_ts"].dt.to_period("Q")
    quarterly = valid.assign(quarter=valid["txn_ts"].dt.to_period("Q")).groupby("quarter").agg(
        gmv=("net_amount", "sum"), orders=("txn_id", "nunique"))
    quarterly["aov"] = (quarterly["gmv"] / quarterly["orders"]).round(2)
    quarterly = quarterly.round(2).reset_index()
    quarterly = add_growth_columns(quarterly, "quarter", "gmv")

    write_parquet(monthly, MARTS_DIR / "mart_monthly.parquet")
    write_parquet(quarterly, MARTS_DIR / "mart_quarterly.parquet")

    # 1c) 月度复购率（时序口径）
    # 口径：当月有购买行为的会员中「当月购买 >=2 次」的比例。
    # 与漏斗里的阶段转化率不同：漏斗是**全窗口累计**的，这里是**按月**的时序指标。
    member_month = (valid[valid["member_type"] == "会员"]
                    .groupby(["month", "member_id"])["txn_id"].nunique()
                    .reset_index(name="orders"))
    monthly_repeat = (member_month.groupby("month")
                      .agg(active_members=("member_id", "nunique"),
                           repeat_members=("orders", lambda s: int((s >= 2).sum())))
                      .reset_index())
    monthly_repeat["repurchase_rate"] = (
        monthly_repeat["repeat_members"] / monthly_repeat["active_members"]).round(4)
    monthly_repeat = monthly_repeat.merge(monthly[["month", "gmv"]], on="month", how="left")
    write_parquet(monthly_repeat, MARTS_DIR / "mart_monthly_repurchase.parquet")

    # 2) 门店对比：GMV / 订单 / 客单价 + 增长率 + 会员结构差异
    store = ok[~ok["is_return"]].groupby("store_id").agg(
        gmv=("net_amount", "sum"), orders=("txn_id", "nunique")).round(2).reset_index()
    store["aov"] = (store["gmv"] / store["orders"]).round(2)

    # 2a) 门店增长率：末期 3 个月 vs 前 3 个月（窗口太短噪声大，太长不灵敏）
    store_txn = ok[~ok["is_return"]].copy()
    store_txn["month"] = store_txn["txn_ts"].dt.to_period("M")
    months = sorted(store_txn["month"].unique())
    if len(months) >= 6:
        late, early = months[-3:], months[-6:-3]
        g_late = store_txn[store_txn["month"].isin(late)].groupby("store_id")["net_amount"].sum()
        g_early = store_txn[store_txn["month"].isin(early)].groupby("store_id")["net_amount"].sum()
        # **按 store_id 对齐后必须转成位置赋值**：
        # reindex 后的 Series 索引是 store_id（1~15），而 store 的索引是 0~14；
        # 直接 `store[col] = series` 会按**索引**对齐 → 门店 1 拿到 NaN、
        # 门店 k 拿到门店 k-1 的值（整体错位一格）。
        # 这里用 `.to_numpy()` 做位置赋值，顺序由 reindex 保证与 store 行序一致。
        store["gmv_growth"] = ((g_late / g_early.replace(0, pd.NA) - 1)
                               .reindex(store["store_id"].to_numpy()).round(4).to_numpy())
    else:
        store["gmv_growth"] = pd.NA

    # 2b) 会员结构差异：该店 VIP 占比 - 全司 VIP 占比
    # 口径：VIP = RFM 分群里的「重要价值客户」（见 features/rfm.py 的 8 类映射）
    # mart_rfm 可能不存在（例如只跑了 gen+etl）：此时 VIP 结构差异列降级为 NaN，
    # 而不是让整个报表构建失败 —— 显式降级好过硬依赖。
    rfm_path = MARTS_DIR / "mart_rfm.parquet"
    vip_ids: set = set()
    if rfm_path.exists():
        rfm_seg = read_parquet(rfm_path)
        vip_ids = set(rfm_seg.loc[rfm_seg["rfm_segment"] == "重要价值客户", "member_id"])
    store["vip_share"] = pd.NA
    store["vip_share_diff"] = pd.NA
    m_store = store_txn[store_txn["member_type"] == "会员"]
    if len(m_store) and vip_ids:
        by_store = m_store.groupby("store_id")["member_id"].nunique()
        vip_by_store = (m_store[m_store["member_id"].isin(vip_ids)]
                        .groupby("store_id")["member_id"].nunique())
        overall_vip = len(vip_ids) / max(len(rfm_seg), 1)
        # 同上：按 store_id 对齐后位置赋值，避免整体错位一格
        share = (vip_by_store.reindex(store["store_id"].to_numpy()).fillna(0).to_numpy()
                 / by_store.reindex(store["store_id"].to_numpy()).replace(0, np.nan).to_numpy())
        store["vip_share"] = np.round(share, 4)
        store["vip_share_diff"] = np.round(share - overall_vip, 4)

    # 3) 品类结构 + ABC
    it = (items.merge(ok[["txn_id", "net_amount", "is_return"]], on="txn_id")
          .merge(products[["sku", "category"]], on="sku"))
    it["line_amount"] = it["qty"] * it["unit_price"]
    cat = it[~it["is_return"]].groupby("category").agg(
        gmv=("line_amount", "sum"), sku_cnt=("sku", "nunique")).reset_index()
    cat["gmv_share"] = (cat["gmv"] / cat["gmv"].sum()).round(4)
    cat = cat.sort_values("gmv", ascending=False).reset_index(drop=True)
    cat["cum_share"] = cat["gmv_share"].cumsum().round(4)
    cat["abc_class"] = pd.cut(cat["cum_share"], [0, 0.7, 0.9, 1.0],
                              labels=["A", "B", "C"], include_lowest=True)

    # 4) 时段热力：星期 × 小时
    ok["weekday"] = ok["txn_ts"].dt.day_of_week
    ok["hour"] = ok["txn_ts"].dt.hour
    pivot = ok[~ok["is_return"]].pivot_table(index="weekday", columns="hour",
                                             values="txn_id", aggfunc="nunique", fill_value=0)

    # 5) 会员漏斗：注册 → 首购 → 复购 → 高频（≥12 次）
    m_txn = ok[ok["member_type"] == "会员"]
    per_member = m_txn.groupby("member_id")["txn_ts"].agg(["nunique", "count"])
    funnel = pd.DataFrame({
        "stage": ["注册会员", "有首购", "有复购", "高频(≥12次)"],
        "count": [len(members), len(per_member),
                  int((per_member["nunique"] >= 2).sum()),
                  int((per_member["nunique"] >= 12).sum())],
    })
    funnel["conversion"] = (funnel["count"] / funnel["count"].iloc[0]).round(4)

    # 出图
    charts.register_chinese_font()
    figures = [
        charts.gmv_trend(monthly),
        charts.monthly_yoy(monthly),
        charts.quarterly_growth(quarterly),
        charts.store_rank(store),
        charts.category_pareto(cat),
        charts.hour_heatmap(pivot),
    ]
    print(f"[reports] figures: {figures}")
    return {"monthly": monthly, "quarterly": quarterly, "store": store,
            "category": cat, "heatmap": pivot, "funnel": funnel,
            "monthly_repurchase": monthly_repeat}


if __name__ == "__main__":
    build_core_reports()
