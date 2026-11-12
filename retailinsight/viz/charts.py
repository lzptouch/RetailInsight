"""图表函数与中文字体注册（统一配色，notebook/报表/看板共用）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from retailinsight.settings import FIGURES_DIR

PALETTE = ["#2E6FBA", "#E8833A", "#54A24B", "#D64550", "#8C63B6", "#B279A2", "#F2A93B", "#4C9F70"]


def register_chinese_font() -> str:
    """macOS 用 PingFang SC，容器兜底 Noto Sans CJK；都找不到时告警降级。"""
    for name in ["PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS", "Heiti SC"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    print("[viz] 警告：未找到中文字体，图表中文可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False
    return "default"


#: 模块导入时注册一次，保证任何入口（CLI / notebook / 看板）产出的图都带中文字体
register_chinese_font()


def _save(fig: plt.Figure, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def gmv_trend(monthly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly["month"].astype(str), monthly["gmv"], marker="o", color=PALETTE[0])
    ax.set_title("月度 GMV 趋势")
    ax.set_ylabel("GMV（元）")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "01_gmv_trend")


def monthly_yoy(monthly: pd.DataFrame) -> str:
    """月度 GMV 同比：月份 × 年份分组柱状图。"""
    df = monthly.copy()
    df["year"] = df["month"].dt.year.astype(str)
    df["m"] = df["month"].dt.month
    pv = df.pivot_table(index="m", columns="year", values="gmv")
    x = np.arange(len(pv.index))
    width = 0.8 / len(pv.columns)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, year in enumerate(pv.columns):
        ax.bar(x + i * width, pv[year] / 1e4, width, label=str(year), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(pv.columns) - 1) / 2, [f"{m}月" for m in pv.index])
    ax.set_title("月度 GMV 同比（按年份对比）")
    ax.set_ylabel("GMV（万元）")
    ax.legend(title="年份")
    return _save(fig, "01b_monthly_yoy")


def quarterly_growth(quarterly: pd.DataFrame) -> str:
    """季度 GMV 柱状图 + 同比标注（数据末期不完整季度不标注同比）。"""
    df = quarterly.copy()
    labels = df["quarter"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, df["gmv"] / 1e4, color=PALETTE[0])
    for i, row in df.iterrows():
        if pd.notna(row["yoy"]):
            ax.annotate(f"同比{row['yoy']:+.1%}", (i, row["gmv"] / 1e4),
                        ha="center", va="bottom", fontsize=8,
                        color="#D64550" if row["yoy"] < 0 else "#54A24B")
    ax.set_title("季度 GMV 与同比")
    ax.set_ylabel("GMV（万元）")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "01c_quarterly_yoy")


def store_rank(store_df: pd.DataFrame) -> str:
    df = store_df.sort_values("gmv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["store_id"].astype(str), df["gmv"], color=PALETTE[1])
    ax.set_title("门店 GMV 排名")
    ax.set_xlabel("GMV（元）")
    return _save(fig, "02_store_rank")


def category_pareto(cat_df: pd.DataFrame, top: int = 20) -> str:
    df = cat_df.sort_values("gmv", ascending=False).head(top)
    cum = df["gmv"].cumsum() / df["gmv"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(df["category"].astype(str), df["gmv"], color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(df["category"].astype(str), cum, color=PALETTE[3], marker=".")
    ax2.axhline(0.7, color="gray", ls="--", lw=1)
    ax.set_title(f"品类销售帕累托（Top{top}）")
    ax.tick_params(axis="x", rotation=60)
    ax2.set_ylabel("累计占比")
    return _save(fig, "03_category_pareto")


def hour_heatmap(pivot: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][:len(pivot.index)])
    ax.set_title("星期 × 小时 销售热力图")
    fig.colorbar(im, ax=ax, label="订单量")
    return _save(fig, "04_hour_heatmap")


def segment_bar(seg_df: pd.DataFrame) -> str:
    df = seg_df.sort_values("member_cnt")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(df["rfm_segment"], df["member_cnt"], color=PALETTE[2])
    ax.set_title("RFM 客群规模分布")
    ax.set_xlabel("会员数")
    return _save(fig, "05_segment_bar")


def segment_value(seg_df: pd.DataFrame) -> str:
    df = seg_df.sort_values("gmv_share", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(df["rfm_segment"], df["gmv_share"] * 100, color=PALETTE[4])
    ax.set_title("RFM 客群 GMV 贡献占比")
    ax.set_xlabel("GMV 占比（%）")
    return _save(fig, "06_segment_value")


def lift_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, cum, color=PALETTE[0], label="模型")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机")
    ax.axvline(0.2, color="gray", ls=":", lw=1)
    ax.set_title("流失预测捕获率曲线（Top20% 参考线）")
    ax.set_xlabel("名单占比")
    ax.set_ylabel("累计捕获流失会员比例")
    ax.legend()
    return _save(fig, "07_lift_curve")


def churn_hist(probs: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs, bins=40, color=PALETTE[3], alpha=0.85)
    ax.set_title("流失概率分布")
    ax.set_xlabel("预测流失概率")
    return _save(fig, "08_churn_hist")


def cluster_elbow(k_metrics: dict, best_silhouette: float | None = None) -> str:
    """肘部法 + 轮廓系数双曲线定 K（计划 §8.1「肘部法+轮廓系数」）。"""
    ks = sorted(k_metrics)
    inertia = [k_metrics[k]["inertia"] for k in ks]
    silhouette = [k_metrics[k]["silhouette"] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, inertia, marker="o", color=PALETTE[0], label="inertia（肘部法）")
    ax.set_xlabel("K（簇数）")
    ax.set_ylabel("inertia", color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(ks, silhouette, marker="s", color=PALETTE[1], label="轮廓系数")
    ax2.axhline(0.35, color="gray", ls="--", lw=1)
    ax2.set_ylabel("轮廓系数", color=PALETTE[1])
    if best_silhouette is not None:
        ax2.annotate(f"最优 silhouette={best_silhouette:.3f}",
                     (ks[int(np.argmax(silhouette))], max(silhouette)),
                     textcoords="offset points", xytext=(10, -14), fontsize=8)
    ax.set_title("聚类定 K：肘部法 + 轮廓系数（虚线=验收线 0.35）")
    ax.set_xticks(ks)
    return _save(fig, "09_cluster_elbow")


def _radar(ax, labels: list[str], values: list[float], color: str, label: str) -> None:
    """在极坐标系上画一条闭合雷达曲线。"""
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    vals = list(values) + [values[0]]
    ax.plot(angles, vals, color=color, lw=1.8, label=label)
    ax.fill(angles, vals, color=color, alpha=0.10)


def cluster_radar(desc: pd.DataFrame, feature_cols: list[str] | None = None) -> str:
    """簇画像雷达图：各簇在 6 维特征上的相对水平（0~1 分位）。"""
    cols = feature_cols or ["active_pct", "frequency_pct", "monetary_pct",
                            "avg_basket", "avg_n_categories", "avg_monetary"]
    labels_map = {
        "active_pct": "活跃度", "frequency_pct": "购买频次", "monetary_pct": "消费额",
        "avg_basket": "客单价", "avg_n_categories": "品类广度", "avg_monetary": "人均消费",
    }
    names = [labels_map.get(c, c) for c in cols]
    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(8.5, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(names, fontsize=9)

    # 各维度统一归一到 0~1，避免量纲差异压扁雷达图
    norm = desc.copy()
    for c in cols:
        col = norm[c].astype(float)
        rng = col.max() - col.min()
        norm[c] = (col - col.min()) / rng if rng > 0 else 0.5

    for i, (_, row) in enumerate(norm.iterrows()):
        _radar(ax, names, [row[c] for c in cols], PALETTE[i % len(PALETTE)],
               str(row.get("cluster_name", f"簇{i}")))
    ax.set_ylim(0, 1)
    ax.set_title("聚类簇画像雷达图（各维度归一化）", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    return _save(fig, "10_cluster_radar")


def segment_radar(profile: pd.DataFrame, by: str = "rfm_segment") -> str:
    """分群雷达图（P1-12）：RFM 8 客群在关键指标上的相对位置。"""
    dims = {"avg_basket": "客单价", "monthly_amount": "月均消费",
            "yearly_freq": "年购买次数", "items_per_txn": "连带率",
            "category_entropy": "品类多样性", "return_rate": "退货率"}
    cols = [c for c in dims if c in profile.columns]
    grp = profile.groupby(by)[cols].mean()
    labels_map = {c: dims[c] for c in cols}
    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(9, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([labels_map[c] for c in cols], fontsize=9)

    norm = grp.copy()
    for c in cols:
        col = norm[c].astype(float)
        rng = col.max() - col.min()
        norm[c] = (col - col.min()) / rng if rng > 0 else 0.5

    for i, (name, row) in enumerate(norm.iterrows()):
        _radar(ax, [labels_map[c] for c in cols], [row[c] for c in cols],
               PALETTE[i % len(PALETTE)], str(name))
    ax.set_ylim(0, 1)
    ax.set_title(f"客群雷达图（by {by}）", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8)
    return _save(fig, "11_segment_radar")
