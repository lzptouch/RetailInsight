"""CLV 进阶法：BG/NBD + Gamma-Gamma（v3.7，替代简化法的**主口径**）。

为什么换：简化法假设「留存期服从几何分布、月均消费恒定」，同一客群的人
会被给出几乎相同的价值，只能排序不能定额。BG/NBD 把「买不买」和「买多少」
拆成两个概率模型，用**个体自己的购买节奏**（frequency / recency / T）推断
「他是低频常客还是正在流失」，Gamma-Gamma 再从「买了的人」里估计客单价。

    购买过程  BG/NBD ：λ ~ Gamma(r, α)、每次购买后以概率 p ~ Beta(a, b) 流失
    金额过程  Gamma-Gamma：客单价 ~ Gamma(p, v)，尺度由个体均值 m̄_x 决定
    未来 H 天期望金额 = E[购买次数(H) | x, t_x, T] × E[客单价 | x, m̄_x]

口径与验收（全部实跑，见 `reports/metrics_clv.json`）：
- 观测窗 2021-01-01~2021-12-31 拟合，预测 2022Q1（90 天）做**时间外回测**，
  与简化法「月均 × 3」同台比较 Spearman / MAE / 偏差
- 实测（17,937 名重复购买客户，91 天回测）：金额 MAE **229.27 vs 270.81**（降 15.3%）、
  Spearman **0.4449 vs 0.4109**、偏差 **−1.44 vs +74.40** 元/人 ——
  简化法系统性高估，BG/NBD 近似无偏

**只在 synthetic 数据上验证过**：参数会随真实数据重估，回测是这条链路的一部分，
不是一次性结论。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from retailinsight.settings import SEED

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    _HAS_LIFETIMES = True
except Exception:  # noqa: BLE001
    BetaGeoFitter = GammaGammaFitter = None
    _HAS_LIFETIMES = False

#: 正则化系数。合成数据 18k 客户实测：0.01 时 r/α/a/b 收敛良好，
#: 再小则 a 趋向 0（=「几乎不流失」，与 churn 标签 30%+ 矛盾）
PENALIZER = 0.01

#: CLV 预测视界（月）—— 与「客户资产 12 个月口径」一致
HORIZON_MONTHS = 12

#: 一个月按 30.44 天（365.25/12），避免 12×30=360 天带来的系统性低估
DAYS_PER_MONTH = 30.4375

#: 回测窗口：2021 全年拟合 → 预测 2022Q1，与流失模型的 90 天标签窗同长度，
#: 因此两条链路的「未来 90 天」是同一段时间，结论可互相引用
BACKTEST_OBS_END = "2021-12-31"
BACKTEST_HOLD_END = "2022-04-01"


def available() -> bool:
    """lifetimes 是否可用。缺失时上层降级到简化法，不静默。"""
    return _HAS_LIFETIMES


def build_summary(txn: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """把交易流水压成 BG/NBD 所需的 RFM 摘要。

    返回列：``frequency``（重复购买次数 = 交易次数 − 1）、``recency``
    （首购到最近一次购买的天数）、``T``（首购到 cutoff 的天数）、
    ``monetary_value``（重复购买的平均净额，只有一次购买时为 0）。

    注意 ``frequency`` 用的是 lifetimes 惯例的「重复购买次数」，
    与画像里的 ``frequency``（去重购买日数）不是同一口径，不能互换。
    """
    t = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    t = t[t["txn_ts"] <= cutoff]
    g = t.groupby("member_id")
    first = g["txn_ts"].min()
    last = g["txn_ts"].max()
    txn_cnt = g.size()
    amount = g["net_amount"].sum()

    s = pd.DataFrame(index=first.index)
    s["T"] = (cutoff - first).dt.days.clip(lower=1)
    s["recency"] = (last - first).dt.days.clip(lower=0)
    # lifetimes 的 recency ≤ T 是硬约束；首购当天只买一次时两者都是 0，需至少 1 天
    s["frequency"] = (txn_cnt - 1).clip(lower=0)
    s["monetary_value"] = np.where(txn_cnt > 1,
                                    amount / txn_cnt,
                                    0.0).round(2)
    return s


def fit_models(summary: pd.DataFrame, penalizer: float = PENALIZER):
    """拟合 BG/NBD 与 Gamma-Gamma，返回 ``(bgf, ggf)``。

    只用 ``frequency > 0`` 的客户拟合 Gamma-Gamma —— 只买过一次的人没有
    「重复购买客单价」可言，强行纳入会把客单价压向首单金额。
    """
    if not _HAS_LIFETIMES:
        raise RuntimeError("未安装 lifetimes，无法拟合 BG/NBD")

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    pos = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(pos["frequency"], pos["monetary_value"])
    return bgf, ggf


def predict_clv(bgf, ggf, summary: pd.DataFrame,
                months: int = HORIZON_MONTHS,
                gross_margin: float = 0.30) -> pd.Series:
    """预测未来 ``months`` 个月的期望毛利（元），索引为 member_id。

    **只买过一次的客户（``frequency == 0``）返回 NaN，不硬估**。
    原因：BG/NBD 在 x=0 处的超几何项参数为负（c = a+b−1 ≈ −0.14），
    lifetimes 会算出 NaN；与其用一条不相干的解析式凑数，不如让上层
    退回简化法 —— 简化法对这类新客本来就更稳（只用历史月均）。
    上层 ``apply_bgnbd_clv`` 会把这些人标成 ``simple_fallback``。
    """
    days = months * DAYS_PER_MONTH
    n_buy = bgf.conditional_expected_number_of_purchases_up_to_time(
        days, summary["frequency"], summary["recency"], summary["T"])

    pos = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    avg = ggf.conditional_expected_average_profit(
        pos["frequency"], pos["monetary_value"]).reindex(summary.index)
    # 群体期望客单价：取训练集中位频次 / 中位客单价处的条件期望，
    # 不依赖个别会员，也不会被极值拉偏
    if pos.empty:
        pop_mean = float(summary["monetary_value"].mean() or 0.0)
    else:
        pop_mean = float(ggf.conditional_expected_average_profit(
            pd.Series([float(pos["frequency"].median())]),
            pd.Series([float(pos["monetary_value"].median())])).iloc[0])
    avg = avg.fillna(pop_mean)

    return (n_buy * avg * gross_margin).round(2).rename("clv_bgnbd")


def _simple_forecast(txn: pd.DataFrame, obs_end: pd.Timestamp,
                     days: int) -> pd.Series:
    """简化法对照：**个人在网日均净额** × 天数（等价于「月均 × 月数」）。

    分母必须是会员自己的在网天数（首购→obs_end），不能是全库跨度 ——
    后者会把早期注册的低频客户日均稀释到接近 0，让对照组显得很弱，
    那是「给自己找对手」而不是做比较。
    """
    t = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    t = t[t["txn_ts"] <= obs_end]
    first = t.groupby("member_id")["txn_ts"].min()
    span = (obs_end - first).dt.days.clip(lower=1)
    daily = t.groupby("member_id")["net_amount"].sum() / span
    return (daily * days).rename("simple")


def backtest(txn: pd.DataFrame, obs_end: pd.Timestamp, hold_end: pd.Timestamp,
             penalizer: float = PENALIZER) -> dict:
    """时间外回测：观测窗拟合 → 预测 holdout 窗口 → 与实际对账。

    同时跑简化法，保证「换了算法」有同台证据，而不是自己说自己好。
    返回可直接写进 `reports/metrics_clv.json` 的 dict。
    """
    hold_days = max((hold_end - obs_end).days, 1)
    obs = txn[txn["txn_ts"] <= obs_end]
    summary_all = build_summary(obs, obs_end)
    # 观测期太短的客户（首购距今不足 30 天）估计不可靠，两边都剔除
    summary_all = summary_all[summary_all["T"] >= 30]
    # 只在**两种方法都能给出个体估计**的客户上比较：BG/NBD 对只买过一次的
    # 客户没有个体估计（见 predict_clv），把他们留在对照组会变成「比谁的
    # 兜底更准」而不是比模型。剔除人数如实写进结果，不藏。
    summary = summary_all[summary_all["frequency"] > 0]
    n_excluded = int(len(summary_all) - len(summary))

    bgf, ggf = fit_models(summary, penalizer)
    months = max(hold_days / DAYS_PER_MONTH, 1.0)
    pred_bg = predict_clv(bgf, ggf, summary, months=months, gross_margin=1.0)
    pred_simple = _simple_forecast(obs, obs_end, hold_days)

    t = txn[(txn["member_type"] == "会员") & (~txn["is_logic_error"])]
    hold = t[(t["txn_ts"] > obs_end) & (t["txn_ts"] <= hold_end)]
    actual = pd.DataFrame({
        "act_amt": hold.groupby("member_id")["net_amount"].sum(),
        "act_n": hold.groupby("member_id").size(),
    })
    cmp = summary.join(actual).fillna({"act_amt": 0.0, "act_n": 0})

    def _score(pred: pd.Series) -> dict:
        p = pred.reindex(cmp.index).astype(float)
        # 用 pandas 自带的秩相关，不引 scipy：scipy 只是 lifetimes 的**传递**依赖，
        # 把它写成我们自己的断言对象，等 lifetimes 换依赖时这里会先崩
        rho = p.corr(cmp["act_amt"], method="spearman")
        return {
            "n": int(len(cmp)),
            "amount_spearman": round(float(rho), 4),
            "amount_mae": round(float(np.mean(np.abs(p - cmp["act_amt"]))), 2),
            "amount_bias": round(float(np.mean(p - cmp["act_amt"])), 2),
        }

    bg = _score(pred_bg)
    sm = _score(pred_simple)
    return {
        "method": "bgnbd_gamma_gamma",
        "penalizer": penalizer,
        "obs_end": str(obs_end.date()),
        "hold_end": str(hold_end.date()),
        "hold_days": hold_days,
        "n_members": int(len(cmp)),
        "n_repeat": int((summary["frequency"] > 0).sum()),
        "n_excluded_single_purchase": n_excluded,
        "bgnbd": bg,
        "simple": sm,
        "winner": "bgnbd" if bg["amount_mae"] < sm["amount_mae"] else "simple",
        "mae_reduction_pct": round((1 - bg["amount_mae"] / max(sm["amount_mae"], 1e-9)) * 100, 1),
        "bgf_params": {k: round(float(v), 4) for k, v in bgf.params_.items()},
        "ggf_params": {k: round(float(v), 4) for k, v in ggf.params_.items()},
        "seed": SEED,
    }
