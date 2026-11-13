"""RFM 口径单测（toy 数据，验证 F 同日合并 / M 净额 / 分群规则）。"""
import pandas as pd

from retailinsight.features.rfm import assign_segment, compute_rfm, score_rfm


def _toy_txn():
    rows = [
        # member 1：高频高值，最近有消费
        (1, "2025-12-20", 500.0, False),
        (1, "2025-12-20", 300.0, False),  # 同日多单 → F 只计 1
        (1, "2025-12-25", 200.0, False),
        (1, "2025-12-26", -50.0, True),   # 退货，计入净额不计入 F 增量
        # member 2：低频低值，长期未消费
        (2, "2024-03-01", 20.0, False),
    ]
    df = pd.DataFrame(rows, columns=["member_id", "ts", "amount", "is_return"])
    df["txn_ts"] = pd.to_datetime(df["ts"])
    df["member_type"] = "会员"
    df["is_logic_error"] = False
    df["net_amount"] = df["amount"]
    df["store_id"] = 1
    df["discount"] = 0.0
    return df.drop(columns=["ts"])


def test_rfm_calculation():
    txn = _toy_txn()
    end = pd.Timestamp("2026-01-01")
    rfm = compute_rfm(txn, end_date=end).set_index("member_id")

    m1, m2 = rfm.loc[1], rfm.loc[2]
    assert m1["recency_days"] == 6           # 2026-01-01 - 2025-12-26
    assert m1["frequency"] == 3              # 12-20 合并为 1 天
    assert m1["monetary"] == 950.0           # 500+300+200-50
    assert m2["recency_days"] == 671
    assert m2["frequency"] == 1
    assert m2["monetary"] == 20.0


def test_segment_rules():
    txn = _toy_txn()
    end = pd.Timestamp("2026-01-01")
    scored = assign_segment(score_rfm(compute_rfm(txn, end_date=end)))
    seg = scored.set_index("member_id")["rfm_segment"]
    # m1: R 高、F 高、M 高 → 重要价值客户；m2: R 低、F 低、M 低 → 流失风险客户
    assert seg[1] == "重要价值客户"
    assert seg[2] == "流失风险客户"
