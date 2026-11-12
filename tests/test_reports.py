"""月度/季度 环比与同比 计算口径单测。"""
import pandas as pd
import pytest

from retailinsight.reports.core_reports import add_growth_columns


def _toy_monthly():
    # 2020-01 ~ 2021-12 共 24 个月，GMV = 月份序号 × 100（严格线性）
    months = pd.period_range("2020-01", periods=24, freq="M")
    return pd.DataFrame({"month": months, "gmv": [100.0 * (i + 1) for i in range(24)]})


def test_mom():
    df = add_growth_columns(_toy_monthly(), "month", "gmv")
    assert pd.isna(df.loc[0, "mom"])
    # 第 2 个月环比 = 200/100 - 1 = 1.0
    assert df.loc[1, "mom"] == pytest.approx(1.0, abs=1e-4)
    # 第 13 个月环比 = 1300/1200 - 1
    assert df.loc[12, "mom"] == pytest.approx(1300 / 1200 - 1, abs=1e-4)


def test_yoy():
    df = add_growth_columns(_toy_monthly(), "month", "gmv")
    # 首年 12 个月无同比
    assert df.loc[:11, "yoy"].isna().all()
    # 2021-01 同比 = 1300/100 - 1 = 12.0
    assert df.loc[12, "yoy"] == pytest.approx(12.0, abs=1e-4)
    # 2021-12 同比 = 2400/1200 - 1 = 1.0
    assert df.loc[23, "yoy"] == pytest.approx(1.0, abs=1e-4)


def test_quarterly_yoy():
    quarters = pd.period_range("2020Q1", periods=8, freq="Q")
    df = pd.DataFrame({"quarter": quarters, "gmv": [100.0, 200, 300, 400, 150, 250, 350, 450]})
    out = add_growth_columns(df, "quarter", "gmv")
    # 2021Q1 同比 = 150/100 - 1 = 0.5；2021Q2 同比 = 250/200 - 1 = 0.25
    assert out.loc[4, "yoy"] == pytest.approx(0.5, abs=1e-4)
    assert out.loc[5, "yoy"] == pytest.approx(0.25, abs=1e-4)
    # 环比 = 与上一季度比
    assert out.loc[4, "mom"] == pytest.approx(150 / 400 - 1, abs=1e-4)
