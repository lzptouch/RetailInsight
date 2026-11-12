"""决策建议报告渲染器单测：KPI 回填、测算公式、模板渲染。"""
import pandas as pd
import pytest

from retailinsight.reports import render_decision_report as rdr


def _kpis():
    return {
        "n_members": 35612, "n_profile": 36000, "n_orders": 310178, "gmv_wan": 4469.0,
        "data_window": "2020-01-01 ~ 2022-07-31", "n_months": 31,
        "n_vip": 6600, "vip_share": "18.5%", "vip_gmv_share": "42.7%",
        "vip_arpu": 2777.5, "vip_arpu_month": 115.7,
        "n_churn_risk": 13526, "churn_risk_share": "38.0%",
        "churn_risk_gmv_share": "9.8%", "churn_risk_avg_recency": 303,
        "top2_share": "36.6%", "top2_gmv_share": "77.8%",
        "n_high_risk": 4989, "high_risk_arpu": 225.0,
        "model_backend": "lightgbm", "model_auc": 0.8243, "model_auc_lr": 0.7972,
        "model_auc_gain": 0.0271, "model_auc_gain_verdict": "未达标",
        "model_ks": 0.4922, "model_lift": 2.34, "model_capture": 0.4688,
        "model_base_rate": 0.3676, "model_lift_upper": 2.72, "model_n_features": 31,
        "model_cohorts": "a / b / c", "model_valid_cohort": "d", "model_test_cohort": "e",
        "threshold_action": 0.24, "threshold_high": 0.61,
        "cluster_space": "rfm3", "cluster_k": 4, "cluster_silhouette": 0.4155,
        "cluster_ari": 0.1345, "cluster_oracle": -0.0561, "cluster_reached": True,
        "n_registered": 36000, "n_first_buy": 35612, "n_repeat": 25390,
        "n_high_freq": 10104, "conv_first": "98.9%", "conv_repeat": "71.3%",
        "conv_high_freq": "39.8%", "n_repeat_not_high": 15286,
        "peak_month": "2021-12", "peak_gmv_wan": 315.9, "trough_month": "2020-01",
        "trough_gmv_wan": 10.3, "season_ratio": 30.6, "nov_gmv_wan": 271.2,
        "weekend_ratio": "29.6%", "peak_hours": "9/10/11",
        "top5_categories": "饮料/乳制品", "top5_share": "56.2%", "n_categories": 60,
        "n_abc_a": 8, "n_rules": 361, "rule_洗发水_护发素": "洗发水→护发素 lift 6.06",
        "rule_火锅底料_牛羊肉": "火锅底料→牛羊肉 lift 3.51",
        "rule_面包糕点_乳制品": "面包糕点→乳制品 lift 1.66",
        "rule_纸尿裤_啤酒": "纸尿裤→啤酒 lift 2.23",
        "store_count": 15, "store_diff_max": 0.0002, "store_diff_mean": 0.0001,
        "store_gmv_total_wan": 13235.1,
        "s2_attr_lo": "15%", "s2_attr_hi": "25%", "s2_save_lo": "15%", "s2_save_hi": "30%",
        "s2_gmv_lo": "20.6", "s2_gmv_hi": "68.7", "s2_cost_wan": "17.8~29.7",
        "s2_roi_lo": "0.7", "s2_roi_hi": "3.9",
        "s5_gmv_lo": "3.7", "s5_gmv_hi": "7.3", "s5_lift_lo": "30%", "s5_lift_hi": "60%",
        "s5_rules_used": "10",
        "s7_ratio": "12.9", "s7_trough_gmv_wan": "24.5", "s7_peak_gmv_wan": "315.9",
        "s7_slow_lo": "31.6", "s7_slow_hi": "63.1", "s7_loss_lo": "9.5", "s7_loss_hi": "25.3",
        "s7_slow_rate_lo": "10%", "s7_slow_rate_hi": "20%",
        "s7_short_rate_lo": "3%", "s7_short_rate_hi": "8%",
        "s8_gmv_lo": "0.0", "s8_gmv_hi": "0.1", "s8_coef_lo": "3%", "s8_coef_hi": "8%",
        "generated_at": "2026-11-12 00:00:00", "excel_name": "r.xlsx",
        "q_last": "2022Q2", "q_last_yoy": "+69.7%", "q_yoy_max": "+684.1%",
        "q_yoy_max_q": "2021Q1", "q_last_incomplete": "2022Q3", "m_last": "2022-07",
        "m_last_yoy": "+85.9%",
    }


def test_calc_s2_formula():
    k = _kpis()
    res = rdr.calc_s2(k)
    # ΔGMV = N × 流失率 × 挽回率 × 月均消费 × 12
    expect_lo = 6600 * 0.15 * 0.15 * 115.7 * 12
    assert float(res["s2_gmv_lo"]) == pytest.approx(rdr._w(expect_lo), abs=0.1)
    # ROI 下界用最高成本、上界用最低成本
    assert float(res["s2_roi_lo"]) < float(res["s2_roi_hi"])


def test_calc_s5_uses_rules():
    rules = pd.DataFrame({
        "antecedents": ["洗发水"], "consequents": ["护发素"],
        "support": [0.05], "confidence": [0.5], "lift": [6.0],
    })
    cat_gmv = pd.Series({"护发素": 1e5})
    res = rdr._calc_s5(rules, cat_gmv)
    assert res["s5_rules_used"] == "1"
    assert float(res["s5_gmv_hi"]) > float(res["s5_gmv_lo"])


def test_calc_s7_excludes_partial_months():
    monthly = pd.DataFrame({
        "month": pd.period_range("2024-01", periods=6, freq="M"),
        "gmv": [1.0e4, 1.0e5, 2.0e5, 3.0e5, 4.0e5, 5.0e5],
    })
    res = rdr._calc_s7(monthly, {})
    # 去掉首尾月后，谷底基准 = 最低 3 个月均值 = (100000+200000+300000)/3 = 200000 元
    assert float(res["s7_trough_gmv_wan"]) == pytest.approx(20.0)
    assert float(res["s7_peak_gmv_wan"]) == pytest.approx(40.0)
    assert res["s7_ratio"] == "2.0"


def test_calc_s8_scales_with_difference():
    store_gmv = pd.Series({1: 1.0e6, 2: 1.0e6})
    diffs = {1: 0.10, 2: 0.20}
    res = rdr._calc_s8(store_gmv, diffs)
    assert float(res["s8_gmv_hi"]) > float(res["s8_gmv_lo"])
    # Σ(门店GMV × 差异度) = 30 万元；× 3% = 0.9 万元
    assert float(res["s8_gmv_lo"]) == pytest.approx(0.9)


def test_render_substitutes_all_placeholders():
    tpl = rdr.TEMPLATE_PATH.read_text(encoding="utf-8")
    out = rdr.render(_kpis())
    assert "{{" not in out and "}}" not in out
    assert "未达标" in out          # 布尔值已转中文
    assert "35612" in out
    # 模板里的每个占位符都必须有对应 KPI
    import re
    for key in re.findall(r"\{\{\s*(\w+)\s*\}\}", tpl):
        assert key in _kpis(), key


def test_render_raises_on_missing_kpi(tmp_path, monkeypatch):
    fake = tmp_path / "t.tpl"
    fake.write_text("{{unknown_key}}", encoding="utf-8")
    monkeypatch.setattr(rdr, "TEMPLATE_PATH", fake)
    with pytest.raises(KeyError):
        rdr.render(_kpis())


def test_pct_and_w_helpers():
    assert rdr._pct(0.1853) == "18.5%"
    assert rdr._w(1e5) == 10.0
