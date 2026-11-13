"""决策建议报告渲染器单测：KPI 回填、测算公式、模板渲染。"""
import pandas as pd
import pytest

from retailinsight.reports import render_decision_report as rdr


def _kpis():
    return {
        "n_members": 35650, "n_profile": 36000, "n_orders": 308956, "gmv_wan": 4453.4,
        "data_window": "2020-01-01 ~ 2022-07-31", "n_months": 31,
        "n_vip": 6531, "vip_share": "18.3%", "vip_gmv_share": "42.5%",
        "vip_arpu": 2784.4, "vip_arpu_month": 116.0,
        "aov_member": 144.0, "monthly_freq_member": 0.5,
        "n_churn_risk": 13485, "churn_risk_share": "37.8%",
        "churn_risk_gmv_share": "9.8%", "churn_risk_avg_recency": 305,
        "top2_share": "36.6%", "top2_gmv_share": "77.8%",
        "n_high_risk": 4666, "high_risk_arpu": 204.3, "high_risk_arpu_month": 8.5,
        "model_backend": "lightgbm", "model_name": "随机森林（RandomForest）",
        "model_name_raw": "random_forest",
        "model_auc": 0.83, "model_auc_lr": 0.7995,
        "model_auc_gain": 0.0316, "model_auc_gain_verdict": "已达标",
        "model_ks": 0.4936, "model_lift": 2.34, "model_capture": 0.4687,
        "model_base_rate": 0.3672, "model_lift_upper": 2.72, "model_n_features": 31,
        "model_cohorts": "a / b / c", "model_valid_cohort": "d", "model_test_cohort": "e",
        "model_gbdt_auc": 0.8179, "model_gbdt_gain": 0.0191, "model_gbdt_ks": 0.4836,
        "model_rf_auc": 0.83, "model_rf_gain": 0.0316,
        "model_selection_reason": "random_forest 同时满足 AUC ≥ 基线+0.03 与最高 KS",
        "model_verdict_paragraph": "主模型由验收线自动选出：**随机森林（RandomForest）**。",
        "threshold_action": 0.28, "threshold_high": 0.63,
        "cluster_space": "rfm3", "cluster_k": 4, "cluster_silhouette": 0.4165,
        "cluster_ari": 0.1339, "cluster_oracle": -0.0563, "cluster_reached": True,
        "n_registered": 36000, "n_first_buy": 35650, "n_repeat": 25485,
        "n_high_freq": 9991, "conv_first": "99.0%", "conv_repeat": "71.5%",
        "conv_high_freq": "39.2%", "n_repeat_not_high": 15494,
        "peak_month": "2021-12", "peak_gmv_wan": 315.4, "trough_month": "2020-01",
        "trough_gmv_wan": 24.7, "season_ratio": 28.4, "nov_gmv_wan": 274.2,
        "weekend_ratio": "29.7%", "peak_hours": "10/9/11",
        "top5_categories": "饮料/乳制品", "top5_share": "57.9%", "n_categories": 60,
        "n_abc_a": 7, "n_rules": 373, "rule_洗发水_护发素": "洗发水→护发素 lift 6.39",
        "rule_火锅底料_牛羊肉": "火锅底料→牛羊肉 lift 3.78",
        "rule_面包糕点_乳制品": "面包糕点→乳制品 lift 1.66",
        "rule_纸尿裤_啤酒": "纸尿裤→啤酒 lift 2.23",
        "store_count": 15, "store_diff_max": 0.1244, "store_diff_mean": 0.0502,
        "store_gmv_total_wan": 13996.5, "scale_line": "",
        "s1_gmv_lo": "22.4", "s1_gmv_hi": "44.8", "s1_touch": "80%",
        "s1_save_lo": "10%", "s1_save_hi": "20%", "s1_arpu_m": "100", "s1_months": "6",
        "s2_attr_lo": "15%", "s2_attr_hi": "25%", "s2_save_lo": "15%", "s2_save_hi": "30%",
        "s2_gmv_lo": "20.5", "s2_gmv_hi": "68.2", "s2_cost_wan": "17.6~29.4",
        "s2_roi_lo": "0.7", "s2_roi_hi": "3.9",
        "s3_gmv_wan": "46.6", "s3_conv": "5%", "s3_freq_gain": "0.35", "s3_aov": "144",
        "s4_cost_all_wan": "36.0", "s4_cost_target_wan": "7.1", "s4_save_wan": "28.9",
        "s4_save_rate": "80%", "s4_n_all": "36,000", "s4_n_target": "7,130",
        "s4_coupon_cost": "10", "s4_top_pct": "20%", "s4_threshold": "0.0802",
        "s5_gmv_lo": "3.6", "s5_gmv_hi": "7.2", "s5_lift_lo": "30%", "s5_lift_hi": "60%",
        "s5_rules_used": "10",
        "s7_ratio": "12.8", "s7_trough_gmv_wan": "24.7", "s7_peak_gmv_wan": "315.4",
        "s7_slow_lo": "31.6", "s7_slow_hi": "63.3", "s7_loss_lo": "9.5", "s7_loss_hi": "25.2",
        "s7_slow_rate_lo": "10%", "s7_slow_rate_hi": "20%",
        "s7_short_rate_lo": "3%", "s7_short_rate_hi": "8%",
        "s8_gmv_lo": "0.0", "s8_gmv_hi": "0.1", "s8_coef_lo": "3%", "s8_coef_hi": "8%",
        "generated_at": "2026-11-13 00:00:00", "excel_name": "r.xlsx",
        "q_last": "2022Q2", "q_last_yoy": "+69.3%", "q_yoy_max": "+657.5%",
        "q_yoy_max_q": "2021Q1", "q_last_incomplete": "2022Q3", "m_last": "2022-07",
        "m_last_yoy": "+85.6%",
    }


def test_calc_s2_formula():
    k = _kpis()
    res = rdr.calc_s2(k)
    # ΔGMV = N × 流失率 × 挽回率 × 月均消费 × 12
    expect_lo = k["n_vip"] * 0.15 * 0.15 * k["vip_arpu_month"] * 12
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
    assert "已达标" in out          # 布尔值已转中文
    assert "35650" in out
    # 模板里的每个占位符都必须有对应 KPI
    import re
    for key in re.findall(r"\{\{\s*(\w+)\s*\}\}", tpl):
        assert key in _kpis(), key


def test_render_scale_line_only_when_small_sample():
    """`scale_line` 非空才出现在报告里；为空时整行占位符消失（不留空行）。

    这是 v3.6 #29 的护栏：`--scale 0.05` 之类的小规模跑批会覆盖 `docs/`
    里的交付文档，首行必须能自证「这不是正式口径」。
    """
    small = rdr.render(dict(_kpis(), scale_line="> ⚠ 本报告基于小规模验证数据生成"))
    assert "小规模验证数据" in small

    full = rdr.render(dict(_kpis(), scale_line=""))
    assert "小规模验证数据" not in full
    assert "\n\n\n" not in full         # 空占位符不许留下连续空行


def test_calc_s1_formula():
    """S1 高危挽留：ΔGMV = N_高危 × 触达率 × 挽回率 × 月均消费 × 6 个月。"""
    k = _kpis()
    res = rdr._calc_s1(k)
    expect_lo = k["n_high_risk"] * 0.80 * 0.10 * 100.0 * 6
    expect_hi = k["n_high_risk"] * 0.80 * 0.20 * 100.0 * 6
    assert float(res["s1_gmv_lo"]) == pytest.approx(rdr._w(expect_lo), abs=0.1)
    assert float(res["s1_gmv_hi"]) == pytest.approx(rdr._w(expect_hi), abs=0.1)


def test_calc_s4_uses_measured_tag():
    """S4 券成本节省：全域人数与精准投放人数都来自实测，节省率必须自洽。"""
    # 标签右偏：Top 20% 的圈选口径应选出 2,000 人（不是固定阈值选出几个）
    profile = pd.DataFrame({
        "discount_sensitivity": [0.9] * 1000 + [0.5] * 1000 + [0.1] * 8000,
    })
    k = {"n_profile": 10000}
    res = rdr._calc_s4(k, profile)
    assert res["s4_n_target"] == "2,000", "应按 Top 20% 圈选"
    assert float(res["s4_cost_all_wan"]) == pytest.approx(10.0)      # 10000 × 10 元
    assert float(res["s4_cost_target_wan"]) == pytest.approx(2.0)   # 2000 × 10 元
    assert float(res["s4_save_rate"].rstrip("%")) == pytest.approx(80.0)


def test_calc_s3_formula():
    """S3 复购→高频：人数来自漏斗，客单价来自 mart_rfm 实测。"""
    k = {"n_repeat_not_high": 10000, "aov_member": 100.0, "monthly_freq_member": 0.5}
    res = rdr._calc_s3(k)
    assert float(res["s3_gmv_wan"]) > 0
    assert res["s3_conv"] == "5%"


def test_render_raises_on_missing_kpi(tmp_path, monkeypatch):
    fake = tmp_path / "t.tpl"
    fake.write_text("{{unknown_key}}", encoding="utf-8")
    monkeypatch.setattr(rdr, "TEMPLATE_PATH", fake)
    with pytest.raises(KeyError):
        rdr.render(_kpis())


def test_pct_and_w_helpers():
    assert rdr._pct(0.1853) == "18.5%"
    assert rdr._w(1e5) == 10.0


def test_profile_cards_scale_warning_only_fires_on_small_runs():
    """画像卡片同样要自证规模：小样本跑批首行必须告警，正式口径不告警。"""
    from retailinsight.reports.profile_cards import PORTFOLIO_MEMBERS, _scale_warning

    assert _scale_warning(PORTFOLIO_MEMBERS) == ""
    assert _scale_warning(PORTFOLIO_MEMBERS + 5000) == ""
    warn = _scale_warning(int(PORTFOLIO_MEMBERS * 0.2))
    assert "小规模验证数据" in warn and "作品集口径" in warn
