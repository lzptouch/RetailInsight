"""流失模型单测：阈值调优、风险分层、评估指标、时间切分。"""
import numpy as np
import pandas as pd
import pytest

from retailinsight.models.churn import COHORT_CUTOFFS, FEATURES, _risk_level, evaluate, tune_thresholds
from retailinsight.settings import COHORT_CUTOFFS as SET_CUTOFFS


def test_features_constant_length():
    assert len(FEATURES) == 31
    assert len(set(FEATURES)) == 31


def test_evaluate_metrics():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    m = evaluate(y, p)
    assert m["auc"] == pytest.approx(1.0)
    assert m["ks"] == pytest.approx(1.0, abs=1e-6)
    assert m["lift_top20"] == pytest.approx(2.0, abs=1e-6)
    assert m["base_rate"] == pytest.approx(0.5)


def test_tune_thresholds_monotonic_and_sane():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    p = np.concatenate([rng.normal(0.2, 0.05, 900), rng.normal(0.7, 0.1, 100)])
    th = tune_thresholds(y, p)
    assert th["action"] <= th["mid"] <= th["high"]
    assert 0.05 <= th["action"] <= 0.95
    # 高危名单应该比行动名单小 → 切点更高
    assert th["high"] >= th["action"]


def test_tune_thresholds_prefers_aggressive_when_fn_costly():
    """漏判成本越高，越应该多标人（切点更低）。"""
    rng = np.random.default_rng(1)
    y = np.array([0] * 900 + [1] * 100)
    p = np.concatenate([rng.normal(0.2, 0.05, 900), rng.normal(0.7, 0.1, 100)])
    costly = tune_thresholds(y, p, cost_fn=20.0)
    cheap = tune_thresholds(y, p, cost_fn=1.0)
    assert costly["action"] <= cheap["action"]


def test_risk_level_uses_tuned_thresholds():
    th = {"mid": 0.5, "high": 0.8}
    prob = pd.Series([0.1, 0.6, 0.9])
    lv = _risk_level(prob, th)
    assert list(lv.astype(str)) == ["低", "中", "高"]


def test_cohort_split_is_time_ordered():
    """训练/验证/测试必须严格按时间先后，且测试集为末期。"""
    assert len(SET_CUTOFFS) >= 4
    assert sorted(SET_CUTOFFS) == SET_CUTOFFS
    assert SET_CUTOFFS[-1] == COHORT_CUTOFFS[-1]
