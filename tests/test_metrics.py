import numpy as np
import pandas as pd
import pytest

import metrics as M


def test_block_bootstrap_constant_values():
    v = pd.Series([2.0] * 70)
    b = M.block_bootstrap(v, pd.Series(np.arange(70) // 7), n=200)
    assert b["mean"] == b["ci_low"] == b["ci_high"] == 2.0
    assert b["share_positive"] == 1.0


def test_block_bootstrap_interval_contains_mean():
    rng = np.random.default_rng(0)
    v = pd.Series(rng.normal(0.5, 1, 700))
    b = M.block_bootstrap(v, pd.Series(np.arange(700) // 7), n=500)
    assert b["ci_low"] < b["mean"] < b["ci_high"]
    assert b["ci_low"] > 0


def test_brier():
    assert M.brier([1, 0], [1, 0]) == 0
    assert M.brier([0.5, 0.5], [1, 0]) == 0.25


def heat_rows():
    return pd.DataFrame({
        "lead": 1, "heatwave": [1, 1, 0, 0], "threshold": 45.0,
        "pred_nwp": [46, 44, 46, 44], "pred_nwp_bias": [46, 46, 44, 44],
        "p_lgbm": [0.6, 0.35, 0.1, 0.4],
    })


def test_warning_scores():
    w = M.warning_scores(heat_rows()).set_index("method")
    raw = w.loc["Weather model, raw (GFS)"]
    assert (raw["hits"], raw["misses"], raw["false_alarms"]) == (1, 1, 1)
    assert raw["POD"] == 0.5 and raw["FAR"] == 0.5 and raw["CSI"] == pytest.approx(1 / 3, abs=1e-3)
    bias = w.loc["Weather model + recent-bias fix"]
    assert (bias["POD"], bias["FAR"]) == (1.0, 0.0)
    cautious = w.loc["LightGBM, chance >= 30% (cautious)"]
    assert (cautious["hits"], cautious["false_alarms"]) == (2, 1)


def test_reliability_groups_by_forecast_chance():
    h = pd.DataFrame({"p": [0.01, 0.02, 0.6, 0.65, 0.62], "heatwave": [0, 0, 1, 1, 0]})
    r = M.reliability(h, "p")
    assert r["days"].tolist() == [2, 3]
    assert r["observed"].tolist() == pytest.approx([0, 2 / 3])


def test_compare_to_best_baseline_sign():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(40, 3, n)
    df = pd.DataFrame({"lead": 1, "issue_date": pd.date_range("2023-01-01", periods=n), "y_true": y,
                       "pred": y + rng.normal(0, 0.5, n)})
    for k, col in enumerate(M.BASELINES):
        df[col] = y + rng.normal(0, 1 + k, n)
    c = M.compare_to_best_baseline(df).iloc[0]
    assert c["best_baseline"] == M.MODELS[M.BASELINES[0]]
    assert c["improvement"] > 0 and c["ci_low"] > 0


def test_coverage_table():
    df = pd.DataFrame({"lead": 1, "y_true": [1.0, 2.0, 3.0, 10.0], "lo": 0.0, "hi": 2.5, "lo_cal": 0.0,
                       "hi_cal": 3.5})
    c = M.coverage_table(df).iloc[0]
    assert c["coverage_raw"] == 0.5 and c["coverage_calibrated"] == 0.75
