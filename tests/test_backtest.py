import numpy as np
import pandas as pd
import pytest

import config
from backtest import Z, normal_cdf, online_conformal, prob_above, retrain_dates, training_rows


def test_normal_cdf():
    assert normal_cdf(0) == pytest.approx(0.5)
    assert normal_cdf(Z) == pytest.approx(0.9)


def test_prob_above():
    assert prob_above(45, 45, 1) == pytest.approx(0.5)
    assert prob_above(45, 55, 1) == pytest.approx(1.0)
    assert prob_above(45, 35, 1) == pytest.approx(0.0, abs=1e-12)
    p = prob_above(45, 44.9, 0.0)              # zero spread is floored, no division by zero
    assert 0 < p < 0.5


def test_online_conformal_fixes_a_too_narrow_interval():
    rng = np.random.default_rng(1)
    days = pd.date_range("2021-01-01", periods=1000)
    y = rng.normal(0, 1, len(days))
    fc = pd.DataFrame({"lead": 1, "issue_date": days, "target_date": days + pd.Timedelta(days=1),
                       "y_true": y, "pred": 0.0, "lo": -0.5, "hi": 0.5})      # covers only ~38%
    out = online_conformal(fc)
    late = out.iloc[400:]
    inside = (late["y_true"] >= late["lo_cal"]) & (late["y_true"] <= late["hi_cal"])
    assert 0.75 < inside.mean() < 0.86
    first = out.iloc[0]
    assert first["q_conformal"] == 0           # nothing observed yet: no change


def test_conformal_uses_only_observed_errors():
    """Errors whose day has not come yet must not move the interval."""
    days = pd.date_range("2021-01-01", periods=600)
    base = pd.DataFrame({"lead": 1, "issue_date": days, "target_date": days + pd.Timedelta(days=1),
                         "y_true": 0.0, "pred": 0.0, "lo": -1.0, "hi": 1.0})
    changed = base.copy()
    changed.loc[changed.index >= 500, "y_true"] = 50.0       # huge errors from day 500 on
    a, b = online_conformal(base), online_conformal(changed)
    assert np.allclose(a["q_conformal"].iloc[:499], b["q_conformal"].iloc[:499])


def test_backtest_retrains_only_on_the_past(rows, backtest):
    fc = backtest
    assert (fc["trained_until"] <= fc["issue_date"]).all()
    dates = retrain_dates("2023-06-01", fc["issue_date"].max())
    expected = fc["issue_date"].map(lambda d: dates[dates <= d].max())
    assert (fc["trained_until"] == expected).all()
    for (lead, r_day), g in fc.groupby(["lead", "trained_until"]):
        n = len(training_rows(rows[rows["lead"] == lead], r_day))
        assert (g["n_train"] == n).all()


def test_backtest_output_is_sane(backtest):
    fc = backtest
    for col in ["p_lgbm", "p_nwp", "p_nwp_bias", "p_climatology"]:
        assert fc[col].between(0, 1).all(), col
    assert (fc["lo_cal"] <= fc["hi_cal"]).all()
    assert set(fc["lead"]) == set(config.LEADS)
    assert fc["pred"].notna().all()
    # the correction learns the made-up warm bias, so it must beat the raw model
    s = fc.dropna(subset=["y_true"])
    assert (s["y_true"] - s["pred"]).abs().mean() < (s["y_true"] - s["pred_nwp"]).abs().mean()
