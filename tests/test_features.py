"""The most important test: a forecast made on day t must not use anything from after t."""
import pandas as pd
import pytest

import config
from backtest import training_rows
from features import BASELINES, FEATURES, make_rows

DAY = pd.Timestamp("2022-05-15")


def future_changed(daily: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Add 100 degC to everything that was NOT known on the evening of day t:
    observations after t, and forecasts issued after t (f_d for day D was issued on D - d)."""
    d = daily.copy()
    d.loc[d["date"] > t, ["tmax", "tmin"]] += 100
    for lead in range(1, config.MAX_PREVIOUS_DAY + 1):
        d.loc[d["date"] - pd.Timedelta(days=lead) > t, f"f{lead}"] += 100
    return d


def rows_of_day(rows, t):
    return rows[rows["issue_date"] == t].sort_values(["city", "lead"]).reset_index(drop=True)


def test_features_do_not_see_the_future(daily, rows):
    changed = make_rows(future_changed(daily, DAY))
    before, after = rows_of_day(rows, DAY), rows_of_day(changed, DAY)
    assert len(before) == len(after) > 0
    pd.testing.assert_frame_equal(before[FEATURES + BASELINES], after[FEATURES + BASELINES])
    # ... while the answer (the future) really did change, so the test can fail
    assert (after["y_true"] - before["y_true"]).dropna().to_numpy() == pytest.approx(100)


def test_the_next_day_does_change(daily, rows):
    """Control: the same change DOES reach the rows of the following day."""
    changed = make_rows(future_changed(daily, DAY))
    nxt = DAY + pd.Timedelta(days=1)
    before, after = rows_of_day(rows, nxt), rows_of_day(changed, nxt)
    assert not before["tmax_t"].equals(after["tmax_t"])


def test_forecast_is_the_run_issued_on_day_t(daily, rows):
    r = rows[(rows["city"] == "delhi") & (rows["lead"] == 3) & (rows["issue_date"] == DAY)].iloc[0]
    target = daily[(daily["city"] == "delhi") & (daily["date"] == DAY + pd.Timedelta(days=3))].iloc[0]
    assert r["f"] == target["f3"]            # made 3 days before the target = on day t
    assert r["f_prev"] == target["f4"]       # the run of the day before
    assert r["target_date"] == DAY + pd.Timedelta(days=3)


def test_bias_fix_baseline_uses_only_past_errors(daily, rows):
    city = daily[daily["city"] == "jaipur"].set_index("date")
    err = (city["tmax"] - city["f1"]).loc[DAY - pd.Timedelta(days=13): DAY]
    r = rows[(rows["city"] == "jaipur") & (rows["lead"] == 1) & (rows["issue_date"] == DAY)].iloc[0]
    assert r["bias14"] == pytest.approx(err.mean())
    assert r["pred_nwp_bias"] == pytest.approx(r["f"] + err.mean())


def test_training_rows_only_have_answers_known_on_retrain_day(rows):
    r_day = pd.Timestamp("2023-03-01")
    train = training_rows(rows, r_day)
    assert (train["target_date"] <= r_day).all()
    assert train["y_true"].notna().all()
    # a 5-day forecast made 2 days before the retrain day has no answer yet
    assert not ((train["lead"] == 5) & (train["issue_date"] == r_day - pd.Timedelta(days=2))).any()
    expected = rows[(rows["target_date"] <= r_day) & rows["y_true"].notna()]
    assert len(train) == len(expected)
