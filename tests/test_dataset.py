import gzip
import json

import numpy as np
import pandas as pd
import pytest

import config
import dataset
from dataset import calendar_day, clean_station, climate_normal, heat_threshold, one_day_spikes


# --- IMD heatwave rule -------------------------------------------------------------------
@pytest.mark.parametrize("normal, expected", [
    (30.0, 40.0),     # cool normal: the 40 degC floor applies
    (37.0, 41.5),     # normal + 4.5
    (40.5, 45.0),     # normal + 4.5 = 45.0
    (42.0, 45.0),     # 45 degC is a heatwave whatever the normal
])
def test_heat_threshold(normal, expected):
    assert heat_threshold(normal) == pytest.approx(expected)


def test_threshold_is_the_imd_rule():
    """One number per day must give exactly the IMD rule with its 'and' / 'or'."""
    tmax, normal = np.meshgrid(np.arange(35, 50, 0.25), np.arange(28, 44, 0.25))   # exact binary steps
    imd = ((tmax >= 40) & (tmax - normal >= 4.5)) | (tmax >= 45)
    assert np.array_equal(tmax >= heat_threshold(normal), imd)


def test_calendar_day_same_for_leap_and_normal_years():
    d = pd.Series(pd.to_datetime(["2023-03-01", "2024-03-01", "2023-12-31", "2024-02-29"]))
    assert calendar_day(d).tolist() == [61, 61, 366, 60]


# --- cleaning --------------------------------------------------------------------------------
def hourly(rows):
    return pd.DataFrame(rows, columns=["time", "temp"]).assign(time=lambda d: pd.to_datetime(d["time"]))


def test_clean_station_rules_and_daily_values():
    day = "2024-05-01"
    rows = [(f"{day} {h:02d}:00", 30.0 + h * 0.2) for h in range(24)]
    rows[14] = (f"{day} 14:00", 44.0)           # the day's max (inside 10:00-18:00)
    rows[20] = (f"{day} 20:00", 46.0)           # evening: not part of the day's max
    rows[3] = (f"{day} 03:00", 21.0)            # night: the day's min
    rows += [
        (f"{day} 12:00", 50.0),                 # same time twice -> median of 30+12*0.2=32.4 and 50
        (f"{day} 13:00", np.nan),               # missing reading
        (f"{day} 15:00", 60.0),                 # impossible value
        ("2024-05-02 11:00", 35.0),             # next day: only 2 daytime readings -> dropped
        ("2024-05-02 12:00", 36.0),
        ("2024-05-09 12:00", 40.0),             # after 'today'
    ]
    daily, counts = clean_station(hourly(rows), today=pd.Timestamp("2024-05-03"))
    assert counts == {"missing_reading": 1, "outside_range": 1, "future_time": 1, "duplicate_time": 1,
                      "days_too_few_readings": 1, "days_one_day_spike": 0}
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["tmax"] == 44.0
    assert row["tmin"] == 21.0
    assert row["n_daytime"] == 9


def test_one_day_spikes():
    d = pd.DataFrame({"date": pd.date_range("2024-05-01", periods=7),
                      "tmax": [30.0, 31.0, 45.0, 31.0, 33.0, 38.0, 44.0]})
    # 45 sticks out 14 degC from both neighbours: an error. 38 -> 44 is heat building: kept.
    assert one_day_spikes(d).tolist() == [False, False, True, False, False, False, False]


def test_climate_normal_counts_years_and_smooths():
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "tmax": 35.0})
    normal, years = climate_normal(df)
    assert years == 30
    assert len(normal) == 366
    assert normal.to_numpy() == pytest.approx(35.0)


def test_read_station_converts_utc_to_indian_time(tmp_path):
    path = tmp_path / "42181.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("2024-05-01,8,41.5,10,20,0,,270,10,,1000,,1\n")
        f.write("2024-05-01,20,30.0,10,20,0,,270,10,,1000,,1\n")
    df = dataset.read_station(path)
    assert df["time"].tolist() == [pd.Timestamp("2024-05-01 13:30"), pd.Timestamp("2024-05-02 01:30")]
    assert df["temp"].tolist() == [41.5, 30.0]


def test_read_forecasts_daily_max_needs_all_daytime_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FORECAST_DIR", tmp_path)
    times = pd.date_range("2024-05-01 00:00", "2024-05-02 23:00", freq="h")
    times = times[times != pd.Timestamp("2024-05-02 15:00")]          # day 2 misses one daytime hour
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    for d in range(1, config.MAX_PREVIOUS_DAY + 1):
        hourly[f"temperature_2m_previous_day{d}"] = [30.0 + d + (t.hour == 15) * 10 + (t.hour == 21) * 20
                                                     for t in times]
    (tmp_path / "delhi_2024.json").write_text(json.dumps(hourly), encoding="utf-8")
    f = dataset.read_forecasts("delhi")
    assert f["f1"].iloc[0] == 41.0            # 15:00 peak counts, the 21:00 value does not
    assert f["f6"].iloc[0] == 46.0
    assert np.isnan(f["f1"].iloc[1])          # incomplete day -> no forecast
