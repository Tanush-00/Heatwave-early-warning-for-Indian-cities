"""Shared test data: a small made-up daily table that looks like the real one.
No test touches the network or the files in data/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from dataset import heat_threshold  # noqa: E402


def synthetic_daily(cities=("delhi", "jaipur"), start="2021-04-01", end="2024-06-30", seed=0) -> pd.DataFrame:
    """Seasonal max temperature + day-to-day weather + a weather model that runs warm by
    a different amount in each city, with errors growing with lead time."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.day_of_year.to_numpy()
    frames = []
    for i, city in enumerate(cities):
        normal = 33 + 8 * np.sin(2 * np.pi * (doy - 50) / 365.25)          # warmest in late May
        anomaly = np.zeros(len(dates))
        for k in range(1, len(dates)):
            anomaly[k] = 0.7 * anomaly[k - 1] + rng.normal(0, 1.8)
        tmax = normal + anomaly
        d = pd.DataFrame({"city": city, "date": dates, "tmax": tmax, "tmin": tmax - 12})
        for lead in range(1, config.MAX_PREVIOUS_DAY + 1):
            d[f"f{lead}"] = tmax + (1.0 + i) + rng.normal(0, 0.5 + 0.4 * lead, len(dates))
        d["normal"] = normal
        d["threshold"] = heat_threshold(normal)
        d.loc[rng.random(len(dates)) < 0.02, ["tmax", "tmin"]] = np.nan            # station gaps
        d["heatwave"] = np.where(d["tmax"].isna(), np.nan, (d["tmax"] >= d["threshold"]).astype(float))
        d["clim_prob"] = np.where(dates.month.isin([4, 5, 6]), 0.05, 0.0)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    return synthetic_daily()


@pytest.fixture(scope="session")
def rows(daily) -> pd.DataFrame:
    from features import make_rows
    return make_rows(daily)


@pytest.fixture(scope="session")
def backtest(rows) -> pd.DataFrame:
    """A short, fast backtest (small models) on the made-up data."""
    from backtest import run_backtest
    return run_backtest(rows, params={"n_estimators": 30}, test_start="2023-06-01", log=lambda *_: None)
