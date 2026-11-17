"""Step 2: clean the station data and build one daily table.

    python dataset.py

For every city and day: observed max temperature (tmax), the climate normal for that
day, the IMD heatwave threshold, whether it was a heatwave day, and what the weather
model forecast for that day 1..6 days earlier (f1..f6).
Output: data/processed/daily.csv, results/data_quality.csv, results/cities.csv
"""
import json
import math
import sys
from datetime import date

import numpy as np
import pandas as pd

import config

METEOSTAT_COLUMNS = ["date", "hour", "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt",
                     "pres", "tsun", "coco"]
IST = pd.Timedelta(hours=5, minutes=30)


# --- station observations -------------------------------------------------------------
def read_station(path) -> pd.DataFrame:
    """Hourly readings (Meteostat time is UTC) -> columns time (IST) and temp."""
    df = pd.read_csv(path, names=METEOSTAT_COLUMNS, compression="gzip", usecols=["date", "hour", "temp"])
    utc = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(df["hour"], unit="h")
    return pd.DataFrame({"time": utc + IST, "temp": df["temp"]})


def clean_station(hourly: pd.DataFrame, today) -> tuple:
    """Clean the hourly readings, then make one row per day:
    tmax = highest reading 10:00-18:00 IST, tmin = lowest reading of the day.
    Returns (daily DataFrame with date, tmax, tmin, n_daytime; {rule: rows removed})."""
    counts = {}
    df = hourly

    def drop(mask, rule):
        counts[rule] = int(mask.sum())
        return df[~mask]

    df = drop(df["time"].isna() | df["temp"].isna(), "missing_reading")
    lo, hi = config.TEMP_RANGE
    df = drop((df["temp"] < lo) | (df["temp"] > hi), "outside_range")
    df = drop(df["time"] >= pd.Timestamp(today) + pd.Timedelta(days=1), "future_time")
    before = len(df)
    df = df.groupby("time", as_index=False)["temp"].median()       # same time twice -> one reading
    counts["duplicate_time"] = before - len(df)

    df = df.assign(date=df["time"].dt.normalize())
    day = df[df["time"].dt.hour.isin(list(config.DAYTIME_HOURS))].groupby("date")["temp"].agg(["max", "count"])
    whole = df.groupby("date")["temp"].agg(["min", "count"])
    daily = pd.DataFrame({"tmax": day["max"], "n_daytime": day["count"]}).join(
        whole.rename(columns={"min": "tmin", "count": "n_day"}), how="outer").reset_index()
    few = daily["n_daytime"].fillna(0) < config.MIN_READINGS_DAYTIME
    counts["days_too_few_readings"] = int(few.sum())
    daily = daily[~few].copy()
    daily.loc[daily["n_day"] < config.MIN_READINGS_DAY, "tmin"] = np.nan
    daily = daily.sort_values("date").reset_index(drop=True)
    spikes = one_day_spikes(daily)
    counts["days_one_day_spike"] = int(spikes.sum())
    daily = daily[~spikes].reset_index(drop=True)
    return daily[["date", "tmax", "tmin", "n_daytime"]], counts


def one_day_spikes(df: pd.DataFrame) -> pd.Series:
    """A value more than SPIKE_DEGREES away from BOTH neighbouring days, in the same
    direction: a typing mistake, not weather (real heat builds up over days)."""
    s = df.set_index("date")["tmax"].asfreq("D")
    prev_gap, next_gap = s - s.shift(1), s - s.shift(-1)
    spike = (((prev_gap > config.SPIKE_DEGREES) & (next_gap > config.SPIKE_DEGREES))
             | ((prev_gap < -config.SPIKE_DEGREES) & (next_gap < -config.SPIKE_DEGREES)))
    return df["date"].map(spike).fillna(False).astype(bool)


def calendar_day(dates: pd.Series) -> pd.Series:
    """Day number 1..366 on a leap-year calendar, so 1 March is always day 61."""
    return pd.to_datetime({"year": 2000, "month": dates.dt.month, "day": dates.dt.day}).dt.day_of_year


def climate_normal(df: pd.DataFrame) -> tuple:
    """Average tmax for each calendar day over NORMAL_YEARS, smoothed over 31 days.
    Returns (Series indexed 1..366, number of years with enough data)."""
    y0, y1 = config.NORMAL_YEARS
    ref = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)]
    per_year = ref.groupby(ref["date"].dt.year).size()
    good_years = int((per_year >= 300).sum())
    if ref.empty:
        return pd.Series(np.nan, index=range(1, 367)), good_years
    mean = ref.groupby(calendar_day(ref["date"]))["tmax"].mean().reindex(range(1, 367))
    wrapped = pd.concat([mean.iloc[-15:], mean, mean.iloc[:15]])
    smooth = wrapped.rolling(31, center=True, min_periods=20).mean().iloc[15:-15]
    smooth.index = range(1, 367)
    return smooth, good_years


def heat_threshold(normal):
    """IMD rule as one number: heatwave if tmax >= this.
    (tmax >= 40 and tmax - normal >= 4.5) or tmax >= 45  <=>  tmax >= min(45, max(40, normal + 4.5))"""
    return np.minimum(config.HEAT_ABSOLUTE, np.maximum(config.HEAT_MIN_TMAX, normal + config.HEAT_DEPARTURE))


def climate_heat_probability(df: pd.DataFrame, normal: pd.Series) -> pd.Series:
    """How often each calendar day (+-7 days) was a heatwave day in NORMAL_YEARS:
    the 'climatology' forecast, which needs no weather model at all."""
    y0, y1 = config.NORMAL_YEARS
    ref = df[(df["date"].dt.year >= y0) & (df["date"].dt.year <= y1)].copy()
    ref["cday"] = calendar_day(ref["date"])
    ref["hw"] = ref["tmax"] >= heat_threshold(ref["cday"].map(normal))
    rate = ref.groupby("cday")["hw"].agg(["sum", "count"]).reindex(range(1, 367), fill_value=0)
    wrapped = pd.concat([rate.iloc[-7:], rate, rate.iloc[:7]])
    window = wrapped.rolling(15, center=True).sum().iloc[7:-7]
    prob = (window["sum"] / window["count"].replace(0, np.nan))
    prob.index = range(1, 367)
    return prob


# --- weather-model forecasts -------------------------------------------------------------
def read_forecasts(city_id: str) -> pd.DataFrame:
    """Daily forecast of the max temperature: the highest hourly value between 10:00 and
    18:00 IST, for each 'previous day' (1 = forecast made 1 day earlier, ...)."""
    frames = []
    for path in sorted(config.FORECAST_DIR.glob(f"{city_id}_*.json")):
        frames.append(pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))))
    if not frames:
        return pd.DataFrame(columns=["date"])
    h = pd.concat(frames, ignore_index=True)
    h["time"] = pd.to_datetime(h["time"])
    h = h.drop_duplicates("time", keep="last")
    h = h[h["time"].dt.hour.isin(list(config.DAYTIME_HOURS))]
    h["date"] = h["time"].dt.normalize()
    out = pd.DataFrame({"date": sorted(h["date"].unique())})
    for d in range(1, config.MAX_PREVIOUS_DAY + 1):
        col = f"temperature_2m_previous_day{d}"
        g = h.groupby("date")[col].agg(["max", "count"])
        complete = g["count"] == len(config.DAYTIME_HOURS)          # all 9 hours present
        out[f"f{d}"] = out["date"].map(g["max"].where(complete))
    return out


# --- everything together ------------------------------------------------------------------
def build(today=None) -> tuple:
    today = today or date.today()
    quality, summary, frames = [], [], []
    cleaned = {}
    for c in config.CITIES:
        path = config.STATION_DIR / f"{c['station']}.csv.gz"
        if not path.exists():
            summary.append({"city": c["id"], "kept": False, "reason": "station file missing"})
            continue
        hourly = read_station(path)
        df, counts = clean_station(hourly, today)
        quality.append({"city": c["id"], "readings_raw": len(hourly), **counts, "days_clean": len(df)})
        cleaned[c["id"]] = df

    # the data ends where most stations end (one late station should not cut everyone short)
    last_dates = pd.Series({k: v["date"].max() for k, v in cleaned.items()})
    end = last_dates.median().normalize()
    start = pd.Timestamp(config.FORECAST_START)
    period = pd.date_range(start, end, freq="D")

    for c in config.CITIES:
        if c["id"] not in cleaned:
            continue
        df = cleaned[c["id"]]
        normal, normal_years = climate_normal(df)
        in_period = df[(df["date"] >= start) & (df["date"] <= end)]
        coverage = len(in_period) / len(period)
        row = {"city": c["id"], "name": c["name"], "station": c["station"], "normal_years": normal_years,
               "coverage_since_2021": round(coverage, 3), "last_date": str(df["date"].max().date())}
        if normal_years < config.MIN_NORMAL_YEARS:
            summary.append({**row, "kept": False, "reason": f"only {normal_years} years for the climate normal"})
            continue
        if coverage < config.MIN_COVERAGE:
            summary.append({**row, "kept": False, "reason": f"max temperature on only {coverage:.0%} of days"})
            continue
        fc = read_forecasts(c["id"])
        d = pd.DataFrame({"date": period})
        d = d.merge(df[["date", "tmax", "tmin"]], on="date", how="left").merge(fc, on="date", how="left")
        cday = calendar_day(d["date"])
        d["normal"] = cday.map(normal)
        d["threshold"] = heat_threshold(d["normal"])
        d["heatwave"] = np.where(d["tmax"].isna(), np.nan, (d["tmax"] >= d["threshold"]).astype(float))
        d["clim_prob"] = cday.map(climate_heat_probability(df, normal))
        d.insert(0, "city", c["id"])
        frames.append(d)
        hw = d[d["date"].dt.month.isin(config.HEAT_SEASON_MONTHS)]["heatwave"]
        summary.append({**row, "kept": True, "reason": "", "heatwave_days": int(hw.sum()),
                        "forecast_days": int(d["f1"].notna().sum())})

    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return daily, pd.DataFrame(quality), pd.DataFrame(summary)


def load_daily(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.DAILY_CSV, parse_dates=["date"])


def main() -> int:
    daily, quality, summary = build()
    if daily.empty:
        print("No city has enough data. Run download.py first.")
        return 1
    for d in (config.PROCESSED_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    daily.to_csv(config.DAILY_CSV, index=False, float_format="%.2f")
    quality.to_csv(config.RESULTS_DIR / "data_quality.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "cities.csv", index=False)
    kept = summary[summary["kept"]]
    print(f"Data quality (readings / days removed per rule):\n{quality.to_string(index=False)}\n")
    print(f"Cities kept: {len(kept)} of {len(summary)}")
    print(summary.drop(columns=["name"]).to_string(index=False))
    print(f"\nDaily table: {len(daily):,} rows, {daily['date'].min().date()} .. {daily['date'].max().date()}")
    hw = daily[daily["date"].dt.month.isin(config.HEAT_SEASON_MONTHS)]
    print(f"Heatwave days (Mar-Jun): {int(hw['heatwave'].sum())} of {int(hw['heatwave'].notna().sum())} "
          f"station-days ({hw['heatwave'].mean():.1%})")
    if math.isnan(daily["f1"].mean()):
        print("WARNING: no forecasts found. Run download.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
