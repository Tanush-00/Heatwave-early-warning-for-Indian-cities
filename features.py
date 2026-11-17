"""One row per (city, forecast day t, lead L).

The forecast is made on the EVENING of day t (that day's max temperature is known by
then) for day t + L. Every column on row t uses only what was known on day t:
- station readings up to day t,
- weather-model forecasts that were issued on or before day t,
- the climate normal (fixed: built from 1991-2020, before any test year).
The answer (y_true = tmax on day t + L) is only used as the target.
"""
import numpy as np
import pandas as pd

import config

FEATURES = ["f", "revision", "tmax_t", "tmin_t", "tmax_3d", "anom_t", "bias7", "bias14",
            "normal_target", "f_anom", "f_minus_thr", "doy_sin", "doy_cos", "city_code"]
BASELINES = ["pred_persistence", "pred_climatology", "pred_nwp", "pred_nwp_bias"]


def city_rows(g: pd.DataFrame, lead: int, code: int) -> pd.DataFrame:
    g = g.set_index("date").asfreq("D")                 # every calendar day, gaps become NaN
    later = lambda s: s.shift(-lead)                    # value of day t + lead, put on row t

    f_for_day = g[f"f{lead}"]                           # forecast made `lead` days before each day
    err = g["tmax"] - f_for_day                         # the model's error on each day (known that evening)

    out = pd.DataFrame(index=g.index)
    out["target_date"] = out.index + pd.Timedelta(days=lead)
    out["y_true"] = later(g["tmax"])
    out["f"] = later(f_for_day)                         # issued on day t
    out["f_prev"] = later(g[f"f{lead + 1}"])            # issued on day t - 1 for the same day
    out["revision"] = out["f"] - out["f_prev"]          # did the newest run change its mind?
    out["tmax_t"] = g["tmax"]
    out["tmin_t"] = g["tmin"]
    out["tmax_3d"] = g["tmax"].rolling(3, min_periods=2).mean()
    out["anom_t"] = g["tmax"] - g["normal"]
    out["bias7"] = err.rolling(7, min_periods=4).mean()            # days t-6 .. t
    out["bias14"] = err.rolling(config.BIAS_WINDOW_DAYS, min_periods=7).mean()
    out["normal_target"] = later(g["normal"])
    out["threshold"] = later(g["threshold"])
    out["heatwave"] = later(g["heatwave"])
    out["clim_prob"] = later(g["clim_prob"])
    out["f_anom"] = out["f"] - out["normal_target"]
    out["f_minus_thr"] = out["f"] - out["threshold"]
    doy = out["target_date"].dt.day_of_year
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["city_code"] = code
    out["lead"] = lead

    # the four simple forecasts every model must beat
    out["pred_persistence"] = out["tmax_t"]                         # "same as today"
    out["pred_climatology"] = out["normal_target"]                  # "a normal day"
    out["pred_nwp"] = out["f"]                                      # the weather model as it is
    out["pred_nwp_bias"] = out["f"] + out["bias14"].fillna(0)       # model minus its recent error
    return out.rename_axis("issue_date").reset_index()


def make_rows(daily: pd.DataFrame, leads=None) -> pd.DataFrame:
    leads = leads or config.LEADS
    codes = {cid: i for i, cid in enumerate(config.CITY_IDS)}
    frames = [city_rows(g, lead, codes.get(city, len(codes)))
              .assign(city=city) for city, g in daily.groupby("city") for lead in leads]
    rows = pd.concat(frames, ignore_index=True)
    return rows[rows["f"].notna()].reset_index(drop=True)
