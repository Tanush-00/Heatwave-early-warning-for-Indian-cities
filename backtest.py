"""Walk-forward backtest: forecast every day of the test period as if it were that day.

- Every RETRAIN_EVERY_DAYS days (date R) the models are retrained, only on rows whose
  answer was already observed: target_date <= R. Forecasts made on days R .. next R use them.
- One set of models per lead (direct forecasting), three LightGBM quantile models each
  (10th, 50th, 90th percentile). They predict the weather model's ERROR, so the forecast
  is "weather model + learned correction".
- The 80% interval is then calibrated online (conformal): widen or narrow it by what the
  last year of ALREADY-OBSERVED errors needed.
- Heatwave probability = chance that tmax >= the IMD threshold, from the forecast and its
  interval (normal curve). Baselines get a probability the same way. A direct LightGBM
  yes/no classifier is also trained and saved (p_classifier) for comparison only.
"""
import math

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

import config
from features import FEATURES

QUANTILES = (config.INTERVAL_LOW, 0.5, config.INTERVAL_HIGH)
Z = 1.2815515655446004                     # N(0,1) 90th percentile: half an 80% interval in SDs
_erf = np.vectorize(math.erf)


def normal_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def prob_above(threshold, mean, sd):
    """P(value >= threshold) for a normal curve with this mean and spread."""
    sd = np.maximum(np.asarray(sd, dtype=float), 0.3)
    return 1.0 - normal_cdf((np.asarray(threshold, dtype=float) - np.asarray(mean, dtype=float)) / sd)


def retrain_dates(start, end) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{config.RETRAIN_EVERY_DAYS}D")


def training_rows(rows: pd.DataFrame, retrain_date) -> pd.DataFrame:
    """Only rows whose answer was known on the retrain day."""
    known = (rows["target_date"] <= pd.Timestamp(retrain_date)) & rows["y_true"].notna()
    return rows[known]


def fit_quantiles(train: pd.DataFrame, params=None) -> list:
    params = {**config.LGBM_PARAMS, **(params or {})}
    target = train["y_true"] - train["f"]
    return [LGBMRegressor(objective="quantile", alpha=q, random_state=config.SEED, **params)
            .fit(train[FEATURES], target) for q in QUANTILES]


def fit_heat_classifier(train: pd.DataFrame, params=None):
    """Direct yes/no model for 'tmax >= heatwave threshold'. Returns None when the training
    rows have no heatwave day yet (then the probability from the quantiles is used)."""
    params = {**config.LGBM_PARAMS, **(params or {})}
    y = (train["y_true"] >= train["threshold"]).astype(int)
    if y.nunique() < 2:
        return None
    return LGBMClassifier(random_state=config.SEED, **params).fit(train[FEATURES], y)


def predict_quantiles(models, rows: pd.DataFrame) -> np.ndarray:
    corr = np.column_stack([m.predict(rows[FEATURES]) for m in models])
    return np.sort(corr, axis=1) + rows["f"].to_numpy()[:, None]       # sorted: lo <= median <= hi


def error_spread(train: pd.DataFrame, column: str) -> pd.Series:
    """Per city: standard deviation of a baseline's past errors (for its probability)."""
    return (train["y_true"] - train[column]).groupby(train["city"]).std()


def online_conformal(fc: pd.DataFrame) -> pd.DataFrame:
    """For each forecast day d: how much did the raw interval need widening (q) to hold
    80% of the errors of the last year that were observed by day d? Then lo -= q, hi += q."""
    fc = fc.copy()
    fc["q_conformal"] = 0.0
    level = config.INTERVAL_HIGH - config.INTERVAL_LOW
    for lead, part in fc.groupby("lead"):
        known = part[part["y_true"].notna()]
        score = np.maximum(known["lo"] - known["y_true"], known["y_true"] - known["hi"])
        tdates = known["target_date"].to_numpy()
        for d in np.sort(part["issue_date"].unique()):
            window = (tdates <= d) & (tdates > d - np.timedelta64(config.CONFORMAL_WINDOW_DAYS, "D"))
            s = score.to_numpy()[window]
            if len(s) < 100:
                continue
            k = min(1.0, math.ceil((len(s) + 1) * level) / len(s))
            fc.loc[part.index[part["issue_date"].to_numpy() == d], "q_conformal"] = np.quantile(s, k)
    fc["lo_cal"] = fc["lo"] - fc["q_conformal"]
    fc["hi_cal"] = fc["hi"] + fc["q_conformal"]
    bad = fc["lo_cal"] > fc["hi_cal"]                       # a very negative q: keep a point interval
    fc.loc[bad, ["lo_cal", "hi_cal"]] = fc.loc[bad, ["pred", "pred"]].to_numpy()
    return fc


def add_probabilities(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    sd_model = (fc["hi_cal"] - fc["lo_cal"]) / (2 * Z)
    # Chosen: the chance from the temperature forecast and its calibrated range. (A direct
    # yes/no classifier, p_classifier, was tried and scored worse: too few heatwave days to
    # learn from. It stays in the output so that comparison can be checked.)
    fc["p_lgbm"] = prob_above(fc["threshold"], fc["pred"], sd_model)
    fc["p_nwp"] = prob_above(fc["threshold"], fc["pred_nwp"], fc["sd_nwp"])
    fc["p_nwp_bias"] = prob_above(fc["threshold"], fc["pred_nwp_bias"], fc["sd_nwp_bias"])
    fc["p_climatology"] = fc["clim_prob"]
    return fc


def run_backtest(rows: pd.DataFrame, params=None, test_start=None, log=print) -> pd.DataFrame:
    test_start = pd.Timestamp(test_start or config.TEST_START)
    test = rows[rows["issue_date"] >= test_start]
    dates = retrain_dates(test_start, test["issue_date"].max())
    pieces = []
    for i, r_date in enumerate(dates):
        stop = dates[i + 1] if i + 1 < len(dates) else test["issue_date"].max() + pd.Timedelta(days=1)
        for lead in sorted(rows["lead"].unique()):
            block = test[(test["lead"] == lead) & (test["issue_date"] >= r_date) & (test["issue_date"] < stop)]
            if block.empty:
                continue
            train = training_rows(rows[rows["lead"] == lead], r_date)
            models = fit_quantiles(train, params)
            q = predict_quantiles(models, block)
            block = block.assign(lo=q[:, 0], pred=q[:, 1], hi=q[:, 2], trained_until=r_date, n_train=len(train))
            clf = fit_heat_classifier(train, params)
            block["p_classifier"] = clf.predict_proba(block[FEATURES])[:, 1] if clf is not None else np.nan
            block["sd_nwp"] = block["city"].map(error_spread(train, "pred_nwp"))
            block["sd_nwp_bias"] = block["city"].map(error_spread(train, "pred_nwp_bias"))
            pieces.append(block)
        if i % 6 == 0:
            log(f"  retrained on data up to {r_date.date()} ({i + 1}/{len(dates)})")
    fc = pd.concat(pieces).sort_values(["lead", "issue_date", "city"])
    return add_probabilities(online_conformal(fc)).reset_index(drop=True)
