"""Scores: MAE, block bootstrap, Brier score, warning scores, reliability, coverage."""
import numpy as np
import pandas as pd

import config

MODELS = {
    "pred_persistence": "Persistence (same as today)",
    "pred_climatology": "Climatology (a normal day)",
    "pred_nwp": "Weather model, raw (GFS)",
    "pred_nwp_bias": "Weather model + recent-bias fix",
    "pred": "LightGBM correction (this project)",
}
BASELINES = [c for c in MODELS if c != "pred"]
PROBS = {
    "p_climatology": "Climatology (how often, in 1991-2020)",
    "p_nwp": "Weather model, raw (GFS)",
    "p_nwp_bias": "Weather model + recent-bias fix",
    "p_lgbm": "LightGBM correction (this project)",
}


def scored(fc: pd.DataFrame) -> pd.DataFrame:
    """Rows every model can be scored on (same rows for all: no model gets easier days)."""
    need = ["y_true", *MODELS, *PROBS, "heatwave"]
    return fc.dropna(subset=need).copy()


def heat_season(df: pd.DataFrame) -> pd.DataFrame:
    return df[pd.to_datetime(df["target_date"]).dt.month.isin(config.HEAT_SEASON_MONTHS)]


def mae_table(df: pd.DataFrame) -> pd.DataFrame:
    out = {MODELS[c]: df.groupby("lead").apply(lambda g, c=c: (g["y_true"] - g[c]).abs().mean())
           for c in MODELS}
    return pd.DataFrame(out).round(3)


def block_ids(df: pd.DataFrame) -> pd.Series:
    days = (pd.to_datetime(df["issue_date"]) - pd.Timestamp(config.TEST_START)).dt.days
    return days // config.BOOTSTRAP_BLOCK_DAYS


def block_bootstrap(values: pd.Series, blocks: pd.Series, n=None, seed=None) -> dict:
    """Mean of `values` with a 95% interval, resampling whole weeks (all cities of a week
    together), because errors of nearby days and nearby cities are linked."""
    n = n or config.BOOTSTRAP_RESAMPLES
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    sums = values.groupby(blocks).sum().to_numpy()
    counts = values.groupby(blocks).size().to_numpy()
    pick = rng.integers(0, len(sums), size=(n, len(sums)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "share_positive": float((means > 0).mean())}


def compare_to_best_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Per lead: MAE of the best baseline minus MAE of LightGBM (positive = LightGBM better)."""
    rows = []
    for lead, g in df.groupby("lead"):
        maes = {c: (g["y_true"] - g[c]).abs().mean() for c in BASELINES}
        best = min(maes, key=maes.get)
        diff = (g["y_true"] - g[best]).abs() - (g["y_true"] - g["pred"]).abs()
        b = block_bootstrap(diff, block_ids(g))
        rows.append({"lead": lead, "best_baseline": MODELS[best], "mae_best_baseline": maes[best],
                     "mae_lgbm": (g["y_true"] - g["pred"]).abs().mean(), "improvement": b["mean"],
                     "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                     "improvement_pct": 100 * b["mean"] / maes[best], "share_lgbm_better": b["share_positive"],
                     "rows": len(g)})
    return pd.DataFrame(rows).round(4)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def brier_table(h: pd.DataFrame) -> pd.DataFrame:
    """Heat season only. Brier score (lower = better) and skill vs climatology (higher = better),
    plus a bootstrap interval for (best other method - LightGBM)."""
    rows = []
    for lead, g in h.groupby("lead"):
        scores = {p: brier(g[p], g["heatwave"]) for p in PROBS}
        others = {p: s for p, s in scores.items() if p != "p_lgbm"}
        best = min(others, key=others.get)
        diff = (g[best] - g["heatwave"]) ** 2 - (g["p_lgbm"] - g["heatwave"]) ** 2
        b = block_bootstrap(diff, block_ids(g))
        row = {"lead": lead, "events": int(g["heatwave"].sum()), "rows": len(g)}
        row.update({f"brier_{p[2:]}": round(s, 4) for p, s in scores.items()})
        row.update({f"skill_{p[2:]}": round(1 - s / scores["p_climatology"], 3)
                    for p, s in scores.items() if p != "p_climatology"})
        row.update({"best_other": PROBS[best], "brier_gain": round(b["mean"], 5),
                    "ci_low": round(b["ci_low"], 5), "ci_high": round(b["ci_high"], 5)})
        if "p_classifier" in g and g["p_classifier"].notna().all():       # tried, not used
            row["brier_classifier_tried"] = round(brier(g["p_classifier"], g["heatwave"]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def warning_scores(h: pd.DataFrame) -> pd.DataFrame:
    """Heat season only. A 'warning' = the forecast says tmax >= heatwave threshold
    (or, for probabilities, p >= the cut-off). POD = share of heatwave days warned,
    FAR = share of warnings that were false alarms, CSI = hits / (hits + misses + false alarms)."""
    rules = {
        "Weather model, raw (GFS)": lambda g: g["pred_nwp"] >= g["threshold"],
        "Weather model + recent-bias fix": lambda g: g["pred_nwp_bias"] >= g["threshold"],
        "LightGBM, chance >= 50%": lambda g: g["p_lgbm"] >= 0.5,
        "LightGBM, chance >= 30% (cautious)": lambda g: g["p_lgbm"] >= 0.3,
    }
    rows = []
    for lead, g in h.groupby("lead"):
        event = g["heatwave"] == 1
        for name, rule in rules.items():
            warn = rule(g)
            hits, misses, false = int((warn & event).sum()), int((~warn & event).sum()), int((warn & ~event).sum())
            rows.append({"lead": lead, "method": name, "warnings": int(warn.sum()), "hits": hits,
                         "misses": misses, "false_alarms": false,
                         "POD": round(hits / max(1, hits + misses), 3),
                         "FAR": round(false / max(1, hits + false), 3),
                         "CSI": round(hits / max(1, hits + misses + false), 3)})
    return pd.DataFrame(rows)


def reliability(h: pd.DataFrame, column: str, bins=(0, 0.05, 0.15, 0.3, 0.5, 0.7, 1.0001)) -> pd.DataFrame:
    """Do days that got a 30% chance really turn out heatwaves ~30% of the time?"""
    cut = pd.cut(h[column], bins=list(bins), right=False)
    t = h.groupby(cut, observed=True).agg(forecast=(column, "mean"), observed=("heatwave", "mean"),
                                          days=("heatwave", "size"))
    return t.reset_index(drop=True).assign(method=column)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in df.groupby("lead"):
        inside_raw = (g["y_true"] >= g["lo"]) & (g["y_true"] <= g["hi"])
        inside_cal = (g["y_true"] >= g["lo_cal"]) & (g["y_true"] <= g["hi_cal"])
        rows.append({"lead": lead, "target": config.INTERVAL_HIGH - config.INTERVAL_LOW,
                     "coverage_raw": round(inside_raw.mean(), 3), "coverage_calibrated": round(inside_cal.mean(), 3),
                     "width_raw": round((g["hi"] - g["lo"]).mean(), 2),
                     "width_calibrated": round((g["hi_cal"] - g["lo_cal"]).mean(), 2)})
    return pd.DataFrame(rows)


def per_city(df: pd.DataFrame, lead: int) -> pd.DataFrame:
    g = df[df["lead"] == lead]
    cols = ["pred_nwp", "pred_nwp_bias", "pred"]
    t = pd.DataFrame({MODELS[c]: (g["y_true"] - g[c]).abs().groupby(g["city"]).mean() for c in cols})
    t["heatwave_days"] = g.groupby("city")["heatwave"].sum().astype(int)
    return t.round(3).sort_values(MODELS["pred"])


def biggest_misses(h: pd.DataFrame, n=15) -> pd.DataFrame:
    g = h.assign(error=h["y_true"] - h["pred"]).reindex(
        (h["y_true"] - h["pred"]).abs().sort_values(ascending=False).index).head(n)
    cols = ["city", "issue_date", "target_date", "lead", "y_true", "pred", "lo_cal", "hi_cal", "pred_nwp",
            "error", "heatwave", "p_lgbm"]
    g = g[cols].copy()
    num = g.select_dtypes("number").columns
    g[num] = g[num].round(2)
    return g
