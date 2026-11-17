"""Step 4: score the backtest, make the charts and write results/summary.md.

    python evaluate.py

Also rewrites the results section of README.md (between the RESULTS markers), so no
number in the README is typed by hand.
"""
import re
import sys

import pandas as pd
from lightgbm import LGBMRegressor

import charts
import config
import metrics as M
from features import FEATURES

README = config.ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
EXAMPLE_LEAD = 3


def load_predictions(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.PREDICTIONS_CSV, parse_dates=["issue_date", "target_date", "trained_until"])


def feature_importance(rows: pd.DataFrame, lead=EXAMPLE_LEAD) -> pd.DataFrame:
    """Gain importance of a median model trained on every observed row (for explanation only)."""
    r = rows[(rows["lead"] == lead) & rows["y_true"].notna()]
    m = LGBMRegressor(objective="quantile", alpha=0.5, random_state=config.SEED, importance_type="gain",
                      **config.LGBM_PARAMS).fit(r[FEATURES], r["y_true"] - r["f"])
    imp = pd.DataFrame({"feature": FEATURES, "gain": m.feature_importances_})
    imp["share_pct"] = (100 * imp["gain"] / imp["gain"].sum()).round(1)
    return imp.sort_values("gain", ascending=False).drop(columns="gain").reset_index(drop=True)


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def gfs_bias(h: pd.DataFrame, lead=EXAMPLE_LEAD) -> pd.Series:
    """Mean (raw weather model - observed) per city: positive = the model runs too warm."""
    g = h[h["lead"] == lead]
    return (g["pred_nwp"] - g["y_true"]).groupby(g["city"]).mean().sort_values()


def gfs_bias_sentence(bias: pd.Series) -> str:
    names = {c["id"]: c["name"] for c in config.CITIES}
    word = lambda v: "too warm" if v > 0 else "too cool"
    lo, hi = bias.index[0], bias.index[-1]
    return (f"**Why a bias fix helps so much:** in March-June the raw weather model's {EXAMPLE_LEAD}-day forecast "
            f"was {abs(bias.iloc[-1]):.1f} degC {word(bias.iloc[-1])} on average at {names.get(hi, hi)}, but "
            f"{abs(bias.iloc[0]):.1f} degC {word(bias.iloc[0])} at {names.get(lo, lo)}: each station has its own "
            "error, so the model must be corrected station by station.")


def results_block(t: dict) -> str:
    cmp_h, brier, cov, warn = t["compare_heat"], t["brier"], t["coverage"], t["warnings"]
    c3 = cmp_h[cmp_h["lead"] == EXAMPLE_LEAD].iloc[0]
    b3 = brier[brier["lead"] == EXAMPLE_LEAD].iloc[0]
    mae = t["mae_heat"].reset_index().rename(columns={"lead": "Days ahead"})
    w3 = warn[warn["lead"] == EXAMPLE_LEAD][["method", "warnings", "hits", "misses", "false_alarms", "POD", "FAR", "CSI"]]
    lines = [
        f"Test period: forecasts made {t['test_from']} .. {t['test_to']} for {t['n_cities']} cities "
        f"({t['rows']:,} scored forecasts; {t['heat_rows']:,} of them for March-June, with "
        f"{t['events']} heatwave days per lead).",
        "",
        "**Error of the max-temperature forecast, March-June** (mean absolute error, degC; lower is better):",
        "",
        md_table(mae),
        "",
        f"**Is LightGBM really better than the best simple method?** At {EXAMPLE_LEAD} days ahead, March-June: "
        f"{c3['mae_lgbm']:.2f} vs {c3['mae_best_baseline']:.2f} degC for *{c3['best_baseline']}*, "
        f"{c3['improvement']:.2f} degC better ({c3['improvement_pct']:.0f}%), 95% interval "
        f"{c3['ci_low']:.2f} to {c3['ci_high']:.2f} degC (block bootstrap over weeks).",
        "",
        md_table(cmp_h[["lead", "best_baseline", "mae_best_baseline", "mae_lgbm", "improvement", "ci_low", "ci_high"]]
                 .rename(columns={"lead": "Days ahead"})),
        "",
        gfs_bias_sentence(t["gfs_bias"]),
        "",
        f"**Heatwave chance** (Brier score, lower is better; skill vs climatology, higher is better), "
        f"{EXAMPLE_LEAD} days ahead: LightGBM {b3['brier_lgbm']:.4f} (skill {b3['skill_lgbm']:.2f}), "
        f"bias-fixed weather model {b3['brier_nwp_bias']:.4f} (skill {b3['skill_nwp_bias']:.2f}), raw weather "
        f"model {b3['brier_nwp']:.4f} (skill {b3['skill_nwp']:.2f}). Gain over the best other method: "
        f"{b3['brier_gain']:.4f}, 95% interval {b3['ci_low']:.4f} to {b3['ci_high']:.4f}"
        + (" (the interval includes 0: not a clear win)." if b3["ci_low"] <= 0 <= b3["ci_high"] else ".")
        + (f" Tried and dropped: a direct LightGBM yes/no classifier scored {b3['brier_classifier_tried']:.4f} "
           "(worse; too few heatwave days to learn from)." if "brier_classifier_tried" in b3 else ""),
        "",
        f"**Heatwave warnings, {EXAMPLE_LEAD} days ahead** (POD = heatwave days warned; FAR = warnings that "
        "were false alarms):",
        "",
        md_table(w3, floatfmt="{:.2f}"),
        "",
        "**80% range** (share of real values inside; should be 0.80): " + ", ".join(
            f"{r.lead} day{'s' if r.lead > 1 else ''}: {r.coverage_raw:.2f} raw -> {r.coverage_calibrated:.2f} "
            f"calibrated" for r in cov.itertuples()) + ".",
        "",
        "![Error by lead](images/mae_by_lead.png)",
        "",
        "![Delhi 2024](images/delhi_2024.png)",
        "",
        "![Reliability](images/reliability.png)",
        "",
        "![Error by city](images/city_mae.png)",
        "",
        "All numbers: `results/` (one CSV per table). Written by `evaluate.py`.",
    ]
    return "\n".join(lines)


def update_readme(block: str) -> bool:
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    new = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text,
                 flags=re.S)
    README.write_text(new, encoding="utf-8")
    return True


def evaluate(fc: pd.DataFrame, rows=None, write_readme=True, log=print) -> dict:
    s = M.scored(fc)
    h = M.heat_season(s)
    t = {
        "mae_all": M.mae_table(s), "mae_heat": M.mae_table(h),
        "compare_all": M.compare_to_best_baseline(s), "compare_heat": M.compare_to_best_baseline(h),
        "brier": M.brier_table(h), "warnings": M.warning_scores(h), "coverage": M.coverage_table(s),
        "reliability": pd.concat([M.reliability(h, p) for p in M.PROBS]),
        "per_city": M.per_city(h, EXAMPLE_LEAD), "misses": M.biggest_misses(h), "gfs_bias": gfs_bias(h),
        "test_from": str(s["issue_date"].min().date()), "test_to": str(s["issue_date"].max().date()),
        "n_cities": s["city"].nunique(), "rows": len(s), "heat_rows": len(h),
        "events": int(h[h["lead"] == EXAMPLE_LEAD]["heatwave"].sum()),
    }
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    t["mae_all"].to_csv(out / "mae_all_year.csv")
    t["mae_heat"].to_csv(out / "mae_heat_season.csv")
    pd.concat([t["compare_all"].assign(period="all year"), t["compare_heat"].assign(period="March-June")]).to_csv(
        out / "lgbm_vs_best_baseline.csv", index=False)
    t["brier"].to_csv(out / "heatwave_brier.csv", index=False)
    t["warnings"].to_csv(out / "heatwave_warnings.csv", index=False)
    t["coverage"].to_csv(out / "interval_coverage.csv", index=False)
    t["reliability"].to_csv(out / "reliability.csv", index=False)
    t["per_city"].join(t["gfs_bias"].rename("raw_model_bias").round(3)).to_csv(out / "error_by_city.csv")
    t["misses"].to_csv(out / "biggest_misses.csv", index=False)
    if rows is not None:
        t["importance"] = feature_importance(rows)
        t["importance"].to_csv(out / "feature_importance.csv", index=False)

    charts.mae_by_lead(t["mae_heat"])
    charts.reliability_chart(t["reliability"])
    charts.season_example(s, "delhi", 2024, EXAMPLE_LEAD)
    charts.city_chart(t["per_city"], EXAMPLE_LEAD)

    block = results_block(t)
    (out / "summary.md").write_text("# Results\n\n" + block.replace("images/", "../images/") + "\n", encoding="utf-8")
    if write_readme and update_readme(block):
        log("README results section updated.")
    log(block)
    return t


def main() -> int:
    if not config.PREDICTIONS_CSV.exists():
        print("No predictions yet. Run run_all.py (or backtest) first.")
        return 1
    from dataset import load_daily
    from features import make_rows
    evaluate(load_predictions(), rows=make_rows(load_daily()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
