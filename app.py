"""Heatwave early-warning dashboard.

    streamlit run app.py

Reads only what run_all.py wrote to results/. Nothing is trained or downloaded here.
"""
import altair as alt
import pandas as pd
import streamlit as st

import config
from metrics import MODELS

# --- look -----------------------------------------------------------------------------
# Colour follows the method, never its rank (same as the PNG charts in images/).
LGBM, BIAS, RAW = MODELS["pred"], MODELS["pred_nwp_bias"], MODELS["pred_nwp"]
COLOR = {LGBM: "#2a78d6", BIAS: "#eb6834", RAW: "#1baf7a",
         MODELS["pred_persistence"]: "#eda100", MODELS["pred_climatology"]: "#e87ba4"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OBSERVED, THRESHOLD = "Observed", "Heatwave threshold (IMD rule)"
WARNING_AT, WATCH_AT = 0.5, 0.3                     # heatwave chance needed for each alert level
BAR_SHAPE = "M-0.3,-1.3L0.3,-1.3L0.3,1.3L-0.3,1.3Z"  # a short upright bar: marks the threshold
NAMES = {c["id"]: c["name"] for c in config.CITIES}
FEATURE_LABELS = {
    "f": "Weather model forecast", "revision": "Change since previous run", "tmax_t": "Today's max",
    "tmin_t": "Today's min", "tmax_3d": "Last 3 days' max", "anom_t": "Today's max minus normal",
    "bias7": "Model error, last 7 days", "bias14": "Model error, last 14 days",
    "normal_target": "Normal for that day", "f_anom": "Forecast minus normal", "f_minus_thr": "Forecast minus threshold",
    "doy_sin": "Season (sine)", "doy_cos": "Season (cosine)", "city_code": "City",
}
PRED_COLUMNS = ["city", "issue_date", "target_date", "lead", "y_true", "pred", "lo_cal", "hi_cal", "pred_nwp",
                "pred_nwp_bias", "pred_persistence", "pred_climatology", "threshold", "heatwave", "p_lgbm",
                "trained_until"]

st.set_page_config(page_title="Heatwave early warning", page_icon="🌡️", layout="wide")


def days(d: int) -> str:
    return f"{d} day{'s' if d != 1 else ''} ahead"


def styled(chart, height=360):
    return (chart.properties(height=height, background=SURFACE)
            .configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(orient="bottom", labelColor=INK2, labelFontSize=12, labelLimit=320, columns=3,
                              symbolStrokeWidth=2.5))


def show(chart, height=360):
    st.altair_chart(styled(chart, height), theme=None, width="stretch")


def series_color(names):
    names = list(names)
    return alt.Color("series:N", scale=alt.Scale(domain=names, range=[COLOR.get(n, INK) for n in names]),
                     legend=alt.Legend(title=None), sort=names)


# --- data -------------------------------------------------------------------------------
def mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading the backtest forecasts ...")
def load_predictions(stamp: float) -> pd.DataFrame:
    """`stamp` = file time: a new run_all.py output is loaded again, not served from the cache."""
    return pd.read_csv(config.PREDICTIONS_CSV, usecols=PRED_COLUMNS,
                       parse_dates=["issue_date", "target_date", "trained_until"])


@st.cache_data
def load_table(name: str, stamp: float, index_col=None) -> pd.DataFrame | None:
    path = config.RESULTS_DIR / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def table(name, index_col=None):
    return load_table(name, mtime(config.RESULTS_DIR / name), index_col=index_col)


def alert(p: float) -> str:
    if p >= WARNING_AT:
        return "🔴 Warning"
    if p >= WATCH_AT:
        return "🟠 Watch"
    return "No alert"


def verdict(p: float, happened: float) -> str:
    if pd.isna(happened):
        return "not observed"
    warned = p >= WATCH_AT
    if happened == 1:
        return "✔ Hit" if warned else "✘ Missed"
    return "✘ False alarm" if warned else "✔ Correct, no alert"


# --- header -----------------------------------------------------------------------------
def header():
    st.title("🌡️ Heatwave early warning for Indian cities")
    st.caption("Can machine learning make a free global weather model (GFS) better at warning of heatwaves, "
               "1 to 5 days ahead? Every number below comes from a walk-forward backtest: each forecast was "
               "made as if it were that day, with only the data that existed then.")
    cmp_ = table("lgbm_vs_best_baseline.csv")
    cov, brier = table("interval_coverage.csv"), table("heatwave_brier.csv")
    if cmp_ is None or cov is None or brier is None:
        return
    c = cmp_[(cmp_["period"] == "March-June") & (cmp_["lead"] == 3)].iloc[0]
    b = brier[brier["lead"] == 3].iloc[0]
    k = st.columns(4)
    k[0].metric("Error, 3 days ahead (Mar-Jun)", f"{c['mae_lgbm']:.2f} °C",
                delta=f"{-c['improvement']:.2f} °C vs best simple method", delta_color="inverse",
                help=f"Mean absolute error of the max-temperature forecast. Best simple method: {c['best_baseline']} "
                     f"({c['mae_best_baseline']:.2f} °C).")
    k[1].metric("Improvement, 95% interval", f"{c['ci_low']:.2f} to {c['ci_high']:.2f} °C",
                help="Block bootstrap over weeks. The whole interval is above 0, so the gain is not luck.")
    k[2].metric("80% range really holds", f"{cov['coverage_calibrated'].mean():.0%}",
                help="Share of real values inside the calibrated 80% range (target: 80%).")
    k[3].metric("Heatwave-chance skill (3 days)", f"{b['skill_lgbm']:.2f}",
                delta=f"{b['skill_lgbm'] - b['skill_nwp_bias']:+.2f} vs bias-fixed model",
                help="Brier skill vs climatology: 0 = no better than 'how often it happens', 1 = perfect. "
                     "The gain over the bias-fixed model is not statistically clear (see Model scores).")


# --- tab 1: bulletin --------------------------------------------------------------------
def busiest_day(g: pd.DataFrame):
    counts = g.groupby("issue_date")["heatwave"].sum()
    return counts[counts == counts.max()].index.max().date()


def bulletin(p: pd.DataFrame):
    st.subheader("Daily heatwave bulletin")
    st.caption("What the system would have said on the evening of the chosen day, next to what really happened. "
               f"Alert levels: **🔴 Warning** = heatwave chance ≥ {WARNING_AT:.0%}, "
               f"**🟠 Watch** = ≥ {WATCH_AT:.0%}.")
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    lead = c1.segmented_control("Forecast for", config.LEADS, default=3, format_func=days, key="b_lead",
                                required=True)
    g = p[p["lead"] == lead]
    issue = c2.date_input("Forecast made on", value=busiest_day(g), min_value=g["issue_date"].min().date(),
                          max_value=g["issue_date"].max().date(), key="b_issue", format="DD/MM/YYYY")
    day = g[g["issue_date"] == pd.Timestamp(issue)].sort_values("p_lgbm", ascending=False)
    if day.empty:
        st.info("No forecast for that day (missing weather-model run). Pick another day.")
        return
    target = day["target_date"].iloc[0]
    c3.markdown(f"**Forecast for {target:%A %d %B %Y}**  \nModels trained on data up to "
                f"{day['trained_until'].iloc[0]:%d %b %Y} (nothing later was used).")

    verdicts = [verdict(a, b) for a, b in zip(day["p_lgbm"], day["heatwave"])]
    k = st.columns(4)
    k[0].metric("Cities with an alert", int((day["p_lgbm"] >= WATCH_AT).sum()), border=True)
    k[1].metric("Heatwave days that happened", int(day["heatwave"].fillna(0).sum()), border=True)
    k[2].metric("Hits / missed", f"{verdicts.count('✔ Hit')} / {verdicts.count('✘ Missed')}", border=True)
    k[3].metric("False alarms", verdicts.count("✘ False alarm"), border=True)

    t = pd.DataFrame({
        "City": day["city"].map(NAMES),
        "Alert": day["p_lgbm"].map(alert),
        "Heatwave chance": (100 * day["p_lgbm"]).round(),
        "Forecast max (°C)": day["pred"],
        "80% range (°C)": [f"{a:.1f} – {b:.1f}" for a, b in zip(day["lo_cal"], day["hi_cal"])],
        "Heatwave if ≥ (°C)": day["threshold"],
        "Raw GFS (°C)": day["pred_nwp"],
        "Observed max (°C)": day["y_true"],
        "Result": verdicts,
    })
    st.dataframe(t, hide_index=True, height=35 * (len(t) + 1) + 3, column_config={
        "Heatwave chance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Forecast max (°C)": st.column_config.NumberColumn(format="%.1f"),
        "Heatwave if ≥ (°C)": st.column_config.NumberColumn(format="%.1f"),
        "Raw GFS (°C)": st.column_config.NumberColumn(format="%.1f"),
        "Observed max (°C)": st.column_config.NumberColumn(format="%.1f"),
    })

    order = list(t["City"])
    band = alt.Chart(t.assign(lo=day["lo_cal"].to_numpy(), hi=day["hi_cal"].to_numpy())).mark_bar(
        height=10, color=COLOR[LGBM], opacity=0.22, cornerRadius=4).encode(
        y=alt.Y("City:N", sort=order, title=None),
        x=alt.X("lo:Q", title="Max temperature (°C)", scale=alt.Scale(zero=False),
                axis=alt.Axis(tickMinStep=1, format=".0f")), x2="hi:Q",
        tooltip=[alt.Tooltip("City:N"), alt.Tooltip("80% range (°C):N")])
    long = pd.concat([
        pd.DataFrame({"City": t["City"], "value": t["Forecast max (°C)"], "series": LGBM}),
        pd.DataFrame({"City": t["City"], "value": t["Heatwave if ≥ (°C)"], "series": THRESHOLD}),
        pd.DataFrame({"City": t["City"], "value": t["Observed max (°C)"], "series": OBSERVED}),
    ]).dropna()
    names = [LGBM, THRESHOLD, OBSERVED]
    pts = alt.Chart(long).mark_point(filled=True, size=110, strokeWidth=1, stroke=SURFACE, opacity=1).encode(
        y=alt.Y("City:N", sort=order), x="value:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=names, range=[COLOR[LGBM], INK2, INK]),
                        legend=alt.Legend(title=None)),
        shape=alt.Shape("series:N", scale=alt.Scale(domain=names, range=["circle", BAR_SHAPE, "diamond"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("City:N"), alt.Tooltip("series:N", title="What"),
                 alt.Tooltip("value:Q", title="°C", format=".1f")])
    st.markdown("**Forecast with its 80% range (shaded bar) vs the heatwave threshold**")
    show(band + pts, height=34 * len(t) + 40)


# --- tab 2: city explorer --------------------------------------------------------------------
def explorer(p: pd.DataFrame):
    st.subheader("City explorer")
    c1, c2, c3, c4 = st.columns([1.5, 1.9, 1.9, 1])
    cities = sorted(p["city"].unique(), key=lambda c: NAMES.get(c, c))
    city = c1.selectbox("City", cities, index=cities.index("delhi") if "delhi" in cities else 0,
                        format_func=lambda c: NAMES.get(c, c))
    lead = c2.segmented_control("Forecast made", config.LEADS, default=3, format_func=days, key="e_lead",
                                required=True)
    years = sorted(p["target_date"].dt.year.unique().tolist())
    year = c3.segmented_control("Year", years, default=2024 if 2024 in years else years[-1], key="e_year",
                                required=True)
    season = c4.toggle("March-June only", value=True)
    g = p[(p["city"] == city) & (p["lead"] == lead) & (p["target_date"].dt.year == year)]
    if season:
        g = g[g["target_date"].dt.month.isin(config.HEAT_SEASON_MONTHS)]
    g = g.sort_values("target_date")
    if g.empty:
        st.info("No forecasts for this choice.")
        return

    show_bias = st.checkbox(f"Also show '{BIAS}'", value=False)
    line_names = [OBSERVED, LGBM, RAW] + ([BIAS] if show_bias else []) + [THRESHOLD]
    cols = {OBSERVED: "y_true", LGBM: "pred", RAW: "pred_nwp", BIAS: "pred_nwp_bias", THRESHOLD: "threshold"}
    long = pd.concat([pd.DataFrame({"target_date": g["target_date"], "value": g[cols[n]], "series": n})
                      for n in line_names])
    colors = {**COLOR, OBSERVED: INK, THRESHOLD: MUTED}
    x = alt.X("target_date:T", title=None, axis=alt.Axis(format="%d %b"))
    band = alt.Chart(g).mark_area(color=COLOR[LGBM], opacity=0.16).encode(
        x=x, y=alt.Y("lo_cal:Q", title="Max temperature (°C)", scale=alt.Scale(zero=False)), y2="hi_cal:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x, y="value:Q",
        color=alt.Color("series:N", sort=line_names, legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=line_names, range=[colors[n] for n in line_names])),
        strokeDash=alt.StrokeDash("series:N", sort=line_names, legend=None,
                                  scale=alt.Scale(domain=line_names,
                                                  range=[[1, 0] if n != THRESHOLD else [5, 4] for n in line_names])))
    hw = g[g["heatwave"] == 1].assign(kind="Heatwave day")
    dots = alt.Chart(hw).mark_point(filled=True, size=55, color=INK, stroke=SURFACE, strokeWidth=1.5).encode(
        x=x, y="y_true:Q", shape=alt.Shape("kind:N", legend=alt.Legend(title=None),
                                           scale=alt.Scale(domain=["Heatwave day"], range=["circle"])))
    hover = alt.selection_point(fields=["target_date"], nearest=True, on="pointerover", empty=False,
                                clear="pointerout")
    rule = alt.Chart(g).mark_rule(color=MUTED, strokeWidth=1).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("target_date:T", title="Day", format="%a %d %b %Y"),
                 alt.Tooltip("y_true:Q", title=OBSERVED, format=".1f"),
                 alt.Tooltip("pred:Q", title=LGBM, format=".1f"),
                 alt.Tooltip("lo_cal:Q", title="80% range from", format=".1f"),
                 alt.Tooltip("hi_cal:Q", title="80% range to", format=".1f"),
                 alt.Tooltip("pred_nwp:Q", title=RAW, format=".1f"),
                 alt.Tooltip("pred_nwp_bias:Q", title=BIAS, format=".1f"),
                 alt.Tooltip("threshold:Q", title="Heatwave if ≥", format=".1f"),
                 alt.Tooltip("p_lgbm:Q", title="Heatwave chance", format=".0%")]).add_params(hover)
    st.markdown(f"**{NAMES.get(city, city)}, {year}: forecasts made {days(lead)}** "
                "(shaded = LightGBM 80% range; hover for the numbers)")
    show(band + lines + dots + rule, height=380)

    scored = g.dropna(subset=["y_true"])
    if scored.empty:
        return
    mae = {name: (scored["y_true"] - scored[col]).abs().mean() for col, name in MODELS.items()}
    k = st.columns(4)
    k[0].metric(f"Error, {LGBM}", f"{mae[LGBM]:.2f} °C", border=True)
    k[1].metric(f"Error, {BIAS}", f"{mae[BIAS]:.2f} °C", border=True)
    k[2].metric(f"Error, {RAW}", f"{mae[RAW]:.2f} °C", border=True)
    events = scored["heatwave"] == 1
    warned = scored["p_lgbm"] >= WATCH_AT
    k[3].metric("Heatwave days warned (chance ≥ 30%)", f"{int((events & warned).sum())} of {int(events.sum())}",
                help=f"False alarms in this period: {int((warned & ~events).sum())}", border=True)


# --- tab 3: model scores -------------------------------------------------------------------
def scores():
    st.subheader("How good is it, overall?")
    period = st.segmented_control("Period", ["March-June", "All year"], default="March-June", key="s_period",
                                  required=True)
    mae = table("mae_heat_season.csv" if period == "March-June" else "mae_all_year.csv", index_col="lead")
    left, right = st.columns(2)
    if mae is not None:
        long = mae.reset_index().melt("lead", var_name="series", value_name="mae")
        ch = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            x=alt.X("lead:O", title="Forecast made this many days ahead", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mae:Q", title="Mean absolute error (°C)", scale=alt.Scale(zero=True)),
            color=series_color(MODELS.values()),
            tooltip=[alt.Tooltip("series:N", title="Method"), alt.Tooltip("lead:O", title="Days ahead"),
                     alt.Tooltip("mae:Q", title="Error (°C)", format=".2f")])
        with left:
            st.markdown(f"**Error of the max-temperature forecast, {period}** (lower is better)")
            show(ch, height=330)
    cmp_ = table("lgbm_vs_best_baseline.csv")
    if cmp_ is not None:
        with right:
            st.markdown("**LightGBM vs the best simple method** (positive = LightGBM better)")
            c = cmp_[cmp_["period"] == ("March-June" if period == "March-June" else "all year")]
            c = c.assign(best_baseline=c["best_baseline"].str.replace("Weather model", "GFS", regex=False))
            st.dataframe(c[["lead", "best_baseline", "mae_best_baseline", "mae_lgbm", "improvement", "ci_low",
                            "ci_high"]].rename(columns={
                                "lead": "Days ahead", "best_baseline": "Best simple method",
                                "mae_best_baseline": "Its error", "mae_lgbm": "LightGBM",
                                "improvement": "Gain (°C)", "ci_low": "95% from", "ci_high": "95% to"}),
                         hide_index=True, column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                                         ["Its error", "LightGBM", "Gain (°C)", "95% from",
                                                          "95% to"]})
            st.caption("95% interval from a block bootstrap that resamples whole weeks (all cities together), "
                       "because errors on nearby days are linked.")
            cov = table("interval_coverage.csv")
            if cov is not None:
                st.markdown("**Does the 80% range hold 80% of the real values?**")
                st.dataframe(cov.rename(columns={"lead": "Days ahead", "coverage_raw": "Before calibration",
                                                 "coverage_calibrated": "After calibration",
                                                 "width_calibrated": "Width (°C)"})
                             [["Days ahead", "Before calibration", "After calibration", "Width (°C)"]],
                             hide_index=True, column_config={
                                 "Before calibration": st.column_config.NumberColumn(format="percent"),
                                 "After calibration": st.column_config.NumberColumn(format="percent"),
                                 "Width (°C)": st.column_config.NumberColumn(format="%.1f")})

    st.divider()
    st.subheader("Heatwave chance and warnings (March-June)")
    left, right = st.columns(2)
    rel = table("reliability.csv")
    if rel is not None:
        prob_names = {"p_lgbm": LGBM, "p_nwp_bias": BIAS, "p_nwp": RAW}
        r = rel[rel["method"].isin(prob_names)].assign(series=lambda d: d["method"].map(prob_names))
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            color=MUTED, strokeDash=[5, 4], strokeWidth=1).encode(x="x:Q", y="y:Q")
        ch = alt.Chart(r).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
            x=alt.X("forecast:Q", title="Forecast chance of a heatwave day", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
            y=alt.Y("observed:Q", title="Share that were heatwave days", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
            color=series_color(prob_names.values()),
            tooltip=[alt.Tooltip("series:N", title="Method"),
                     alt.Tooltip("forecast:Q", title="Said (avg)", format=".0%"),
                     alt.Tooltip("observed:Q", title="Happened", format=".0%"),
                     alt.Tooltip("days:Q", title="Forecasts in this group")])
        with left:
            st.markdown("**Chance said vs how often it happened** (dashed = perfectly honest)")
            show(diag + ch, height=360)
    with right:
        brier = table("heatwave_brier.csv")
        if brier is not None:
            st.markdown("**Brier score** (lower is better) **and skill vs climatology** (higher is better)")
            b = pd.DataFrame({
                "Days ahead": brier["lead"], "LightGBM": brier["brier_lgbm"],
                "Bias-fixed GFS": brier["brier_nwp_bias"], "Raw GFS": brier["brier_nwp"],
                "Skill LightGBM": brier["skill_lgbm"], "Skill bias-fixed": brier["skill_nwp_bias"],
                "Gain, 95% interval": [f"{a:+.4f} to {c:+.4f}" for a, c in zip(brier["ci_low"], brier["ci_high"])]})
            st.dataframe(b, hide_index=True, column_config={
                k: st.column_config.NumberColumn(format="%.4f") for k in ["LightGBM", "Bias-fixed GFS", "Raw GFS"]}
                | {k: st.column_config.NumberColumn(format="%.2f") for k in ["Skill LightGBM", "Skill bias-fixed"]})
            unclear = (brier["ci_low"] <= 0).any()
            st.caption("Gain = Brier score of the best other method minus LightGBM's (block bootstrap). "
                       + ("Where the interval includes 0, LightGBM's chances are a little better but the test "
                          "summers are not enough to prove it." if unclear else "All intervals are above 0."))
        warn = table("heatwave_warnings.csv")
        if warn is not None:
            lead = st.segmented_control("Warnings made", config.LEADS, default=3, format_func=days,
                                        key="s_warn_lead", required=True)
            w = warn[warn["lead"] == lead][["method", "hits", "misses", "false_alarms", "POD", "FAR", "CSI"]]
            st.dataframe(w.rename(columns={"method": "Warn when", "hits": "Hits", "misses": "Missed",
                                           "false_alarms": "False alarms"}), hide_index=True,
                         column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ["POD", "FAR", "CSI"]})
            st.caption("POD = share of heatwave days that were warned; FAR = share of warnings that were false "
                       "alarms; CSI = hits / (hits + misses + false alarms).")

    st.divider()
    left, right = st.columns(2)
    cities = table("error_by_city.csv", index_col=0)
    if cities is not None:
        three = [RAW, BIAS, LGBM]
        long = cities.reset_index(names="city").melt("city", value_vars=three, var_name="series", value_name="mae")
        long["City"] = long["city"].map(NAMES)
        order = cities.sort_values(LGBM).index.map(NAMES).tolist()
        span = long.groupby("City")["mae"].agg(["min", "max"]).reset_index()
        ch = (alt.Chart(span).mark_rule(color=AXIS).encode(y=alt.Y("City:N", sort=order, title=None),
                                                         x=alt.X("min:Q", title="Mean absolute error (°C)"),
                                                         x2="max:Q")
              + alt.Chart(long).mark_point(filled=True, size=90, stroke=SURFACE, strokeWidth=1.5).encode(
                    y=alt.Y("City:N", sort=order), x="mae:Q", color=series_color(three),
                    tooltip=[alt.Tooltip("City:N"), alt.Tooltip("series:N", title="Method"),
                             alt.Tooltip("mae:Q", title="Error (°C)", format=".2f")]))
        with left:
            st.markdown("**Error by city, March-June, 3 days ahead**")
            show(ch, height=34 * len(order) + 40)
    imp = table("feature_importance.csv")
    if imp is not None:
        imp = imp.assign(feature=imp["feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
        ch = alt.Chart(imp.head(10)).mark_bar(color=COLOR[LGBM], cornerRadiusEnd=4, height=14).encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("share_pct:Q", title="Share of total gain (%)"),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("share_pct:Q", title="% of gain", format=".1f")])
        with right:
            st.markdown("**What the model uses most** (3 days ahead, gain importance)")
            show(ch, height=34 * min(10, len(imp)) + 40)
            st.caption("The weather model's own recent error at that station matters most: the correction is "
                       "mostly 'learn each station's bias', plus how unusual the forecast is for the season.")


# --- tab 4 + 5 -------------------------------------------------------------------------------
def misses():
    st.subheader("Biggest misses (March-June)")
    st.caption("The 15 forecasts that were furthest off. Almost all are days that turned out 10-15 °C colder "
               "than forecast: the typical sign of a pre-monsoon thunderstorm or rain day, which neither the "
               "weather model nor the correction saw coming. Note the low heatwave chance on those days: the "
               "misses did not cause false heatwave warnings.")
    m = table("biggest_misses.csv")
    if m is not None:
        m = m.assign(city=m["city"].map(NAMES), heatwave=m["heatwave"].map({1: "yes", 0: "no"})).rename(columns={
            "city": "City", "issue_date": "Made on", "target_date": "For", "lead": "Days ahead",
            "y_true": "Observed", "pred": "LightGBM", "lo_cal": "80% from", "hi_cal": "80% to",
            "pred_nwp": "Raw GFS", "error": "Error", "heatwave": "Heatwave day", "p_lgbm": "Heatwave chance"})
        st.dataframe(m, hide_index=True, column_config={
            "Heatwave chance": st.column_config.NumberColumn(format="percent"),
            **{k: st.column_config.NumberColumn(format="%.1f") for k in
               ["Observed", "LightGBM", "80% from", "80% to", "Raw GFS", "Error"]}})


def data_tab():
    st.subheader("Data and method")
    st.markdown(
        "- **Observations:** hourly readings from IMD airport stations (Meteostat bulk data). The day's max is "
        "the highest reading 10:00-18:00 IST.\n"
        "- **Weather model:** archived GFS forecasts for each station's location, 1-6 days before each day "
        "(Open-Meteo Previous Runs API, from April 2021).\n"
        "- **Heatwave (IMD rule, plains):** max ≥ 40 °C and ≥ 4.5 °C above the 1991-2020 normal, or max ≥ 45 °C.\n"
        "- **Model:** LightGBM quantile regression of the weather model's error; retrained every 30 days on "
        "answers already observed; 80% range calibrated on the last year of observed errors.\n"
        "- **Test period:** forecasts made from 1 January 2023 on. Full details: README.md.")
    cities = table("cities.csv")
    if cities is not None:
        st.markdown("**Stations** (kept if ≥ 10 years for the normal and data on ≥ 80% of days since 2021)")
        st.dataframe(cities, hide_index=True)
    q = table("data_quality.csv")
    if q is not None:
        st.markdown("**Cleaning** (hourly readings removed, then days removed, per rule)")
        st.dataframe(q.assign(city=q["city"].map(NAMES)), hide_index=True)


def main():
    header()
    if not config.PREDICTIONS_CSV.exists():
        st.warning("No results yet. Run `python run_all.py` first, then reload this page.")
        return
    p = load_predictions(mtime(config.PREDICTIONS_CSV))
    tabs = st.tabs(["Warning bulletin", "City explorer", "Model scores", "Biggest misses", "Data & method"])
    with tabs[0]:
        bulletin(p)
    with tabs[1]:
        explorer(p)
    with tabs[2]:
        scores()
    with tabs[3]:
        misses()
    with tabs[4]:
        data_tab()


main()
