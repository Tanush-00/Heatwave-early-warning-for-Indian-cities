"""The four PNG charts for the README (matplotlib, light theme).

Colours follow the model, never its rank: LightGBM is always blue, the bias-fixed weather
model always orange, the raw weather model always aqua, and so on (a fixed, colour-blind
checked order). Every chart also has its numbers in a CSV in results/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from metrics import MODELS, PROBS  # noqa: E402

COLOR = {"pred": "#2a78d6", "pred_nwp_bias": "#eb6834", "pred_nwp": "#1baf7a",
         "pred_persistence": "#eda100", "pred_climatology": "#e87ba4"}
PROB_COLOR = {"p_lgbm": COLOR["pred"], "p_nwp_bias": COLOR["pred_nwp_bias"], "p_nwp": COLOR["pred_nwp"],
              "p_climatology": COLOR["pred_climatology"]}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LINE, MARK = 1.6, 6


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=12)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def _figure(w=8, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, ax, name, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=ncol,
                        frameon=False, fontsize=8.5, labelcolor=INK2)
        leg.set_in_layout(True)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.IMAGES_DIR / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def mae_by_lead(mae_heat: pd.DataFrame) -> None:
    fig, ax = _figure()
    leads = list(mae_heat.index)
    for col, label in MODELS.items():
        ax.plot(leads, mae_heat[label], color=COLOR[col], linewidth=LINE, marker="o", markersize=MARK,
                label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for col in ("pred", "pred_nwp_bias"):
        y = mae_heat[MODELS[col]].iloc[-1]
        ax.annotate(f"{y:.2f}", (leads[-1], y), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=8.5)
    ax.set_xticks(leads, [f"{d} day{'s' if d > 1 else ''}" for d in leads])
    ax.set_ylim(bottom=0)
    _style(ax, "Error of max-temperature forecasts, March-June (lower is better)",
           "Forecast made this many days ahead", "Mean absolute error (degC)")
    _save(fig, ax, "mae_by_lead.png")


def reliability_chart(rel: pd.DataFrame) -> None:
    fig, ax = _figure(6.4, 5.2)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("perfectly honest", (0.72, 0.74), rotation=38, color=MUTED, fontsize=8)
    for col in ("p_lgbm", "p_nwp_bias", "p_nwp"):
        t = rel[rel["method"] == col]
        ax.plot(t["forecast"], t["observed"], color=PROB_COLOR[col], linewidth=LINE, marker="o",
                markersize=MARK, label=PROBS[col], markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Heatwave chance said vs how often it happened", "Forecast chance of a heatwave day",
           "Share of those days that were heatwave days")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    _save(fig, ax, "reliability.png", ncol=1)


def season_example(fc: pd.DataFrame, city="delhi", year=2024, lead=3) -> None:
    g = fc[(fc["city"] == city) & (fc["lead"] == lead)]
    g = g[(g["target_date"] >= f"{year}-04-01") & (g["target_date"] <= f"{year}-06-30")].sort_values("target_date")
    if g.empty:
        return
    fig, ax = _figure(9, 4.6)
    x = g["target_date"]
    ax.fill_between(x, g["lo_cal"], g["hi_cal"], color=COLOR["pred"], alpha=0.16, linewidth=0,
                    label="LightGBM 80% range")
    ax.plot(x, g["threshold"], color=MUTED, linewidth=1.2, linestyle="--", label="Heatwave threshold (IMD rule)")
    ax.plot(x, g["pred_nwp"], color=COLOR["pred_nwp"], linewidth=LINE, label=MODELS["pred_nwp"])
    ax.plot(x, g["pred"], color=COLOR["pred"], linewidth=LINE, label=MODELS["pred"])
    ax.plot(x, g["y_true"], color=INK, linewidth=LINE, label="Observed")
    hw = g[g["heatwave"] == 1]
    ax.scatter(hw["target_date"], hw["y_true"], s=28, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.2,
               label="Heatwave day")
    name = next((c["name"] for c in config.CITIES if c["id"] == city), city)
    _style(ax, f"{name}, April-June {year}: forecasts made {lead} days ahead", "", "Max temperature (degC)")
    _save(fig, ax, f"{city}_{year}.png")


def city_chart(cities: pd.DataFrame, lead: int) -> None:
    t = cities.sort_values(MODELS["pred"], ascending=False)
    fig, ax = _figure(7.5, 0.42 * len(t) + 1.6)
    y = range(len(t))
    cols = ["pred_nwp", "pred_nwp_bias", "pred"]
    lo = t[[MODELS[c] for c in cols]].min(axis=1)
    hi = t[[MODELS[c] for c in cols]].max(axis=1)
    ax.hlines(y, lo, hi, color=AXIS, linewidth=1)
    for c in cols:
        ax.scatter(t[MODELS[c]], y, s=46, color=COLOR[c], label=MODELS[c], zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
    names = {c["id"]: c["name"] for c in config.CITIES}
    ax.set_yticks(list(y), [names.get(i, i) for i in t.index])
    ax.set_xlim(left=0)
    _style(ax, f"Error by city, March-June, {lead} days ahead", "Mean absolute error (degC)", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    _save(fig, ax, "city_mae.png", ncol=1)
