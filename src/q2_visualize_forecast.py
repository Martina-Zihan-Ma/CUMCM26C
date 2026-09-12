#!/usr/bin/env python3
"""Paper figures for Question 2 forecasting; reads existing outputs only.

Figure contract: January profiles describe exploitable intraday and lagged
structure; a median-quality operational day shows the realised rolling forecast
without selecting an unusually good result.  All power curves remain in kW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_FILE = ROOT / "data" / "processed" / "attachment2_actual_long.csv"
FORECAST_FILE = ROOT / "outputs" / "question2" / "dynamic_forecasts.csv"
METRICS_FILE = ROOT / "outputs" / "question2" / "forecast_metrics.csv"
OUT = ROOT / "outputs" / "question2" / "figures"
SKILL_SCRIPTS = Path("/Users/mazihan/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


N = 144
YEAR_START, YEAR_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
FORMAL_START, FORMAL_END = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")
JAN_START, JAN_END = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
TICKS = np.arange(0, 1441, 240)
TICK_LABELS = ["0:00", "4:00", "8:00", "12:00", "16:00", "20:00", "24:00"]
X_MINUTES = np.arange(1, N + 1) * 10  # right endpoints: index 0 is 00:10
ACTUAL_COLOR, FORECAST_COLOR = "#2F3E4E", "#C44E52"
PV_MEAN, PV_BAND = "#3B7A8F", "#9ECAE1"
WEEKDAY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#6C757D"]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "xtick.major.width": .7, "ytick.major.width": .7,
})


def check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise ValueError(message)
    checks.append(f"PASS: {message}")


def is_complete_daily_grid(frame: pd.DataFrame) -> bool:
    counts = frame.groupby("date").size()
    indices = frame.groupby("date")["time_index"].apply(
        lambda x: x.astype(int).sort_values().tolist() == list(range(N))
    )
    return bool((counts == N).all() and indices.all())


def label_minutes(value: object) -> int:
    """Convert the stored right-endpoint label; 00:00 means the next midnight."""
    text = str(value).strip().replace("+1", "")
    parts = text.split(":")
    hour, minute = int(parts[0]), int(parts[1])
    result = hour * 60 + minute
    return 1440 if result == 0 else result


def read_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    checks: list[str] = []
    for path in (ACTUAL_FILE, FORECAST_FILE, METRICS_FILE):
        check(path.exists(), f"Required input exists: {path}", checks)
    actual = pd.read_csv(ACTUAL_FILE)
    forecast = pd.read_csv(FORECAST_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    actual["date"] = pd.to_datetime(actual["date"]).dt.normalize()
    forecast["date"] = pd.to_datetime(forecast["date"]).dt.normalize()
    required_actual = {"date", "time_label", "time_index", "load_kw", "pv_actual_kw"}
    required_forecast = {"date", "time_label", "time_index", "actual_load_kw", "actual_pv_kw", "forecast_load_kw", "forecast_pv_kw"}
    check(required_actual <= set(actual), "Actual-data fields are present", checks)
    check(required_forecast <= set(forecast), "Forecast-data fields are present", checks)
    check(actual["date"].min() == YEAR_START and actual["date"].max() == YEAR_END, "Actual data cover 2025-01-01 to 2025-12-31", checks)
    check(len(actual) == 365 * N and is_complete_daily_grid(actual), "Actual data have 365 x 144 complete slots", checks)
    check(not actual.duplicated(["date", "time_index"]).any(), "Actual date + time_index has no duplicates", checks)
    check(np.isfinite(actual[["load_kw", "pv_actual_kw"]].to_numpy(float)).all() and (actual[["load_kw", "pv_actual_kw"]] >= 0).all().all(), "Actual load and PV are finite and non-negative", checks)
    formal = forecast.loc[forecast["date"].between(FORMAL_START, FORMAL_END)].copy()
    check(formal["date"].min() == FORMAL_START and formal["date"].max() == FORMAL_END, "Formal forecast dates cover 2025-02-01 to 2025-12-31", checks)
    check(len(formal) == 334 * N and is_complete_daily_grid(formal), "Formal forecast has 334 x 144 complete slots", checks)
    check(not formal.duplicated(["date", "time_index"]).any(), "Formal forecast date + time_index has no duplicates", checks)
    check(np.isfinite(formal[["forecast_load_kw", "forecast_pv_kw"]].to_numpy(float)).all() and (formal[["forecast_load_kw", "forecast_pv_kw"]] >= 0).all().all(), "Forecast load and PV are finite and non-negative", checks)
    if "history_end_date" in forecast:
        history = pd.to_datetime(formal["history_end_date"]).dt.normalize()
        check((history < formal["date"]).all(), "history_end_date is strictly before each target date", checks)
    actual_map = actual[["date", "time_index", "time_label"]].copy()
    formal_map = formal[["date", "time_index", "time_label"]].copy()
    mapping = actual_map.merge(formal_map, on=["date", "time_index"], suffixes=("_actual", "_forecast"), validate="one_to_one")
    check(len(mapping) == len(formal) and (mapping["time_label_actual"].astype(str) == mapping["time_label_forecast"].astype(str)).all(), "Actual/forecast date-time mappings are identical", checks)
    expected_minutes = pd.Series(range(N)).map(lambda t: (t + 1) * 10).to_numpy()
    source_minutes = actual.loc[actual["date"] == YEAR_START].sort_values("time_index")["time_label"].map(label_minutes).to_numpy()
    check(np.array_equal(source_minutes, expected_minutes), "time_index maps to right endpoints 00:10 through 24:00 without shift", checks)
    check({"period", "method", "load_mae_kw", "load_rmse_kw", "pv_mae_kw", "pv_rmse_kw"} <= set(metrics), "Aggregate metric fields are present", checks)
    return actual, formal, metrics, checks


def daily_curve_correlation(wide: pd.DataFrame, lag: int) -> tuple[float, int]:
    values = []
    for i in range(lag, len(wide)):
        a, b = wide.iloc[i].to_numpy(float), wide.iloc[i - lag].to_numpy(float)
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)), len(values)


def periodicity(actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    january = actual.loc[actual["date"].between(JAN_START, JAN_END)].copy()
    load = january.pivot(index="date", columns="time_index", values="load_kw").sort_index()
    pv = january.pivot(index="date", columns="time_index", values="pv_actual_kw").sort_index()
    rows = []
    for series, curves in (("load", load), ("pv", pv)):
        for lag in (1, 7):
            corr, pairs = daily_curve_correlation(curves, lag)
            rows.append({"period": "2025-01-01 to 2025-01-31", "series": series, "lag_days": lag, "mean_profile_correlation": corr, "paired_days": pairs})
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    january["weekday"] = january["date"].dt.day_name()
    profiles = january.groupby(["weekday", "time_index"], observed=True)["load_kw"].mean().unstack("time_index").reindex(weekdays)
    pv_summary = pd.DataFrame({"mean": pv.mean(), "q10": pv.quantile(.10), "q90": pv.quantile(.90)})
    return pd.DataFrame(rows), profiles, pv_summary


def select_representative_day(formal: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    grouped = formal.groupby("date", sort=True)
    rows = []
    for date, day in grouped:
        le = day["forecast_load_kw"].to_numpy(float) - day["actual_load_kw"].to_numpy(float)
        pe = day["forecast_pv_kw"].to_numpy(float) - day["actual_pv_kw"].to_numpy(float)
        rows.append({"date": date, "load_mae_kw": np.abs(le).mean(), "load_rmse_kw": np.sqrt(np.mean(le ** 2)), "pv_mae_kw": np.abs(pe).mean(), "pv_rmse_kw": np.sqrt(np.mean(pe ** 2)), "actual_pv_daily_sum_kw": day["actual_pv_kw"].sum()})
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["load_rmse_percentile"] = daily["load_rmse_kw"].rank(method="average", pct=True)
    daily["pv_rmse_percentile"] = daily["pv_rmse_kw"].rank(method="average", pct=True)
    daily["combined_error_percentile"] = daily[["load_rmse_percentile", "pv_rmse_percentile"]].mean(axis=1)
    threshold = daily["actual_pv_daily_sum_kw"].quantile(.25)
    daily["pv_daily_sum_q25_kw"] = threshold
    daily["eligible_pv_day"] = daily["actual_pv_daily_sum_kw"] >= threshold
    daily["distance_to_median_score"] = np.where(daily["eligible_pv_day"], np.abs(daily["combined_error_percentile"] - .50), np.nan)
    selected_index = daily.loc[daily["eligible_pv_day"]].sort_values(["distance_to_median_score", "date"]).index[0]
    daily["selected_representative_day"] = False
    daily.loc[selected_index, "selected_representative_day"] = True
    return daily, pd.Timestamp(daily.loc[selected_index, "date"])


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1440)
    ax.set_xticks(TICKS, TICK_LABELS)
    ax.grid(axis="y", color="#D9D9D9", lw=.55)
    ax.set_axisbelow(True)


def label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-.12, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save(fig: plt.Figure, stem: str) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=OUT / f"{stem}.alignment.json", overlay_svg=OUT / f"{stem}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_periodicity(metrics: pd.DataFrame, profiles: pd.DataFrame, pv_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.5))
    fig.subplots_adjust(left=.08, right=.99, bottom=.16, top=.70, wspace=.24)
    ax = axes[0]
    for color, weekday in zip(WEEKDAY_COLORS, profiles.index):
        ax.plot(X_MINUTES, profiles.loc[weekday].to_numpy(float), lw=1.05, color=color, label=weekday)
    style_axis(ax); ax.set_ylabel("Load power (kW)"); ax.set_xlabel("Time of day"); ax.set_title("Weekday-specific average load profiles", loc="left", fontsize=7.2, y=1.33); label_panel(ax, "a")
    ax.legend(ncol=4, fontsize=5.2, loc="lower left", bbox_to_anchor=(0, 1.02, 1, .12), mode="expand", columnspacing=.7, handlelength=1.6)
    load_lines = metrics.loc[metrics["series"] == "load"].set_index("lag_days")["mean_profile_correlation"]
    ax.text(.985, 1.34, f"Mean curve correlation\nlag 1: {load_lines[1]:.3f}\nlag 7: {load_lines[7]:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color="#3C3C3C", clip_on=False)
    ax = axes[1]
    ax.fill_between(X_MINUTES, pv_summary["q10"].to_numpy(float), pv_summary["q90"].to_numpy(float), color=PV_BAND, alpha=.22, label="P10–P90 band")
    ax.plot(X_MINUTES, pv_summary["mean"].to_numpy(float), color=PV_MEAN, lw=1.35, label="Mean PV")
    style_axis(ax); ax.set_ylabel("PV power (kW)"); ax.set_xlabel("Time of day"); ax.set_title("Mean PV profile with 10th–90th percentile band", loc="left", fontsize=7.2, y=1.33); label_panel(ax, "b")
    ax.legend(fontsize=5.2, loc="lower left", bbox_to_anchor=(0, 1.02, 1, .12), mode="expand", handlelength=1.8)
    pv_lines = metrics.loc[metrics["series"] == "pv"].set_index("lag_days")["mean_profile_correlation"]
    ax.text(.985, 1.34, f"Mean curve correlation\nlag 1: {pv_lines[1]:.3f}\nlag 7: {pv_lines[7]:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color="#3C3C3C", clip_on=False)
    save(fig, "q2_periodicity")


def plot_forecast(formal: pd.DataFrame, daily: pd.DataFrame, representative: pd.Timestamp) -> None:
    day = formal.loc[formal["date"] == representative].sort_values("time_index")
    selected = daily.loc[daily["selected_representative_day"]].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.55), sharex=True, constrained_layout=True)
    for ax, actual_col, forecast_col, ylabel, title, mae, rmse, letter in [
        (axes[0], "actual_load_kw", "forecast_load_kw", "Load power (kW)", "Actual and forecast load", selected["load_mae_kw"], selected["load_rmse_kw"], "a"),
        (axes[1], "actual_pv_kw", "forecast_pv_kw", "PV power (kW)", "Actual and forecast PV", selected["pv_mae_kw"], selected["pv_rmse_kw"], "b"),
    ]:
        ax.plot(X_MINUTES, day[actual_col].to_numpy(float), color=ACTUAL_COLOR, lw=1.2, label="Actual")
        ax.plot(X_MINUTES, day[forecast_col].to_numpy(float), color=FORECAST_COLOR, lw=1.1, ls="--", label="Forecast")
        style_axis(ax); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=7.5, pad=8); label_panel(ax, letter)
        ax.legend(loc="upper left", fontsize=6.1, ncol=2, handlelength=2.0)
        ax.text(.985, 1.015, f"MAE = {mae:.1f} kW   RMSE = {rmse:.1f} kW", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.1, color="#3C3C3C", clip_on=False)
    axes[0].tick_params(labelbottom=True)
    axes[1].set_xlabel(f"Time of day — representative date: {representative.date().isoformat()}")
    save(fig, "q2_forecast_vs_actual")


def write_report(checks: list[str], metrics: pd.DataFrame, daily: pd.DataFrame, representative: pd.Timestamp) -> None:
    rep = daily.loc[daily["selected_representative_day"]].iloc[0]
    p = metrics.pivot(index="series", columns="lag_days", values="mean_profile_correlation")
    report = [
        "Q2 FORECAST VISUALIZATION REPORT", "=" * 32,
        f"Actual input: {ACTUAL_FILE}", f"Forecast input: {FORECAST_FILE}", f"Aggregate metrics input: {METRICS_FILE}",
        "Actual fields used: date, time_label, time_index, load_kw, pv_actual_kw.",
        "Forecast fields used: date, time_label, time_index, actual_load_kw, actual_pv_kw, forecast_load_kw, forecast_pv_kw, history_end_date.",
        "Actual dimensions/date range: 52560 rows; 2025-01-01 to 2025-12-31.",
        "Formal forecast dimensions/date range: 48096 rows; 2025-02-01 to 2025-12-31.",
        "Periodicity window: 2025-01-01 to 2025-01-31; correlations are means of Pearson correlations between exactly lagged 144-point daily curves.",
        f"Load correlations: lag1={p.loc['load', 1]:.6f}, lag7={p.loc['load', 7]:.6f}.",
        f"PV correlations: lag1={p.loc['pv', 1]:.6f}, lag7={p.loc['pv', 7]:.6f}.",
        f"Representative date: {representative.date().isoformat()}.",
        "Selection: rank daily load-RMSE and PV-RMSE across all 334 formal days; average the two percentile ranks; exclude actual-PV daily sums below their 25th percentile; select the eligible score nearest 0.50.",
        f"Representative errors: load MAE={rep['load_mae_kw']:.6f} kW, RMSE={rep['load_rmse_kw']:.6f} kW (RMSE percentile={rep['load_rmse_percentile']:.3f}); PV MAE={rep['pv_mae_kw']:.6f} kW, RMSE={rep['pv_rmse_kw']:.6f} kW (RMSE percentile={rep['pv_rmse_percentile']:.3f}).",
        f"Representative combined percentile score={rep['combined_error_percentile']:.6f}; PV eligibility threshold={rep['pv_daily_sum_q25_kw']:.6f} kW-sum.",
        f"Outputs: {OUT / 'q2_periodicity.png'}, {OUT / 'q2_periodicity.pdf'}, {OUT / 'q2_forecast_vs_actual.png'}, {OUT / 'q2_forecast_vs_actual.pdf'}.",
        "Data/field issues: none found. The stored final right-endpoint label is 00:00:00 rather than 0:00+1; it is validated and displayed as the 24:00 boundary, with no circular shift.", "", "Validation checks:", *checks,
    ]
    (OUT / "q2_forecast_visualization_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual, formal, _, checks = read_and_validate()
    period_metrics, load_profiles, pv_summary = periodicity(actual)
    period_metrics.to_csv(OUT / "q2_periodicity_metrics.csv", index=False, encoding="utf-8-sig")
    daily, representative = select_representative_day(formal)
    daily.to_csv(OUT / "q2_representative_day_selection.csv", index=False, encoding="utf-8-sig")
    plot_periodicity(period_metrics, load_profiles, pv_summary)
    plot_forecast(formal, daily, representative)
    write_report(checks, period_metrics, daily, representative)


if __name__ == "__main__":
    main()
