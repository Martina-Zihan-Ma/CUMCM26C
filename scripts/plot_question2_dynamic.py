#!/usr/bin/env python3
"""Publication-style dynamic-battery figures from the verified Q2 outputs.

Figure contract: causal battery dispatch absorbs forecast mismatch before
emergency purchase, lowering cumulative operating cost against the same
forecast no-storage baseline.  The dispatch day is selected as the earliest
day with maximal realized emergency energy, so the mechanism is visible
without manual cherry-picking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs" / "question2", ROOT / "outputs" / "question2" / "figures"
sys.path.insert(0, "/Users/mazihan/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

N, DT = 144, 1 / 6
COLORS = {"load": "#2F3E4E", "pv": "#56B4E9", "grid": "#0072B2", "emergency": "#D55E00", "charge": "#009E73", "discharge": "#CC79A7", "soc": "#6A5ACD", "baseline": "#7F7F7F"}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "legend.frameon": False})


def style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_ylabel(ylabel); ax.grid(axis="y", color="#D9D9D9", lw=.55); ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str, multi: bool = False) -> None:
    FIG.mkdir(parents=True, exist_ok=True); fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=FIG / f"{stem}.alignment.json", overlay_svg=FIG / f"{stem}.alignment.svg", tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=multi, strict=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plan = pd.read_csv(OUT / "question2_plan.csv", parse_dates=["date"])
    dispatch = pd.read_csv(OUT / "question2_dispatch.csv", parse_dates=["date"])
    emergency = pd.read_csv(OUT / "question2_emergency.csv", parse_dates=["date"])
    daily = pd.read_csv(OUT / "question2_daily_summary.csv", parse_dates=["date"])
    actual = pd.read_csv(ROOT / "data" / "processed" / "attachment2_actual_long.csv", parse_dates=["date"])
    forecast = pd.read_csv(OUT / "dynamic_forecasts.csv", parse_dates=["date"])
    merged = plan.merge(dispatch, on=["date", "time_index"], validate="one_to_one").merge(emergency, on=["date", "time_index"], validate="one_to_one").merge(actual[["date", "time_index", "load_kw", "pv_actual_kw"]], on=["date", "time_index"], validate="one_to_one")
    day = daily.sort_values(["actual_emergency_kwh", "date"], ascending=[False, True]).iloc[0].date
    x = np.arange(N) * 10 + 10
    d = merged.loc[merged.date == day].sort_values("time_index")
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 4.65), sharex=True)
    fig.subplots_adjust(left=.10, right=.79, bottom=.11, top=.94, hspace=.43)
    axes[0].plot(x, d.load_kw * DT, label="Actual load", color=COLORS["load"], lw=1.15)
    axes[0].plot(x, d.pv_actual_kw * DT, label="Actual PV", color=COLORS["pv"], lw=1.1)
    axes[0].plot(x, d.grid_purchase_kwh, label="Planned grid", color=COLORS["grid"], lw=1.05)
    axes[0].plot(x, d.emergency_purchase_kwh, label="Emergency", color=COLORS["emergency"], lw=1.05)
    style(axes[0], "Energy per 10 min (kWh)"); axes[0].set_title("Causal real-time supply", loc="left", fontsize=7.5); axes[0].legend(fontsize=5.6, loc="upper left", bbox_to_anchor=(1.02, 1.0)); axes[0].text(-.11, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=8)
    axes[1].plot(x, d.charge_kwh, label="Charge", color=COLORS["charge"], lw=1.05)
    axes[1].plot(x, d.discharge_kwh, label="Discharge", color=COLORS["discharge"], lw=1.05)
    ax2 = axes[1].twinx(); ax2.plot(x, d.soc_end_kwh, label="SOC", color=COLORS["soc"], lw=1.25)
    # Stop bounds before the right-axis labels so they remain unobscured.
    ax2.hlines([1200, 10800], 0, 1440, color="#4D4D4D", lw=.8, ls="--")
    style(axes[1], "Energy per 10 min (kWh)"); ax2.set_ylabel("SOC (kWh)"); ax2.tick_params(pad=14); ax2.spines["top"].set_visible(False); axes[1].text(-.11, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axes[1].legend(handles + h2, labels + l2, ncol=3, fontsize=5.8, loc="lower left", bbox_to_anchor=(0, 1.02))
    for ax in axes: ax.set_xlim(0, 1470); ax.set_xticks(range(0, 1441, 240), ["0:00", "4:00", "8:00", "12:00", "16:00", "20:00", "24:00"])
    axes[1].set_xlabel(f"Time of day — representative date: {day.date().isoformat()}")
    save(fig, "q2_dynamic_dispatch_soc", multi=True)
    formal = forecast.loc[forecast.date.between("2025-02-01", "2025-12-31")].copy()
    base = formal.merge(actual[["date", "time_index", "load_kw", "pv_actual_kw"]], on=["date", "time_index"], validate="one_to_one").merge(plan[["date", "time_index", "price_yuan_per_kwh"]], on=["date", "time_index"], validate="one_to_one")
    base_grid = np.maximum(base.forecast_load_kw - base.forecast_pv_kw, 0) * DT
    base_emergency = np.maximum(base.load_kw * DT - base.pv_actual_kw * DT - base_grid, 0)
    base["baseline_cost"] = base_grid * base.price_yuan_per_kwh + base_emergency * 5 * base.price_yuan_per_kwh
    baseline_daily = base.groupby("date").baseline_cost.sum().reindex(daily.date)
    fig, ax = plt.subplots(figsize=(7.09, 3.2), constrained_layout=True)
    ax.plot(daily.date, np.cumsum(baseline_daily), color=COLORS["baseline"], lw=1.3, label="No-storage baseline")
    ax.plot(daily.date, np.cumsum(daily.total_cost_yuan), color=COLORS["grid"], lw=1.35, label="Dynamic battery")
    style(ax, "Cumulative cost (yuan)"); ax.set_xlabel("Date"); ax.set_title("Cumulative operating cost", loc="left", fontsize=7.5); ax.legend(fontsize=6.2); save(fig, "q2_cumulative_cost_dynamic")
    fig, ax = plt.subplots(figsize=(7.09, 3.2), constrained_layout=True)
    ax.bar(daily.date, daily.planned_cost_yuan, width=.9, color=COLORS["grid"], label="Planned grid cost")
    ax.bar(daily.date, daily.actual_emergency_cost_yuan, width=.9, bottom=daily.planned_cost_yuan, color=COLORS["emergency"], label="Emergency cost")
    style(ax, "Daily cost (yuan)"); ax.set_xlabel("Date"); ax.set_title("Daily cost components under dynamic dispatch", loc="left", fontsize=7.5); ax.legend(fontsize=6.2, ncol=2); save(fig, "q2_daily_cost_dynamic")
    print(f"Representative dispatch date: {day.date().isoformat()}")


if __name__ == "__main__": main()
