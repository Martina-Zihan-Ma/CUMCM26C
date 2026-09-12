#!/usr/bin/env python3
"""Publication-grade Question 2 figures generated from validated outputs.

Figure contract
---------------
Core conclusion: The optimized storage strategy lowers cumulative realised cost
relative to the no-storage strategy while maintaining an SOC trajectory above
the dynamic end-of-day reserve.
Archetype: two independent quantitative single-panel figures.
Source data: validated Question 2 CSV outputs; no observations are excluded.
Statistics: deterministic full-year operational simulation (n = 334 days), so
no inferential uncertainty interval is applicable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "question2"
FIG = OUT / "figures" / "nature"
SKILL_SCRIPTS = Path("/Users/mazihan/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


# Required editable-vector and journal-width settings.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

COLORS = {
    "optimized": "#0F4D92",
    "baseline": "#767676",
    "reserve": "#B64342",
    "soc": "#42949E",
    "bound": "#4D4D4D",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Run the rendered layout gate, then produce editable and print exports."""
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=FIG / f"{stem}.alignment.json",
        overlay_svg=FIG / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def baseline_daily_cost() -> pd.DataFrame:
    """Recreate only the specified no-storage settlement for visual comparison."""
    forecast = pd.read_csv(OUT / "dynamic_forecasts.csv", parse_dates=["date"])
    actual = pd.read_csv(ROOT / "data" / "processed" / "attachment2_actual_long.csv", parse_dates=["date"])
    price = pd.read_csv(ROOT / "data" / "processed" / "attachment1_standard_day.csv")
    date_start, date_end = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")
    forecast = forecast.loc[forecast["date"].between(date_start, date_end)].copy()
    actual = actual.loc[actual["date"].between(date_start, date_end), ["date", "time_index", "load_kw", "pv_actual_kw"]]
    x = forecast.merge(actual, on=["date", "time_index"], validate="one_to_one")
    x = x.merge(price[["time_index", "price_yuan_per_kwh"]], on="time_index", validate="many_to_one")
    grid = np.maximum(x["forecast_load_kw"] - x["forecast_pv_kw"], 0.0) / 6
    balance = grid + x["pv_actual_kw"] / 6 - x["load_kw"] / 6
    emergency = np.maximum(-balance, 0.0)
    x["total_cost_yuan"] = grid * x["price_yuan_per_kwh"] + 5 * emergency * x["price_yuan_per_kwh"]
    return x.groupby("date", as_index=False)["total_cost_yuan"].sum()


def draw_cost() -> None:
    daily = pd.read_csv(OUT / "question2_daily_summary.csv", parse_dates=["date"])
    baseline = baseline_daily_cost()
    assert len(daily) == len(baseline) == 334
    assert np.isclose(daily["total_cost_yuan"].sum(), 14587923.272416, rtol=1e-8)
    assert np.isclose(baseline["total_cost_yuan"].sum(), 18940040.161099, rtol=1e-8)

    x = np.arange(1, len(daily) + 1)
    optimized_cum = daily["total_cost_yuan"].cumsum().to_numpy() / 1e6
    baseline_cum = baseline["total_cost_yuan"].cumsum().to_numpy() / 1e6
    final_saving = baseline_cum[-1] - optimized_cum[-1]

    fig, ax = plt.subplots(figsize=(7.09, 2.75))  # 180 mm wide at journal scale
    ax.plot(x, baseline_cum, color=COLORS["baseline"], lw=1.25, label="No-storage baseline")
    ax.plot(x, optimized_cum, color=COLORS["optimized"], lw=1.65, label="Optimized storage")
    ax.fill_between(x, optimized_cum, baseline_cum, color=COLORS["optimized"], alpha=.12, linewidth=0)
    ax.set(xlim=(1, 334), ylim=(0, baseline_cum[-1] * 1.07), xlabel="Operating day (2025)", ylabel="Cumulative cost (million yuan)")
    ax.set_xticks([1, 59, 120, 181, 243, 304, 334], ["Feb", "Apr", "Jun", "Aug", "Oct", "Dec", "Dec 31"])
    ax.legend(loc="upper left", handlelength=2.5)
    ax.annotate(
        f"Annual saving\n{final_saving:.2f} million yuan\n({100 * final_saving / baseline_cum[-1]:.2f}%)",
        xy=(235, 18.2),
        color=COLORS["optimized"], ha="left", va="center",
    )
    ax.text(-.075, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.91)
    save_figure(fig, "q2_cumulative_cost")


def draw_soc() -> None:
    daily = pd.read_csv(OUT / "question2_daily_summary.csv", parse_dates=["date"])
    assert len(daily) == 334
    assert (daily["soc_end_kwh"] >= daily["reserve_kwh"] - 1e-5).all()
    x = np.arange(1, len(daily) + 1)
    fig, ax = plt.subplots(figsize=(7.09, 2.75))
    ax.fill_between(x, 1200, daily["reserve_kwh"], color=COLORS["reserve"], alpha=.08, linewidth=0)
    ax.plot(x, daily["reserve_kwh"], color=COLORS["reserve"], lw=1.0, ls="--", label="Dynamic terminal reserve")
    ax.plot(x, daily["soc_end_kwh"], color=COLORS["soc"], lw=1.35, label="End-of-day SOC")
    ax.axhline(1200, color=COLORS["bound"], lw=.75, ls=":", label="SOC lower bound")
    ax.set(xlim=(1, 334), ylim=(800, 11200), xlabel="Operating day (2025)", ylabel="Energy (kWh)")
    ax.set_xticks([1, 59, 120, 181, 243, 304, 334], ["Feb", "Apr", "Jun", "Aug", "Oct", "Dec", "Dec 31"])
    ax.legend(
        loc="upper center", bbox_to_anchor=(.5, 1.19), ncol=3,
        columnspacing=1.2, handlelength=2.3,
    )
    ax.text(330, 1200, "1,200", color=COLORS["bound"], ha="right", va="bottom", fontsize=6.5)
    ax.text(-.075, 1.04, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=.105, right=.985, bottom=.23, top=.82)
    save_figure(fig, "q2_soc_reserve")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    draw_cost()
    draw_soc()
    (FIG / "qa_notes.txt").write_text(
        "Figure contract: full 334-day deterministic operational simulation; no observations excluded.\n"
        "Panel a: cumulative realised cost, optimized storage versus the specified same-forecast no-storage settlement.\n"
        "Panel b: end-of-day SOC versus dynamic terminal reserve; SOC is checked for every day.\n"
        "No statistical intervals: these are deterministic full-period accounting outputs, not sampled replicates.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
