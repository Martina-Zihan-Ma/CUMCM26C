#!/usr/bin/env python3
"""Render the verified Q2 dynamic cumulative-cost figure from current outputs.

Figure contract: under identical dynamic forecasts, causal battery dispatch
reduces realised cumulative cost versus the no-storage baseline.  All 334
formal operating days are retained; costs are shown in million yuan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "question2"
FIG = OUT / "figures"
sys.path.insert(0, "/Users/mazihan/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def main() -> None:
    plan = pd.read_csv(OUT / "question2_plan.csv", parse_dates=["date"])
    daily = pd.read_csv(OUT / "question2_daily_summary.csv", parse_dates=["date"])
    actual = pd.read_csv(
        ROOT / "data" / "processed" / "attachment2_actual_long.csv",
        parse_dates=["date"],
    )
    forecast = pd.read_csv(OUT / "dynamic_forecasts.csv", parse_dates=["date"])

    formal = forecast.loc[
        forecast["date"].between("2025-02-01", "2025-12-31")
    ].copy()
    baseline = (
        formal.merge(
            actual[["date", "time_index", "load_kw", "pv_actual_kw"]],
            on=["date", "time_index"],
            validate="one_to_one",
        )
        .merge(
            plan[["date", "time_index", "price_yuan_per_kwh"]],
            on=["date", "time_index"],
            validate="one_to_one",
        )
    )
    baseline_grid = np.maximum(
        baseline["forecast_load_kw"] - baseline["forecast_pv_kw"], 0
    ) / 6
    baseline_emergency = np.maximum(
        baseline["load_kw"] / 6 - baseline["pv_actual_kw"] / 6 - baseline_grid,
        0,
    )
    baseline["cost_yuan"] = baseline["price_yuan_per_kwh"] * (
        baseline_grid + 5 * baseline_emergency
    )
    baseline_daily = baseline.groupby("date")["cost_yuan"].sum().reindex(daily["date"])

    fig, ax = plt.subplots(figsize=(7.09, 3.20), constrained_layout=True)
    ax.plot(
        daily["date"], baseline_daily.cumsum() / 1e6,
        color="#7F7F7F", linewidth=1.30, label="No-storage baseline",
    )
    ax.plot(
        daily["date"], daily["total_cost_yuan"].cumsum() / 1e6,
        color="#0072B2", linewidth=1.45, label="Dynamic battery",
    )
    ax.set(
        xlabel="Date (2025)",
        ylabel="Cumulative cost (million yuan)",
        xlim=(daily["date"].min(), daily["date"].max()),
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, loc="upper left")

    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / "q2_cumulative_cost_dynamic"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
