#!/usr/bin/env python3
"""Paper-ready summary graphics and metric table for Question 2 outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "question2"
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(OUT / "question2_daily_summary.csv", parse_dates=["date"])
    comparison = pd.read_csv(OUT / "strategy_comparison.csv")
    metrics = pd.read_csv(OUT / "forecast_metrics.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date"], daily["total_cost_yuan"], lw=1.1, color="#1677b8", label="Optimized total cost")
    ax.plot(daily["date"], daily["planned_cost_yuan"], lw=.9, color="#5e9f65", label="Planned cost")
    ax.plot(daily["date"], daily["actual_emergency_cost_yuan"], lw=.9, color="#d45d45", label="Emergency cost")
    ax.set(xlabel="Date", ylabel="Yuan/day", title="Daily operating cost of the proposed strategy")
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "daily_cost_components.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["date"], daily["soc_end_kwh"], lw=1.2, color="#694c9f", label="End-of-day SOC")
    ax.plot(daily["date"], daily["reserve_kwh"], lw=.9, color="#d48a35", label="Dynamic minimum reserve")
    ax.axhline(1200, color="black", lw=.8, ls="--", label="SOC lower bound")
    ax.set(xlabel="Date", ylabel="kWh", title="Interday SOC trajectory and dynamic reserve")
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "soc_and_reserve.png", dpi=220)
    plt.close(fig)

    display = comparison[["strategy", "total_cost_yuan", "emergency_purchase_kwh", "spill_kwh"]].copy()
    display.to_csv(OUT / "paper_metrics.csv", index=False, encoding="utf-8-sig")
    feb = metrics.loc[metrics["period"] == "feb_dec"].copy()
    feb.to_csv(OUT / "paper_forecast_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
