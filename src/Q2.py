#!/usr/bin/env python3
"""
Q2 Step 2 — day-ahead stochastic optimization with reserve sensitivity.

Final design keeps TWO complementary risk controls:
1. SERVICE_LEVEL quantile constraint: intraday coverage of uncertain net demand.
2. Dynamic terminal reserve: interday battery safety margin.

CVaR is intentionally removed because it overlaps with the service-quantile control
(both target scenario-side emergency risk) while adding complexity. The reserve is
not redundant: it controls carry-over SOC across days.

The script tests reserve quantiles 0.75/0.80/0.85/0.90, compares emergency days
first, then emergency energy and total cost, and automatically selects the best
quantile. Only one summary line per quantile is printed; no daily log is printed.

The battery plan is fixed at 0:00. Actual Load/PV never trigger intraday
re-optimization; they only determine realized emergency purchase and spill.
"""

from __future__ import annotations

import os
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
PRICE_FILE = ROOT / "data" / "processed" / "attachment1_standard_day.csv"
ACTUAL_FILE = ROOT / "data" / "processed" / "attachment2_actual_long.csv"
FORECAST_FILE = ROOT / "outputs" / "question2" / "dynamic_forecasts.csv"
TEMPLATE_FILE = ROOT / "results" / "result2_template.xlsx"
OUT = ROOT / "outputs" / "question2"

JAN_START = pd.Timestamp("2025-01-01")
JAN_END = pd.Timestamp("2025-01-31")
START = pd.Timestamp("2025-02-01")
END = pd.Timestamp("2025-12-31")
DATES = pd.date_range(START, END, freq="D")

N = 144
DT = 1.0 / 6.0

ETA_C = 0.90
ETA_D = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_JAN1 = 6000.0
MAX_INTERVAL = 5000.0 * DT

EMERGENCY_MULTIPLIER = 5.0

# Residual scenarios.
RESIDUAL_DAYS = 30
RESIDUAL_HALF_LIFE = 14.0
SCENARIOS = 20
MIN_RESIDUAL_DAYS = 5
RNG_SEED = 20260911

# Risk control.  CVaR removed: service quantile + reserve are retained.
SERVICE_LEVEL = 0.95

# Reserve sensitivity: choose by emergency days, then emergency kWh, then total cost.
RESERVE_QUANTILES = (0.75, 0.80, 0.85, 0.90)
RESERVE_MEDIAN_Q = 0.50
MIN_JAN_CALIBRATION_DAYS = 5
JAN_LOAD_SAME_WEEKDAY = 4
JAN_LOAD_RECENT = 7
JAN_PV_DAYS = 7
JAN_PV_DECAY = 0.82

CYCLE_EPS = 1e-6
SPILL_EPS = 1e-8

# Optional fast debug run, e.g. Q2_MAX_DAYS=5 python src/q2_optimize.py
MAX_DAYS = int(os.environ.get("Q2_MAX_DAYS", "0"))


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------

def read_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    for path in [PRICE_FILE, ACTUAL_FILE, FORECAST_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    price_df = pd.read_csv(PRICE_FILE).sort_values("time_index")
    actual = pd.read_csv(ACTUAL_FILE)
    forecast = pd.read_csv(FORECAST_FILE)

    actual["date"] = pd.to_datetime(actual["date"]).dt.normalize()
    forecast["date"] = pd.to_datetime(forecast["date"]).dt.normalize()

    if "history_end_date" in forecast:
        forecast["history_end_date"] = pd.to_datetime(
            forecast["history_end_date"]
        ).dt.normalize()

    if price_df["time_index"].astype(int).tolist() != list(range(N)):
        raise ValueError("Price time_index must be 0..143.")

    required_actual = {"date", "time_index", "load_kw", "pv_actual_kw"}
    required_forecast = {
        "date", "time_index",
        "forecast_load_kw", "forecast_pv_kw",
    }
    if required_actual - set(actual):
        raise ValueError(f"Actual data missing {sorted(required_actual - set(actual))}.")
    if required_forecast - set(forecast):
        raise ValueError(
            "Run q2_forecast_final.py first. Missing forecast columns: "
            f"{sorted(required_forecast - set(forecast))}"
        )

    for name, df in [("actual", actual), ("forecast", forecast)]:
        for date, part in df.groupby("date"):
            idx = part.sort_values("time_index")["time_index"].astype(int).tolist()
            if idx != list(range(N)):
                raise ValueError(f"{name} {date}: time_index must be 0..143.")

    if "history_end_date" in forecast:
        if not (forecast["history_end_date"] < forecast["date"]).all():
            raise RuntimeError("Forecast leakage: history_end_date >= target date.")

    return price_df["price_yuan_per_kwh"].to_numpy(float), actual, forecast


def day_values(df: pd.DataFrame, date: pd.Timestamp, col: str) -> np.ndarray:
    x = df.loc[df["date"] == date].sort_values("time_index")
    if len(x) != N:
        raise ValueError(f"{date.date()}: {col} has {len(x)} rows, expected {N}.")
    return x[col].to_numpy(float)


# ----------------------------------------------------------------------
# January initialization and reserve rule
# ----------------------------------------------------------------------

def january_load_forecast(actual: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    history = actual.loc[
        (actual["date"] >= JAN_START) & (actual["date"] < date)
    ]
    same = sorted(
        pd.Timestamp(d) for d in history["date"].unique()
        if pd.Timestamp(d).weekday() == date.weekday()
    )
    selected = same[-JAN_LOAD_SAME_WEEKDAY:] if same else sorted(
        pd.Timestamp(d) for d in history["date"].unique()
    )[-JAN_LOAD_RECENT:]
    return np.maximum(
        np.vstack([day_values(history, d, "load_kw") for d in selected]).mean(axis=0),
        0.0,
    )


def january_pv_forecast(actual: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    dates = sorted(
        pd.Timestamp(d) for d in actual["date"].unique()
        if JAN_START <= pd.Timestamp(d) < date
    )[-JAN_PV_DAYS:]
    if not dates:
        raise ValueError(f"No January PV history before {date.date()}.")

    dates = list(reversed(dates))
    weights = np.array([JAN_PV_DECAY ** i for i in range(len(dates))], dtype=float)
    weights /= weights.sum()
    profiles = np.vstack([day_values(actual, d, "pv_actual_kw") for d in dates])
    return np.maximum(np.average(profiles, axis=0, weights=weights), 0.0)


def positive_net_energy(load_kw: np.ndarray, pv_kw: np.ndarray) -> float:
    return float(np.maximum(load_kw - pv_kw, 0.0).sum() * DT)


def forecast_error_risk(
    load_hat: np.ndarray,
    pv_hat: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
) -> float:
    # Positive actual-minus-forecast net-demand error energy.
    error_kw = (load_actual - load_hat) - (pv_actual - pv_hat)
    return float(np.maximum(error_kw, 0.0).sum() * DT)


def january_calibration(actual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in pd.date_range(JAN_START + pd.Timedelta(days=1), JAN_END):
        load_hat = january_load_forecast(actual, date)
        pv_hat = january_pv_forecast(actual, date)
        load_actual = day_values(actual, date, "load_kw")
        pv_actual = day_values(actual, date, "pv_actual_kw")
        rows.append({
            "date": date,
            "forecast_net_kwh": positive_net_energy(load_hat, pv_hat),
            "risk_kwh": forecast_error_risk(
                load_hat, pv_hat, load_actual, pv_actual
            ),
        })
    return pd.DataFrame(rows)


@dataclass
class ReserveRule:
    q50: float
    q_high: float
    reserve_quantile: float
    mean_net_kwh: float


def fit_reserve_rule(
    calibration: pd.DataFrame,
    reserve_quantile: float,
) -> ReserveRule:
    risk = calibration["risk_kwh"].to_numpy(float)
    return ReserveRule(
        q50=float(np.quantile(risk, RESERVE_MEDIAN_Q)),
        q_high=float(np.quantile(risk, reserve_quantile)),
        reserve_quantile=float(reserve_quantile),
        mean_net_kwh=float(calibration["forecast_net_kwh"].mean()),
    )


def reserve_for_day(
    load_hat: np.ndarray,
    pv_hat: np.ndarray,
    rule: ReserveRule,
) -> tuple[float, float, float]:
    """Dynamic minimum terminal SOC calibrated from empirical forecast-error risk."""
    net_kwh = positive_net_energy(load_hat, pv_hat)
    ratio = net_kwh / rule.mean_net_kwh if rule.mean_net_kwh > 1e-12 else 1.0

    # Same cleaned lower-bound rule as the previous version, generalized from Q85
    # to a tested high quantile Qq.
    center = SOC_MIN + (rule.q_high / ETA_D) * ratio
    reserve = center - (rule.q_high - rule.q50) / ETA_D
    reserve = float(np.clip(reserve, SOC_MIN, SOC_MAX))
    return reserve, float(center), float(ratio)


def causal_january_rule(
    calibration: pd.DataFrame,
    date: pd.Timestamp,
    reserve_quantile: float,
) -> ReserveRule | None:
    history = calibration.loc[calibration["date"] < date]
    return (
        fit_reserve_rule(history, reserve_quantile)
        if len(history) >= MIN_JAN_CALIBRATION_DAYS
        else None
    )


# ----------------------------------------------------------------------
# Residual scenarios
# ----------------------------------------------------------------------

def residual_record(
    date: pd.Timestamp,
    load_hat: np.ndarray,
    pv_hat: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
) -> dict:
    return {
        "date": date.normalize(),
        "load_error_kw": (load_actual - load_hat).astype(float),
        "pv_error_kw": (pv_actual - pv_hat).astype(float),
    }


def recent_residuals(history: list[dict], date: pd.Timestamp) -> list[dict]:
    x = sorted(
        [r for r in history if r["date"] < date],
        key=lambda r: r["date"],
    )
    return x[-RESIDUAL_DAYS:]


def sample_scenarios(
    date: pd.Timestamp,
    load_hat: np.ndarray,
    pv_hat: np.ndarray,
    residual_history: list[dict],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = recent_residuals(residual_history, date)

    if len(residuals) < MIN_RESIDUAL_DAYS:
        return (
            np.maximum(load_hat, 0.0)[None, :] * DT,
            np.maximum(pv_hat, 0.0)[None, :] * DT,
        )

    ages = np.array([(date - r["date"]).days for r in residuals], dtype=float)
    weights = np.exp(-np.log(2.0) * ages / RESIDUAL_HALF_LIFE)
    weights /= weights.sum()

    selected = rng.choice(
        len(residuals), size=SCENARIOS, replace=True, p=weights
    )

    load_scenarios = []
    pv_scenarios = []
    for i in selected:
        r = residuals[int(i)]
        load_scenarios.append(
            np.maximum(load_hat + r["load_error_kw"], 0.0) * DT
        )
        pv_scenarios.append(
            np.maximum(pv_hat + r["pv_error_kw"], 0.0) * DT
        )

    return np.vstack(load_scenarios), np.vstack(pv_scenarios)


# ----------------------------------------------------------------------
# Stochastic LP
# ----------------------------------------------------------------------

@dataclass
class VarIndex:
    grid: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    emergency: np.ndarray
    spill: np.ndarray


def make_index(k: int) -> tuple[VarIndex, int]:
    p = 0
    grid = np.arange(p, p + N); p += N
    charge = np.arange(p, p + N); p += N
    discharge = np.arange(p, p + N); p += N
    soc = np.arange(p, p + N + 1); p += N + 1
    emergency = np.arange(p, p + k * N).reshape(k, N); p += k * N
    spill = np.arange(p, p + k * N).reshape(k, N); p += k * N
    return VarIndex(grid, charge, discharge, soc, emergency, spill), p


def solve_plan(
    price: np.ndarray,
    load_scenarios: np.ndarray,
    pv_scenarios: np.ndarray,
    initial_soc: float,
    terminal_reserve: float | None,
) -> dict:
    """
    Stochastic linear program.

    Risk protection is deliberately kept simple:
      (i) service-quantile constraint for intraday net-demand uncertainty;
      (ii) dynamic terminal reserve for interday SOC carry-over.

    CVaR is removed to avoid overlapping intraday risk controls.
    """
    k = load_scenarios.shape[0]
    if load_scenarios.shape != (k, N) or pv_scenarios.shape != (k, N):
        raise ValueError("Bad scenario shape.")

    ix, nv = make_index(k)
    net_scenarios = load_scenarios - pv_scenarios
    service_floor = np.quantile(net_scenarios, SERVICE_LEVEL, axis=0)

    # Objective = planned purchase cost + expected emergency cost.
    c = np.zeros(nv)
    c[ix.grid] = price
    c[ix.charge] = CYCLE_EPS
    c[ix.discharge] = CYCLE_EPS
    for w in range(k):
        c[ix.emergency[w]] = EMERGENCY_MULTIPLIER * price / k
        c[ix.spill[w]] = SPILL_EPS / k

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[ix.charge] = MAX_INTERVAL
    ub[ix.discharge] = MAX_INTERVAL
    lb[ix.soc] = SOC_MIN
    ub[ix.soc] = SOC_MAX
    lb[ix.soc[0]] = ub[ix.soc[0]] = initial_soc
    if terminal_reserve is not None:
        lb[ix.soc[N]] = max(lb[ix.soc[N]], terminal_reserve)

    # Equalities: scenario energy balance + SOC dynamics.
    Aeq = lil_matrix((k * N + N, nv), dtype=float)
    beq = np.zeros(k * N + N)
    row = 0

    for w in range(k):
        for t in range(N):
            # G + E + PV + D = Load + C + Spill
            Aeq[row, ix.grid[t]] = 1.0
            Aeq[row, ix.charge[t]] = -1.0
            Aeq[row, ix.discharge[t]] = 1.0
            Aeq[row, ix.emergency[w, t]] = 1.0
            Aeq[row, ix.spill[w, t]] = -1.0
            beq[row] = load_scenarios[w, t] - pv_scenarios[w, t]
            row += 1

    for t in range(N):
        Aeq[row, ix.soc[t + 1]] = 1.0
        Aeq[row, ix.soc[t]] = -1.0
        Aeq[row, ix.charge[t]] = -ETA_C
        Aeq[row, ix.discharge[t]] = 1.0 / ETA_D
        row += 1

    # Inequalities: battery power envelope + service quantile.
    Aub = lil_matrix((2 * N, nv), dtype=float)
    bub = np.zeros(2 * N)
    row = 0

    for t in range(N):
        Aub[row, ix.charge[t]] = 1.0
        Aub[row, ix.discharge[t]] = 1.0
        bub[row] = MAX_INTERVAL
        row += 1

    for t in range(N):
        # G + D - C >= Q_SERVICE_LEVEL(net demand)
        Aub[row, ix.grid[t]] = -1.0
        Aub[row, ix.discharge[t]] = -1.0
        Aub[row, ix.charge[t]] = 1.0
        bub[row] = -float(service_floor[t])
        row += 1

    result = linprog(
        c=c,
        A_ub=Aub.tocsr(),
        b_ub=bub,
        A_eq=Aeq.tocsr(),
        b_eq=beq,
        bounds=list(zip(lb, ub)),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    x = result.x
    grid = x[ix.grid].copy()
    charge = x[ix.charge].copy()
    discharge = x[ix.discharge].copy()
    soc = x[ix.soc].copy()

    scenario_emergency_cost = np.array([
        np.dot(x[ix.emergency[w]], EMERGENCY_MULTIPLIER * price)
        for w in range(k)
    ])

    return {
        "grid": grid,
        "charge": charge,
        "discharge": discharge,
        "soc_start": soc[:-1],
        "soc_end": soc[1:],
        "terminal_soc": float(soc[-1]),
        "expected_emergency_cost": float(scenario_emergency_cost.mean()),
        "simultaneous": int(((charge > 1e-7) & (discharge > 1e-7)).sum()),
    }


def evaluate_plan(
    grid: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    balance = grid + pv_kwh + discharge - load_kwh - charge
    emergency = np.maximum(-balance, 0.0)
    spill = np.maximum(balance, 0.0)
    error = np.max(np.abs(
        grid + emergency + pv_kwh + discharge
        - load_kwh - charge - spill
    ))
    return emergency, spill, float(error)


# ----------------------------------------------------------------------
# January warm-up
# ----------------------------------------------------------------------

def warm_up_january(
    price: np.ndarray,
    actual: pd.DataFrame,
    calibration: pd.DataFrame,
    reserve_quantile: float,
) -> tuple[float, list[dict], pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    soc = SOC_JAN1
    residuals: list[dict] = []
    rows = [{
        "date": JAN_START,
        "soc_start_kwh": soc,
        "reserve_kwh": np.nan,
        "soc_end_kwh": soc,
    }]

    for date in pd.date_range(JAN_START + pd.Timedelta(days=1), JAN_END):
        load_hat = january_load_forecast(actual, date)
        pv_hat = january_pv_forecast(actual, date)
        rule = causal_january_rule(calibration, date, reserve_quantile)
        reserve = (
            reserve_for_day(load_hat, pv_hat, rule)[0]
            if rule is not None
            else None
        )

        load_s, pv_s = sample_scenarios(date, load_hat, pv_hat, residuals, rng)
        plan = solve_plan(price, load_s, pv_s, soc, reserve)

        load_actual = day_values(actual, date, "load_kw")
        pv_actual = day_values(actual, date, "pv_actual_kw")

        residuals.append(
            residual_record(
                date, load_hat, pv_hat, load_actual, pv_actual
            )
        )
        rows.append({
            "date": date,
            "soc_start_kwh": soc,
            "reserve_kwh": np.nan if reserve is None else reserve,
            "soc_end_kwh": plan["terminal_soc"],
        })
        soc = plan["terminal_soc"]

    return float(soc), residuals, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Strategy runs
# ----------------------------------------------------------------------

def proposed_forecast(
    forecast: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        day_values(forecast, date, "forecast_load_kw"),
        day_values(forecast, date, "forecast_pv_kw"),
    )


def run_optimized_strategy(
    name: str,
    price: np.ndarray,
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    rule: ReserveRule,
    initial_soc: float,
    january_residuals: list[dict],
    dates: list[pd.Timestamp],
    keep_detail: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Same seed for both forecast methods -> same bootstrap draw positions.
    rng = np.random.default_rng(RNG_SEED + 1)
    residuals = deepcopy(january_residuals)
    soc = float(initial_soc)

    plan_rows, dispatch_rows, emergency_rows, daily_rows = [], [], [], []
    max_balance_error = 0.0
    simultaneous = 0

    for i, date in enumerate(dates, 1):
        load_hat, pv_hat = proposed_forecast(forecast, date)
        reserve, reserve_center, net_ratio = reserve_for_day(
            load_hat, pv_hat, rule
        )
        load_s, pv_s = sample_scenarios(
            date, load_hat, pv_hat, residuals, rng
        )
        plan = solve_plan(price, load_s, pv_s, soc, reserve)

        load_actual_kw = day_values(actual, date, "load_kw")
        pv_actual_kw = day_values(actual, date, "pv_actual_kw")
        emergency, spill, balance_error = evaluate_plan(
            plan["grid"], plan["charge"], plan["discharge"],
            load_actual_kw * DT, pv_actual_kw * DT,
        )

        max_balance_error = max(max_balance_error, balance_error)
        simultaneous += plan["simultaneous"]

        planned_cost = float(np.dot(plan["grid"], price))
        emergency_cost = float(
            np.dot(emergency, EMERGENCY_MULTIPLIER * price)
        )

        daily_rows.append({
            "strategy": name,
            "date": date,
            "soc_start_kwh": soc,
            "reserve_kwh": reserve,
            "reserve_center_kwh": reserve_center,
            "net_ratio_to_january": net_ratio,
            "soc_end_kwh": plan["terminal_soc"],
            "planned_purchase_kwh": float(plan["grid"].sum()),
            "planned_cost_yuan": planned_cost,
            "expected_emergency_cost_yuan": plan["expected_emergency_cost"],
            "actual_emergency_kwh": float(emergency.sum()),
            "actual_emergency_cost_yuan": emergency_cost,
            "actual_emergency_intervals": int((emergency > 1e-8).sum()),
            "spill_kwh": float(spill.sum()),
            "total_cost_yuan": planned_cost + emergency_cost,
        })

        if keep_detail:
            for t in range(N):
                plan_rows.append({
                    "date": date,
                    "time_index": t,
                    "grid_purchase_kwh": float(plan["grid"][t]),
                    "price_yuan_per_kwh": float(price[t]),
                    "grid_cost_yuan": float(plan["grid"][t] * price[t]),
                })
                dispatch_rows.append({
                    "date": date,
                    "time_index": t,
                    "charge_kwh": float(plan["charge"][t]),
                    "discharge_kwh": float(plan["discharge"][t]),
                    "soc_start_kwh": float(plan["soc_start"][t]),
                    "soc_end_kwh": float(plan["soc_end"][t]),
                    "spill_kwh": float(spill[t]),
                })
                emergency_rows.append({
                    "date": date,
                    "time_index": t,
                    "emergency_purchase_kwh": float(emergency[t]),
                    "emergency_cost_yuan": float(
                        emergency[t] * EMERGENCY_MULTIPLIER * price[t]
                    ),
                })

        residuals.append(
            residual_record(
                date, load_hat, pv_hat, load_actual_kw, pv_actual_kw
            )
        )
        soc = plan["terminal_soc"]


    daily = pd.DataFrame(daily_rows)
    daily.attrs["max_balance_error"] = max_balance_error
    daily.attrs["simultaneous"] = simultaneous

    return (
        pd.DataFrame(plan_rows),
        pd.DataFrame(dispatch_rows),
        pd.DataFrame(emergency_rows),
        daily,
    )


def no_storage_same_forecast(
    price: np.ndarray,
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for date in dates:
        load_hat, pv_hat = proposed_forecast(forecast, date)
        load_actual = day_values(actual, date, "load_kw") * DT
        pv_actual = day_values(actual, date, "pv_actual_kw") * DT

        grid = np.maximum(load_hat - pv_hat, 0.0) * DT
        balance = grid + pv_actual - load_actual
        emergency = np.maximum(-balance, 0.0)
        spill = np.maximum(balance, 0.0)

        planned_cost = float(np.dot(grid, price))
        emergency_cost = float(
            np.dot(emergency, EMERGENCY_MULTIPLIER * price)
        )

        rows.append({
            "strategy": "no_storage_same_forecast",
            "date": date,
            "planned_purchase_kwh": float(grid.sum()),
            "planned_cost_yuan": planned_cost,
            "actual_emergency_kwh": float(emergency.sum()),
            "actual_emergency_cost_yuan": emergency_cost,
            "actual_emergency_intervals": int((emergency > 1e-8).sum()),
            "spill_kwh": float(spill.sum()),
            "total_cost_yuan": planned_cost + emergency_cost,
        })

    return pd.DataFrame(rows)


def summarize_strategy(daily: pd.DataFrame) -> dict:
    emergency_kwh = float(daily["actual_emergency_kwh"].sum())
    emergency_cost = float(daily["actual_emergency_cost_yuan"].sum())
    return {
        "strategy": daily["strategy"].iloc[0],
        "planned_purchase_kwh": float(daily["planned_purchase_kwh"].sum()),
        "planned_cost_yuan": float(daily["planned_cost_yuan"].sum()),
        "emergency_purchase_kwh": emergency_kwh,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": float(daily["total_cost_yuan"].sum()),
        "spill_kwh": float(daily["spill_kwh"].sum()),
        "emergency_days": int((daily["actual_emergency_kwh"] > 1e-8).sum()),
        "emergency_intervals": int(daily["actual_emergency_intervals"].sum()),
        "avg_normal_price_of_emergency": (
            emergency_cost / (EMERGENCY_MULTIPLIER * emergency_kwh)
            if emergency_kwh > 1e-12 else np.nan
        ),
    }


# ----------------------------------------------------------------------
# Official workbook for the final strategy
# ----------------------------------------------------------------------

def compress_emergency(emergency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, day in emergency.loc[
        emergency["emergency_purchase_kwh"] > 1e-8
    ].groupby("date"):
        idx = day.sort_values("time_index")["time_index"].astype(int).tolist()
        values = day.sort_values("time_index")["emergency_purchase_kwh"].to_numpy(float)

        start = 0
        for j in range(1, len(idx) + 1):
            end_group = j == len(idx) or idx[j] != idx[j - 1] + 1
            if end_group:
                a = idx[start]
                b = idx[j - 1] + 1
                start_min = a * 10
                end_min = b * 10
                fmt = lambda m: "24:00" if m == 1440 else f"{m // 60}:{m % 60:02d}"
                rows.append({
                    "date": pd.Timestamp(date),
                    "period": f"{fmt(start_min)}-{fmt(end_min)}",
                    "emergency_kwh": float(values[start:j].sum()),
                })
                start = j
    return pd.DataFrame(rows)


def find_template() -> Path:
    """
    Locate the official blank result2 workbook without changing its structure.
    """
    candidates = [
        ROOT / "results" / "result2_template.xlsx",
        ROOT / "results" / "result2.xlsx",
        ROOT / "results" / "result2(3).xlsx",
        ROOT / "result2_template.xlsx",
        ROOT / "result2(3).xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find the official result2 template. "
        "Put result2_template.xlsx (or result2(3).xlsx) under results/."
    )


def _write_emergency_sample(
    ws,
    compressed: pd.DataFrame,
    date: pd.Timestamp,
    rows: list[int],
) -> None:
    """
    Fill ONLY the rows already reserved by the official template.

    If a sampled date has more emergency periods than available template rows,
    the final available row combines all remaining periods so that no emergency
    energy is lost while the official worksheet structure stays unchanged.
    """
    group = (
        compressed.loc[compressed["date"] == date]
        .sort_values("period")
        .reset_index(drop=True)
    )

    # Clear only the result cells; keep the template date / ellipsis structure.
    for r in rows:
        ws.cell(r, 2).value = None
        ws.cell(r, 3).value = None

    if group.empty:
        ws.cell(rows[0], 2).value = "无"
        ws.cell(rows[0], 3).value = 0.0
        return

    capacity = len(rows)

    # Fits directly.
    if len(group) <= capacity:
        for j, rec in group.iterrows():
            ws.cell(rows[j], 2).value = str(rec["period"])
            ws.cell(rows[j], 3).value = float(rec["emergency_kwh"])
        return

    # Keep the first capacity-1 periods; combine the rest in the last row.
    for j in range(capacity - 1):
        rec = group.iloc[j]
        ws.cell(rows[j], 2).value = str(rec["period"])
        ws.cell(rows[j], 3).value = float(rec["emergency_kwh"])

    rest = group.iloc[capacity - 1:]
    ws.cell(rows[-1], 2).value = "；".join(rest["period"].astype(str))
    ws.cell(rows[-1], 3).value = float(rest["emergency_kwh"].sum())


def write_result2(
    plan: pd.DataFrame,
    dispatch: pd.DataFrame,
    emergency: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> Path:
    """
    Fill the official result2 workbook STRICTLY in place.

    Sheet 1:
        Fill all Feb-Dec dates already present in the template.

    Sheet 2:
        Fill only the three sample dates already shown by the template:
        2025-02-01, 2025-02-02, 2025-12-31.
        No rows or dates are added.

    Sheet 3:
        Fill only the same three sample dates already shown by the template.
        No rows or dates are added; the ellipsis row is preserved.

    Output:
        outputs/question2/result2.xlsx
    """
    template = find_template()
    wb = load_workbook(template)

    expected_sheets = ["计划购电量", "充放电量", "紧急购电量"]
    if wb.sheetnames != expected_sheets:
        raise ValueError(
            "The workbook is not the official result2 template. "
            f"Found sheets: {wb.sheetnames}"
        )

    if len(dates) != 334:
        raise ValueError("Official result2.xlsx requires all 334 Feb-Dec days.")

    # ==============================================================
    # Sheet 1 — 计划购电量
    # Keep the official 334 date rows and fill values only.
    # ==============================================================
    ws = wb["计划购电量"]

    if ws.max_row != 335 or ws.max_column != 147:
        raise ValueError(
            "计划购电量 template shape changed; expected 335 rows x 147 columns."
        )

    for r, date in enumerate(dates, start=2):
        template_date = pd.Timestamp(ws.cell(r, 1).value).normalize()
        if template_date != date.normalize():
            raise ValueError(
                f"计划购电量 row {r} date mismatch: "
                f"{template_date.date()} != {date.date()}"
            )

        day = plan.loc[plan["date"] == date].sort_values("time_index")
        if len(day) != N:
            raise ValueError(f"{date.date()}: final plan does not have 144 slots.")

        values = day["grid_purchase_kwh"].to_numpy(float)

        # Fill only the blank numeric cells B:EQ.
        for t, value in enumerate(values):
            ws.cell(r, 2 + t).value = float(value)

        ws.cell(r, 146).value = float(values.sum())
        ws.cell(r, 147).value = float(day["grid_cost_yuan"].sum())

    # ==============================================================
    # Sheet 2 — 充放电量
    # IMPORTANT: do NOT expand to every day.
    # The official template shows only Feb 1, Feb 2, ..., Dec 31.
    # ==============================================================
    ws = wb["充放电量"]

    sample_blocks = {
        pd.Timestamp("2025-02-01"): list(range(2, 8)),
        pd.Timestamp("2025-02-02"): list(range(8, 14)),
        pd.Timestamp("2025-12-31"): list(range(15, 21)),
    }

    # Preserve row 14 ("⁝") exactly as supplied.
    for date, rows in sample_blocks.items():
        day = dispatch.loc[dispatch["date"] == date].sort_values("time_index")
        if len(day) != N:
            raise ValueError(f"{date.date()}: dispatch does not have 144 slots.")

        for block, r in enumerate(rows):
            part = day.iloc[24 * block: 24 * (block + 1)]

            # A/B/E are already provided by the official template.
            # Fill C/D and the two SOC cells only.
            ws.cell(r, 3).value = float(part["charge_kwh"].sum())
            ws.cell(r, 4).value = float(part["discharge_kwh"].sum())

            if block == 0:
                ws.cell(r, 6).value = float(day["soc_start_kwh"].iloc[0])
            elif block == 1:
                ws.cell(r, 6).value = float(day["soc_end_kwh"].iloc[-1])

    # ==============================================================
    # Sheet 3 — 紧急购电量
    # IMPORTANT: do NOT list all 334 days.
    # Use only the rows/dates already reserved in the template.
    # ==============================================================
    ws = wb["紧急购电量"]
    compressed = compress_emergency(emergency)

    # Template layout:
    #   Feb 1: rows 2-4
    #   Feb 2: rows 5-7
    #   row 8: "⁝" (must remain untouched)
    #   Dec 31: row 9
    _write_emergency_sample(
        ws, compressed, pd.Timestamp("2025-02-01"), [2, 3, 4]
    )
    _write_emergency_sample(
        ws, compressed, pd.Timestamp("2025-02-02"), [5, 6, 7]
    )
    _write_emergency_sample(
        ws, compressed, pd.Timestamp("2025-12-31"), [9]
    )

    output = OUT / "result2_final.xlsx"
    wb.save(output)
    return output


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    price, actual, forecast = read_inputs()

    dates = list(DATES)
    if MAX_DAYS > 0:
        dates = dates[:MAX_DAYS]

    calibration = january_calibration(actual)
    no_storage = no_storage_same_forecast(price, actual, forecast, dates)
    baseline_stats = summarize_strategy(no_storage)

    # --------------------------------------------------------------
    # Reserve-quantile sensitivity.
    # Selection priority requested for Q2:
    #   1) minimum emergency days
    #   2) minimum emergency energy
    #   3) minimum total cost
    # --------------------------------------------------------------
    sensitivity_rows = []
    state_by_q = {}

    print("Q2 reserve-quantile sensitivity")
    print("================================")
    print(
        f"Risk design: service quantile={SERVICE_LEVEL:.2f} + dynamic reserve; "
        "CVaR removed."
    )
    print()

    for q in RESERVE_QUANTILES:
        rule = fit_reserve_rule(calibration, q)
        feb1_soc, january_residuals, january_warmup = warm_up_january(
            price, actual, calibration, q
        )

        _, _, _, daily = run_optimized_strategy(
            name=f"reserve_q{q:.2f}",
            price=price,
            actual=actual,
            forecast=forecast,
            rule=rule,
            initial_soc=feb1_soc,
            january_residuals=january_residuals,
            dates=dates,
            keep_detail=False,
        )

        stats = summarize_strategy(daily)
        sensitivity_rows.append({
            "reserve_quantile": q,
            "q50_risk_kwh": rule.q50,
            "q_high_risk_kwh": rule.q_high,
            "feb1_soc_kwh": feb1_soc,
            **stats,
        })
        state_by_q[q] = (rule, feb1_soc, january_residuals, january_warmup)

        print(
            f"Q={q:.2f} | emergency days {stats['emergency_days']:3d}/{len(dates)} "
            f"| emergency {stats['emergency_purchase_kwh']:,.1f} kWh "
            f"| emergency cost {stats['emergency_cost_yuan']:,.0f} yuan "
            f"| total cost {stats['total_cost_yuan']:,.0f} yuan"
        )

    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["emergency_days", "emergency_purchase_kwh", "total_cost_yuan", "reserve_quantile"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    selected_q = float(sensitivity.iloc[0]["reserve_quantile"])
    rule, feb1_soc, january_residuals, january_warmup = state_by_q[selected_q]

    print()
    print(
        f"Selected reserve quantile: Q={selected_q:.2f} "
        "(priority: emergency days -> emergency kWh -> total cost)"
    )

    # Re-run ONLY the selected quantile with detailed outputs for result2.xlsx.
    plan, dispatch, emergency, final_daily = run_optimized_strategy(
        name="optimized_proposed",
        price=price,
        actual=actual,
        forecast=forecast,
        rule=rule,
        initial_soc=feb1_soc,
        january_residuals=january_residuals,
        dates=dates,
        keep_detail=True,
    )

    final_stats = summarize_strategy(final_daily)
    comparison = pd.DataFrame([
        summarize_strategy(no_storage),
        final_stats,
    ])

    no_storage_cost = float(baseline_stats["total_cost_yuan"])
    comparison["saving_vs_no_storage_yuan"] = (
        no_storage_cost - comparison["total_cost_yuan"]
    )
    comparison["saving_vs_no_storage_pct"] = (
        100.0 * comparison["saving_vs_no_storage_yuan"] / no_storage_cost
    )

    # Save outputs.
    calibration.to_csv(
        OUT / "january_reserve_calibration.csv", index=False, encoding="utf-8-sig"
    )
    january_warmup.to_csv(
        OUT / "january_warmup.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity.to_csv(
        OUT / "reserve_quantile_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(
        OUT / "strategy_comparison.csv", index=False, encoding="utf-8-sig"
    )
    final_daily.to_csv(
        OUT / "question2_daily_summary.csv", index=False, encoding="utf-8-sig"
    )
    plan.to_csv(OUT / "question2_plan.csv", index=False, encoding="utf-8-sig")
    dispatch.to_csv(
        OUT / "question2_dispatch.csv", index=False, encoding="utf-8-sig"
    )
    emergency.to_csv(
        OUT / "question2_emergency.csv", index=False, encoding="utf-8-sig"
    )

    if len(dates) == len(DATES):
        try:
            excel = write_result2(plan, dispatch, emergency, dates)
        except FileNotFoundError:
            excel = None
    else:
        excel = None

    continuity = (
        final_daily["soc_end_kwh"].iloc[:-1].to_numpy()
        - final_daily["soc_start_kwh"].iloc[1:].to_numpy()
        if len(final_daily) > 1 else np.array([0.0])
    )
    terminal = final_daily["soc_end_kwh"].to_numpy(float)
    reserve = final_daily["reserve_kwh"].to_numpy(float)
    on_reserve = int(np.isclose(terminal, reserve, atol=1e-5).sum())

    saving_yuan = baseline_stats["total_cost_yuan"] - final_stats["total_cost_yuan"]
    saving_pct = 100.0 * saving_yuan / baseline_stats["total_cost_yuan"]

    summary = f"""Q2 FINAL SUMMARY
================
Selected reserve quantile:     {selected_q:.2f}
Service quantile:              {SERVICE_LEVEL:.2f}
Risk controls retained:        service quantile + dynamic reserve
Risk control removed:          CVaR

Final strategy
--------------
Total purchase cost:           {final_stats['total_cost_yuan']:,.2f} yuan
Planned purchase cost:         {final_stats['planned_cost_yuan']:,.2f} yuan
Emergency purchase cost:       {final_stats['emergency_cost_yuan']:,.2f} yuan
Emergency purchase energy:     {final_stats['emergency_purchase_kwh']:,.2f} kWh
Emergency days:                {final_stats['emergency_days']} / {len(final_daily)}
Emergency intervals:           {final_stats['emergency_intervals']}
Spill energy:                  {final_stats['spill_kwh']:,.2f} kWh

No-storage baseline
-------------------
Total purchase cost:           {baseline_stats['total_cost_yuan']:,.2f} yuan
Emergency purchase cost:       {baseline_stats['emergency_cost_yuan']:,.2f} yuan

Economic improvement
--------------------
Cost saving:                   {saving_yuan:,.2f} yuan
Cost saving rate:              {saving_pct:.2f}%

Battery / feasibility
---------------------
Feb-1 initial SOC:             {feb1_soc:,.2f} kWh
Terminal SOC range:            {terminal.min():,.2f} to {terminal.max():,.2f} kWh
Days on minimum reserve:       {on_reserve} / {len(final_daily)}
SOC continuity error:          {np.max(np.abs(continuity)):.3e} kWh
Balance error:                 {final_daily.attrs['max_balance_error']:.3e} kWh
Simultaneous C/D count:        {final_daily.attrs['simultaneous']}

Output workbook
---------------
{excel if excel is not None else 'Not written (template absent or debug mode)'}
"""
    (OUT / "question2_summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":
    main()
