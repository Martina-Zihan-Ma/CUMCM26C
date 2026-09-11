#!/usr/bin/env python3
"""
Q2 Step 1 — unified causal day-ahead Load/PV forecasting.

Design
------
1. January is the warm-up/calibration period.
2. The SAME forecasting functions are used from January 2 onward.
3. For target day d, only observations strictly before d are used.
4. Load uses a rolling Ridge model once enough complete history exists;
   otherwise the same function falls back to a causal historical average.
5. PV always uses a causal exponentially weighted recent-history profile,
   with a mild scale correction learned only from earlier forecast errors.

Outputs
-------
outputs/question2/dynamic_forecasts.csv
    Forecasts for 2025-01-02 to 2025-12-31.  January rows are retained so
    the optimization script can use exactly the same forecast model during
    SOC warm-up and risk calibration.
outputs/question2/forecast_metrics.csv
    MAE / RMSE / bias, reported for January warm-up and Feb-Dec separately.
outputs/question2/january_diagnostics.csv
    Simple January-only descriptive diagnostics (not a second model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "attachment2_actual_long.csv"
OUT = ROOT / "outputs" / "question2"

YEAR_START = pd.Timestamp("2025-01-01")
FORECAST_START = pd.Timestamp("2025-01-02")
OPERATION_START = pd.Timestamp("2025-02-01")
YEAR_END = pd.Timestamp("2025-12-31")
N = 144

# Load forecast.
TRAIN_DAYS = 28
RIDGE_ALPHA = 1.0
MIN_RIDGE_DAYS = 7
SAME_WEEKDAY_DAYS = 4
RECENT_LOAD_DAYS = 7

# PV forecast.
PV_DAYS = 7
PV_DECAY = 0.82
PV_RATIO_DAYS = 3
PV_RATIO_STRENGTH = 0.50

# Simple benchmark.
BASELINE_DAYS = 7


# ----------------------------------------------------------------------
# Input and reshaping
# ----------------------------------------------------------------------

def read_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    required = {"date", "time_label", "time_index", "load_kw", "pv_actual_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["time_index"] = df["time_index"].astype(int)
    df = df.sort_values(["date", "time_index"]).reset_index(drop=True)

    if df["date"].min() > YEAR_START or df["date"].max() < YEAR_END:
        raise ValueError("Full 2025 actual Load/PV data are required.")

    counts = df.groupby("date").size()
    if not (counts == N).all():
        bad = counts[counts != N]
        raise ValueError(f"Each day must have {N} rows. Bad days: {bad.index.tolist()[:5]}")

    for d, part in df.groupby("date", sort=False):
        idx = part["time_index"].tolist()
        if idx != list(range(N)):
            raise ValueError(f"{d.date()}: time_index must be 0..{N - 1}.")

    return df


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="time_index", values=col).sort_index()


# ----------------------------------------------------------------------
# January descriptive diagnostics only
# ----------------------------------------------------------------------

def mean_profile_corr(x: pd.DataFrame, lag: int) -> float:
    vals: list[float] = []
    for i in range(lag, len(x)):
        a = x.iloc[i].to_numpy(float)
        b = x.iloc[i - lag].to_numpy(float)
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)) if vals else np.nan


def january_diagnostics(load: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    jan_load = load.loc["2025-01-01":"2025-01-31"]
    jan_pv = pv.loc["2025-01-01":"2025-01-31"]

    rows = []
    for lag in [1, 7, 14, 21]:
        rows.append({
            "series": "load",
            "metric": f"profile_corr_lag_{lag}",
            "value": mean_profile_corr(jan_load, lag),
        })
    for lag in [1, 7]:
        rows.append({
            "series": "pv",
            "metric": f"profile_corr_lag_{lag}",
            "value": mean_profile_corr(jan_pv, lag),
        })

    daily = jan_load.sum(axis=1).rename("load_total").to_frame()
    daily["weekday"] = daily.index.day_name()
    for weekday, g in daily.groupby("weekday"):
        rows.append({
            "series": "load",
            "metric": f"weekday_mean_{weekday}",
            "value": float(g["load_total"].mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Unified Load forecast
# ----------------------------------------------------------------------

LOAD_FEATURES = [
    "lag1",
    "lag7",
    "mean7",
    "same_weekday",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "tod_sin1",
    "tod_cos1",
    "tod_sin2",
    "tod_cos2",
]


def build_load_features(load: pd.DataFrame) -> pd.DataFrame:
    dates = load.index
    cols = load.columns

    same_weekday = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for wd in range(7):
        dts = [d for d in dates if d.weekday() == wd]
        x = load.loc[dts]
        same_weekday.loc[dts] = (
            x.shift(1).rolling(SAME_WEEKDAY_DAYS, min_periods=1).mean().to_numpy()
        )

    feature_wide = {
        "load_kw": load,
        "lag1": load.shift(1),
        "lag7": load.shift(7),
        "mean7": load.shift(1).rolling(RECENT_LOAD_DAYS, min_periods=1).mean(),
        "same_weekday": same_weekday,
    }

    pieces = [
    x.stack(dropna=False).rename(name)
    for name, x in feature_wide.items()]
    table = pd.concat(pieces, axis=1).reset_index()
    table.columns = ["date", "time_index"] + list(feature_wide)

    doy = table["date"].dt.dayofyear.to_numpy(float)
    dow = table["date"].dt.weekday.to_numpy(float)
    slot = table["time_index"].to_numpy(float)

    table["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    table["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    table["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    table["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    table["tod_sin1"] = np.sin(2 * np.pi * slot / N)
    table["tod_cos1"] = np.cos(2 * np.pi * slot / N)
    table["tod_sin2"] = np.sin(4 * np.pi * slot / N)
    table["tod_cos2"] = np.cos(4 * np.pi * slot / N)
    return table


def fallback_load_forecast(load: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    """Causal fallback used only when Ridge lacks enough complete history."""
    history = [d for d in load.index if d < date]
    if not history:
        raise ValueError(f"No Load history before {date.date()}.")

    recent = history[-RECENT_LOAD_DAYS:]
    recent_mean = load.loc[recent].mean(axis=0).to_numpy(float)

    same = [d for d in history if d.weekday() == date.weekday()][-SAME_WEEKDAY_DAYS:]
    if same:
        same_mean = load.loc[same].mean(axis=0).to_numpy(float)
        pred = 0.70 * same_mean + 0.30 * recent_mean
    else:
        pred = recent_mean

    return np.maximum(pred, 0.0)


def predict_load(
    load: pd.DataFrame,
    features: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[np.ndarray, str]:
    test = features.loc[features["date"] == date].sort_values("time_index").copy()
    if len(test) != N:
        raise ValueError(f"{date.date()}: Load test row count != {N}.")

    train = features.loc[
        (features["date"] < date)
        & (features["date"] >= date - pd.Timedelta(days=TRAIN_DAYS))
    ].dropna(subset=LOAD_FEATURES + ["load_kw"])

    complete_days = train["date"].nunique()
    test_complete = not test[LOAD_FEATURES].isna().any().any()

    if complete_days < MIN_RIDGE_DAYS or not test_complete:
        return fallback_load_forecast(load, date), "historical_fallback"

    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[LOAD_FEATURES], train["load_kw"])
    pred = np.maximum(model.predict(test[LOAD_FEATURES]), 0.0)
    return pred, "rolling_ridge"


# ----------------------------------------------------------------------
# Unified PV forecast
# ----------------------------------------------------------------------

def pv_base_from_history(pv: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    prior = [d for d in pv.index if d < date][-PV_DAYS:]
    if not prior:
        raise ValueError(f"No PV history before {date.date()}.")

    newest_first = list(reversed(prior))
    weights = np.array([PV_DECAY ** i for i in range(len(newest_first))], dtype=float)
    weights /= weights.sum()
    profiles = np.vstack([pv.loc[d].to_numpy(float) for d in newest_first])
    return np.average(profiles, axis=0, weights=weights)


def predict_pv(
    pv: pd.DataFrame,
    date: pd.Timestamp,
    ratios: dict[pd.Timestamp, float],
) -> tuple[np.ndarray, float]:
    base = pv_base_from_history(pv, date)
    recent = [d for d in sorted(ratios) if d < date][-PV_RATIO_DAYS:]
    recent_ratio = float(np.mean([ratios[d] for d in recent])) if recent else 1.0
    scale = 1.0 + PV_RATIO_STRENGTH * (recent_ratio - 1.0)
    scale = float(np.clip(scale, 0.5, 1.5))
    return np.maximum(scale * base, 0.0), scale


def update_pv_ratio(
    pv: pd.DataFrame,
    date: pd.Timestamp,
    ratios: dict[pd.Timestamp, float],
) -> None:
    base = pv_base_from_history(pv, date)
    if base.sum() > 1e-9:
        ratios[date] = float(pv.loc[date].sum() / base.sum())


# ----------------------------------------------------------------------
# Baseline and metrics
# ----------------------------------------------------------------------

def baseline_predictions(
    load: pd.DataFrame,
    pv: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    history = [d for d in load.index if d < date]
    if not history:
        raise ValueError(f"No history before {date.date()}.")
    recent = history[-BASELINE_DAYS:]
    return (
        load.loc[recent].mean(axis=0).to_numpy(float),
        pv.loc[recent].mean(axis=0).to_numpy(float),
    )


def metric_row(
    period: str,
    method: str,
    load_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
) -> dict:
    le = load_pred - load_actual
    pe = pv_pred - pv_actual
    return {
        "period": period,
        "method": method,
        "load_abs_sum": float(np.abs(le).sum()),
        "load_sq_sum": float((le ** 2).sum()),
        "load_err_sum": float(le.sum()),
        "pv_abs_sum": float(np.abs(pe).sum()),
        "pv_sq_sum": float((pe ** 2).sum()),
        "pv_err_sum": float(pe.sum()),
        "n": int(len(le)),
    }


def aggregate_metrics(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    out = []
    for (period, method), g in raw.groupby(["period", "method"], sort=False):
        n = int(g["n"].sum())
        out.append({
            "period": period,
            "method": method,
            "load_mae_kw": g["load_abs_sum"].sum() / n,
            "load_rmse_kw": np.sqrt(g["load_sq_sum"].sum() / n),
            "load_bias_kw": g["load_err_sum"].sum() / n,
            "pv_mae_kw": g["pv_abs_sum"].sum() / n,
            "pv_rmse_kw": np.sqrt(g["pv_sq_sum"].sum() / n),
            "pv_bias_kw": g["pv_err_sum"].sum() / n,
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Walk-forward forecasting
# ----------------------------------------------------------------------

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_data()
    load = wide(df, "load_kw")
    pv = wide(df, "pv_actual_kw")
    features = build_load_features(load)
    ratios: dict[pd.Timestamp, float] = {}

    labels = (
        df[["time_index", "time_label"]]
        .drop_duplicates("time_index")
        .set_index("time_index")["time_label"]
        .to_dict()
    )

    rows: list[dict] = []
    metric_rows: list[dict] = []

    target_dates = [d for d in load.index if FORECAST_START <= d <= YEAR_END]
    for date in target_dates:
        # Both forecasts use only information available before target-day 0:00.
        load_pred, load_model = predict_load(load, features, date)
        pv_pred, pv_scale = predict_pv(pv, date, ratios)
        baseline_load, baseline_pv = baseline_predictions(load, pv, date)

        load_actual = load.loc[date].to_numpy(float)
        pv_actual = pv.loc[date].to_numpy(float)
        period = "january_warmup" if date < OPERATION_START else "feb_dec"

        metric_rows.append(metric_row(
            period, "proposed", load_pred, pv_pred, load_actual, pv_actual
        ))
        metric_rows.append(metric_row(
            period, "rolling_mean_baseline",
            baseline_load, baseline_pv, load_actual, pv_actual
        ))

        for t in range(N):
            rows.append({
                "date": date.date().isoformat(),
                "time_index": t,
                "time_label": labels[t],
                "actual_load_kw": float(load_actual[t]),
                "actual_pv_kw": float(pv_actual[t]),
                "forecast_load_kw": float(load_pred[t]),
                "forecast_pv_kw": float(pv_pred[t]),
                "baseline_load_kw": float(baseline_load[t]),
                "baseline_pv_kw": float(baseline_pv[t]),
                "load_residual_kw": float(load_actual[t] - load_pred[t]),
                "pv_residual_kw": float(pv_actual[t] - pv_pred[t]),
                "load_model": load_model,
                "pv_scale": pv_scale,
                "history_end_date": (date - pd.Timedelta(days=1)).date().isoformat(),
            })

        # Today's actual PV can affect forecasts only after today's forecast is finished.
        update_pv_ratio(pv, date, ratios)

    forecasts = pd.DataFrame(rows)
    metrics = aggregate_metrics(metric_rows)
    diagnostics = january_diagnostics(load, pv)

    forecasts.to_csv(OUT / "dynamic_forecasts.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "january_diagnostics.csv", index=False, encoding="utf-8-sig")

    feb = metrics.loc[metrics["period"] == "feb_dec"]
    proposed = feb.loc[feb["method"] == "proposed"].iloc[0]
    baseline = feb.loc[feb["method"] == "rolling_mean_baseline"].iloc[0]

    print("Q2 Step 1 complete")
    print("------------------")
    print(metrics.to_string(index=False))
    print()
    print(
        "Feb-Dec proposed vs rolling-mean baseline:"
        f"\n  Load MAE improvement = "
        f"{100 * (1 - proposed['load_mae_kw'] / baseline['load_mae_kw']):.2f}%"
        f"\n  PV MAE improvement   = "
        f"{100 * (1 - proposed['pv_mae_kw'] / baseline['pv_mae_kw']):.2f}%"
    )
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    run()
