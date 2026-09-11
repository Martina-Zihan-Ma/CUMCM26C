#!/usr/bin/env python3
"""
Q2 Step 1 — causal Load/PV forecasting + forecast baselines.

Outputs
-------
outputs/question2/dynamic_forecasts.csv
    Proposed forecasts plus simple leakage-free baseline forecasts.
outputs/question2/forecast_metrics.csv
    MAE / RMSE / bias for all forecast methods.
outputs/question2/january_diagnostics.csv
    January-only evidence used to freeze the Load structure.

Operational period: 2025-02-01 to 2025-12-31.
For target day d, every forecast uses only observations before d.
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

START = pd.Timestamp("2025-02-01")
JAN_START = pd.Timestamp("2025-01-01")
JAN_END = pd.Timestamp("2025-01-31")
N = 144

# Proposed Load model.
TRAIN_DAYS = 21
RIDGE_ALPHA = 1.0
REGIME_GAP = 0.15

# Proposed PV model.
PV_DAYS = 7
PV_DECAY = 0.82
PV_RATIO_DAYS = 3
PV_RATIO_STRENGTH = 0.50

# Simple history-average baseline used again in Step 2.
BASELINE_LOAD_WEEKDAYS = 4
BASELINE_PV_DAYS = 7


def read_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    required = {"date", "time_label", "time_index", "load_kw", "pv_actual_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["time_index"] = df["time_index"].astype(int)
    df = df.sort_values(["date", "time_index"]).reset_index(drop=True)

    counts = df.groupby("date").size()
    if not (counts == N).all():
        raise ValueError(f"Each day must have {N} rows.")

    for d, part in df.groupby("date", sort=False):
        if part["time_index"].tolist() != list(range(N)):
            raise ValueError(f"{d.date()}: time_index must be 0..{N - 1}.")

    if df["date"].min() > JAN_START or df["date"].max() < JAN_END:
        raise ValueError("January 1-31 data are required.")

    return df


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="time_index", values=col).sort_index()


# ----------------------------------------------------------------------
# January-only structure identification
# ----------------------------------------------------------------------

def mean_profile_corr(x: pd.DataFrame, lag: int) -> float:
    values = []
    for i in range(lag, len(x)):
        a = x.iloc[i].to_numpy(float)
        b = x.iloc[i - lag].to_numpy(float)
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else np.nan


def identify_low_load_weekdays(
    df: pd.DataFrame,
) -> tuple[set[int], pd.DataFrame]:
    jan = df.loc[df["date"].between(JAN_START, JAN_END)]
    daily = jan.groupby("date", as_index=False).agg(load_total=("load_kw", "sum"))
    daily["weekday_num"] = daily["date"].dt.weekday
    daily["weekday"] = daily["date"].dt.day_name()

    stats = (
        daily.groupby(["weekday_num", "weekday"], as_index=False)
        .agg(mean_load=("load_total", "mean"), std_load=("load_total", "std"))
        .sort_values("mean_load")
        .reset_index(drop=True)
    )

    means = stats["mean_load"].to_numpy(float)
    gaps = (means[1:] - means[:-1]) / np.maximum(means[:-1], 1e-12)
    split = int(np.argmax(gaps))
    low = (
        set(stats.iloc[: split + 1]["weekday_num"].astype(int))
        if float(gaps[split]) >= REGIME_GAP
        else set()
    )
    stats["low_load_regime"] = stats["weekday_num"].isin(low)
    stats["largest_relative_gap"] = float(gaps[split])
    return low, stats


def january_diagnostics(
    df: pd.DataFrame,
    low_weekdays: set[int],
    weekday_stats: pd.DataFrame,
) -> pd.DataFrame:
    jan = df.loc[df["date"].between(JAN_START, JAN_END)]
    load = wide(jan, "load_kw")
    pv = wide(jan, "pv_actual_kw")

    rows = []
    for lag in [1, 7, 14, 21]:
        rows.append({
            "section": "profile_correlation",
            "metric": f"load_lag_{lag}",
            "value": mean_profile_corr(load, lag),
            "detail": "January only",
        })
    for lag in [1, 7]:
        rows.append({
            "section": "profile_correlation",
            "metric": f"pv_lag_{lag}",
            "value": mean_profile_corr(pv, lag),
            "detail": "January only",
        })

    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows.append({
        "section": "regime",
        "metric": "low_load_weekdays",
        "value": np.nan,
        "detail": ",".join(names[w] for w in sorted(low_weekdays)) or "none",
    })

    for _, r in weekday_stats.sort_values("weekday_num").iterrows():
        rows.append({
            "section": "weekday_mean_load",
            "metric": r["weekday"],
            "value": float(r["mean_load"]),
            "detail": "low" if bool(r["low_load_regime"]) else "regular",
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Proposed Load model
# ----------------------------------------------------------------------

LOAD_FEATURES = [
    "lag1", "lag7", "lag14", "lag21",
    "mean7", "same_weekday", "regime_mean", "low_load_day",
    "doy_sin", "doy_cos",
    "tod_sin1", "tod_cos1", "tod_sin2", "tod_cos2",
]


def build_load_features(
    load: pd.DataFrame,
    low_weekdays: set[int],
) -> pd.DataFrame:
    dates = load.index
    cols = load.columns

    same_weekday = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for wd in range(7):
        dts = [d for d in dates if d.weekday() == wd]
        x = load.loc[dts]
        same_weekday.loc[dts] = (
            x.shift(1).rolling(4, min_periods=1).mean().to_numpy()
        )

    regime_mean = pd.DataFrame(index=dates, columns=cols, dtype=float)
    if low_weekdays:
        for low_flag in [False, True]:
            dts = [
                d for d in dates
                if ((d.weekday() in low_weekdays) == low_flag)
            ]
            x = load.loc[dts]
            regime_mean.loc[dts] = (
                x.shift(1).rolling(8, min_periods=1).mean().to_numpy()
            )
    else:
        regime_mean.loc[:, :] = (
            load.shift(1).rolling(8, min_periods=1).mean().to_numpy()
        )

    feature_wide = {
        "load_kw": load,
        "lag1": load.shift(1),
        "lag7": load.shift(7),
        "lag14": load.shift(14),
        "lag21": load.shift(21),
        "mean7": load.shift(1).rolling(7, min_periods=1).mean(),
        "same_weekday": same_weekday,
        "regime_mean": regime_mean,
    }

    pieces = [x.stack(dropna=False).rename(name)
    for name, x in feature_wide.items()]
    table = pd.concat(pieces, axis=1).reset_index()
    table.columns = ["date", "time_index"] + list(feature_wide)

    doy = table["date"].dt.dayofyear.to_numpy(float)
    slot = table["time_index"].to_numpy(float)
    year_angle = 2 * np.pi * doy / 365.0
    day_angle = 2 * np.pi * slot / N

    table["low_load_day"] = (
        table["date"].dt.weekday.isin(low_weekdays).astype(float)
    )
    table["doy_sin"] = np.sin(year_angle)
    table["doy_cos"] = np.cos(year_angle)
    table["tod_sin1"] = np.sin(day_angle)
    table["tod_cos1"] = np.cos(day_angle)
    table["tod_sin2"] = np.sin(2 * day_angle)
    table["tod_cos2"] = np.cos(2 * day_angle)
    return table


def predict_load(
    features: pd.DataFrame,
    date: pd.Timestamp,
) -> np.ndarray:
    train = features.loc[
        (features["date"] < date)
        & (features["date"] >= date - pd.Timedelta(days=TRAIN_DAYS))
    ].dropna(subset=LOAD_FEATURES + ["load_kw"])

    test = (
        features.loc[features["date"] == date]
        .sort_values("time_index")
        .copy()
    )

    if len(test) != N or train.empty:
        raise ValueError(f"Invalid Load training/test data for {date.date()}.")
    if test[LOAD_FEATURES].isna().any().any():
        raise ValueError(f"Missing Load features for {date.date()}.")

    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[LOAD_FEATURES], train["load_kw"])
    return np.maximum(model.predict(test[LOAD_FEATURES]), 0.0)


# ----------------------------------------------------------------------
# Proposed PV model
# ----------------------------------------------------------------------

def weighted_pv_base(pv: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    prior = [d for d in pv.index if d < date][-PV_DAYS:]
    if len(prior) < PV_DAYS:
        raise ValueError(f"Insufficient PV history before {date.date()}.")

    prior = list(reversed(prior))  # newest first
    weights = np.array([PV_DECAY ** i for i in range(len(prior))], dtype=float)
    weights /= weights.sum()
    profiles = np.vstack([pv.loc[d].to_numpy(float) for d in prior])
    return np.average(profiles, axis=0, weights=weights)


def initial_pv_ratios(
    pv: pd.DataFrame,
    before: pd.Timestamp,
) -> dict[pd.Timestamp, float]:
    ratios = {}
    dates = list(pv.index)
    for i, d in enumerate(dates):
        if d >= before:
            break
        if i < PV_DAYS:
            continue
        base = weighted_pv_base(pv, d)
        if base.sum() > 1e-9:
            ratios[d] = float(pv.loc[d].sum() / base.sum())
    return ratios


def predict_pv(
    pv: pd.DataFrame,
    date: pd.Timestamp,
    ratios: dict[pd.Timestamp, float],
) -> tuple[np.ndarray, float]:
    base = weighted_pv_base(pv, date)
    recent = [d for d in sorted(ratios) if d < date][-PV_RATIO_DAYS:]
    recent_ratio = float(np.mean([ratios[d] for d in recent])) if recent else 1.0
    scale = 1.0 + PV_RATIO_STRENGTH * (recent_ratio - 1.0)
    return np.maximum(scale * base, 0.0), float(scale)


def update_pv_ratio(
    pv: pd.DataFrame,
    date: pd.Timestamp,
    ratios: dict[pd.Timestamp, float],
) -> None:
    base = weighted_pv_base(pv, date)
    if base.sum() > 1e-9:
        ratios[date] = float(pv.loc[date].sum() / base.sum())


# ----------------------------------------------------------------------
# Leakage-free baselines
# ----------------------------------------------------------------------

def baseline_predictions(
    load: pd.DataFrame,
    pv: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    history = [d for d in load.index if d < date]
    if len(history) < 7:
        raise ValueError(f"Insufficient history before {date.date()}.")

    prev = history[-1]
    last7 = history[-7:]
    same_weekdays = [d for d in history if d.weekday() == date.weekday()][
        -BASELINE_LOAD_WEEKDAYS:
    ]

    previous_day = (
        load.loc[prev].to_numpy(float),
        pv.loc[prev].to_numpy(float),
    )
    rolling_7day = (
        load.loc[last7].mean(axis=0).to_numpy(float),
        pv.loc[last7].mean(axis=0).to_numpy(float),
    )
    historical_mean = (
        load.loc[same_weekdays].mean(axis=0).to_numpy(float),
        pv.loc[last7[-BASELINE_PV_DAYS:]].mean(axis=0).to_numpy(float),
    )

    return {
        "previous_day": previous_day,
        "rolling_7day": rolling_7day,
        "historical_mean": historical_mean,
    }


def metric_row(
    method: str,
    load_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_actual: np.ndarray,
    pv_actual: np.ndarray,
) -> dict:
    load_err = load_pred - load_actual
    pv_err = pv_pred - pv_actual
    return {
        "method": method,
        "load_abs_sum": float(np.abs(load_err).sum()),
        "load_sq_sum": float((load_err ** 2).sum()),
        "load_err_sum": float(load_err.sum()),
        "pv_abs_sum": float(np.abs(pv_err).sum()),
        "pv_sq_sum": float((pv_err ** 2).sum()),
        "pv_err_sum": float(pv_err.sum()),
        "n": int(len(load_err)),
    }


def aggregate_metrics(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    out = []
    for method, g in raw.groupby("method", sort=False):
        n = int(g["n"].sum())
        out.append({
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
# Walk-forward
# ----------------------------------------------------------------------

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_data()
    load = wide(df, "load_kw")
    pv = wide(df, "pv_actual_kw")

    low_weekdays, weekday_stats = identify_low_load_weekdays(df)
    diagnostics = january_diagnostics(df, low_weekdays, weekday_stats)
    features = build_load_features(load, low_weekdays)
    ratios = initial_pv_ratios(pv, START)

    labels = (
        df[["time_index", "time_label"]]
        .drop_duplicates("time_index")
        .set_index("time_index")["time_label"]
        .to_dict()
    )

    rows = []
    metric_rows = []

    for date in [d for d in load.index if d >= START]:
        # All predictions are completed before target-day actuals are used.
        load_pred = predict_load(features, date)
        pv_pred, pv_scale = predict_pv(pv, date, ratios)
        baselines = baseline_predictions(load, pv, date)

        load_actual = load.loc[date].to_numpy(float)
        pv_actual = pv.loc[date].to_numpy(float)

        metric_rows.append(
            metric_row("proposed", load_pred, pv_pred, load_actual, pv_actual)
        )
        for method, (b_load, b_pv) in baselines.items():
            metric_rows.append(
                metric_row(method, b_load, b_pv, load_actual, pv_actual)
            )

        hist_load, hist_pv = baselines["historical_mean"]

        for t in range(N):
            rows.append({
                "date": date.date().isoformat(),
                "time_index": t,
                "time_label": labels[t],
                "actual_load_kw": float(load_actual[t]),
                "actual_pv_kw": float(pv_actual[t]),
                "forecast_load_kw": float(load_pred[t]),
                "forecast_pv_kw": float(pv_pred[t]),
                "baseline_load_kw": float(hist_load[t]),
                "baseline_pv_kw": float(hist_pv[t]),
                "load_residual_kw": float(load_actual[t] - load_pred[t]),
                "pv_residual_kw": float(pv_actual[t] - pv_pred[t]),
                "pv_scale": pv_scale,
                "history_end_date": (date - pd.Timedelta(days=1)).date().isoformat(),
            })

        # Only now may date's observed PV affect future forecasts.
        update_pv_ratio(pv, date, ratios)

    forecasts = pd.DataFrame(rows)
    metrics = aggregate_metrics(metric_rows)

    forecasts.to_csv(OUT / "dynamic_forecasts.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(
        OUT / "january_diagnostics.csv", index=False, encoding="utf-8-sig"
    )

    proposed = metrics.loc[metrics["method"] == "proposed"].iloc[0]
    baseline = metrics.loc[metrics["method"] == "historical_mean"].iloc[0]

    print("Q2 Step 1 complete")
    print("------------------")
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(
        "January low-load weekdays:",
        ", ".join(weekday_names[w] for w in sorted(low_weekdays)) or "none",
    )
    print()
    print(metrics.to_string(index=False))
    print()
    print(
        "Proposed vs historical-mean baseline:"
        f"\n  Load MAE improvement = "
        f"{100 * (1 - proposed['load_mae_kw'] / baseline['load_mae_kw']):.2f}%"
        f"\n  PV MAE improvement   = "
        f"{100 * (1 - proposed['pv_mae_kw'] / baseline['pv_mae_kw']):.2f}%"
    )
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    run()
