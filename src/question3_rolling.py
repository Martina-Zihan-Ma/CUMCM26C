"""第三问的数据对齐、预测版本选择与滚动执行。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from question3_model import (
    DEFAULT_SOC_STEP_KWH,
    ETA_C,
    ETA_D,
    MAX_INTERVAL_ENERGY,
    SOC_MAX,
    SOC_MIN,
    StorageParameters,
    UP_ADJUSTMENT_MULTIPLIER,
    DOWN_CANCELLATION_PENALTY,
    optimize_dispatch,
    validate_planned_dispatch,
)


N_PER_DAY = 144
UPDATE_INDICES = (0, 36, 72, 108)
UPDATE_HOURS = (0, 6, 12, 18)
INITIAL_SOC_KWH = 6000.0
TERMINAL_SOC_KWH = None  # 24h滚动模式下不再强制日末回到6000
EMERGENCY_PRICE_MULTIPLIER = 5.0


SPECIFIED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


@dataclass
class DataBundle:
    actual: pd.DataFrame
    forecast_10min: pd.DataFrame
    forecast_hourly: pd.DataFrame
    standard_price: pd.DataFrame
    dynamic_price: pd.DataFrame
    model_base: pd.DataFrame
    q2_load_forecast: pd.DataFrame


def load_data_bundle(processed_dir: Path) -> DataBundle:
    actual = pd.read_csv(processed_dir / "attachment2_actual_long.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])
    fc10 = pd.read_csv(processed_dir / "attachment3_forecast_10min_long.csv", encoding="utf-8-sig", parse_dates=["issue_time", "target_time"])
    fch = pd.read_csv(processed_dir / "attachment3_forecast_hourly_long.csv", encoding="utf-8-sig", parse_dates=["date", "issue_time", "target_time"])
    standard = pd.read_csv(processed_dir / "attachment1_standard_day.csv", encoding="utf-8-sig")
    dynamic = pd.read_csv(processed_dir / "attachment4_price_long.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])
    base = pd.read_csv(processed_dir / "model_base_2025.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])

    # Reuse Question 2's finalized load forecast.  Q2 and Q3 share the same load
    # information set; Q3's new intraday information is the PV forecast update.
    q2_path = processed_dir.parents[1] / "outputs" / "question2" / "dynamic_forecasts.csv"
    if not q2_path.exists():
        raise FileNotFoundError(
            f"缺少第二问负荷预测文件：{q2_path}。请将 dynamic_forecasts.csv 放在 outputs/question2/ 下。"
        )
    q2_load = pd.read_csv(q2_path, encoding="utf-8-sig", parse_dates=["date", "history_end_date"])
    required_q2_cols = {"date", "time_index", "forecast_load_kw", "history_end_date"}
    missing = required_q2_cols.difference(q2_load.columns)
    if missing:
        raise KeyError(f"dynamic_forecasts.csv 缺少字段：{sorted(missing)}")
    q2_load = q2_load.sort_values(["date", "time_index"]).reset_index(drop=True)
    if q2_load.duplicated(["date", "time_index"]).any():
        raise ValueError("dynamic_forecasts.csv 存在重复 date/time_index")
    if not q2_load.groupby(q2_load.date.dt.normalize()).size().eq(N_PER_DAY).all():
        raise ValueError("dynamic_forecasts.csv 必须每天包含144个10分钟负荷预测")
    q2_load["forecast_load_kwh"] = q2_load["forecast_load_kw"].astype(float) / 6.0
    # time_index=0 corresponds to interval 00:00-00:10 and therefore target/end 00:10;
    # time_index=143 ends at next-day 00:00.
    q2_load["target_time"] = q2_load["date"].dt.normalize() + pd.to_timedelta(
        (q2_load["time_index"].astype(int) + 1) * 10, unit="min"
    )
    if q2_load.target_time.duplicated().any():
        raise ValueError("dynamic_forecasts.csv 映射后的 target_time 必须唯一")
    q2_load = q2_load.set_index("target_time", drop=False).sort_index()

    assert len(actual) == len(dynamic) == len(base) == 365 * N_PER_DAY
    assert len(standard) == N_PER_DAY and standard.time_index.tolist() == list(range(N_PER_DAY))
    assert actual.groupby(actual.date.dt.normalize()).size().eq(N_PER_DAY).all()
    assert dynamic.groupby(dynamic.date.dt.normalize()).size().eq(N_PER_DAY).all()
    assert not actual.datetime.duplicated().any() and not dynamic.datetime.duplicated().any()
    assert not fc10.duplicated(["issue_time", "target_time"]).any()
    assert not fch.duplicated(["issue_time", "target_time"]).any()
    assert fch.groupby("issue_time").size().eq(24).all()
    assert fc10.groupby("issue_time").size().eq(139).all()
    assert np.allclose(actual.load_kwh, actual.load_kw / 6)
    assert np.allclose(actual.pv_actual_kwh, actual.pv_actual_kw / 6)
    assert np.allclose(fc10.pv_forecast_kwh, fc10.pv_forecast_kw / 6)
    assert np.allclose(base.load_kwh, actual.load_kwh)
    assert np.allclose(base.pv_actual_kwh, actual.pv_actual_kwh)
    assert np.allclose(base.price_yuan_per_kwh, dynamic.price_yuan_per_kwh)
    return DataBundle(actual, fc10, fch, standard, dynamic, base, q2_load)


def day_actual(bundle: DataBundle, day: str | pd.Timestamp) -> pd.DataFrame:
    date = pd.Timestamp(day).normalize()
    result = bundle.actual.loc[bundle.actual.date.dt.normalize().eq(date)].sort_values("time_index").copy()
    assert len(result) == N_PER_DAY and result.time_index.tolist() == list(range(N_PER_DAY))
    result["period_start"] = date + pd.to_timedelta(result.time_index * 10, unit="min")
    result["period_end"] = result.period_start + pd.Timedelta(minutes=10)
    assert result.period_end.iloc[-1] == date + pd.Timedelta(days=1)
    return result.reset_index(drop=True)



def _historical_load_fallback_at(
    bundle: DataBundle,
    actual_index: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
) -> tuple[float, pd.Timestamp, str, str, int]:
    """Causal fallback used only when the Q2 forecast is not yet available.

    This mainly covers 2025-01-01 warm-up and the cross-midnight preview portion of a
    6:00/12:00/18:00 rolling window.  Those preview rows are not executed before the
    next forecast update, but they still must not use a next-day Q2 forecast that was
    created using the yet-unobserved remainder of the current day.
    """
    history_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
    history_times: list[pd.Timestamp] = []
    history_values: list[float] = []
    used_weights: list[float] = []
    for lag_week, weight in enumerate(history_weights, start=1):
        candidate = target - pd.Timedelta(days=7 * lag_week)
        if candidate < issue and candidate in actual_index.index:
            history_times.append(candidate)
            history_values.append(float(actual_index.loc[candidate, "load_kwh"]))
            used_weights.append(float(weight))

    if history_values:
        w = np.asarray(used_weights, dtype=float)
        w /= w.sum()
        value = float(np.dot(w, np.asarray(history_values, dtype=float)))
        source_time = max(history_times)
        source = f"causal_preview_fallback_same_weekday_{len(history_values)}w"
        source_times = ";".join(ts.strftime("%Y-%m-%d %H:%M:%S") for ts in history_times)
        return value, source_time, source, source_times, len(history_values)

    standard_idx = int(((target - pd.Timedelta(minutes=10)).hour * 60 + (target - pd.Timedelta(minutes=10)).minute) // 10) % N_PER_DAY
    standard_row = bundle.standard_price.sort_values("time_index").iloc[standard_idx]
    if "load_kwh" not in standard_row.index:
        raise KeyError("attachment1_standard_day.csv 缺少 load_kwh，无法进行年初 cold start")
    return (
        float(standard_row["load_kwh"]),
        issue - pd.Timedelta(seconds=1),
        "attachment1_standard_day_cold_start",
        "",
        0,
    )


def _load_forecast_at(
    bundle: DataBundle,
    actual_index: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
) -> tuple[float, pd.Timestamp, str, str, int]:
    """Return Q2's load forecast whenever it is causally available.

    `dynamic_forecasts.csv` contains one 144-slot day-ahead load forecast per date.
    For the current service day its `history_end_date` is the previous day, so the
    forecast is available at 0:00 and can be reused unchanged at 6:00/12:00/18:00.
    A next-day row whose history_end_date is the current day is *not* available during
    today's intraday update; such cross-midnight preview rows use a causal fallback.
    """
    issue = pd.Timestamp(issue)
    target = pd.Timestamp(target)
    q2 = bundle.q2_load_forecast
    if target in q2.index:
        row = q2.loc[target]
        history_end = pd.Timestamp(row["history_end_date"]).normalize()
        # Q2 forecast for service day D is valid during D only if it was built using
        # data no later than D-1.  Strictly require the history day to precede issue day.
        if history_end < issue.normalize():
            source_time = history_end + pd.Timedelta(hours=23, minutes=59, seconds=59)
            return (
                float(row["forecast_load_kwh"]),
                source_time,
                "question2_dynamic_forecasts",
                source_time.strftime("%Y-%m-%d %H:%M:%S"),
                1,
            )

    return _historical_load_fallback_at(bundle, actual_index, issue, target)

def _pv_forecast_at(
    bundle: DataBundle,
    actual_index: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
) -> float:
    """Reconstruct the causal PV forecast for one historical issue/target pair."""
    horizon = int((target - issue).total_seconds() // 60)
    if not (10 <= horizon <= 24 * 60):
        raise ValueError("PV forecast target must be within (issue, issue+24h]")

    simulation_start = pd.Timestamp("2025-01-01 00:00:00")
    if issue in actual_index.index:
        anchor_pv_kw = float(actual_index.loc[issue, "pv_actual_kw"])
    elif issue == simulation_start:
        anchor_pv_kw = 0.0
    else:
        raise KeyError(f"缺少历史发布时间锚点实际PV：{issue}")

    hourly = bundle.forecast_hourly.loc[bundle.forecast_hourly.issue_time.eq(issue)].set_index("target_time")
    if hourly.empty:
        raise KeyError(f"缺少历史小时预测版本：{issue}")
    first_hour_time = issue + pd.Timedelta(hours=1)
    first_hour_kw = float(hourly.loc[first_hour_time, "pv_forecast_kw"])
    if horizon < 60:
        pv_kw = anchor_pv_kw + (first_hour_kw - anchor_pv_kw) * horizon / 60.0
    else:
        version = bundle.forecast_10min.loc[bundle.forecast_10min.issue_time.eq(issue)].set_index("target_time")
        if target not in version.index:
            raise KeyError(f"历史预测版本缺少目标时刻：issue={issue}, target={target}")
        pv_kw = float(version.loc[target, "pv_forecast_kw"])
    return max(pv_kw, 0.0) / 6.0



def build_forecast_window(bundle: DataBundle, day: str | pd.Timestamp, start_index: int) -> pd.DataFrame:
    """构造从发布时间起未来整整24小时（144个10分钟段）的因果预测窗口。

    每次 0:00/6:00/12:00/18:00 都使用该 issue_time 已发布的未来24小时
    光伏预测。窗口可以跨过当天24:00；执行仍只取前6小时。
    """
    date = pd.Timestamp(day).normalize()
    if start_index not in UPDATE_INDICES:
        raise ValueError("start_index 必须对应 0:00、6:00、12:00 或 18:00")
    issue = date + pd.Timedelta(minutes=10 * start_index)
    actual_index = bundle.actual.set_index("datetime")

    # 附件2按10分钟区间记录实际值。2025-01-01 00:00 是全年仿真的
    # 唯一起点，此时不存在更早的“已观测实际PV”可作为首小时插值锚点。
    # 为避免读取 00:00 之后的实际值造成未来信息泄漏，对这个唯一 cold-start
    # 时刻使用夜间光伏物理值 0 kW 作为锚点；从其余发布时间开始，仍严格使用
    # issue_time 当时已经存在的实际PV观测。
    simulation_start = pd.Timestamp("2025-01-01 00:00:00")
    if issue in actual_index.index:
        anchor_pv_kw = float(actual_index.loc[issue, "pv_actual_kw"])
        anchor_source = "observed_at_issue"
    elif issue == simulation_start:
        anchor_pv_kw = 0.0
        anchor_source = "jan1_midnight_zero_cold_start"
    else:
        raise KeyError(f"缺少发布时间锚点的已发生实际值：{issue}")
    version = bundle.forecast_10min.loc[bundle.forecast_10min.issue_time.eq(issue)].set_index("target_time").sort_index()
    hourly = bundle.forecast_hourly.loc[bundle.forecast_hourly.issue_time.eq(issue)].set_index("target_time").sort_index()
    assert len(version) == 139 and len(hourly) == 24

    first_hour_time = issue + pd.Timedelta(hours=1)
    first_hour_kw = float(hourly.loc[first_hour_time, "pv_forecast_kw"])
    # 固定未来24小时：+10min, +20min, ..., +1440min
    targets = issue + pd.to_timedelta(np.arange(1, N_PER_DAY + 1) * 10, unit="min")
    rows: list[dict[str, object]] = []
    for target in targets:
        horizon = int((target - issue).total_seconds() // 60)
        if horizon < 60:
            pv_kw = anchor_pv_kw + (first_hour_kw - anchor_pv_kw) * horizon / 60.0
            source = "observed_anchor_linear_to_h1"
        else:
            if target not in version.index:
                raise KeyError(f"单一预测版本缺少目标时刻：issue={issue}, target={target}")
            pv_kw = float(version.loc[target, "pv_forecast_kw"])
            source = "attachment3_10min_interpolation"

        load_kwh, load_source_time, load_source, load_source_times, history_count = _load_forecast_at(
            bundle, actual_index, issue, target
        )

        rows.append({
            "issue_time": issue,
            "target_time": target,
            "forecast_horizon_minutes": horizon,
            "pv_forecast_kw": max(pv_kw, 0.0),
            "pv_forecast_kwh": max(pv_kw, 0.0) / 6.0,
            "pv_forecast_source": source,
            "pv_anchor_observed_at_issue_kw": anchor_pv_kw,
            "pv_anchor_observed_time": issue,
            "pv_anchor_source": anchor_source,
            "load_forecast_kwh": load_kwh,
            "load_forecast_source_time": load_source_time,
            "load_forecast_source_times": load_source_times,
            "load_forecast_history_count": history_count,
            "load_forecast_source": load_source,
            "information_cutoff": issue,
        })
    result = pd.DataFrame(rows)

    # No extra safety quantile is added. Q3 directly reuses the Q2 load forecast;
    # forecast uncertainty is handled by 6-hour PV updates, real-time battery recourse,
    # and 5x emergency purchase only after storage can no longer cover the realized shortage.
    result["optimization_load_kwh"] = result["load_forecast_kwh"]

    assert len(result) == N_PER_DAY
    assert result.issue_time.nunique() == 1 and result.issue_time.iloc[0] == issue
    assert (result.target_time > result.issue_time).all()
    assert (result.load_forecast_source_time < result.issue_time).all()
    assert result.load_forecast_history_count.between(0, 4).all()
    assert int(result.forecast_horizon_minutes.iloc[-1]) == 24 * 60
    return result


def _execute_plan_block(
    actual_day: pd.DataFrame,
    plan: pd.DataFrame,
    base_plan: pd.DataFrame,
    forecast_window: pd.DataFrame,
    start_index: int,
    stop_index: int,
    strategy: str,
    soc0_kwh: float,
) -> pd.DataFrame:
    """Execute the committed grid purchase against realized load/PV.

    The grid-purchase decision is the committed decision. Storage charge/discharge is
    physical recourse and therefore adapts to the realized net load in every 10-minute
    interval. In particular, the model NEVER buys 5x emergency electricity merely to
    preserve a previously planned battery-charging action.

    Execution priority for each interval:
      1) scheduled grid purchase + realized PV serve realized community load;
      2) if energy is still short, discharge storage as far as power/SOC allow;
      3) only the remaining load shortage is emergency purchase;
      4) if there is surplus, charge storage as far as power/SOC allow, then spill/curtail.

    SOC is propagated sequentially from ``soc0_kwh`` using the ACTUAL executed battery
    action. Hence forecast errors can move realized SOC away from the nominal DP path,
    and the next 6-hour rolling optimization starts from this realized SOC.
    """
    count = stop_index - start_index
    p = plan.iloc[:count].reset_index(drop=True)
    f = forecast_window.iloc[:count].reset_index(drop=True)
    a = actual_day.iloc[start_index:stop_index].reset_index(drop=True)
    base = base_plan.iloc[start_index:stop_index].reset_index(drop=True)

    scheduled_grid = p.scheduled_grid_kwh.to_numpy(float)
    actual_load = a.load_kwh.to_numpy(float)
    actual_pv = a.pv_actual_kwh.to_numpy(float)

    charge = np.zeros(count, dtype=float)
    discharge = np.zeros(count, dtype=float)
    emergency = np.zeros(count, dtype=float)
    curtailed_pv = np.zeros(count, dtype=float)
    overpurchase_spill = np.zeros(count, dtype=float)
    soc_start = np.zeros(count, dtype=float)
    soc_end = np.zeros(count, dtype=float)

    soc = float(soc0_kwh)
    if not (SOC_MIN - 1e-9 <= soc <= SOC_MAX + 1e-9):
        raise ValueError(f"执行块初始SOC越界: {soc}")

    for i in range(count):
        soc_start[i] = soc

        # Positive residual means the realized community load is not yet covered.
        # Negative residual means scheduled grid + realized PV leave a surplus.
        residual_load = actual_load[i] - scheduled_grid[i] - actual_pv[i]

        if residual_load > 1e-12:
            # Load-first rule: battery discharge is real-time physical recourse.
            max_discharge_soc = max(0.0, (soc - SOC_MIN) * ETA_D)
            discharge[i] = min(residual_load, MAX_INTERVAL_ENERGY, max_discharge_soc)
            remaining_shortage = residual_load - discharge[i]
            emergency[i] = max(remaining_shortage, 0.0)
        else:
            # Surplus is stored whenever physically possible; only the remaining energy
            # is curtailed/spilled. This also naturally reduces a nominal charge when
            # the realized surplus is smaller than forecast.
            surplus = -residual_load
            max_charge_soc = max(0.0, (SOC_MAX - soc) / ETA_C)
            charge[i] = min(surplus, MAX_INTERVAL_ENERGY, max_charge_soc)
            spill = max(surplus - charge[i], 0.0)

            # Keep the previous reporting convention: attribute spill to PV first and
            # classify any remainder as scheduled-grid overpurchase spill.
            curtailed_pv[i] = min(actual_pv[i], spill)
            overpurchase_spill[i] = max(spill - curtailed_pv[i], 0.0)

        soc = soc + ETA_C * charge[i] - discharge[i] / ETA_D
        # Numerical clipping only; feasibility above already enforces the bounds.
        if soc < SOC_MIN and soc > SOC_MIN - 1e-8:
            soc = SOC_MIN
        if soc > SOC_MAX and soc < SOC_MAX + 1e-8:
            soc = SOC_MAX
        if not (SOC_MIN - 1e-7 <= soc <= SOC_MAX + 1e-7):
            raise AssertionError(f"执行后SOC越界: {soc}")
        soc_end[i] = soc

    up = np.maximum(scheduled_grid - base.scheduled_grid_kwh.to_numpy(float), 0.0)
    down = np.maximum(base.scheduled_grid_kwh.to_numpy(float) - scheduled_grid, 0.0)
    price = base_plan.price_yuan_per_kwh.iloc[start_index:stop_index].to_numpy(float)
    base_cost = price * base.scheduled_grid_kwh.to_numpy(float)
    if strategy == "rolling":
        adjustment_cost = UP_ADJUSTMENT_MULTIPLIER * price * up - DOWN_CANCELLATION_PENALTY * price * down
    else:
        up[:] = 0.0
        down[:] = 0.0
        adjustment_cost = np.zeros(count)

    emergency_cost = EMERGENCY_PRICE_MULTIPLIER * price * emergency
    result = pd.DataFrame(
        {
            "date": a.date.dt.strftime("%Y-%m-%d"),
            "time_index": a.time_index.astype(int),
            "period_start": a.period_start,
            "period_end": a.period_end,
            "actual_load_kw": a.load_kw,
            "actual_load_kwh": a.load_kwh,
            "actual_pv_kw": a.pv_actual_kw,
            "actual_pv_kwh": a.pv_actual_kwh,
            "load_forecast_kwh": f.load_forecast_kwh,
            "optimization_load_kwh": f.optimization_load_kwh,
            "pv_forecast_used_kw": f.pv_forecast_kw,
            "pv_forecast_used_kwh": f.pv_forecast_kwh,
            "forecast_issue_time": f.issue_time,
            "forecast_horizon_minutes": f.forecast_horizon_minutes.astype(int),
            "pv_forecast_source": f.pv_forecast_source,
            "pv_anchor_observed_time": f.pv_anchor_observed_time,
            "load_forecast_source_time": f.load_forecast_source_time,
            "base_plan_grid_kwh": base.scheduled_grid_kwh,
            "scheduled_grid_kwh": scheduled_grid,
            # Explicit Q3 representation: q^(k) = q^(0) + a^+ - a^- .
            "adjustment_delta_kwh": up - down,
            "adjustment_changed": (up + down) > 1e-8,
            "up_adjustment_kwh": up,
            "down_adjustment_kwh": down,
            "emergency_purchase_kwh": emergency,
            "total_grid_purchase_kwh": scheduled_grid + emergency,
            "grid_purchase_kwh": scheduled_grid + emergency,
            # Executed battery actions and SOC (not the nominal DP actions).
            "charge_input_kwh": charge,
            "discharge_output_kwh": discharge,
            "curtailed_pv_kwh": curtailed_pv,
            "overpurchase_spill_kwh": overpurchase_spill,
            "soc_start_kwh": soc_start,
            "soc_end_kwh": soc_end,
            "charge_state_binary": (charge > 1e-8).astype(int),
            # Preserve nominal battery decisions for audit/comparison only.
            "planned_charge_input_kwh": p.charge_input_kwh.to_numpy(float),
            "planned_discharge_output_kwh": p.discharge_output_kwh.to_numpy(float),
            "planned_soc_start_kwh": p.soc_start_kwh.to_numpy(float),
            "planned_soc_end_kwh": p.soc_end_kwh.to_numpy(float),
            "price_yuan_per_kwh": price,
            "base_plan_cost_yuan": base_cost,
            "adjustment_cost_yuan": adjustment_cost,
            "emergency_cost_yuan": emergency_cost,
            "grid_cost_yuan": base_cost + adjustment_cost + emergency_cost,
            "strategy": strategy,
        }
    )
    return result



def _rolling_price_24h(bundle: DataBundle, start_index: int) -> np.ndarray:
    """将附件1标准日电价按当前时刻循环展开为未来24小时144点。"""
    day_price = bundle.standard_price.sort_values("time_index").price_yuan_per_kwh.to_numpy(float)
    return np.r_[day_price[start_index:], day_price[:start_index]]


def _base_plan_vector_24h(base_plan: pd.DataFrame, start_index: int) -> np.ndarray:
    """当前日剩余时段使用0:00原计划；跨过午夜的预览时段尚无次日原计划，用NaN标记。"""
    remaining = base_plan.scheduled_grid_kwh.iloc[start_index:].to_numpy(float)
    preview = np.full(start_index, np.nan, dtype=float)
    return np.r_[remaining, preview]

def simulate_day(
    bundle: DataBundle,
    day: str,
    strategy: str = "rolling",
    parameters: StorageParameters | None = None,
    soc0_kwh: float = INITIAL_SOC_KWH,
    terminal_soc_kwh: float | None = TERMINAL_SOC_KWH,
    update_indices: tuple[int, ...] = UPDATE_INDICES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按固定24小时预测窗口进行滚动优化。

    rolling: 每次优化未来144个10分钟段，但只执行到下一次更新（6小时）。
    static_forecast: 0:00优化未来24小时并执行当天144段，不做日内更新。

    SOC只在整个仿真最开始由调用方给定；窗口间由已执行块末端SOC连续传递。
    terminal_soc_kwh 默认 None，不再强制每个窗口/每天回到6000。
    """
    parameters = parameters or StorageParameters(soc_step_kwh=DEFAULT_SOC_STEP_KWH)
    if strategy not in {"rolling", "static_forecast"}:
        raise ValueError("strategy 必须为 rolling 或 static_forecast")
    actual = day_actual(bundle, day)

    f0 = build_forecast_window(bundle, day, 0)
    price0 = _rolling_price_24h(bundle, 0)
    base_plan = optimize_dispatch(
        f0.optimization_load_kwh, f0.pv_forecast_kwh, price0,
        soc0_kwh, terminal_soc_kwh, parameters=parameters,
    )
    base_plan.insert(0, "time_index", np.arange(N_PER_DAY))
    base_plan.insert(0, "target_time", f0.target_time.to_numpy())

    logs: list[pd.DataFrame] = []
    executed: list[pd.DataFrame] = []
    current_soc = float(soc0_kwh)
    starts: tuple[int, ...] = update_indices if strategy == "rolling" else (0,)
    if not starts or starts[0] != 0 or any(start not in UPDATE_INDICES for start in starts):
        raise ValueError("rolling 的 update_indices 必须为以0开始的题设更新时间子集")

    for position, start in enumerate(starts):
        forecast = build_forecast_window(bundle, day, start)
        price24 = _rolling_price_24h(bundle, start)
        stop = N_PER_DAY if strategy == "static_forecast" else (
            starts[position + 1] if position + 1 < len(starts) else N_PER_DAY
        )

        if start == 0:
            plan = base_plan.drop(columns=["target_time", "time_index"]).copy()
            base_vector = base_plan.scheduled_grid_kwh.to_numpy(float)
        else:
            base_vector = _base_plan_vector_24h(base_plan, start)
            plan = optimize_dispatch(
                forecast.optimization_load_kwh,
                forecast.pv_forecast_kwh,
                price24,
                current_soc,
                terminal_soc_kwh,
                base_plan_grid_kwh=base_vector,
                parameters=parameters,
            )

        if len(plan) != N_PER_DAY or len(forecast) != N_PER_DAY:
            raise AssertionError("每次滚动优化必须恰好覆盖未来144个10分钟时段")
        check = validate_planned_dispatch(plan, parameters)
        if not (
            check["planned_max_energy_balance_residual_kwh"] < 1e-6
            and check["planned_max_soc_transition_residual_kwh"] < 1e-6
            and check["planned_soc_bounds_ok"]
            and check["planned_power_bounds_ok"]
            and check["planned_simultaneous_charge_discharge_count"] == 0
        ):
            raise AssertionError(f"窗口计划验证失败：{check}")

        run_id = f"{day}T{start // 6:02d}:00_{strategy}"
        log = forecast.copy()
        log.insert(0, "optimization_run_id", run_id)
        log.insert(1, "update_time", forecast.issue_time)
        log.insert(2, "forecast_coverage_start", forecast.target_time.iloc[0])
        log.insert(3, "forecast_coverage_end", forecast.target_time.iloc[-1])
        log["execution_start_index"] = start
        log["execution_stop_index_exclusive"] = stop
        log["executed_in_this_update"] = np.arange(N_PER_DAY) < (stop - start)
        log["planned_grid_kwh"] = plan.scheduled_grid_kwh.to_numpy(float)
        log["planned_charge_kwh"] = plan.charge_input_kwh.to_numpy(float)
        log["planned_discharge_kwh"] = plan.discharge_output_kwh.to_numpy(float)
        log["planned_curtailment_kwh"] = plan.planned_curtailment_kwh.to_numpy(float)
        log["planned_soc_start_kwh"] = plan.soc_start_kwh.to_numpy(float)
        log["planned_soc_end_kwh"] = plan.soc_end_kwh.to_numpy(float)
        log["base_plan_grid_kwh"] = base_vector
        log["soc_grid_step_kwh"] = parameters.soc_step_kwh
        logs.append(log)

        block = _execute_plan_block(actual, plan, base_plan, forecast, start, stop, strategy, current_soc)
        if executed:
            previous_end = float(executed[-1].soc_end_kwh.iloc[-1])
            this_start = float(block.soc_start_kwh.iloc[0])
            if abs(previous_end - this_start) > 1e-7:
                raise AssertionError(f"滚动块SOC不连续：{previous_end} -> {this_start}")
        executed.append(block)
        current_soc = float(block.soc_end_kwh.iloc[-1])

    result = pd.concat(executed, ignore_index=True)
    versions = pd.concat(logs, ignore_index=True)
    assert len(result) == N_PER_DAY and result.time_index.tolist() == list(range(N_PER_DAY))
    result["cumulative_grid_cost_yuan"] = result.grid_cost_yuan.cumsum()
    result["cumulative_grid_kwh"] = result.total_grid_purchase_kwh.cumsum()
    result["cumulative_emergency_kwh"] = result.emergency_purchase_kwh.cumsum()
    result["cumulative_curtailed_pv_kwh"] = result.curtailed_pv_kwh.cumsum()
    return result, versions, base_plan


def simulate_static_from_base_plan(
    bundle: DataBundle,
    day: str,
    base_plan: pd.DataFrame,
) -> pd.DataFrame:
    """复用滚动仿真的0:00原计划，执行不更新预测的静态对照。"""
    actual = day_actual(bundle, day)
    forecast = build_forecast_window(bundle, day, 0)
    plan = base_plan.drop(columns=["target_time", "time_index"]).copy()
    result = _execute_plan_block(
        actual, plan, base_plan, forecast, 0, N_PER_DAY, "static_forecast",
        float(base_plan.soc_start_kwh.iloc[0]),
    )
    result["cumulative_grid_cost_yuan"] = result.grid_cost_yuan.cumsum()
    result["cumulative_grid_kwh"] = result.total_grid_purchase_kwh.cumsum()
    result["cumulative_emergency_kwh"] = result.emergency_purchase_kwh.cumsum()
    result["cumulative_curtailed_pv_kwh"] = result.curtailed_pv_kwh.cumsum()
    return result


def simulate_oracle_day(
    bundle: DataBundle,
    day: str,
    parameters: StorageParameters | None = None,
    soc0_kwh: float = INITIAL_SOC_KWH,
    terminal_soc_kwh: float | None = TERMINAL_SOC_KWH,
) -> pd.DataFrame:
    parameters = parameters or StorageParameters()
    actual = day_actual(bundle, day)
    price = bundle.standard_price.sort_values("time_index").price_yuan_per_kwh.to_numpy(float)
    plan = optimize_dispatch(actual.load_kwh, actual.pv_actual_kwh, price, soc0_kwh, terminal_soc_kwh, parameters=parameters)
    result = pd.DataFrame(
        {
            "date": pd.Timestamp(day).strftime("%Y-%m-%d"), "time_index": actual.time_index.astype(int),
            "period_start": actual.period_start, "period_end": actual.period_end,
            "actual_load_kw": actual.load_kw, "actual_load_kwh": actual.load_kwh,
            "actual_pv_kw": actual.pv_actual_kw, "actual_pv_kwh": actual.pv_actual_kwh,
            "load_forecast_kwh": actual.load_kwh, "pv_forecast_used_kw": actual.pv_actual_kw,
            "pv_forecast_used_kwh": actual.pv_actual_kwh, "forecast_issue_time": pd.NaT,
            "forecast_horizon_minutes": -1, "pv_forecast_source": "oracle_actual_not_executable",
            "load_forecast_source_time": pd.NaT, "base_plan_grid_kwh": plan.scheduled_grid_kwh,
            "scheduled_grid_kwh": plan.scheduled_grid_kwh, "adjustment_delta_kwh": 0.0,
            "adjustment_changed": False, "up_adjustment_kwh": 0.0,
            "down_adjustment_kwh": 0.0, "emergency_purchase_kwh": 0.0,
            "total_grid_purchase_kwh": plan.scheduled_grid_kwh, "grid_purchase_kwh": plan.scheduled_grid_kwh,
            "charge_input_kwh": plan.charge_input_kwh, "discharge_output_kwh": plan.discharge_output_kwh,
            "curtailed_pv_kwh": plan.planned_curtailment_kwh, "overpurchase_spill_kwh": 0.0,
            "soc_start_kwh": plan.soc_start_kwh, "soc_end_kwh": plan.soc_end_kwh,
            "charge_state_binary": plan.charge_state_binary.astype(int), "price_yuan_per_kwh": price,
            "base_plan_cost_yuan": price * plan.scheduled_grid_kwh, "adjustment_cost_yuan": 0.0,
            "emergency_cost_yuan": 0.0, "grid_cost_yuan": price * plan.scheduled_grid_kwh,
            "strategy": "oracle_actual",
        }
    )
    result["cumulative_grid_cost_yuan"] = result.grid_cost_yuan.cumsum()
    result["cumulative_grid_kwh"] = result.total_grid_purchase_kwh.cumsum()
    result["cumulative_emergency_kwh"] = 0.0
    result["cumulative_curtailed_pv_kwh"] = result.curtailed_pv_kwh.cumsum()
    return result


def summarize_schedule(schedule: pd.DataFrame) -> dict[str, object]:
    pv = float(schedule.actual_pv_kwh.sum())
    curtailed = float(schedule.curtailed_pv_kwh.sum())
    return {
        "date": str(schedule.date.iloc[0]), "strategy": str(schedule.strategy.iloc[0]),
        "scheduled_grid_kwh": float(schedule.scheduled_grid_kwh.sum()),
        "emergency_purchase_kwh": float(schedule.emergency_purchase_kwh.sum()),
        "total_grid_purchase_kwh": float(schedule.total_grid_purchase_kwh.sum()),
        "base_plan_cost_yuan": float(schedule.base_plan_cost_yuan.sum()),
        "adjustment_cost_yuan": float(schedule.adjustment_cost_yuan.sum()),
        "emergency_cost_yuan": float(schedule.emergency_cost_yuan.sum()),
        "total_grid_cost_yuan": float(schedule.grid_cost_yuan.sum()),
        "curtailed_pv_kwh": curtailed, "overpurchase_spill_kwh": float(schedule.overpurchase_spill_kwh.sum()),
        "pv_consumption_rate": 1.0 - curtailed / pv if pv > 0 else 1.0,
        "charge_input_kwh": float(schedule.charge_input_kwh.sum()),
        "discharge_output_kwh": float(schedule.discharge_output_kwh.sum()),
        "soc_initial_kwh": float(schedule.soc_start_kwh.iloc[0]),
        "soc_terminal_kwh": float(schedule.soc_end_kwh.iloc[-1]),
        "soc_min_kwh": float(min(schedule.soc_start_kwh.min(), schedule.soc_end_kwh.min())),
        "soc_max_kwh": float(max(schedule.soc_start_kwh.max(), schedule.soc_end_kwh.max())),
    }


def validate_executed_schedule(schedule: pd.DataFrame) -> dict[str, object]:
    balance = schedule.scheduled_grid_kwh + schedule.emergency_purchase_kwh + schedule.actual_pv_kwh + schedule.discharge_output_kwh - schedule.actual_load_kwh - schedule.charge_input_kwh - schedule.curtailed_pv_kwh - schedule.overpurchase_spill_kwh
    transition = schedule.soc_end_kwh - schedule.soc_start_kwh - ETA_C * schedule.charge_input_kwh + schedule.discharge_output_kwh / ETA_D
    expected_cost = schedule.base_plan_cost_yuan + schedule.adjustment_cost_yuan + schedule.emergency_cost_yuan
    issue_time = pd.to_datetime(schedule.forecast_issue_time)
    source_time = pd.to_datetime(schedule.load_forecast_source_time)
    pv_anchor_time = pd.to_datetime(schedule.pv_anchor_observed_time) if "pv_anchor_observed_time" in schedule else pd.Series(pd.NaT, index=schedule.index)
    return {
        "row_count_144_ok": bool(len(schedule) == N_PER_DAY),
        "time_index_complete_ok": bool(schedule.time_index.tolist() == list(range(N_PER_DAY))),
        "max_energy_balance_residual_kwh": float(balance.abs().max()),
        "max_soc_transition_residual_kwh": float(transition.abs().max()),
        "soc_bounds_ok": bool(schedule.soc_start_kwh.between(SOC_MIN - 1e-7, SOC_MAX + 1e-7).all() and schedule.soc_end_kwh.between(SOC_MIN - 1e-7, SOC_MAX + 1e-7).all()),
        "power_bounds_ok": bool((schedule[["charge_input_kwh", "discharge_output_kwh"]] <= MAX_INTERVAL_ENERGY + 1e-7).all().all()),
        "simultaneous_charge_discharge_count": int(((schedule.charge_input_kwh > 1e-8) & (schedule.discharge_output_kwh > 1e-8)).sum()),
        # If 5x emergency power is used, storage must already be discharging at the
        # maximum physically available amount for that interval.
        "emergency_only_after_max_discharge_ok": bool((
            (schedule.emergency_purchase_kwh <= 1e-8)
            | (
                (schedule.charge_input_kwh <= 1e-8)
                & (np.abs(
                    schedule.discharge_output_kwh.to_numpy(float)
                    - np.minimum(
                        MAX_INTERVAL_ENERGY,
                        np.maximum(0.0, (schedule.soc_start_kwh.to_numpy(float) - SOC_MIN) * ETA_D),
                    )
                ) <= 1e-6)
            )
        ).all()),
        "adjustment_up_down_simultaneous_count": int(((schedule.up_adjustment_kwh > 1e-8) & (schedule.down_adjustment_kwh > 1e-8)).sum()),
        "adjustment_identity_max_residual_kwh": float((
            schedule.scheduled_grid_kwh - schedule.base_plan_grid_kwh
            - schedule.up_adjustment_kwh + schedule.down_adjustment_kwh
        ).abs().max()),
        "grid_nonnegative_ok": bool((schedule[["scheduled_grid_kwh", "emergency_purchase_kwh", "total_grid_purchase_kwh"]] >= -1e-9).all().all()),
        "curtailment_nonnegative_ok": bool((schedule[["curtailed_pv_kwh", "overpurchase_spill_kwh"]] >= -1e-9).all().all()),
        "initial_soc_ok": bool(SOC_MIN - 1e-7 <= schedule.soc_start_kwh.iloc[0] <= SOC_MAX + 1e-7),
        "terminal_soc_ok": bool(SOC_MIN - 1e-7 <= schedule.soc_end_kwh.iloc[-1] <= SOC_MAX + 1e-7),
        "soc_continuity_max_residual_kwh": float(np.max(np.r_[0.0, np.abs(schedule.soc_start_kwh.iloc[1:].to_numpy() - schedule.soc_end_kwh.iloc[:-1].to_numpy())])),
        "forecast_issue_hours_ok": bool(issue_time.dt.hour.isin(UPDATE_HOURS).all()),
        "forecast_issue_not_after_execution_ok": bool((issue_time <= pd.to_datetime(schedule.period_start)).all()),
        "load_source_precedes_issue_ok": bool((source_time < issue_time).all()),
        # Under the original dataset alignment convention, the PV anchor stamped at the
        # issue time is treated as information available at that issue; never allow a later anchor.
        "pv_anchor_not_after_issue_ok": bool((pv_anchor_time <= issue_time).all()),
        "cost_sum_residual_yuan": float(abs(schedule.grid_cost_yuan.sum() - expected_cost.sum())),
        "cumulative_cost_residual_yuan": float(abs(schedule.cumulative_grid_cost_yuan.iloc[-1] - schedule.grid_cost_yuan.sum())),
        "executed_decisions_unique_ok": bool(schedule.time_index.is_unique),
    }
