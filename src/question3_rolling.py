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
    optimize_dispatch,
    validate_planned_dispatch,
)


N_PER_DAY = 144
UPDATE_INDICES = (0, 36, 72, 108)
UPDATE_HOURS = (0, 6, 12, 18)
INITIAL_SOC_KWH = 6000.0
TERMINAL_SOC_KWH = 6000.0
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


def load_data_bundle(processed_dir: Path) -> DataBundle:
    actual = pd.read_csv(processed_dir / "attachment2_actual_long.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])
    fc10 = pd.read_csv(processed_dir / "attachment3_forecast_10min_long.csv", encoding="utf-8-sig", parse_dates=["issue_time", "target_time"])
    fch = pd.read_csv(processed_dir / "attachment3_forecast_hourly_long.csv", encoding="utf-8-sig", parse_dates=["date", "issue_time", "target_time"])
    standard = pd.read_csv(processed_dir / "attachment1_standard_day.csv", encoding="utf-8-sig")
    dynamic = pd.read_csv(processed_dir / "attachment4_price_long.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])
    base = pd.read_csv(processed_dir / "model_base_2025.csv", encoding="utf-8-sig", parse_dates=["date", "datetime"])
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
    return DataBundle(actual, fc10, fch, standard, dynamic, base)


def day_actual(bundle: DataBundle, day: str | pd.Timestamp) -> pd.DataFrame:
    date = pd.Timestamp(day).normalize()
    result = bundle.actual.loc[bundle.actual.date.dt.normalize().eq(date)].sort_values("time_index").copy()
    assert len(result) == N_PER_DAY and result.time_index.tolist() == list(range(N_PER_DAY))
    result["period_start"] = date + pd.to_timedelta(result.time_index * 10, unit="min")
    result["period_end"] = result.period_start + pd.Timedelta(minutes=10)
    assert result.period_end.iloc[-1] == date + pd.Timedelta(days=1)
    return result.reset_index(drop=True)


def build_forecast_window(bundle: DataBundle, day: str | pd.Timestamp, start_index: int) -> pd.DataFrame:
    """构造从发布时间到当天24:00的单一、因果预测版本。"""
    date = pd.Timestamp(day).normalize()
    if start_index not in UPDATE_INDICES:
        raise ValueError("start_index 必须对应 0:00、6:00、12:00 或 18:00")
    issue = date + pd.Timedelta(minutes=10 * start_index)
    actual_index = bundle.actual.set_index("datetime")
    if issue not in actual_index.index:
        raise KeyError(f"缺少发布时间锚点的已发生实际值：{issue}")
    anchor_pv_kw = float(actual_index.loc[issue, "pv_actual_kw"])
    version = bundle.forecast_10min.loc[bundle.forecast_10min.issue_time.eq(issue)].set_index("target_time").sort_index()
    hourly = bundle.forecast_hourly.loc[bundle.forecast_hourly.issue_time.eq(issue)].set_index("target_time").sort_index()
    assert len(version) == 139 and len(hourly) == 24
    first_hour_time = issue + pd.Timedelta(hours=1)
    first_hour_kw = float(hourly.loc[first_hour_time, "pv_forecast_kw"])
    targets = date + pd.to_timedelta(np.arange(start_index + 1, N_PER_DAY + 1) * 10, unit="min")
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
        load_source_time = target - pd.Timedelta(days=7)
        if load_source_time >= issue or load_source_time not in actual_index.index:
            raise AssertionError("负荷预测历史源不满足信息边界")
        load_kwh = float(actual_index.loc[load_source_time, "load_kwh"])
        rows.append(
            {
                "issue_time": issue,
                "target_time": target,
                "forecast_horizon_minutes": horizon,
                "pv_forecast_kw": max(pv_kw, 0.0),
                "pv_forecast_kwh": max(pv_kw, 0.0) / 6.0,
                "pv_forecast_source": source,
                "pv_anchor_observed_at_issue_kw": anchor_pv_kw,
                "pv_anchor_observed_time": issue,
                "load_forecast_kwh": load_kwh,
                "load_forecast_source_time": load_source_time,
                "load_forecast_source": "actual_same_interval_7_days_ago",
                "information_cutoff": issue,
            }
        )
    result = pd.DataFrame(rows)
    assert len(result) == N_PER_DAY - start_index
    assert result.issue_time.nunique() == 1 and result.issue_time.iloc[0] == issue
    assert (result.target_time > result.issue_time).all()
    assert (result.load_forecast_source_time < result.issue_time).all()
    return result


def _execute_plan_block(
    actual_day: pd.DataFrame,
    plan: pd.DataFrame,
    base_plan: pd.DataFrame,
    forecast_window: pd.DataFrame,
    start_index: int,
    stop_index: int,
    strategy: str,
) -> pd.DataFrame:
    count = stop_index - start_index
    p = plan.iloc[:count].reset_index(drop=True)
    f = forecast_window.iloc[:count].reset_index(drop=True)
    a = actual_day.iloc[start_index:stop_index].reset_index(drop=True)
    base = base_plan.iloc[start_index:stop_index].reset_index(drop=True)
    shortage = a.load_kwh + p.charge_input_kwh - p.scheduled_grid_kwh - a.pv_actual_kwh - p.discharge_output_kwh
    emergency = shortage.clip(lower=0.0)
    surplus = (-shortage).clip(lower=0.0)
    curtailed_pv = np.minimum(a.pv_actual_kwh.to_numpy(float), surplus.to_numpy(float))
    overpurchase_spill = surplus.to_numpy(float) - curtailed_pv
    up = np.maximum(p.scheduled_grid_kwh.to_numpy(float) - base.scheduled_grid_kwh.to_numpy(float), 0.0)
    down = np.maximum(base.scheduled_grid_kwh.to_numpy(float) - p.scheduled_grid_kwh.to_numpy(float), 0.0)
    price = base_plan.price_yuan_per_kwh.iloc[start_index:stop_index].to_numpy(float)
    base_cost = price * base.scheduled_grid_kwh.to_numpy(float)
    if strategy == "rolling":
        adjustment_cost = 1.5 * price * up - 0.5 * price * down
    else:
        up[:] = 0.0
        down[:] = 0.0
        adjustment_cost = np.zeros(count)
    emergency_cost = EMERGENCY_PRICE_MULTIPLIER * price * emergency.to_numpy(float)
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
            "pv_forecast_used_kw": f.pv_forecast_kw,
            "pv_forecast_used_kwh": f.pv_forecast_kwh,
            "forecast_issue_time": f.issue_time,
            "forecast_horizon_minutes": f.forecast_horizon_minutes.astype(int),
            "pv_forecast_source": f.pv_forecast_source,
            "load_forecast_source_time": f.load_forecast_source_time,
            "base_plan_grid_kwh": base.scheduled_grid_kwh,
            "scheduled_grid_kwh": p.scheduled_grid_kwh,
            "up_adjustment_kwh": up,
            "down_adjustment_kwh": down,
            "emergency_purchase_kwh": emergency,
            "total_grid_purchase_kwh": p.scheduled_grid_kwh.to_numpy(float) + emergency.to_numpy(float),
            "grid_purchase_kwh": p.scheduled_grid_kwh.to_numpy(float) + emergency.to_numpy(float),
            "charge_input_kwh": p.charge_input_kwh,
            "discharge_output_kwh": p.discharge_output_kwh,
            "curtailed_pv_kwh": curtailed_pv,
            "overpurchase_spill_kwh": overpurchase_spill,
            "soc_start_kwh": p.soc_start_kwh,
            "soc_end_kwh": p.soc_end_kwh,
            "charge_state_binary": p.charge_state_binary.astype(int),
            "price_yuan_per_kwh": price,
            "base_plan_cost_yuan": base_cost,
            "adjustment_cost_yuan": adjustment_cost,
            "emergency_cost_yuan": emergency_cost,
            "grid_cost_yuan": base_cost + adjustment_cost + emergency_cost,
            "strategy": strategy,
        }
    )
    return result


def simulate_day(
    bundle: DataBundle,
    day: str,
    strategy: str = "rolling",
    parameters: StorageParameters | None = None,
    soc0_kwh: float = INITIAL_SOC_KWH,
    terminal_soc_kwh: float = TERMINAL_SOC_KWH,
    update_indices: tuple[int, ...] = UPDATE_INDICES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = parameters or StorageParameters(soc_step_kwh=DEFAULT_SOC_STEP_KWH)
    if strategy not in {"rolling", "static_forecast"}:
        raise ValueError("strategy 必须为 rolling 或 static_forecast")
    actual = day_actual(bundle, day)
    price = bundle.standard_price.sort_values("time_index").price_yuan_per_kwh.to_numpy(float)
    f0 = build_forecast_window(bundle, day, 0)
    base_plan = optimize_dispatch(f0.load_forecast_kwh, f0.pv_forecast_kwh, price, soc0_kwh, terminal_soc_kwh, parameters=parameters)
    base_plan.insert(0, "time_index", np.arange(N_PER_DAY))
    base_plan.insert(0, "target_time", f0.target_time.to_numpy())
    logs: list[pd.DataFrame] = []
    executed: list[pd.DataFrame] = []
    current_soc = soc0_kwh
    starts: tuple[int, ...] = update_indices if strategy == "rolling" else (0,)
    if not starts or starts[0] != 0 or any(start not in UPDATE_INDICES for start in starts):
        raise ValueError("rolling 的 update_indices 必须为以0开始的题设更新时间子集")
    for position, start in enumerate(starts):
        forecast = build_forecast_window(bundle, day, start)
        stop = N_PER_DAY if strategy == "static_forecast" else (
            starts[position + 1] if position + 1 < len(starts) else N_PER_DAY
        )
        if start == 0:
            plan = base_plan.drop(columns=["target_time", "time_index"]).copy()
        else:
            plan = optimize_dispatch(
                forecast.load_forecast_kwh,
                forecast.pv_forecast_kwh,
                price[start:],
                current_soc,
                terminal_soc_kwh,
                base_plan_grid_kwh=base_plan.scheduled_grid_kwh.iloc[start:].to_numpy(float),
                parameters=parameters,
            )
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
        log["executed_in_this_update"] = np.arange(start, N_PER_DAY) < stop
        log["planned_grid_kwh"] = plan.scheduled_grid_kwh.to_numpy(float)
        log["planned_charge_kwh"] = plan.charge_input_kwh.to_numpy(float)
        log["planned_discharge_kwh"] = plan.discharge_output_kwh.to_numpy(float)
        log["planned_curtailment_kwh"] = plan.planned_curtailment_kwh.to_numpy(float)
        log["planned_soc_start_kwh"] = plan.soc_start_kwh.to_numpy(float)
        log["planned_soc_end_kwh"] = plan.soc_end_kwh.to_numpy(float)
        log["base_plan_grid_kwh"] = base_plan.scheduled_grid_kwh.iloc[start:].to_numpy(float)
        log["soc_grid_step_kwh"] = parameters.soc_step_kwh
        logs.append(log)
        block = _execute_plan_block(actual, plan, base_plan, forecast, start, stop, strategy)
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
    result = _execute_plan_block(actual, plan, base_plan, forecast, 0, N_PER_DAY, "static_forecast")
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
    terminal_soc_kwh: float = TERMINAL_SOC_KWH,
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
            "scheduled_grid_kwh": plan.scheduled_grid_kwh, "up_adjustment_kwh": 0.0,
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
        "pv_forecast_mae_kw": float((schedule.pv_forecast_used_kw - schedule.actual_pv_kw).abs().mean()),
        "load_forecast_mae_kw": float(((schedule.load_forecast_kwh - schedule.actual_load_kwh) * 6).abs().mean()),
    }


def validate_executed_schedule(schedule: pd.DataFrame) -> dict[str, object]:
    balance = schedule.scheduled_grid_kwh + schedule.emergency_purchase_kwh + schedule.actual_pv_kwh + schedule.discharge_output_kwh - schedule.actual_load_kwh - schedule.charge_input_kwh - schedule.curtailed_pv_kwh - schedule.overpurchase_spill_kwh
    transition = schedule.soc_end_kwh - schedule.soc_start_kwh - ETA_C * schedule.charge_input_kwh + schedule.discharge_output_kwh / ETA_D
    expected_cost = schedule.base_plan_cost_yuan + schedule.adjustment_cost_yuan + schedule.emergency_cost_yuan
    issue_time = pd.to_datetime(schedule.forecast_issue_time)
    source_time = pd.to_datetime(schedule.load_forecast_source_time)
    return {
        "row_count_144_ok": bool(len(schedule) == N_PER_DAY),
        "time_index_complete_ok": bool(schedule.time_index.tolist() == list(range(N_PER_DAY))),
        "max_energy_balance_residual_kwh": float(balance.abs().max()),
        "max_soc_transition_residual_kwh": float(transition.abs().max()),
        "soc_bounds_ok": bool(schedule.soc_start_kwh.between(SOC_MIN - 1e-7, SOC_MAX + 1e-7).all() and schedule.soc_end_kwh.between(SOC_MIN - 1e-7, SOC_MAX + 1e-7).all()),
        "power_bounds_ok": bool((schedule[["charge_input_kwh", "discharge_output_kwh"]] <= MAX_INTERVAL_ENERGY + 1e-7).all().all()),
        "simultaneous_charge_discharge_count": int(((schedule.charge_input_kwh > 1e-8) & (schedule.discharge_output_kwh > 1e-8)).sum()),
        "grid_nonnegative_ok": bool((schedule[["scheduled_grid_kwh", "emergency_purchase_kwh", "total_grid_purchase_kwh"]] >= -1e-9).all().all()),
        "curtailment_nonnegative_ok": bool((schedule[["curtailed_pv_kwh", "overpurchase_spill_kwh"]] >= -1e-9).all().all()),
        "initial_soc_ok": bool(abs(schedule.soc_start_kwh.iloc[0] - INITIAL_SOC_KWH) < 1e-7),
        "terminal_soc_ok": bool(abs(schedule.soc_end_kwh.iloc[-1] - TERMINAL_SOC_KWH) < 1e-7),
        "soc_continuity_max_residual_kwh": float(np.max(np.r_[0.0, np.abs(schedule.soc_start_kwh.iloc[1:].to_numpy() - schedule.soc_end_kwh.iloc[:-1].to_numpy())])),
        "forecast_issue_hours_ok": bool(issue_time.dt.hour.isin(UPDATE_HOURS).all()),
        "forecast_issue_not_after_execution_ok": bool((issue_time <= pd.to_datetime(schedule.period_start)).all()),
        "load_source_precedes_issue_ok": bool((source_time < issue_time).all()),
        "cost_sum_residual_yuan": float(abs(schedule.grid_cost_yuan.sum() - expected_cost.sum())),
        "cumulative_cost_residual_yuan": float(abs(schedule.cumulative_grid_cost_yuan.iloc[-1] - schedule.grid_cost_yuan.sum())),
        "executed_decisions_unique_ok": bool(schedule.time_index.is_unique),
    }
