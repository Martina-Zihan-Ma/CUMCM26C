"""第三问储能调度的离散状态动态规划模型。

模型把 SOC 离散为有限状态。每个 10 分钟时段只允许 SOC 上升（充电）、
下降（放电）或不变，因而无需二元变量也能严格排除同时充放电。动态规划
枚举全部可行状态转移，得到给定 SOC 网格上的全局最优解。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


ETA_C = 0.90
ETA_D = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
MAX_POWER_KW = 5000.0
MAX_INTERVAL_ENERGY = MAX_POWER_KW / 6.0
DEFAULT_SOC_STEP_KWH = 120.0
THROUGHPUT_TIE_BREAK = 1e-9


@dataclass(frozen=True)
class StorageParameters:
    eta_c: float = ETA_C
    eta_d: float = ETA_D
    soc_min_kwh: float = SOC_MIN
    soc_max_kwh: float = SOC_MAX
    max_interval_energy_kwh: float = MAX_INTERVAL_ENERGY
    soc_step_kwh: float = DEFAULT_SOC_STEP_KWH


def settlement_cost(
    scheduled_grid_kwh: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    base_plan_grid_kwh: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回计划成本、上调量、下调量和净调整成本。

    下调解释为取消电量不再按全价购买、但支付电价 50% 的违约费，故相对
    原计划的净调整项为 ``+1.5*p*up - 0.5*p*down``。
    """
    grid = np.asarray(scheduled_grid_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    if base_plan_grid_kwh is None:
        zero = np.zeros_like(grid)
        return price * grid, zero, zero, zero
    base = np.asarray(base_plan_grid_kwh, dtype=float)
    up = np.maximum(grid - base, 0.0)
    down = np.maximum(base - grid, 0.0)
    plan_cost = price * base
    adjustment_cost = 1.5 * price * up - 0.5 * price * down
    return plan_cost, up, down, adjustment_cost


def _state_grid(parameters: StorageParameters, soc0: float, terminal_soc: float | None) -> np.ndarray:
    p = parameters
    regular = np.arange(p.soc_min_kwh, p.soc_max_kwh + p.soc_step_kwh / 2, p.soc_step_kwh)
    extras = [p.soc_min_kwh, p.soc_max_kwh, soc0]
    if terminal_soc is not None:
        extras.append(float(terminal_soc))
    states = np.unique(np.r_[regular, extras])
    states = states[(states >= p.soc_min_kwh - 1e-9) & (states <= p.soc_max_kwh + 1e-9)]
    return np.sort(states.astype(float))


def optimize_dispatch(
    load_forecast_kwh: Sequence[float],
    pv_forecast_kwh: Sequence[float],
    price_yuan_per_kwh: Sequence[float],
    soc0_kwh: float,
    terminal_soc_kwh: float | None = None,
    base_plan_grid_kwh: Sequence[float] | None = None,
    parameters: StorageParameters | None = None,
) -> pd.DataFrame:
    """求解一个滚动窗口，并返回逐时段最优计划。

    terminal_soc_kwh=None 时，末端 SOC 不被强制固定；动态规划从所有
    可行末端状态中选择当前 24 小时预测窗口下成本最低的状态。
    """
    p = parameters or StorageParameters()
    load = np.asarray(load_forecast_kwh, dtype=float)
    pv = np.asarray(pv_forecast_kwh, dtype=float)
    price = np.asarray(price_yuan_per_kwh, dtype=float)
    n = len(load)
    if n == 0 or len(pv) != n or len(price) != n:
        raise ValueError("负荷、光伏和电价必须是相同长度的非空序列")
    if not (p.soc_min_kwh <= soc0_kwh <= p.soc_max_kwh):
        raise ValueError("窗口初始 SOC 越界")
    if terminal_soc_kwh is not None and not (p.soc_min_kwh <= terminal_soc_kwh <= p.soc_max_kwh):
        raise ValueError("窗口末端 SOC 越界")
    base = None if base_plan_grid_kwh is None else np.asarray(base_plan_grid_kwh, dtype=float)
    if base is not None and len(base) != n:
        raise ValueError("原计划购电量长度与窗口不一致")

    states = _state_grid(p, soc0_kwh, terminal_soc_kwh)
    k = len(states)
    delta = states[None, :] - states[:, None]
    charge = np.where(delta >= -1e-12, np.maximum(delta, 0.0) / p.eta_c, 0.0)
    discharge = np.where(delta < -1e-12, -delta * p.eta_d, 0.0)
    feasible = (charge <= p.max_interval_energy_kwh + 1e-9) & (
        discharge <= p.max_interval_energy_kwh + 1e-9
    )

    start_idx = int(np.argmin(np.abs(states - soc0_kwh)))
    terminal_idx = None if terminal_soc_kwh is None else int(np.argmin(np.abs(states - terminal_soc_kwh)))
    if abs(states[start_idx] - soc0_kwh) > 1e-8:
        raise RuntimeError("SOC 初始状态没有进入状态网格")
    if terminal_idx is not None and abs(states[terminal_idx] - float(terminal_soc_kwh)) > 1e-8:
        raise RuntimeError("SOC 末端状态没有进入状态网格")

    dp = np.full(k, np.inf)
    dp[start_idx] = 0.0
    parents = np.full((n, k), -1, dtype=np.int32)
    net = load - pv
    for t in range(n):
        raw_grid = net[t] + charge - discharge
        grid = np.maximum(raw_grid, 0.0)
        if base is None or not np.isfinite(base[t]):
            # No committed base plan for this preview interval (e.g. after midnight).
            # Price it as an ordinary future purchase; only executed/current-day intervals
            # are settled against the 0:00 base plan.
            stage = price[t] * grid
        else:
            up = np.maximum(grid - base[t], 0.0)
            down = np.maximum(base[t] - grid, 0.0)
            stage = price[t] * base[t] + 1.5 * price[t] * up - 0.5 * price[t] * down
        stage = stage + THROUGHPUT_TIE_BREAK * (charge + discharge)
        total = dp[:, None] + np.where(feasible, stage, np.inf)
        parents[t] = np.argmin(total, axis=0)
        dp = np.min(total, axis=0)
    if terminal_idx is None:
        terminal_idx = int(np.argmin(dp))
    if not np.isfinite(dp[terminal_idx]):
        raise RuntimeError("给定 SOC 网格下窗口不可行")

    state_indices = np.empty(n + 1, dtype=np.int32)
    state_indices[-1] = terminal_idx
    for t in range(n - 1, -1, -1):
        state_indices[t] = parents[t, state_indices[t + 1]]
    if state_indices[0] != start_idx:
        raise RuntimeError("动态规划回溯未回到窗口初始 SOC")

    soc_start = states[state_indices[:-1]]
    soc_end = states[state_indices[1:]]
    ds = soc_end - soc_start
    c = np.where(ds >= -1e-9, np.maximum(ds, 0.0) / p.eta_c, 0.0)
    d = np.where(ds < -1e-9, -ds * p.eta_d, 0.0)
    raw_grid = net + c - d
    g = np.maximum(raw_grid, 0.0)
    w = np.maximum(-raw_grid, 0.0)
    if base is None:
        plan_cost, up, down, adjustment = settlement_cost(g, price, None)
    else:
        finite = np.isfinite(base)
        plan_cost = np.where(finite, price * base, price * g)
        up = np.where(finite, np.maximum(g - base, 0.0), 0.0)
        down = np.where(finite, np.maximum(base - g, 0.0), 0.0)
        adjustment = np.where(finite, 1.5 * price * up - 0.5 * price * down, 0.0)
    return pd.DataFrame(
        {
            "load_forecast_kwh": load,
            "pv_forecast_kwh": pv,
            "price_yuan_per_kwh": price,
            "scheduled_grid_kwh": g,
            "charge_input_kwh": c,
            "discharge_output_kwh": d,
            "planned_curtailment_kwh": w,
            "soc_start_kwh": soc_start,
            "soc_end_kwh": soc_end,
            "charge_state_binary": (c > 1e-8).astype(int),
            "base_plan_grid_kwh": g if base is None else base,
            "up_adjustment_kwh": up,
            "down_adjustment_kwh": down,
            "base_plan_cost_yuan": plan_cost,
            "adjustment_cost_yuan": adjustment,
            "optimization_stage_cost_yuan": plan_cost + adjustment,
            "soc_grid_step_kwh": p.soc_step_kwh,
        }
    )


def validate_planned_dispatch(schedule: pd.DataFrame, parameters: StorageParameters | None = None) -> dict[str, float | int | bool]:
    p = parameters or StorageParameters()
    balance = schedule.scheduled_grid_kwh + schedule.pv_forecast_kwh + schedule.discharge_output_kwh - schedule.load_forecast_kwh - schedule.charge_input_kwh - schedule.planned_curtailment_kwh
    transition = schedule.soc_end_kwh - schedule.soc_start_kwh - p.eta_c * schedule.charge_input_kwh + schedule.discharge_output_kwh / p.eta_d
    return {
        "planned_max_energy_balance_residual_kwh": float(balance.abs().max()),
        "planned_max_soc_transition_residual_kwh": float(transition.abs().max()),
        "planned_soc_bounds_ok": bool(schedule.soc_start_kwh.between(p.soc_min_kwh - 1e-7, p.soc_max_kwh + 1e-7).all() and schedule.soc_end_kwh.between(p.soc_min_kwh - 1e-7, p.soc_max_kwh + 1e-7).all()),
        "planned_power_bounds_ok": bool((schedule[["charge_input_kwh", "discharge_output_kwh"]] <= p.max_interval_energy_kwh + 1e-7).all().all()),
        "planned_simultaneous_charge_discharge_count": int(((schedule.charge_input_kwh > 1e-8) & (schedule.discharge_output_kwh > 1e-8)).sum()),
        "planned_grid_nonnegative_ok": bool((schedule.scheduled_grid_kwh >= -1e-9).all()),
        "planned_curtailment_nonnegative_ok": bool((schedule.planned_curtailment_kwh >= -1e-9).all()),
    }
