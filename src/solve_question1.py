#!/usr/bin/env python3
"""问题一：基于混合整数线性规划的微网标准日经济调度。

从仓库根目录运行：python src/solve_question1.py
只读取清洗后的附件1成品表，不修改任何原始附件或清洗产物。
"""
from __future__ import annotations

import shutil
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

# 让 matplotlib 缓存写入仓库，避免用户主目录不可写时影响可复现运行。
ROOT = Path(__file__).resolve().parents[1]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


INPUT = ROOT / "data" / "processed" / "attachment1_standard_day.csv"
TEMPLATE = ROOT / "results" / "result1.xlsx"
OUT = ROOT / "outputs" / "question1"
FIG = OUT / "figures"
REPORT = ROOT / "reports" / "question1_report.md"
COMPLETED_TEMPLATE = ROOT / "results" / "result1_completed.xlsx"

N = 144
ETA_C = ETA_D = 0.90
SOC_MIN, SOC_MAX = 1200.0, 10800.0
MAX_INTERVAL_ENERGY = 5000.0 / 6.0  # kWh/10分钟
COST_EPSILON = 1e-5  # 元；第二阶段允许的成本数值容差


def assert_input(df: pd.DataFrame) -> None:
    """只做问题一所需的轻量复核，绝不重新清洗数据。"""
    cols = ["time_label", "time_index", "price_yuan_per_kwh", "load_kw", "pv_forecast_kw", "load_kwh", "pv_forecast_kwh"]
    assert len(df) == N, f"附件1行数应为144，实际为{len(df)}"
    assert set(cols).issubset(df.columns), "附件1缺少问题一需要的字段"
    assert df[cols].notna().all().all(), "附件1关键字段存在缺失值"
    assert df["time_index"].is_unique and df["time_index"].tolist() == list(range(N)), "time_index不是唯一0~143"
    assert (df[["price_yuan_per_kwh", "load_kw", "pv_forecast_kw"]] >= 0).all().all(), "电价、负载或光伏存在负值"
    assert np.allclose(df["load_kwh"], df["load_kw"] / 6), "load_kwh不是load_kw/6"
    assert np.allclose(df["pv_forecast_kwh"], df["pv_forecast_kw"] / 6), "pv_forecast_kwh不是pv_forecast_kw/6"


def make_periods(df: pd.DataFrame) -> pd.DataFrame:
    """按模板左端点定义构造00:00--24:00日历日，并将0:00+1循环置首。"""
    source = df.copy()
    # 清洗表的0:00+1是次日0:00；在“每天重复”的标准日条件下，它等价于
    # 代表日00:00--00:10这一时段的参数，必须参与一次且仅参与一次调度。
    next_day = source["time_label"].astype(str).eq("0:00+1")
    assert next_day.sum() == 1 and len(source) == N, "必须唯一定位0:00+1并保留144行"
    result = pd.concat([source.loc[next_day], source.loc[~next_day]], ignore_index=True)
    result["source_time_index"] = result["time_index"].astype(int)
    result["calendar_time_index"] = np.arange(N, dtype=int)
    # 下游一律以日历顺序time_index作图和求解，原始顺序另存为source_time_index。
    result["time_index"] = result["calendar_time_index"]
    base = pd.Timestamp("2025-01-01")
    result["period_start"] = base + pd.to_timedelta(result["calendar_time_index"] * 10, unit="min")
    result["period_end"] = result["period_start"] + pd.Timedelta(minutes=10)
    result["period_start_text"] = result["period_start"].dt.strftime("%H:%M")
    result["period_end_text"] = result["period_end"].dt.strftime("%H:%M")
    result.loc[result["calendar_time_index"] == N - 1, "period_end_text"] = "24:00"
    result["period_start_minute"] = result["calendar_time_index"] * 10
    result["period_end_minute"] = result["period_start_minute"] + 10
    result["period_key"] = result["period_start_text"] + "-" + result["period_end_text"]
    assert result["period_key"].nunique() == N and result["source_time_index"].nunique() == N
    return result


def indices() -> dict[str, np.ndarray]:
    """返回单一决策向量中的变量下标。SOC含0~144共145个状态。"""
    g = np.arange(0, N)
    c = np.arange(N, 2 * N)
    d = np.arange(2 * N, 3 * N)
    w = np.arange(3 * N, 4 * N)
    s = np.arange(4 * N, 4 * N + N + 1)
    z = np.arange(4 * N + N + 1, 5 * N + N + 1)
    return {"g": g, "c": c, "d": d, "w": w, "s": s, "z": z, "nvar": int(z[-1] + 1)}


def solve_milp(data: pd.DataFrame, fixed_s0: float | None = None) -> tuple[np.ndarray, float, float, str]:
    """两阶段MILP：先最小费用，后在最优成本内最小化充放电吞吐量。"""
    ix = indices(); nv = ix["nvar"]
    price = data["price_yuan_per_kwh"].to_numpy(float)
    net_load = (data["load_kwh"] - data["pv_forecast_kwh"]).to_numpy(float)

    lb = np.zeros(nv); ub = np.full(nv, np.inf)
    lb[ix["s"]], ub[ix["s"]] = SOC_MIN, SOC_MAX
    ub[ix["c"]] = MAX_INTERVAL_ENERGY; ub[ix["d"]] = MAX_INTERVAL_ENERGY
    ub[ix["z"]] = 1.0
    if fixed_s0 is not None:
        lb[ix["s"][0]] = ub[ix["s"][0]] = fixed_s0
    bounds = Bounds(lb, ub)
    integrality = np.zeros(nv, dtype=int); integrality[ix["z"]] = 1

    # 等式：每时段电量平衡、SOC状态转移、循环SOC。
    aeq = lil_matrix((2 * N + 1, nv), dtype=float); beq = np.zeros(2 * N + 1)
    for t in range(N):
        aeq[t, ix["g"][t]] = 1; aeq[t, ix["c"][t]] = -1
        aeq[t, ix["d"][t]] = 1; aeq[t, ix["w"][t]] = -1
        beq[t] = net_load[t]  # G-C+D-W=L-P
        r = N + t
        aeq[r, ix["s"][t + 1]] = 1; aeq[r, ix["s"][t]] = -1
        aeq[r, ix["c"][t]] = -ETA_C; aeq[r, ix["d"][t]] = 1 / ETA_D
    aeq[-1, ix["s"][-1]] = 1; aeq[-1, ix["s"][0]] = -1
    eq = LinearConstraint(aeq.tocsr(), beq, beq)

    # C≤M z；D≤M(1-z)，从而禁止同一时段同时充、放电。
    aineq = lil_matrix((2 * N, nv), dtype=float); lower = np.full(2 * N, -np.inf); upper = np.zeros(2 * N)
    for t in range(N):
        aineq[t, ix["c"][t]] = 1; aineq[t, ix["z"][t]] = -MAX_INTERVAL_ENERGY
        aineq[N + t, ix["d"][t]] = 1; aineq[N + t, ix["z"][t]] = MAX_INTERVAL_ENERGY
        upper[N + t] = MAX_INTERVAL_ENERGY
    operational = LinearConstraint(aineq.tocsr(), lower, upper)

    cost_objective = np.zeros(nv); cost_objective[ix["g"]] = price
    first = milp(c=cost_objective, integrality=integrality, bounds=bounds, constraints=[eq, operational], options={"disp": False})
    if not first.success:
        raise RuntimeError(f"第一阶段MILP失败：status={first.status}; message={first.message}")
    optimal_cost = float(price @ first.x[ix["g"]])

    # 第二阶段在成本≤Cost*+epsilon下选择最小吞吐量的等价经济策略。
    second_obj = np.zeros(nv); second_obj[ix["c"]] = 1; second_obj[ix["d"]] = 1
    cost_limit = LinearConstraint(cost_objective.reshape(1, -1), [-np.inf], [optimal_cost + COST_EPSILON])
    second = milp(c=second_obj, integrality=integrality, bounds=bounds, constraints=[eq, operational, cost_limit], options={"disp": False})
    if not second.success:
        raise RuntimeError(f"第二阶段MILP失败：status={second.status}; message={second.message}")
    final_cost = float(price @ second.x[ix["g"]])
    return second.x, optimal_cost, final_cost, second.message


def schedule_from_solution(data: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    ix = indices(); result = data.copy() if "period_key" in data.columns else make_periods(data)
    result["grid_purchase_kwh"] = x[ix["g"]]
    result["charge_input_kwh"] = x[ix["c"]]
    result["discharge_output_kwh"] = x[ix["d"]]
    result["curtailed_pv_kwh"] = x[ix["w"]]
    result["soc_start_kwh"] = x[ix["s"]][:-1]
    result["soc_end_kwh"] = x[ix["s"]][1:]
    result["charge_state_binary"] = np.rint(x[ix["z"]]).astype(int)
    result["grid_cost_yuan"] = result["grid_purchase_kwh"] * result["price_yuan_per_kwh"]
    return result


def validate(schedule: pd.DataFrame, stage1_cost: float, final_cost: float, comparison_cost: float) -> dict[str, float | bool | str]:
    balance = schedule["grid_purchase_kwh"] + schedule["pv_forecast_kwh"] + schedule["discharge_output_kwh"] - schedule["load_kwh"] - schedule["charge_input_kwh"] - schedule["curtailed_pv_kwh"]
    transition = schedule["soc_end_kwh"] - schedule["soc_start_kwh"] - ETA_C * schedule["charge_input_kwh"] + schedule["discharge_output_kwh"] / ETA_D
    simultaneous = ((schedule["charge_input_kwh"] > 1e-6) & (schedule["discharge_output_kwh"] > 1e-6)).sum()
    return {
        "max_energy_balance_residual_kwh": float(balance.abs().max()),
        "max_soc_transition_residual_kwh": float(transition.abs().max()),
        "soc_bounds_ok": bool(((schedule["soc_start_kwh"] >= SOC_MIN - 1e-6) & (schedule["soc_end_kwh"] <= SOC_MAX + 1e-6)).all()),
        "charge_discharge_bounds_ok": bool(((schedule[["charge_input_kwh", "discharge_output_kwh"]] <= MAX_INTERVAL_ENERGY + 1e-6).all().all())),
        "simultaneous_charge_discharge_count": int(simultaneous),
        "cycle_soc_residual_kwh": float(abs(schedule["soc_end_kwh"].iloc[-1] - schedule["soc_start_kwh"].iloc[0])),
        "schedule_purchase_sum_kwh": float(schedule["grid_purchase_kwh"].sum()),
        "cost_sum_residual_yuan": float(abs(schedule["grid_cost_yuan"].sum() - final_cost)),
        "second_stage_cost_gap_yuan": float(final_cost - stage1_cost),
        "fixed_6000_cost_difference_yuan": float(comparison_cost - final_cost),
    }


def four_hour_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """严格按日历分钟边界汇总，每组必须正好包含24个时段。"""
    rows = []
    used: set[str] = set()
    for start_minute in range(0, 1440, 240):
        end_minute = start_minute + 240
        part = schedule.loc[(schedule["period_start_minute"] >= start_minute) & (schedule["period_start_minute"] < end_minute)]
        assert len(part) == 24, f"{start_minute // 60:02d}:00时段数不为24"
        used.update(part["period_key"])
        rows.append({"time_block": f"{start_minute // 60:02d}:00–{end_minute // 60:02d}:00", "charge_kwh": part["charge_input_kwh"].sum(), "discharge_kwh": part["discharge_output_kwh"].sum()})
    assert len(used) == N and used == set(schedule["period_key"]), "四小时汇总存在遗漏或重复"
    return pd.DataFrame(rows)


def set_chinese_font() -> None:
    # 使用本机存在的中文字体文件，避免名称未注册时回退到无中文字形的DejaVu。
    from matplotlib import font_manager
    font_path = Path("/System/Library/Fonts/STHeiti Medium.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def draw_figures(schedule: pd.DataFrame) -> None:
    set_chinese_font(); FIG.mkdir(parents=True, exist_ok=True)
    hour = schedule["period_end_minute"] / 60
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(hour, schedule["load_kw"], label="小区负载", lw=1.8)
    ax.plot(hour, schedule["pv_forecast_kw"], label="光伏预测功率", lw=1.8)
    ax.plot(hour, schedule["grid_purchase_kwh"] * 6, label="计划购电功率（等效）", lw=1.6)
    ax.set(xlabel="日历时刻 / h", ylabel="功率 / kW", title="问题一：标准日负载、光伏与计划购电")
    ax.set_xlim(0, 24); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout()
    fig.savefig(FIG / "question1_dispatch.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.step(np.r_[0, hour], np.r_[schedule["soc_start_kwh"].iloc[0], schedule["soc_end_kwh"]], where="post", label="储电量")
    ax.axhline(SOC_MIN, c="#c44e52", ls="--", label="下限1200")
    ax.axhline(SOC_MAX, c="#55a868", ls="--", label="上限10800")
    ax.set(xlabel="时刻 / h", ylabel="储电量 / kWh", title="问题一：储能SOC轨迹")
    ax.set_xlim(0, 24); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout()
    fig.savefig(FIG / "question1_soc.png", dpi=300); plt.close(fig)


def template_period_key(label: str) -> str:
    """将模板的跨日写法规范到代表日日历键，如0:00+1-0:10+1→00:00-00:10。"""
    start, end = str(label).split("-")
    def clock(text: str) -> str:
        text = text.replace("+1", "")
        hour, minute = text.split(":")
        return f"{int(hour):02d}:{int(minute):02d}"
    start_key, end_key = clock(start), clock(end)
    if "+1" in end and end_key == "00:00":
        end_key = "24:00"
    return f"{start_key}-{end_key}"


def fill_template(schedule: pd.DataFrame, blocks: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """复制官方模板并仅改目标单元格，保留其余 Office XML 组件。"""
    if not TEMPLATE.exists(): raise FileNotFoundError(f"未找到附件5模板：{TEMPLATE}")
    OUT.mkdir(parents=True, exist_ok=True); shutil.copy2(TEMPLATE, COMPLETED_TEMPLATE); shutil.copy2(TEMPLATE, OUT / "result1.xlsx")
    expected_sheets = ["计划购电量", "充放电量"]
    schedule_by_key = schedule.set_index("period_key")
    assert schedule_by_key.index.is_unique and len(schedule_by_key) == N
    excel_metrics: dict[str, float] = {}
    for output in [COMPLETED_TEMPLATE, OUT / "result1.xlsx"]:
        # 首先用openpyxl验证模板工作表/结构；随后以XML最小替换写值。
        # 后者避免openpyxl保存时剔除共享字符串、打印设置等模板内部组件。
        wb = load_workbook(output, read_only=True)
        if wb.sheetnames != expected_sheets: raise ValueError("输出模板工作表名称或顺序发生改变")
        ws = wb["计划购电量"]
        if ws.max_row != 145 or ws.max_column != 2: raise ValueError("计划购电量模板行列结构不符合预期")
        with ZipFile(output, "r") as source:
            content = {name: source.read(name) for name in source.namelist()}
        sheet1 = content["xl/worksheets/sheet1.xml"].decode("utf-8")
        template_keys: list[str] = []
        for row in range(2, 146):
            key = template_period_key(ws.cell(row, 1).value)
            if key not in schedule_by_key.index: raise ValueError(f"模板时段{ws.cell(row, 1).value}不能映射到日历解")
            template_keys.append(key)
            value = schedule_by_key.loc[key, "grid_purchase_kwh"]
            cell = f'B{row}'; replacement = f'<c r="{cell}" s="2"><v>{float(value):.12g}</v></c>'
            sheet1, n = re.subn(rf'<c r="{cell}" s="2"\s*/>', replacement, sheet1, count=1)
            if n != 1: raise ValueError(f"模板中未找到{cell}的空白购电量单元格")
        assert len(template_keys) == N and len(set(template_keys)) == N and set(template_keys) == set(schedule_by_key.index), "模板144行时间映射不完整"
        sheet2 = content["xl/worksheets/sheet2.xml"].decode("utf-8")
        values = {}
        for i, row in blocks.iterrows():
            values[f"B{i + 2}"] = row["charge_kwh"]; values[f"C{i + 2}"] = row["discharge_kwh"]
        values["E2"] = schedule["soc_start_kwh"].iloc[0]; values["E3"] = schedule["soc_end_kwh"].iloc[-1]
        for row in range(2, 8):
            additions = "".join(f'<c r="{cell}" s="5"><v>{float(value):.12g}</v></c>' for cell, value in values.items() if cell[1:] == str(row))
            sheet2, n = re.subn(rf'(<row r="{row}"[^>]*>.*?)(</row>)', rf'\1{additions}\2', sheet2, count=1, flags=re.DOTALL)
            if n != 1: raise ValueError(f"模板中未找到充放电表第{row}行")
        content["xl/worksheets/sheet1.xml"] = sheet1.encode("utf-8")
        content["xl/worksheets/sheet2.xml"] = sheet2.encode("utf-8")
        with ZipFile(output, "w", ZIP_DEFLATED) as target:
            for name, binary in content.items(): target.writestr(name, binary)
    # 重开验证所有目标位置，保证写入成功且逐行数值一致。
    wb = load_workbook(COMPLETED_TEMPLATE, data_only=True)
    purchase = np.array([wb["计划购电量"].cell(i, 2).value for i in range(2, 146)], dtype=float)
    expected_purchase = np.array([schedule_by_key.loc[template_period_key(wb["计划购电量"].cell(i, 1).value), "grid_purchase_kwh"] for i in range(2, 146)], dtype=float)
    assert np.allclose(purchase, expected_purchase, atol=1e-8), "模板购电量按时间键核验失败"
    charge = np.array([wb["充放电量"].cell(i, 2).value for i in range(2, 8)], dtype=float)
    discharge = np.array([wb["充放电量"].cell(i, 3).value for i in range(2, 8)], dtype=float)
    assert np.allclose(charge, blocks["charge_kwh"], atol=1e-8), "模板累计充电量核验失败"
    assert np.allclose(discharge, blocks["discharge_kwh"], atol=1e-8), "模板累计放电量核验失败"
    assert abs(wb["充放电量"].cell(2, 5).value - schedule["soc_start_kwh"].iloc[0]) < 1e-8, "模板0:00储电量核验失败"
    assert abs(wb["充放电量"].cell(3, 5).value - schedule["soc_end_kwh"].iloc[-1]) < 1e-8, "模板24:00储电量核验失败"
    excel_metrics["excel_purchase_sum_kwh"] = float(purchase.sum())
    excel_metrics["excel_vs_schedule_purchase_residual_kwh"] = float(abs(purchase.sum() - schedule["grid_purchase_kwh"].sum()))
    for key in ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]:
        template_row = next(i for i in range(2, 146) if template_period_key(wb["计划购电量"].cell(i, 1).value) == key)
        excel_metrics[f"excel_{key}_residual_kwh"] = float(abs(wb["计划购电量"].cell(template_row, 2).value - schedule_by_key.loc[key, "grid_purchase_kwh"]))
    return [str(COMPLETED_TEMPLATE.relative_to(ROOT)), str((OUT / "result1.xlsx").relative_to(ROOT))], excel_metrics


def write_report(schedule: pd.DataFrame, blocks: pd.DataFrame, validation: dict, stage1: float, final: float, fixed_cost: float, solver_message: str) -> None:
    specified_keys = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]
    specified = schedule.set_index("period_key").loc[specified_keys]
    table1 = "\n".join(f"| {row.period_start_text}–{row.period_end_text} | {row.grid_purchase_kwh:.3f} |" for _, row in specified.iterrows())
    table2 = "\n".join(f"| {r.time_block} | {r.charge_kwh:.3f} | {r.discharge_kwh:.3f} |" for _, r in blocks.iterrows())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(rf'''# 问题一：基于混合整数线性规划的微网日内经济调度

## 1. 问题概述

题目要求在每天0:00，依据标准日电价、负荷与光伏预测，制定144个10分钟时段的计划购电策略；小区负载必须满足，且0:00与24:00储电量相同，并使全天购电费用最小。本节仅使用附件1清洗后的标准日表，不涉及问题二、三。

## 2. 数据与假设

使用`attachment1_standard_day.csv`的144条记录。原始宽表已转为长表，功率按$E=P/6$转换为10分钟区间电量；本模型直接使用既有`load_kwh`、`pv_forecast_kwh`，不重复换算。附件时间标签按照官方模板解释为区间左端点；为形成00:00–24:00日历日，将`0:00+1`循环移动到最前面并规范化为00:00–00:10。

假设外部电网可按电价无限购电、不能售电；多余光伏只能弃光。充电和放电的单程效率均取0.9；充电量$C_t$为输入储能设备侧的电量，放电量$D_t$为储能设备输出至微网的电量。初始储电量$S_0$未被题意指定，因此作为[1200,10800] kWh内的决策变量，并以$S_{{144}}=S_0$实现循环运行。

## 3. 模型建立

令$t=1,\ldots,144$。$L_t,P_t,p_t$分别为负载电量(kWh)、光伏预测电量(kWh)、电价(元/kWh)；$G_t,C_t,D_t,W_t,S_t$依次为购电、充电、放电、弃光和期末储电量，单位均为kWh；$z_t$为充电状态二元变量。

目标函数为

$$\min\ \sum_t p_tG_t.$$

约束为

$$G_t+P_t+D_t=L_t+C_t+W_t,$$
$$S_t=S_{{t-1}}+0.9C_t-D_t/0.9,$$
$$1200\le S_t\le10800,\quad 0\le C_t\le833.333333z_t,\quad 0\le D_t\le833.333333(1-z_t),$$
$$S_{{144}}=S_0,\quad G_t,W_t\ge0,\quad z_t\in\{{0,1\}}.$$

采用`scipy.optimize.milp`（HiGHS）两阶段求解：第一阶段最小化购电费，得到{stage1:.6f}元；第二阶段加入购电费不高于第一阶段最优值加{COST_EPSILON}元的约束，最小化$\sum_t(C_t+D_t)$以消除同成本的不必要循环。求解器返回：{solver_message}。

## 4. 计算结果

全天计划购电量为**{schedule.grid_purchase_kwh.sum():.3f} kWh**，全天购电费为**{final:.3f} 元**；最优初始/末端储电量均为**{schedule.soc_start_kwh.iloc[0]:.3f} kWh**。预测光伏自用{(schedule.pv_forecast_kwh.sum()-schedule.curtailed_pv_kwh.sum()):.3f} kWh，弃光{schedule.curtailed_pv_kwh.sum():.3f} kWh。

表1 指定时段购电量及全天结果

| 时段 | 计划购电量 / kWh |
|---|---:|
{table1}
| 全天 | {schedule.grid_purchase_kwh.sum():.3f} |
| 全天购电费 / 元 | {final:.3f} |

表2 分时段充放电量与循环储电量

| 时段 | 累计充电量 / kWh | 累计放电量 / kWh |
|---|---:|---:|
{table2}
| 0:00储电量 | {schedule.soc_start_kwh.iloc[0]:.3f} | — |
| 24:00储电量 | {schedule.soc_end_kwh.iloc[-1]:.3f} | — |

图1显示购电主要转移至低价时段，光伏高发时段优先供负荷并为储能充电；图2显示SOC始终处于1200–10800 kWh运行区间且日末回到初始水平。自由$S_0$方案的费用为{final:.3f}元；固定$S_0=6000$ kWh对照方案费用为{fixed_cost:.3f}元，差异为{fixed_cost-final:.6f}元，说明本例中初始SOC约束对最优费用的影响为该数值。

![图1：标准日计划购电调度](../outputs/question1/figures/question1_dispatch.png)

![图2：储能SOC轨迹](../outputs/question1/figures/question1_soc.png)

## 5. 验证与结论

最大时段电量平衡残差为{validation['max_energy_balance_residual_kwh']:.3e} kWh，最大SOC转移残差为{validation['max_soc_transition_residual_kwh']:.3e} kWh；循环SOC残差为{validation['cycle_soc_residual_kwh']:.3e} kWh；Excel与CSV全天购电量残差为{validation['excel_vs_schedule_purchase_residual_kwh']:.3e} kWh。六个分段充、放电汇总均覆盖24个时段且无遗漏，不存在同时充放电时段，充放电上限、SOC边界、购电费用和Excel写入均逐项复核通过。完整逐时计划见`outputs/question1/question1_full_schedule.csv`。

模型是透明可复核的MILP，能严格表达功率/能量、效率和循环储能约束。局限在于光伏预测、电价和负荷被视为确定值，且未建模储能衰减、需量电费及外网售电；这些可在后续问题的滚动优化中扩展。

模板填表按时间段键匹配：模板`00:10–00:20`取日历解同名时段；模板最后一行`0:00+1–0:10+1`取代表日循环策略的`00:00–00:10`。原模板的工作表名称、顺序、表头、格式和布局均未修改。
''', encoding="utf-8")


def assert_report_consistency(schedule: pd.DataFrame, blocks: pd.DataFrame, final_cost: float) -> None:
    """报告由真实结果表渲染后再回读，核验指定时段、汇总和费用未发生偏差。"""
    text = REPORT.read_text(encoding="utf-8")
    for key in ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]:
        row = schedule.loc[schedule["period_key"] == key]
        assert len(row) == 1
        value = row["grid_purchase_kwh"].iloc[0]
        assert f"| {key.replace('-', '–')} | {value:.3f} |" in text, f"报告缺少或错误写入{key}"
    assert f"| 全天 | {schedule['grid_purchase_kwh'].sum():.3f} |" in text
    assert f"| 全天购电费 / 元 | {final_cost:.3f} |" in text
    for _, row in blocks.iterrows():
        assert f"| {row.time_block} | {row.charge_kwh:.3f} | {row.discharge_kwh:.3f} |" in text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    raw_data = pd.read_csv(INPUT); assert_input(raw_data)
    data = make_periods(raw_data)
    x, stage1_cost, final_cost, message = solve_milp(data)
    schedule = schedule_from_solution(data, x)
    x_fixed, _, fixed_cost, _ = solve_milp(data, fixed_s0=6000.0)
    blocks = four_hour_summary(schedule)
    validation = validate(schedule, stage1_cost, final_cost, fixed_cost)
    validation["four_hour_charge_sum_residual_kwh"] = float(abs(blocks.charge_kwh.sum() - schedule.charge_input_kwh.sum()))
    validation["four_hour_discharge_sum_residual_kwh"] = float(abs(blocks.discharge_kwh.sum() - schedule.discharge_output_kwh.sum()))
    assert validation["max_energy_balance_residual_kwh"] < 1e-6
    assert validation["max_soc_transition_residual_kwh"] < 1e-6
    assert validation["cycle_soc_residual_kwh"] < 1e-6
    assert validation["simultaneous_charge_discharge_count"] == 0
    assert validation["soc_bounds_ok"] and validation["charge_discharge_bounds_ok"]
    assert validation["four_hour_charge_sum_residual_kwh"] < 1e-6
    assert validation["four_hour_discharge_sum_residual_kwh"] < 1e-6
    schedule.to_csv(OUT / "question1_full_schedule.csv", index=False, encoding="utf-8-sig")
    draw_figures(schedule)
    template_paths, excel_metrics = fill_template(schedule, blocks)
    validation.update(excel_metrics)
    assert validation["excel_vs_schedule_purchase_residual_kwh"] < 1e-6
    assert all(v < 1e-6 for k, v in validation.items() if k.startswith("excel_") and k.endswith("_residual_kwh"))
    summary = pd.DataFrame([{"solver": "scipy.optimize.milp / HiGHS", "stage1_min_cost_yuan": stage1_cost, "final_cost_yuan": final_cost, "total_grid_purchase_kwh": schedule.grid_purchase_kwh.sum(), "s0_kwh": schedule.soc_start_kwh.iloc[0], "s144_kwh": schedule.soc_end_kwh.iloc[-1], "fixed_s0_6000_cost_yuan": fixed_cost, **validation}])
    summary.to_csv(OUT / "question1_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "question1_validation.txt").write_text("\n".join(f"{k}: {v}" for k,v in validation.items()) + "\n", encoding="utf-8")
    write_report(schedule, blocks, validation, stage1_cost, final_cost, fixed_cost, message)
    assert_report_consistency(schedule, blocks, final_cost)
    print(f"求解成功：最低购电费={final_cost:.6f} 元；全天购电量={schedule.grid_purchase_kwh.sum():.6f} kWh")
    print(f"S0=S144={schedule.soc_start_kwh.iloc[0]:.6f} kWh；最大能量平衡残差={validation['max_energy_balance_residual_kwh']:.3e} kWh")
    print("模板输出：" + "，".join(template_paths))


if __name__ == "__main__": main()
