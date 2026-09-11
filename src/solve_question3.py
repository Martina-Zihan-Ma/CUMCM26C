#!/usr/bin/env python3
"""运行 C 题第三问并生成 CSV、图表、验证、日志和报告。"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from question3_model import (  # noqa: E402
    DEFAULT_SOC_STEP_KWH, ETA_C, ETA_D, MAX_INTERVAL_ENERGY, MAX_POWER_KW,
    SOC_MAX, SOC_MIN, StorageParameters,
)
from question3_rolling import (  # noqa: E402
    EMERGENCY_PRICE_MULTIPLIER, INITIAL_SOC_KWH, N_PER_DAY, SPECIFIED_DATES,
    TERMINAL_SOC_KWH, UPDATE_HOURS, load_data_bundle, simulate_day,
    simulate_oracle_day, simulate_static_from_base_plan, summarize_schedule,
    validate_executed_schedule,
)

DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "question3"
FIG = OUT / "figures"
REPORT = ROOT / "reports" / "question3_report.md"
FULL_SCHEDULE = OUT / "question3_full_period_rolling_schedule.csv"
MAIN_PARAMETERS = StorageParameters(soc_step_kwh=DEFAULT_SOC_STEP_KWH)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, str, np.ndarray]],
    y_label: str,
    fixed_y: tuple[float, float] | None = None,
) -> None:
    width, height = 1400, 700
    left, right, top, bottom = 105, 35, 90, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(30), font(20), font(16)
    draw.text((left, 20), title, fill="#202124", font=title_font)
    all_values = np.concatenate([np.asarray(values, float) for _, _, values in series])
    ymin, ymax = fixed_y or (float(np.nanmin(all_values)), float(np.nanmax(all_values)))
    if abs(ymax - ymin) < 1e-9:
        ymin, ymax = ymin - 1, ymax + 1
    if fixed_y is None:
        pad = 0.08 * (ymax - ymin)
        ymin, ymax = ymin - pad, ymax + pad
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    draw.line((x0, y0, x1, y0), fill="#555", width=2)
    draw.line((x0, y0, x0, y1), fill="#555", width=2)
    for hour in range(0, 25, 3):
        x = x0 + (x1 - x0) * hour / 24
        draw.line((x, y0, x, y0 + 7), fill="#555", width=1)
        draw.text((x - 12, y0 + 12), str(hour), fill="#444", font=small_font)
    for j in range(6):
        value = ymin + (ymax - ymin) * j / 5
        y = y0 - (y0 - y1) * j / 5
        draw.line((x0, y, x1, y), fill="#e5e7eb", width=1)
        draw.text((8, y - 10), f"{value:.0f}", fill="#555", font=small_font)
    draw.text((width // 2 - 30, height - 35), "Time / h", fill="#333", font=body_font)
    draw.text((8, top - 30), y_label, fill="#333", font=body_font)
    legend_x = left + 20
    for name, color, values in series:
        values = np.asarray(values, float)
        points = [
            (
                x0 + (x1 - x0) * (i + 1) / len(values),
                y0 - (y0 - y1) * (value - ymin) / (ymax - ymin),
            )
            for i, value in enumerate(values)
        ]
        draw.line(points, fill=color, width=3)
        draw.line((legend_x, top + 15, legend_x + 35, top + 15), fill=color, width=5)
        draw.text((legend_x + 44, top + 2), name, fill="#222", font=small_font)
        legend_x += int(80 + draw.textlength(name, font=small_font))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def draw_day_figures(day: str, rolling: pd.DataFrame) -> None:
    stem = day.replace("-", "")
    line_chart(
        FIG / f"{stem}_dispatch.png", f"Question 3 rolling dispatch - {day}",
        [("Actual load", "#202124", rolling.actual_load_kw.to_numpy()),
         ("Actual PV", "#f59e0b", rolling.actual_pv_kw.to_numpy()),
         ("Scheduled grid", "#2563eb", rolling.scheduled_grid_kwh.to_numpy() * 6),
         ("Emergency", "#dc2626", rolling.emergency_purchase_kwh.to_numpy() * 6)],
        "Power / kW",
    )
    line_chart(FIG / f"{stem}_soc.png", f"Storage SOC - {day}",
               [("SOC", "#16a34a", rolling.soc_end_kwh.to_numpy())],
               "Energy / kWh", (SOC_MIN, SOC_MAX))
    line_chart(FIG / f"{stem}_forecast_error.png", f"PV forecast error - {day}",
               [("Forecast - actual", "#7c3aed", (rolling.pv_forecast_used_kw - rolling.actual_pv_kw).to_numpy())],
               "Error / kW")
    line_chart(FIG / f"{stem}_update_change.png", f"Purchase-plan changes - {day}",
               [("00:00 base plan", "#6b7280", rolling.base_plan_grid_kwh.to_numpy() * 6),
                ("Executed schedule", "#0284c7", rolling.scheduled_grid_kwh.to_numpy() * 6)],
               "Power equivalent / kW")


def contiguous_emergency_blocks(schedule: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day, group in schedule.groupby("date", sort=True):
        group = group.sort_values("time_index").reset_index(drop=True)
        active = group.emergency_purchase_kwh.to_numpy(float) > tolerance
        i = 0
        while i < N_PER_DAY:
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < N_PER_DAY and active[j + 1]:
                j += 1
            def clock(minute: int) -> str:
                return "24:00" if minute == 1440 else f"{minute // 60:02d}:{minute % 60:02d}"
            rows.append({
                "date": day,
                "emergency_period": f"{clock(i * 10)}-{clock((j + 1) * 10)}",
                "emergency_purchase_kwh": float(group.emergency_purchase_kwh.iloc[i:j + 1].sum()),
                "start_time_index": i,
                "end_time_index_inclusive": j,
            })
            i = j + 1
    return pd.DataFrame(rows, columns=["date", "emergency_period", "emergency_purchase_kwh", "start_time_index", "end_time_index_inclusive"])


def selected_charge_blocks(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for day, schedule in selected.items():
        for start in range(0, N_PER_DAY, 24):
            part = schedule.iloc[start:start + 24]
            rows.append({
                "date": day,
                "time_block": f"{start // 6:02d}:00-{(start + 24) // 6:02d}:00",
                "charge_input_kwh": float(part.charge_input_kwh.sum()),
                "discharge_output_kwh": float(part.discharge_output_kwh.sum()),
                "soc_00_kwh": float(schedule.soc_start_kwh.iloc[0]) if start == 0 else np.nan,
                "soc_24_kwh": float(schedule.soc_end_kwh.iloc[-1]) if start == 24 else np.nan,
            })
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] + ["---:" for _ in columns[1:]]) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def write_report(
    selected_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    resolution: pd.DataFrame,
) -> None:
    selected = selected_summary.copy()
    selected["pv_consumption_rate"] *= 100
    selected_view = selected[[
        "date", "strategy", "total_grid_cost_yuan", "total_grid_purchase_kwh",
        "emergency_purchase_kwh", "curtailed_pv_kwh", "pv_consumption_rate",
    ]]
    selected_view.columns = ["日期", "策略", "总费用/元", "总购电/kWh", "紧急购电/kWh", "弃光/kWh", "光伏消纳率/%"]
    period = daily_summary.groupby("strategy", as_index=False).agg(
        天数=("date", "count"), 总费用_元=("total_grid_cost_yuan", "sum"),
        总购电_kWh=("total_grid_purchase_kwh", "sum"), 紧急购电_kWh=("emergency_purchase_kwh", "sum"),
        弃光_kWh=("curtailed_pv_kwh", "sum"), 充电_kWh=("charge_input_kwh", "sum"),
        放电_kWh=("discharge_output_kwh", "sum"),
    )
    period.columns = ["策略", "天数", "总费用/元", "总购电/kWh", "紧急购电/kWh", "弃光/kWh", "充电/kWh", "放电/kWh"]
    pivot = daily_summary.pivot(index="date", columns="strategy", values="total_grid_cost_yuan")
    savings = pivot["static_forecast"] - pivot["rolling"]
    better_days = int((savings > 1e-7).sum())
    total_saving = float(savings.sum())
    mean_saving = float(savings.mean())
    max_balance = float(validation.max_energy_balance_residual_kwh.max())
    max_transition = float(validation.max_soc_transition_residual_kwh.max())
    sens_view = sensitivity[["date", "eta", "total_grid_cost_yuan", "emergency_purchase_kwh", "curtailed_pv_kwh"]].copy()
    sens_view.columns = ["日期", "充放电效率", "滚动总费用/元", "紧急购电/kWh", "弃光/kWh"]
    res_view = resolution[["date", "soc_step_kwh", "total_grid_cost_yuan", "total_grid_purchase_kwh"]].copy()
    res_view.columns = ["日期", "SOC步长/kWh", "滚动总费用/元", "总购电/kWh"]

    text = rf"""# 第三问：基于分时预测更新的微网滚动购电策略

## 1. 问题重述与题面边界

PDF 第三问规定：每天 0:00、6:00、12:00 和 18:00 可获得未来 24 小时整点光伏发电功率预报；0:00 形成当日计划购电，后三个发布时间可以调整尚未执行的购电量。计划量高于调整量的部分按交易时刻电价的 50% 计违约，调整量高于计划量的部分按交易时刻电价的 1.5 倍计价；总费用包括计划购电、紧急购电和调整相关费用。结果文件覆盖 2025-02-01 至 2025-12-31，论文重点给出 2025-03-20、06-21、09-23、12-21。

第三问继承问题二“紧急购电按当时电价 5 倍”的规则。第三问原文明示使用附件1电价，因此本模型把附件1的144点电价作为每天重复的交易电价。附件4属于问题四，只做数据一致性检查，不进入第三问目标函数。

## 2. 数据与预处理

模型读取 `attachment1_standard_day.csv`、`attachment2_actual_long.csv`、`attachment3_forecast_hourly_long.csv`、`attachment3_forecast_10min_long.csv` 和 `model_base_2025.csv`，并读取 `attachment4_price_long.csv` 交叉核对。原始附件1至4的工作表、日期数、每日144点、缺失、重复键和单位已与 `scripts/clean_data.py` 核对。

功率按 $E=P/6$ 转为10分钟电量。每个日期的 `time_index=0,...,143` 原序对应 00:00-00:10 至 23:50-24:00；`0:00+1` 只表示最后区间的结束，没有循环移动。

## 3. 信息边界

附件3每个 `issue_time` 给出未来第1至24小时整点值；清洗表仅在 +60 至 +1440 分钟内线性插值。发布后首小时的五个十分钟点，以发布时间已观测到的光伏功率和本版本 +1 小时预测线性插值，不使用未来实际值。未来负荷没有题给预报，沿用问题二回测最优基线：目标时段前7天同一时刻实际负荷，且始终满足 `load_forecast_source_time < issue_time`。

每次优化完整记录 `issue_time`、`target_time`、提前量、光伏锚点、负荷历史源、优化计划和实际执行标志。附件2的当天实际值只用于时段执行后的紧急购电、弃光/溢出和评价，不回填预测。

## 4. 优化模型

令 $L_t,P_t,p_t$ 为预测负荷电量、预测光伏电量和附件1电价，$G_t,C_t,D_t,W_t,S_t$ 为常规购电、充电输入、放电输出、预测弃光和期末SOC：

$$G_t+P_t+D_t=L_t+C_t+W_t,$$
$$S_t=S_{{t-1}}+0.9C_t-D_t/0.9,$$
$$1200\le S_t\le10800,\qquad 0\le C_t,D_t\le5000/6.$$

状态转移只允许充电、放电或不变，从结构上严格禁止同段同时充放电。每天0:00与24:00均设为6000 kWh，继承附录1给定初值并消除有限窗口末端放空。

0:00 最小化 $\sum p_tG_t^0$。更新时相对原计划定义 $U_t=(G_t-G_t^0)_+$、$R_t=(G_t^0-G_t)_+$，最小化

$$\sum_t[p_tG_t^0+1.5p_tU_t-0.5p_tR_t].$$

下调口径解释为：取消量不再按全价购买，但支付50%违约费，因此相对原计划净减少 $0.5p_tR_t$。执行时若实际净负荷高于预测，差额按 $5p_t$ 紧急购电；若低于预测，实际弃光和已购电溢出分列，避免把电网溢出误称为弃光。

求解采用离散SOC动态规划。主步长为 {DEFAULT_SOC_STEP_KWH:.0f} kWh，每个窗口枚举全部可行状态转移，得到该网格上的全局最优解；另以60/240 kWh做数值分辨率检查。

## 5. 滚动机制

0:00先优化144段并固定原计划，只执行00:00-06:00。6:00使用新版预测和执行后SOC重算剩余108段，只执行到12:00；12:00和18:00同理。后续窗口不能修改已执行行，每个窗口只使用一个当时已发布的预测版本。

## 6. 四个指定日期

{md_table(selected_view)}

`oracle_actual` 使用全天实际负荷和实际光伏，只是同一模型下不可执行的事后最优参照。符合信息边界、可实际执行的是 `rolling`。

## 7. 全期对照与预测更新价值

{md_table(period)}

2025-02-01至12-31共334天中，滚动方案有 {better_days} 天费用低于静态方案。累计静态减滚动费用为 {total_saving:.3f} 元，日均为 {mean_saving:.3f} 元。该比较同时计入计划、上下调净结算和5倍紧急购电，所以是否更新不能只看常规购电量。

完整334天滚动执行表为 `question3_full_period_rolling_schedule.csv`，题目给定格式的汇总工作簿为 `result3.xlsx`。四个指定日的调度、SOC、预测误差和更新前后计划图位于 `outputs/question3/figures/`。

## 8. 预测误差影响

逐时结果同时保留实际值、采用的预测值、预测发布时间、紧急购电和弃光。实际光伏低于预测通常增加5倍紧急购电，实际光伏高于预测通常增加弃光或已购电溢出；滚动和静态的费用差反映了预测更新、调整结算与误差暴露的综合效果。

## 9. 敏感性分析

充放电效率敏感性：

{md_table(sens_view)}

SOC网格分辨率敏感性（下表为真实执行后的回测成本，不应用于判断DP预测目标的网格最优性；对应预测目标见 `question3_dp_grid_audit.csv`）：

{md_table(res_view)}

## 10. 验证结果

四个指定日三种策略和全期滚动/静态均检查了电量平衡、SOC转移、SOC边界、功率上限、充放互斥、购电/弃光非负、初末SOC、窗口连续、执行索引唯一、预测发布时间、历史负荷来源及费用求和。所有布尔检查通过；最大电量平衡残差为 {max_balance:.3e} kWh，最大SOC转移残差为 {max_transition:.3e} kWh。

## 11. 优点、局限与结论

模型把原计划、调整计划和紧急购电分开，逐窗口封闭信息集，并保存全部输入版本和执行状态。动态规划仅在给定离散SOC状态/动作空间和既定结算口径下对预测目标全局最优；真实执行回测成本会随预测误差而不必随网格单调。

局限包括：题面未完全展开下调结算的会计口径，本文采用“取消全价购买并支付50%违约费”的净结算解释；负荷预测仅为问题二的前一周同期基线；SOC离散存在小幅数值误差；预测过剩时可能出现已购电溢出，本文单列处理。若官方另有下调结算口径，只需替换 `settlement_cost`。

滚动结果是本题可执行答案；静态方案衡量不更新预报的代价，事后最优只给出信息完全时的参考下界。核心代码为 `src/question3_model.py`、`src/question3_rolling.py` 和 `src/solve_question3.py`。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    bundle = load_data_bundle(DATA)
    days = pd.date_range("2025-02-01", "2025-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    if FULL_SCHEDULE.exists():
        FULL_SCHEDULE.unlink()
    daily_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    selected_summary_rows: list[dict[str, object]] = []
    selected_timeseries: list[pd.DataFrame] = []
    selected_rolling: dict[str, pd.DataFrame] = {}
    first_write = True
    for number, day in enumerate(days, 1):
        rolling, versions, base_plan = simulate_day(bundle, day, "rolling", MAIN_PARAMETERS)
        static = simulate_static_from_base_plan(bundle, day, base_plan)
        rolling.to_csv(
            FULL_SCHEDULE, mode="w" if first_write else "a", header=first_write, index=False,
            encoding="utf-8-sig" if first_write else "utf-8", date_format="%Y-%m-%d %H:%M:%S",
        )
        first_write = False
        for schedule in [static, rolling]:
            daily_rows.append(summarize_schedule(schedule))
            check = validate_executed_schedule(schedule)
            validation_rows.append({"date": day, "strategy": schedule.strategy.iloc[0], **check})
            failed = [
                key for key, value in check.items()
                if (key.endswith("_ok") and not value)
                or ("residual" in key and float(value) > 1e-6)
                or (key.endswith("count") and int(value) != 0)
            ]
            if failed:
                raise AssertionError(f"{day} {schedule.strategy.iloc[0]} 验证失败：{failed}")
        if day in SPECIFIED_DATES:
            oracle = simulate_oracle_day(bundle, day, MAIN_PARAMETERS)
            selected_rolling[day] = rolling
            save_csv(rolling, OUT / f"{day.replace('-', '')}_rolling_schedule.csv")
            save_csv(versions, OUT / f"{day.replace('-', '')}_forecast_versions.csv")
            for schedule in [static, rolling, oracle]:
                selected_summary_rows.append(summarize_schedule(schedule))
                selected_timeseries.append(schedule)
            oracle_check = validate_executed_schedule(oracle)
            for key in ["forecast_issue_hours_ok", "forecast_issue_not_after_execution_ok", "load_source_precedes_issue_ok"]:
                oracle_check[key] = True
            validation_rows.append({"date": day, "strategy": "oracle_actual", **oracle_check})
            draw_day_figures(day, rolling)
        if number % 50 == 0 or number == len(days):
            print(f"第三问全年进度：{number}/{len(days)}")

    daily_summary = pd.DataFrame(daily_rows)
    validation = pd.DataFrame(validation_rows)
    selected_summary = pd.DataFrame(selected_summary_rows)
    selected_ts = pd.concat(selected_timeseries, ignore_index=True)
    save_csv(daily_summary, OUT / "question3_full_period_daily_summary.csv")
    save_csv(validation, OUT / "question3_validation_summary.csv")
    save_csv(selected_summary, OUT / "question3_strategy_comparison_selected_dates.csv")
    save_csv(selected_ts, OUT / "question3_selected_strategy_timeseries.csv")
    for day in SPECIFIED_DATES:
        stem = day.replace("-", "")
        save_csv(selected_summary.loc[selected_summary.date.eq(day)], OUT / f"{stem}_cost_summary.csv")
        records = validation.loc[validation.date.eq(day)].to_dict(orient="records")
        (OUT / f"{stem}_validation.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    full = pd.read_csv(FULL_SCHEDULE, encoding="utf-8-sig")
    emergency_blocks = contiguous_emergency_blocks(full)
    charge_blocks = selected_charge_blocks(selected_rolling)
    save_csv(emergency_blocks, OUT / "question3_emergency_blocks.csv")
    save_csv(charge_blocks, OUT / "question3_selected_charge_blocks.csv")
    dates = days
    planned = full.pivot(index="date", columns="time_index", values="base_plan_grid_kwh").loc[dates]
    adjusted = full.pivot(index="date", columns="time_index", values="scheduled_grid_kwh").loc[dates]
    rolling_daily = daily_summary.loc[daily_summary.strategy.eq("rolling")].set_index("date").loc[dates]
    payload = {
        "dates": dates,
        "planned_grid_kwh": planned.to_numpy(float).tolist(),
        "adjusted_grid_kwh": adjusted.to_numpy(float).tolist(),
        "planned_total_kwh": planned.sum(axis=1).to_numpy(float).tolist(),
        "planned_cost_yuan": rolling_daily.base_plan_cost_yuan.to_numpy(float).tolist(),
        "adjusted_total_kwh": adjusted.sum(axis=1).to_numpy(float).tolist(),
        "adjusted_settlement_cost_yuan": (rolling_daily.base_plan_cost_yuan + rolling_daily.adjustment_cost_yuan).to_numpy(float).tolist(),
        "selected_charge_blocks": charge_blocks.astype(object).where(pd.notna(charge_blocks), None).to_dict(orient="records"),
        "emergency_blocks": emergency_blocks.to_dict(orient="records"),
    }
    (OUT / "question3_workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    sensitivity_rows = []
    for eta in [0.85, 0.90, 0.95]:
        parameters = StorageParameters(eta_c=eta, eta_d=eta, soc_step_kwh=DEFAULT_SOC_STEP_KWH)
        for day in SPECIFIED_DATES:
            schedule, _, _ = simulate_day(bundle, day, "rolling", parameters)
            sensitivity_rows.append({"eta": eta, **summarize_schedule(schedule)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_csv(sensitivity, OUT / "question3_efficiency_sensitivity.csv")

    resolution_rows = []
    for step in [240.0, 120.0, 60.0]:
        parameters = StorageParameters(soc_step_kwh=step)
        schedule, _, _ = simulate_day(bundle, SPECIFIED_DATES[0], "rolling", parameters)
        resolution_rows.append({"soc_step_kwh": step, **summarize_schedule(schedule)})
    resolution = pd.DataFrame(resolution_rows)
    save_csv(resolution, OUT / "question3_resolution_sensitivity.csv")
    write_report(selected_summary, daily_summary, validation, sensitivity, resolution)

    inputs = [ROOT / "C题.pdf"] + sorted(DATA.glob("*.csv"))
    log = {
        "run_time_local": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "date_range": [days[0], days[-1]],
        "specified_dates": list(SPECIFIED_DATES),
        "forecast_update_hours": list(UPDATE_HOURS),
        "execution_block_hours": 6,
        "storage": {
            "eta_c": ETA_C, "eta_d": ETA_D, "soc_min_kwh": SOC_MIN, "soc_max_kwh": SOC_MAX,
            "max_power_kw": MAX_POWER_KW, "max_interval_energy_kwh": MAX_INTERVAL_ENERGY,
            "initial_soc_kwh": INITIAL_SOC_KWH, "terminal_soc_kwh": TERMINAL_SOC_KWH,
            "soc_step_kwh": DEFAULT_SOC_STEP_KWH,
        },
        "cost_rules": {
            "emergency_multiplier": EMERGENCY_PRICE_MULTIPLIER,
            "up_adjustment_multiplier": 1.5,
            "down_cancellation_penalty_multiplier": 0.5,
        },
        "price_source": "attachment1_standard_day.csv (PDF Question 3)",
        "excluded_from_question3_objective": "attachment4_price_long.csv (PDF Question 4 only)",
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    log["outputs"] = sorted(str(path.relative_to(ROOT)) for path in OUT.rglob("*") if path.is_file())
    (OUT / "reproducibility_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("第三问计算完成。")
    print(selected_summary[["date", "strategy", "total_grid_cost_yuan", "total_grid_purchase_kwh", "emergency_purchase_kwh"]].to_string(index=False))


if __name__ == "__main__":
    main()
