#!/usr/bin/env python3
"""清洗2026年国赛C题附件数据，生成可用于预测与微网优化的标准长表。

用法：在仓库根目录运行 ``python scripts/clean_data.py``。
本脚本仅读取 data/ 下的原始 Excel，所有生成物写入 data/processed/ 和 outputs/。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
PROCESSED_DIR = RAW_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "data_quality_report.txt"
TEN_MINUTES_IN_DAY = 144
EXPECTED_DAYS_2025 = 365


class QualityReport:
    """收集检查信息，并同时在报告中给出通过/异常状态。"""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def check(self, name: str, passed: bool, detail: str) -> None:
        status = "通过" if passed else "异常"
        self.checks.append((name, passed, detail))
        self.lines.append(f"[{status}] {name}: {detail}")

    def save(self) -> None:
        REPORT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def display_value(value: Any) -> str:
    """让报告中的 numpy/pandas 标量和时间戳保持易读。"""
    if pd.isna(value):
        return "NA"
    return str(value)


def report_raw_sheet(path: Path, sheet_name: str, df: pd.DataFrame, report: QualityReport) -> None:
    """按题目要求记录每个原始工作表的结构和基础质量情况。"""
    report.add(f"\n文件：{path.name}；工作表：{sheet_name}")
    report.add(f"数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    report.add("列名：" + ", ".join(map(str, df.columns)))
    report.add("数据类型：" + "; ".join(f"{c}={dtype}" for c, dtype in df.dtypes.items()))
    report.add("缺失值数量：" + "; ".join(f"{c}={int(n)}" for c, n in df.isna().sum().items()))
    report.add(f"重复行数量：{int(df.duplicated().sum())}")
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        report.add("数值范围：无数值列")
    else:
        ranges = [f"{c}=[{display_value(numeric[c].min())}, {display_value(numeric[c].max())}]" for c in numeric]
        report.add("数值范围：" + "; ".join(ranges))


def parse_time_label(label: Any) -> tuple[str, pd.Timedelta]:
    """解析原始时间标签；0:00+1 明确表示原始日期的次日 00:00。"""
    text = str(label).strip()
    if text == "0:00+1":
        return text, pd.Timedelta(days=1)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"无法解析时间标签：{label!r}")
    return text, pd.Timedelta(hours=parsed.hour, minutes=parsed.minute, seconds=parsed.second)


def date_gaps(dates: pd.Series) -> list[str]:
    """返回给定日期范围中缺失的自然日，供质量报告使用。"""
    dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return []
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    existing = pd.DatetimeIndex(dates.unique())
    return [d.strftime("%Y-%m-%d") for d in full.difference(existing)]


def validate_wide_time_columns(df: pd.DataFrame, date_col: str, source: str, report: QualityReport) -> list[str]:
    """验证宽表的 144 个列标签及每个原始日期对应的时段数。"""
    time_cols = list(df.columns[1:])
    report.check(f"{source}时间列数量", len(time_cols) == TEN_MINUTES_IN_DAY, f"{len(time_cols)}（期望 144）")
    parsed: list[pd.Timedelta] = []
    for col in time_cols:
        _, offset = parse_time_label(col)
        parsed.append(offset)
    report.check(
        f"{source}时间标签唯一", len(set(map(str, time_cols))) == len(time_cols),
        f"重复标签数={len(time_cols) - len(set(map(str, time_cols)))}",
    )
    report.check(
        f"{source}时间标签严格递增", parsed == sorted(parsed),
        f"首列={time_cols[0]!s}；末列={time_cols[-1]!s}",
    )
    duplicate_dates = int(pd.to_datetime(df[date_col], errors="coerce").duplicated().sum())
    report.check(f"{source}重复原始日期", duplicate_dates == 0, f"重复数={duplicate_dates}")
    missing = date_gaps(df[date_col])
    report.check(f"{source}缺失日期", not missing, f"缺失数={len(missing)}；样例={missing[:5]}")
    # 宽表中每一行含同一组时间列，因此逐日时段数就是时间列数。
    bad_days = int((pd.Series(len(time_cols), index=df.index) != TEN_MINUTES_IN_DAY).sum())
    report.check(f"{source}每个原始日期144个10分钟时段", bad_days == 0, f"异常日期数={bad_days}")
    return time_cols


def numeric_column(series: pd.Series, name: str, report: QualityReport) -> pd.Series:
    """转换数值并报告原数据中非空但不能转换为数值的单元格数量。"""
    converted = pd.to_numeric(series, errors="coerce")
    invalid = int((series.notna() & converted.isna()).sum())
    report.check(f"{name}非数值数据", invalid == 0, f"非数值单元格数={invalid}")
    return converted


def wide_to_long(
    df: pd.DataFrame, value_name: str, source: str, report: QualityReport
) -> pd.DataFrame:
    """将“日期×10分钟列”的宽表规范化为日期、标签、索引和 datetime 长表。"""
    date_col = df.columns[0]
    time_cols = validate_wide_time_columns(df, date_col, source, report)
    long = df.melt(id_vars=[date_col], value_vars=time_cols, var_name="time_label", value_name=value_name)
    long = long.rename(columns={date_col: "date"})
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    label_to_index = {str(label): index for index, label in enumerate(time_cols)}
    label_to_offset = {str(label): parse_time_label(label)[1] for label in time_cols}
    long["time_label"] = long["time_label"].astype(str)
    long["time_index"] = long["time_label"].map(label_to_index).astype("int64")
    # 不能将 0:00+1 当作当天零点：其 datetime 必须跨到下一天。
    long["datetime"] = long["date"] + long["time_label"].map(label_to_offset)
    long[value_name] = numeric_column(long[value_name], f"{source}{value_name}", report)
    long = long[["date", "time_label", "time_index", "datetime", value_name]].sort_values("datetime").reset_index(drop=True)
    report.check(f"{source}长表重复datetime", not long["datetime"].duplicated().any(), f"重复数={int(long['datetime'].duplicated().sum())}")
    return long


def clean_attachment1(report: QualityReport) -> pd.DataFrame:
    path = RAW_DIR / "附件1.xlsx"
    df = pd.read_excel(path, sheet_name="Sheet1")
    report_raw_sheet(path, "Sheet1", df, report)
    required = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(df.columns) != required:
        raise ValueError(f"附件1列名与预期不一致：{list(df.columns)}")
    labels = df["时间"].astype(str).tolist()
    report.check("附件1时间点数量", len(labels) == TEN_MINUTES_IN_DAY, f"{len(labels)}（期望 144）")
    offsets = [parse_time_label(x)[1] for x in labels]
    report.check("附件1时间标签唯一", len(set(labels)) == len(labels), f"重复数={len(labels)-len(set(labels))}")
    report.check("附件1时间标签严格递增", offsets == sorted(offsets), f"首={labels[0]}；末={labels[-1]}")
    result = pd.DataFrame({"time_label": labels, "time_index": range(len(df))})
    result["price_yuan_per_kwh"] = numeric_column(df["电价"], "附件1电价", report)
    result["load_kw"] = numeric_column(df["小区负载"], "附件1负载", report)
    result["pv_forecast_kw"] = numeric_column(df["光伏发电预测功率"], "附件1光伏预测", report)
    result["load_kwh"] = result["load_kw"] / 6
    result["pv_forecast_kwh"] = result["pv_forecast_kw"] / 6
    report.check("附件1负负载", not (result["load_kw"] < 0).any(), f"数量={int((result['load_kw'] < 0).sum())}")
    report.check("附件1负光伏功率", not (result["pv_forecast_kw"] < 0).any(), f"数量={int((result['pv_forecast_kw'] < 0).sum())}")
    report.check("附件1负电价", not (result["price_yuan_per_kwh"] < 0).any(), f"数量={int((result['price_yuan_per_kwh'] < 0).sum())}")
    report.check("附件1kWh转换", (result["load_kwh"] == result["load_kw"] / 6).all() and (result["pv_forecast_kwh"] == result["pv_forecast_kw"] / 6).all(), "功率×10/60")
    return result


def clean_attachment2(report: QualityReport) -> pd.DataFrame:
    path = RAW_DIR / "附件2.xlsx"
    load_raw = pd.read_excel(path, sheet_name="小区负载")
    pv_raw = pd.read_excel(path, sheet_name="光伏发电实际功率")
    report_raw_sheet(path, "小区负载", load_raw, report)
    report_raw_sheet(path, "光伏发电实际功率", pv_raw, report)
    load = wide_to_long(load_raw, "load_kw", "附件2-负载", report)
    pv = wide_to_long(pv_raw, "pv_actual_kw", "附件2-实际光伏", report)
    merged = load.merge(pv, on=["date", "time_label", "time_index", "datetime"], how="outer", validate="one_to_one", indicator=True)
    report.check("附件2负载和光伏合并完整", (merged["_merge"] == "both").all(), f"非双方匹配数={int((merged['_merge'] != 'both').sum())}")
    merged = merged.drop(columns="_merge").sort_values("datetime").reset_index(drop=True)
    merged["load_kwh"] = merged["load_kw"] / 6
    merged["pv_actual_kwh"] = merged["pv_actual_kw"] / 6
    report.check("附件2实际数据行数", len(merged) == EXPECTED_DAYS_2025 * TEN_MINUTES_IN_DAY, f"{len(merged)}（期望 52560）")
    report.check("附件2重复datetime", not merged["datetime"].duplicated().any(), f"重复数={int(merged['datetime'].duplicated().sum())}")
    report.check("附件2负负载", not (merged["load_kw"] < 0).any(), f"数量={int((merged['load_kw'] < 0).sum())}")
    report.check("附件2负光伏功率", not (merged["pv_actual_kw"] < 0).any(), f"数量={int((merged['pv_actual_kw'] < 0).sum())}")
    report.check("附件2kWh转换", ((merged["load_kwh"] - merged["load_kw"] / 6).abs() < 1e-12).all() and ((merged["pv_actual_kwh"] - merged["pv_actual_kw"] / 6).abs() < 1e-12).all(), "功率×10/60")
    return merged


def clean_attachment3(report: QualityReport) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RAW_DIR / "附件3.xlsx"
    raw = pd.read_excel(path, sheet_name="Sheet1")
    report_raw_sheet(path, "Sheet1", raw, report)
    expected_cols = ["日期", "预报时刻"] + [f"预报{i}小时" for i in range(1, 25)]
    if list(raw.columns) != expected_cols:
        raise ValueError("附件3列名或预报小时范围与预期不一致，请先检查原始模板。")
    blank_dates = int(raw["日期"].isna().sum())
    # 原表按视觉分组保留空白日期；向下填充后每一行都获得所属日期。
    raw["日期"] = raw["日期"].ffill()
    report.add(f"附件3日期列向下填充：原始空白单元格={blank_dates}，未填充缺失={int(raw['日期'].isna().sum())}")
    raw["date"] = pd.to_datetime(raw["日期"], errors="coerce").dt.normalize()
    date_row_counts = raw.groupby("date").size()
    report.check("附件3日期每个日期4行（重复日期为预报版本结构）", (date_row_counts == 4).all(), f"异常日期数={int((date_row_counts != 4).sum())}")
    missing_dates = date_gaps(raw["date"])
    report.check("附件3缺失日期", not missing_dates, f"缺失数={len(missing_dates)}；样例={missing_dates[:5]}")
    raw["forecast_release_time"] = raw["预报时刻"].astype(str)
    release_offsets = raw["forecast_release_time"].map(lambda value: parse_time_label(value)[1])
    raw["issue_time"] = raw["date"] + release_offsets
    horizons = [f"预报{i}小时" for i in range(1, 25)]
    hourly = raw.melt(id_vars=["date", "forecast_release_time", "issue_time"], value_vars=horizons, var_name="horizon_label", value_name="pv_forecast_kw")
    hourly["forecast_horizon_hour"] = hourly["horizon_label"].str.extract(r"预报(\d+)小时").astype("int64")
    hourly["target_time"] = hourly["issue_time"] + pd.to_timedelta(hourly["forecast_horizon_hour"], unit="h")
    hourly["pv_forecast_kw"] = numeric_column(hourly["pv_forecast_kw"], "附件3光伏预测", report)
    hourly["pv_forecast_kwh_1h"] = hourly["pv_forecast_kw"]
    hourly = hourly[["date", "forecast_release_time", "issue_time", "forecast_horizon_hour", "target_time", "pv_forecast_kw", "pv_forecast_kwh_1h"]].sort_values(["issue_time", "target_time"]).reset_index(drop=True)
    daily_releases = hourly.groupby("date")["issue_time"].nunique()
    report.check("附件3每天4次光伏预报", (daily_releases == 4).all(), f"异常日期数={int((daily_releases != 4).sum())}")
    horizons_per_issue = hourly.groupby("issue_time").size()
    report.check("附件3每次预报24个小时", (horizons_per_issue == 24).all(), f"异常发布时间数={int((horizons_per_issue != 24).sum())}")
    report.check("附件3小时预测行数", len(hourly) == EXPECTED_DAYS_2025 * 4 * 24, f"{len(hourly)}（期望 35040）")
    duplicate_version = int(hourly.duplicated(["issue_time", "target_time"]).sum())
    report.check("附件3重复issue_time+target_time", duplicate_version == 0, f"重复数={duplicate_version}")
    report.check("附件3负光伏功率", not (hourly["pv_forecast_kw"] < 0).any(), f"数量={int((hourly['pv_forecast_kw'] < 0).sum())}")

    ten_minute_parts: list[pd.DataFrame] = []
    for issue_time, group in hourly.groupby("issue_time", sort=False):
        series = group.set_index("target_time")["pv_forecast_kw"].sort_index()
        # 原始预测从“预报1小时”开始，没有第0小时值。故只在[+60min, +1440min]
        # 内做线性时间插值；绝不对发布后首小时进行外推或虚构预测值。
        target_index = pd.date_range(series.index.min(), series.index.max(), freq="10min")
        interpolated = series.reindex(target_index).interpolate(method="time").clip(lower=0)
        # 兼容较早 pandas：Series.reset_index 尚不支持 names= 参数。
        part = interpolated.rename("pv_forecast_kw").reset_index().rename(columns={"index": "target_time"})
        part.insert(0, "issue_time", issue_time)
        part["forecast_horizon_minutes"] = ((part["target_time"] - part["issue_time"]).dt.total_seconds() / 60).round().astype("int64")
        part["pv_forecast_kwh"] = part["pv_forecast_kw"] / 6
        ten_minute_parts.append(part)
    ten_minute = pd.concat(ten_minute_parts, ignore_index=True)
    ten_minute = ten_minute[["issue_time", "target_time", "forecast_horizon_minutes", "pv_forecast_kw", "pv_forecast_kwh"]].sort_values(["issue_time", "target_time"]).reset_index(drop=True)
    report.add("附件3十分钟插值首小时处理：源数据不存在预报0小时；每个版本仅输出发布后60至1440分钟（共139个10分钟点），首小时不外推、不编造。")
    report.check("附件3十分钟预测无负功率", not (ten_minute["pv_forecast_kw"] < 0).any(), f"数量={int((ten_minute['pv_forecast_kw'] < 0).sum())}")
    report.check("附件3十分钟kWh转换", ((ten_minute["pv_forecast_kwh"] - ten_minute["pv_forecast_kw"] / 6).abs() < 1e-12).all(), "功率×10/60")
    return hourly, ten_minute


def clean_attachment4(report: QualityReport) -> pd.DataFrame:
    path = RAW_DIR / "附件4.xlsx"
    raw = pd.read_excel(path, sheet_name="Sheet1")
    report_raw_sheet(path, "Sheet1", raw, report)
    price = wide_to_long(raw, "price_yuan_per_kwh", "附件4-电价", report)
    report.check("附件4电价数据行数", len(price) == EXPECTED_DAYS_2025 * TEN_MINUTES_IN_DAY, f"{len(price)}（期望 52560）")
    report.check("附件4重复datetime", not price["datetime"].duplicated().any(), f"重复数={int(price['datetime'].duplicated().sum())}")
    report.check("附件4负电价", not (price["price_yuan_per_kwh"] < 0).any(), f"数量={int((price['price_yuan_per_kwh'] < 0).sum())}")
    return price


def report_missing_and_ranges(name: str, df: pd.DataFrame, report: QualityReport) -> None:
    """记录成品表的缺失值、重复行和数值范围。"""
    missing = int(df.isna().sum().sum())
    report.check(f"{name}缺失值", missing == 0, f"总数={missing}；逐列={df.isna().sum().to_dict()}")
    report.add(f"{name}重复行数量：{int(df.duplicated().sum())}")
    ranges = {col: [display_value(df[col].min()), display_value(df[col].max())] for col in df.select_dtypes(include="number")}
    report.add(f"{name}数值范围：{ranges}")


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    return path


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = QualityReport()
    report.add("2026年全国大学生数学建模竞赛C题：数据质量与清洗报告")
    report.add("原始附件未被修改；本报告由 scripts/clean_data.py 自动生成。")

    attachment1 = clean_attachment1(report)
    attachment2 = clean_attachment2(report)
    attachment3_hourly, attachment3_10min = clean_attachment3(report)
    attachment4 = clean_attachment4(report)

    # 使用明确的原始日期+时段序号合并，避免跨日的 0:00+1 语义被错误折叠。
    model_base = attachment2.merge(
        attachment4[["date", "time_label", "time_index", "datetime", "price_yuan_per_kwh"]],
        on=["date", "time_label", "time_index", "datetime"], how="left", validate="one_to_one", indicator=True,
    )
    report.check("基础表合并动态电价完整", (model_base["_merge"] == "both").all(), f"未匹配数={int((model_base['_merge'] != 'both').sum())}")
    model_base = model_base.drop(columns="_merge")
    report.check("基础表合并前后行数一致", len(model_base) == len(attachment2), f"合并前={len(attachment2)}；合并后={len(model_base)}")
    model_base = model_base[["date", "time_label", "time_index", "datetime", "load_kw", "load_kwh", "pv_actual_kw", "pv_actual_kwh", "price_yuan_per_kwh"]]

    outputs = {
        "attachment1_standard_day.csv": attachment1,
        "attachment2_actual_long.csv": attachment2,
        "attachment3_forecast_hourly_long.csv": attachment3_hourly,
        "attachment3_forecast_10min_long.csv": attachment3_10min,
        "attachment4_price_long.csv": attachment4,
        "model_base_2025.csv": model_base,
    }
    report.add("\n成品表检查：")
    for filename, frame in outputs.items():
        report_missing_and_ranges(filename, frame, report)
    paths = {filename: save_csv(frame, filename) for filename, frame in outputs.items()}
    report.add("\n验收汇总：")
    for name, passed, detail in report.checks:
        report.add(f"{'通过' if passed else '异常'} | {name} | {detail}")
    report.save()

    print("数据清洗完成。生成文件：")
    for filename, path in paths.items():
        print(f"- {path.relative_to(ROOT)}：{outputs[filename].shape[0]} 行 × {outputs[filename].shape[1]} 列")
    print(f"- {REPORT_PATH.relative_to(ROOT)}")
    failures = [name for name, passed, _ in report.checks if not passed]
    print("质量检查结果：" + ("全部通过" if not failures else "发现异常：" + "；".join(failures)))


if __name__ == "__main__":
    main()
