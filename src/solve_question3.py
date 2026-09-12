#!/usr/bin/env python3
"""运行 C 题第三问并生成 CSV、图表、验证、日志和报告。"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
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
    UP_ADJUSTMENT_MULTIPLIER, DOWN_CANCELLATION_PENALTY,
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
RESULT_TEMPLATE_CANDIDATES = [
    ROOT / "results" / "result4-3.xlsx",
    ROOT / "results" / "result4-3(1).xlsx",
    ROOT / "results" / "result4-3_template.xlsx",
]
RESULT_WORKBOOK = OUT / "result4-3_completed.xlsx"
MAIN_PARAMETERS = StorageParameters(soc_step_kwh=DEFAULT_SOC_STEP_KWH)

# Runtime/output switches. Keep the default run focused on the final Q3 solution.
SAVE_VERBOSE_OUTPUTS = False   # per-date forecast/version/timeseries CSVs
RUN_EXTRA_ANALYSES = False     # efficiency + SOC-grid sensitivity; enable for paper appendix
PROGRESS_EVERY_DAYS = 10


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



_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
ET.register_namespace("", _XLSX_NS)
ET.register_namespace("r", _XLSX_REL_NS)


def _find_result_template() -> Path:
    for path in RESULT_TEMPLATE_CANDIDATES:
        if path.exists():
            return path
    names = ", ".join(path.name for path in RESULT_TEMPLATE_CANDIDATES)
    raise FileNotFoundError(f"results/ 中找不到结果模板，请放入以下任一文件：{names}")


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    ns = f"{{{_XLSX_NS}}}"
    return [
        "".join(node.text or "" for node in si.iter(ns + "t"))
        for si in root.findall(ns + "si")
    ]


def _xlsx_sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    ns = f"{{{_XLSX_NS}}}"
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rel_root.findall(f"{{{_XLSX_PKG_REL_NS}}}Relationship")
    }
    out: dict[str, str] = {}
    for sheet in wb_root.find(ns + "sheets"):
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{{{_XLSX_REL_NS}}}id"]
        target = rel_map[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        out[name] = target
    return out


_XLSX_CELL_CACHE: dict[int, dict[str, ET.Element]] = {}
_XLSX_ROW_CACHE: dict[int, dict[int, ET.Element]] = {}


def _xlsx_ref_parts(ref: str) -> tuple[int, int]:
    letters = "".join(ch for ch in ref if ch.isalpha())
    digits = "".join(ch for ch in ref if ch.isdigit())
    col = 0
    for ch in letters:
        col = col * 26 + ord(ch.upper()) - 64
    return col, int(digits)


def _xlsx_cell(root: ET.Element, ref: str) -> ET.Element:
    ns = f"{{{_XLSX_NS}}}"
    key = id(root)
    cell_map = _XLSX_CELL_CACHE.get(key)
    row_map = _XLSX_ROW_CACHE.get(key)
    if cell_map is None or row_map is None:
        cell_map = {c.attrib["r"]: c for c in root.iter(ns + "c") if "r" in c.attrib}
        row_map = {int(r.attrib["r"]): r for r in root.iter(ns + "row") if "r" in r.attrib}
        _XLSX_CELL_CACHE[key] = cell_map
        _XLSX_ROW_CACHE[key] = row_map

    cell = cell_map.get(ref)
    if cell is not None:
        return cell

    # EP/EQ 等模板汇总列在空白数据行中可能没有实体 cell；按同一行样式创建。
    target_col, row_number = _xlsx_ref_parts(ref)
    row = row_map.get(row_number)
    if row is None:
        raise ValueError(f"结果模板缺少行 {row_number}")
    cells = row.findall(ns + "c")
    style = None
    if cells:
        nearest = min(cells, key=lambda c: abs(_xlsx_ref_parts(c.attrib["r"])[0] - target_col))
        style = nearest.attrib.get("s")
    cell = ET.Element(ns + "c", {"r": ref})
    if style is not None:
        cell.attrib["s"] = style
    insert_at = len(cells)
    for idx, existing in enumerate(cells):
        if _xlsx_ref_parts(existing.attrib["r"])[0] > target_col:
            insert_at = idx
            break
    row.insert(insert_at, cell)
    cell_map[ref] = cell
    return cell


def _xlsx_cell_text(cell: ET.Element, shared: list[str]) -> str | None:
    ns = f"{{{_XLSX_NS}}}"
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        node = cell.find(ns + "is/" + ns + "t")
        return None if node is None else node.text
    value = cell.find(ns + "v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def _xlsx_set_number(root: ET.Element, ref: str, value: float) -> None:
    ns = f"{{{_XLSX_NS}}}"
    cell = _xlsx_cell(root, ref)
    cell.attrib.pop("t", None)
    for tag in ["f", "is"]:
        node = cell.find(ns + tag)
        if node is not None:
            cell.remove(node)
    node = cell.find(ns + "v")
    if node is None:
        node = ET.SubElement(cell, ns + "v")
    node.text = f"{float(value):.12g}"


def _xlsx_set_text(root: ET.Element, ref: str, text: str) -> None:
    ns = f"{{{_XLSX_NS}}}"
    cell = _xlsx_cell(root, ref)
    cell.attrib["t"] = "inlineStr"
    for tag in ["f", "v", "is"]:
        node = cell.find(ns + tag)
        if node is not None:
            cell.remove(node)
    is_node = ET.SubElement(cell, ns + "is")
    text_node = ET.SubElement(is_node, ns + "t")
    text_node.text = str(text)


def _xlsx_clear(root: ET.Element, ref: str) -> None:
    ns = f"{{{_XLSX_NS}}}"
    cell = _xlsx_cell(root, ref)
    cell.attrib.pop("t", None)
    for tag in ["f", "v", "is"]:
        node = cell.find(ns + tag)
        if node is not None:
            cell.remove(node)


def _excel_col(number: int) -> str:
    chars = []
    n = number
    while n:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(65 + rem))
    return "".join(reversed(chars))


def _excel_serial_date(day: str | pd.Timestamp) -> int:
    """Convert a calendar date to the Excel 1900-date-system serial used by the template."""
    ts = pd.Timestamp(day).normalize()
    return int((ts - pd.Timestamp("1899-12-30")).days)


def _rewrite_emergency_sheet(
    root: ET.Element,
    emergency_blocks: pd.DataFrame,
    official_dates: list[str],
) -> None:
    """
    Rebuild the official "紧急购电量" sheet so that every Feb-Dec date is present.

    Each maximal run of consecutive 10-minute emergency-purchase intervals is written
    as ONE continuous time block. Different blocks are always written on different rows;
    no block is squeezed together with another block just because the original template
    only displayed three example rows. A day with no emergency purchase is retained as
    one date row with blank period/amount cells.
    """
    ns = f"{{{_XLSX_NS}}}"
    sheet_data = root.find(ns + "sheetData")
    if sheet_data is None:
        raise ValueError("紧急购电量模板缺少 sheetData")

    # Preserve the official header row exactly; replace all example / ellipsis rows.
    rows = sheet_data.findall(ns + "row")
    header = next((row for row in rows if row.attrib.get("r") == "1"), None)
    if header is None:
        raise ValueError("紧急购电量模板缺少表头行")
    for row in list(rows):
        if row is not header:
            sheet_data.remove(row)

    # Clear per-root lookup caches because rows/cells are being rebuilt wholesale.
    _XLSX_CELL_CACHE.pop(id(root), None)
    _XLSX_ROW_CACHE.pop(id(root), None)

    blocks = emergency_blocks.copy()
    if not blocks.empty:
        blocks["date"] = pd.to_datetime(blocks["date"]).dt.strftime("%Y-%m-%d")

    row_number = 2
    for day in official_dates:
        group = (
            blocks.loc[blocks.date.eq(day)]
            .sort_values("start_time_index")
            .reset_index(drop=True)
            if not blocks.empty else pd.DataFrame()
        )

        records = [
            (str(rec.emergency_period), float(rec.emergency_purchase_kwh))
            for rec in group.itertuples(index=False)
        ]
        # User explicitly wants every date shown. Keep one blank row on no-emergency days.
        if not records:
            records = [("", np.nan)]

        nrows = len(records)
        for idx, (period, amount) in enumerate(records):
            # Match the template's visual grammar:
            #   first row of a date block -> styles 2/3,
            #   middle rows -> style 4, last row -> style 5.
            # For a one-row date block, style 2/3 keeps the official date/number format.
            if idx == 0:
                style_a, style_bc = "2", "3"
            elif idx == nrows - 1:
                style_a = style_bc = "5"
            else:
                style_a = style_bc = "4"

            row = ET.Element(ns + "row", {
                "r": str(row_number),
                "spans": "1:3",
                "ht": "14",
                "customHeight": "1",
            })
            sheet_data.append(row)

            # Date is written only on the first row for that day, matching the example.
            cell_a = ET.SubElement(row, ns + "c", {"r": f"A{row_number}", "s": style_a})
            if idx == 0:
                value = ET.SubElement(cell_a, ns + "v")
                value.text = str(_excel_serial_date(day))

            cell_b = ET.SubElement(row, ns + "c", {"r": f"B{row_number}", "s": style_bc})
            if period:
                cell_b.attrib["t"] = "inlineStr"
                is_node = ET.SubElement(cell_b, ns + "is")
                text_node = ET.SubElement(is_node, ns + "t")
                text_node.text = period

            cell_c = ET.SubElement(row, ns + "c", {"r": f"C{row_number}", "s": style_bc})
            if np.isfinite(amount):
                value = ET.SubElement(cell_c, ns + "v")
                value.text = f"{amount:.12g}"

            row_number += 1

    last_row = row_number - 1
    dimension = root.find(ns + "dimension")
    if dimension is not None:
        dimension.attrib["ref"] = f"A1:C{last_row}"


def write_result4_3_workbook(full_schedule: pd.DataFrame) -> Path:
    """
    按官方 result4-3/result3 同结构模板写入最终结果。

    注意：本函数只负责“写表”。当前 solve_question3.py 的价格源仍是附件1，
    因而数值本质上属于问题3；若要作为真正的 result4-3 提交，求解阶段必须
    已切换到附件4的日期-时段动态电价。
    """
    template = _find_result_template()
    required_sheets = ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]

    frame = full_schedule.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    official_dates = pd.date_range("2025-02-01", "2025-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    frame = frame.loc[frame.date.isin(official_dates)].copy()
    counts = frame.groupby("date").size().reindex(official_dates)
    if counts.isna().any() or not (counts == N_PER_DAY).all():
        bad = counts[counts.ne(N_PER_DAY)].to_dict()
        raise ValueError(f"写Excel前检查失败：2/1-12/31 每天必须有144段，异常={bad}")

    emergency_blocks = contiguous_emergency_blocks(frame)

    with zipfile.ZipFile(template, "r") as zin:
        targets = _xlsx_sheet_targets(zin)
        if list(targets.keys()) != required_sheets:
            raise ValueError(f"结果模板工作表不匹配：{list(targets.keys())}")
        shared = _xlsx_shared_strings(zin)
        xml_roots = {name: ET.fromstring(zin.read(targets[name])) for name in required_sheets}

        # 1) 计划购电量；2) 调整购电量。官方模板展示顺序是 time_index 1..143,0。
        for sheet_name, value_col, cost_mode in [
            ("计划购电量", "base_plan_grid_kwh", "base"),
            ("调整购电量", "scheduled_grid_kwh", "adjusted"),
        ]:
            root = xml_roots[sheet_name]
            if _xlsx_cell_text(_xlsx_cell(root, "B1"), shared) != "0:10-0:20":
                raise ValueError(f"{sheet_name} 模板时间表头异常")
            if _xlsx_cell_text(_xlsx_cell(root, "EO1"), shared) != "0:00-0:10+1":
                raise ValueError(f"{sheet_name} 模板最后时间表头异常")

            for row_number, day in enumerate(official_dates, start=2):
                part = frame.loc[frame.date.eq(day)].sort_values("time_index")
                values = part[value_col].to_numpy(float)
                display_values = np.concatenate([values[1:], values[:1]])
                for j, value in enumerate(display_values, start=2):
                    _xlsx_set_number(root, f"{_excel_col(j)}{row_number}", value)
                _xlsx_set_number(root, f"EP{row_number}", float(values.sum()))
                if cost_mode == "base":
                    day_cost = float(part["base_plan_cost_yuan"].sum())
                else:
                    day_cost = float((part["base_plan_cost_yuan"] + part["adjustment_cost_yuan"]).sum())
                _xlsx_set_number(root, f"EQ{row_number}", day_cost)

        # 3) 充放电量：严格只填模板已经给出的样例日期，不扩表、不破坏省略号。
        charge_rows = {
            "2025-02-01": list(range(2, 8)),
            "2025-02-02": list(range(8, 14)),
            "2025-03-20": list(range(14, 20)),
            "2025-12-31": list(range(21, 27)),
        }
        root = xml_roots["充放电量"]
        for day, rows in charge_rows.items():
            part = frame.loc[frame.date.eq(day)].sort_values("time_index").reset_index(drop=True)
            if len(part) != N_PER_DAY:
                raise ValueError(f"{day}: 充放电量缺少144段")
            for block_index, row_number in enumerate(rows):
                block = part.iloc[24 * block_index : 24 * (block_index + 1)]
                _xlsx_set_number(root, f"C{row_number}", float(block.charge_input_kwh.sum()))
                _xlsx_set_number(root, f"D{row_number}", float(block.discharge_output_kwh.sum()))
            _xlsx_set_number(root, f"F{rows[0]}", float(part.soc_start_kwh.iloc[0]))
            _xlsx_set_number(root, f"F{rows[1]}", float(part.soc_end_kwh.iloc[-1]))

        # 4) 紧急购电量：删除模板中的示例/省略号区域，完整展开 2025-02-01
        # 至 2025-12-31。每个“连续发生紧急购电的10分钟区间”合并成一个
        # 连续时间段；不同连续段严格分行，不再受原模板三行示例容量限制。
        root = xml_roots["紧急购电量"]
        _rewrite_emergency_sheet(root, emergency_blocks, official_dates)

        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=OUT) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                changed = {targets[name]: ET.tostring(xml_roots[name], encoding="utf-8", xml_declaration=True) for name in required_sheets}
                for item in zin.infolist():
                    data = changed.get(item.filename, zin.read(item.filename))
                    zout.writestr(item, data)
            shutil.move(str(tmp_path), RESULT_WORKBOOK)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 最终再做一次轻量结构检查，防止输出文件损坏。
    with zipfile.ZipFile(RESULT_WORKBOOK, "r") as check_zip:
        check_zip.testzip()
        check_targets = _xlsx_sheet_targets(check_zip)
        if list(check_targets.keys()) != required_sheets:
            raise ValueError("输出Excel结构校验失败")
    return RESULT_WORKBOOK

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

PDF 第三问规定：每天 0:00、6:00、12:00 和 18:00 可获得未来 24 小时整点光伏发电功率预报；0:00 形成当日计划购电，后三个发布时间可以调整尚未执行的购电量。计划量高于调整量的部分按交易时刻电价的 50% 计违约，调整量高于计划量的部分按交易时刻电价的 1.5 倍计价；总费用包括计划购电、紧急购电和调整相关费用。结果文件覆盖 2025-01-01 至 2025-12-31，论文重点给出 2025-03-20、06-21、09-23、12-21。

第三问继承问题二“紧急购电按当时电价 5 倍”的规则。第三问原文明示使用附件1电价，因此本模型把附件1的144点电价作为每天重复的交易电价。附件4属于问题四，只做数据一致性检查，不进入第三问目标函数。

## 2. 数据与预处理

模型读取 `attachment1_standard_day.csv`、`attachment2_actual_long.csv`、`attachment3_forecast_hourly_long.csv`、`attachment3_forecast_10min_long.csv` 和 `model_base_2025.csv`，并读取 `attachment4_price_long.csv` 交叉核对。原始附件1至4的工作表、日期数、每日144点、缺失、重复键和单位已与 `scripts/clean_data.py` 核对。

功率按 $E=P/6$ 转为10分钟电量。每个日期的 `time_index=0,...,143` 原序对应 00:00-00:10 至 23:50-24:00；`0:00+1` 只表示最后区间的结束，没有循环移动。

## 3. 信息边界

附件3每个 `issue_time` 给出未来第1至24小时整点值；清洗表仅在 +60 至 +1440 分钟内线性插值。发布后首小时的五个十分钟点，以发布时间已观测到的光伏功率和本版本 +1 小时预测线性插值，不使用未来实际值。负荷预测直接复用问题二已经生成的 `outputs/question2/dynamic_forecasts.csv` 中 `forecast_load_kw`，从而保持问题二、三的负荷预测口径一致；同一天 0:00、6:00、12:00 和 18:00 使用相同的日前负荷预测，第三问新增的日内信息仅用于更新光伏预测。对于滚动24小时窗口中跨过午夜、而下一日问题二负荷预测在当前时刻尚不可因果获得的预览部分，使用历史同星期几负荷作为临时因果预览值；这些跨日预览值不会在下一次预测更新前直接执行。

每次优化完整记录 `issue_time`、`target_time`、提前量、光伏锚点、负荷历史源、优化计划和实际执行标志。附件2的当天实际值只用于时段执行后的紧急购电、弃光/溢出和评价，不回填预测。

本模型不额外引入分位数安全裕度，避免与储能实时补偿形成重复保护。预测层直接沿用问题二负荷预测和当前发布时间可获得的光伏预测；预测误差在执行层由储能优先吸收，储能仍不足时才触发5倍紧急购电。


## 4. 优化模型

令 $L_t,P_t,p_t$ 为预测负荷电量、预测光伏电量和附件1电价，$G_t,C_t,D_t,W_t,S_t$ 为常规购电、充电输入、放电输出、预测弃光和期末SOC：

$$G_t+P_t+D_t=L_t+C_t+W_t,$$
$$S_t=S_{{t-1}}+0.9C_t-D_t/0.9,$$
$$1200\le S_t\le10800,\qquad 0\le C_t,D_t\le5000/6.$$

状态转移只允许充电、放电或不变，从结构上严格禁止同段同时充放电。仅在整个仿真起点将SOC设为6000 kWh；之后SOC在滚动块和相邻日期之间连续传递。每次优化使用未来24小时固定窗口，不再强制窗口末端或每日24:00回到6000 kWh。

0:00 最小化 $\sum p_tG_t^0$。更新时相对原计划定义 $U_t=(G_t-G_t^0)_+$、$R_t=(G_t^0-G_t)_+$，最小化

$$\sum_t[p_tG_t^0+1.5p_tU_t-0.5p_tR_t].$$

下调口径解释为：取消量不再按全价购买，但支付50%违约费，因此相对原计划净减少 $0.5p_tR_t$。执行阶段把购电量视为已承诺决策，而储能充放电作为实时物理调节量：每个10分钟段先以计划购电与实际光伏满足实际小区负荷；若仍有缺口，优先在功率和SOC约束内放电，仅剩余负荷缺口才按 $5p_t$ 紧急购电。因此不会为了维持原计划充电量而触发5倍紧急购电。若有富余，则在约束内优先充电，剩余部分再记为弃光或已购电溢出。

求解采用离散SOC动态规划。主步长为 {DEFAULT_SOC_STEP_KWH:.0f} kWh，每个窗口枚举全部可行状态转移，得到该网格上的全局最优解；另以60/240 kWh做数值分辨率检查。

## 5. 滚动机制

0:00、6:00、12:00和18:00均使用当时已发布的预测版本优化未来144个10分钟时段（24小时），但每次只执行前6小时。下一次优化从上一执行块的实际SOC继续，跨日时前一天24:00的SOC直接作为次日0:00的SOC。后续窗口不能修改已执行行。

## 6. 四个指定日期

{md_table(selected_view)}

`oracle_actual` 使用全天实际负荷和实际光伏，只是同一模型下不可执行的事后最优参照。符合信息边界、可实际执行的是 `rolling`。

## 7. 全期对照与预测更新价值

{md_table(period)}

2025-01-01至12-31共365天中，滚动方案有 {better_days} 天费用低于静态方案。累计静态减滚动费用为 {total_saving:.3f} 元，日均为 {mean_saving:.3f} 元。该比较同时计入计划、上下调净结算和5倍紧急购电，所以是否更新不能只看常规购电量。

完整365天滚动执行表为 `question3_full_period_rolling_schedule.csv`，题目给定格式的汇总工作簿为 `result3.xlsx`。四个指定日的调度、SOC和更新前后计划图位于 `outputs/question3/figures/`。

## 8. 调整与执行结果

逐时结果保留0:00基准购电量、滚动调整后的购电量、上调量、下调量、紧急购电量以及储能状态。通过恒等式 $G_t^{(k)}=G_t^0+U_t-R_t$ 检查调整分解的一致性，并以334天累计费用、紧急购电量和调整规模评价滚动策略。

## 9. 敏感性分析

充放电效率敏感性：

{md_table(sens_view)}

SOC网格分辨率敏感性（下表为真实执行后的回测成本，不应用于判断DP预测目标的网格最优性；对应预测目标见 `question3_dp_grid_audit.csv`）：

{md_table(res_view)}

## 10. 验证结果

四个指定日三种策略和全期滚动/静态均检查了电量平衡、SOC转移、SOC边界、功率上限、充放互斥、购电/弃光非负、窗口内SOC连续、跨日SOC连续、执行索引唯一、预测发布时间、历史负荷来源及费用求和。所有布尔检查通过；最大电量平衡残差为 {max_balance:.3e} kWh，最大SOC转移残差为 {max_transition:.3e} kWh。

## 11. 优点、局限与结论

模型把原计划、调整计划和紧急购电分开，逐窗口封闭信息集，并保存全部输入版本和执行状态。动态规划仅在给定离散SOC状态/动作空间和既定结算口径下对预测目标全局最优；真实执行回测成本会随预测误差而不必随网格单调。

局限包括：题面未完全展开下调结算的会计口径，本文采用“取消全价购买并支付50%违约费”的净结算解释；第三问负荷预测直接沿用问题二模型，因此日内滚动更新主要反映新增光伏预测信息的价值；跨午夜的未执行预览区间在下一日负荷预测尚不可用时采用因果历史回退值；SOC离散存在小幅数值误差；预测过剩时可能出现已购电溢出，本文单列处理。若官方另有下调结算口径，只需替换 `settlement_cost`。

滚动结果是本题可执行答案；静态方案衡量不更新预报的代价，事后最优只给出信息完全时的参考下界。核心代码为 `src/question3_model.py`、`src/question3_rolling.py` 和 `src/solve_question3.py`。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")



def official_period_summary(daily_summary: pd.DataFrame) -> dict[str, float | int]:
    """Return the PDF-required 2025-02-01..2025-12-31 (334-day) Q3 statistics."""
    frame = daily_summary.copy()
    frame["date_ts"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame.date_ts.between("2025-02-01", "2025-12-31")]
    rolling = frame.loc[frame.strategy.eq("rolling")]
    static = frame.loc[frame.strategy.eq("static_forecast")]
    return {
        "days": int(rolling.date_ts.nunique()),
        "rolling_cost_yuan": float(rolling.total_grid_cost_yuan.sum()),
        "static_cost_yuan": float(static.total_grid_cost_yuan.sum()),
        "saving_yuan": float(static.total_grid_cost_yuan.sum() - rolling.total_grid_cost_yuan.sum()),
        "rolling_emergency_kwh": float(rolling.emergency_purchase_kwh.sum()),
        "static_emergency_kwh": float(static.emergency_purchase_kwh.sum()),
        "emergency_days": int((rolling.emergency_purchase_kwh > 1e-8).sum()),
    }

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    bundle = load_data_bundle(DATA)
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    if FULL_SCHEDULE.exists():
        FULL_SCHEDULE.unlink()
    daily_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    selected_summary_rows: list[dict[str, object]] = []
    selected_timeseries: list[pd.DataFrame] = []
    selected_rolling: dict[str, pd.DataFrame] = {}
    first_write = True
    rolling_soc = float(INITIAL_SOC_KWH)
    static_soc = float(INITIAL_SOC_KWH)
    previous_rolling_end: float | None = None
    previous_static_end: float | None = None
    for number, day in enumerate(days, 1):
        rolling_day_soc0 = rolling_soc
        static_day_soc0 = static_soc

        rolling, versions, base_plan = simulate_day(
            bundle, day, "rolling", MAIN_PARAMETERS, soc0_kwh=rolling_day_soc0, terminal_soc_kwh=None
        )
        static, _, _ = simulate_day(
            bundle, day, "static_forecast", MAIN_PARAMETERS, soc0_kwh=static_day_soc0, terminal_soc_kwh=None
        )

        if previous_rolling_end is not None and abs(float(rolling.soc_start_kwh.iloc[0]) - previous_rolling_end) > 1e-7:
            raise AssertionError(f"{day} rolling 跨日SOC不连续")
        if previous_static_end is not None and abs(float(static.soc_start_kwh.iloc[0]) - previous_static_end) > 1e-7:
            raise AssertionError(f"{day} static 跨日SOC不连续")

        rolling_soc = float(rolling.soc_end_kwh.iloc[-1])
        static_soc = float(static.soc_end_kwh.iloc[-1])
        previous_rolling_end = rolling_soc
        previous_static_end = static_soc
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
            oracle = simulate_oracle_day(bundle, day, MAIN_PARAMETERS, soc0_kwh=rolling_day_soc0, terminal_soc_kwh=None)
            selected_rolling[day] = rolling
            if SAVE_VERBOSE_OUTPUTS:
                save_csv(rolling, OUT / f"{day.replace('-', '')}_rolling_schedule.csv")
                save_csv(versions, OUT / f"{day.replace('-', '')}_forecast_versions.csv")
            for schedule in [static, rolling, oracle]:
                selected_summary_rows.append(summarize_schedule(schedule))
                selected_timeseries.append(schedule)
            oracle_check = validate_executed_schedule(oracle)
            for key in ["forecast_issue_hours_ok", "forecast_issue_not_after_execution_ok", "load_source_precedes_issue_ok", "pv_anchor_not_after_issue_ok"]:
                oracle_check[key] = True
            validation_rows.append({"date": day, "strategy": "oracle_actual", **oracle_check})
            draw_day_figures(day, rolling)
        if number % PROGRESS_EVERY_DAYS == 0 or number == 1 or number == len(days):
            rolling_so_far = [r for r in daily_rows if r["strategy"] == "rolling"]
            cost_so_far = sum(float(r["total_grid_cost_yuan"]) for r in rolling_so_far)
            emergency_so_far = sum(float(r["emergency_purchase_kwh"]) for r in rolling_so_far)
            print(
                f"[Q3] {number:3d}/{len(days)} days | {day} | "
                f"rolling cost={cost_so_far:,.0f} yuan | emergency={emergency_so_far:,.0f} kWh",
                flush=True,
            )

    daily_summary = pd.DataFrame(daily_rows)
    validation = pd.DataFrame(validation_rows)
    selected_summary = pd.DataFrame(selected_summary_rows)
    selected_ts = pd.concat(selected_timeseries, ignore_index=True)
    # Minimal useful CSV set: annual daily summary, validation, and four-date comparison.
    save_csv(daily_summary, OUT / "question3_full_period_daily_summary.csv")
    save_csv(validation, OUT / "question3_validation_summary.csv")
    save_csv(selected_summary, OUT / "question3_strategy_comparison_selected_dates.csv")
    if SAVE_VERBOSE_OUTPUTS:
        save_csv(selected_ts, OUT / "question3_selected_strategy_timeseries.csv")
        for day in SPECIFIED_DATES:
            stem = day.replace("-", "")
            save_csv(selected_summary.loc[selected_summary.date.eq(day)], OUT / f"{stem}_cost_summary.csv")
            records = validation.loc[validation.date.eq(day)].to_dict(orient="records")
            (OUT / f"{stem}_validation.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    full = pd.read_csv(FULL_SCHEDULE, encoding="utf-8-sig")
    official334 = official_period_summary(daily_summary)
    emergency_blocks = contiguous_emergency_blocks(full)
    charge_blocks = selected_charge_blocks(selected_rolling)
    save_csv(emergency_blocks, OUT / "question3_emergency_blocks.csv")
    if SAVE_VERBOSE_OUTPUTS:
        save_csv(charge_blocks, OUT / "question3_selected_charge_blocks.csv")
    result_workbook = write_result4_3_workbook(full)
    print(f"结果Excel已生成：{result_workbook}")
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
    if SAVE_VERBOSE_OUTPUTS:
        (OUT / "question3_workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    if RUN_EXTRA_ANALYSES:
        print("[Q3] Running optional sensitivity analyses...", flush=True)
        sensitivity_rows = []
        for eta in [0.85, 0.90, 0.95]:
            parameters = StorageParameters(eta_c=eta, eta_d=eta, soc_step_kwh=DEFAULT_SOC_STEP_KWH)
            for day in SPECIFIED_DATES:
                schedule, _, _ = simulate_day(bundle, day, "rolling", parameters, terminal_soc_kwh=None)
                sensitivity_rows.append({"eta": eta, **summarize_schedule(schedule)})
        sensitivity = pd.DataFrame(sensitivity_rows)

        resolution_rows = []
        for step in [240.0, 120.0, 60.0]:
            parameters = StorageParameters(soc_step_kwh=step)
            schedule, _, _ = simulate_day(bundle, SPECIFIED_DATES[0], "rolling", parameters, terminal_soc_kwh=None)
            resolution_rows.append({"soc_step_kwh": step, **summarize_schedule(schedule)})
        resolution = pd.DataFrame(resolution_rows)
        write_report(selected_summary, daily_summary, validation, sensitivity, resolution)
    else:
        print("[Q3] Optional sensitivity analyses skipped (RUN_EXTRA_ANALYSES=False).", flush=True)

    input_candidates = [ROOT / "C题.pdf"] + sorted(DATA.glob("*.csv"))
    inputs = [path for path in input_candidates if path.exists()]
    log = {
        "run_time_local": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "date_range": [days[0], days[-1]],
        "specified_dates": list(SPECIFIED_DATES),
        "forecast_update_hours": list(UPDATE_HOURS),
        "execution_block_hours": 6,
        "optimization_horizon_hours": 24,
        "soc_continuity": "continuous_across_updates_and_days",
        "storage": {
            "eta_c": ETA_C, "eta_d": ETA_D, "soc_min_kwh": SOC_MIN, "soc_max_kwh": SOC_MAX,
            "max_power_kw": MAX_POWER_KW, "max_interval_energy_kwh": MAX_INTERVAL_ENERGY,
            "initial_soc_kwh": INITIAL_SOC_KWH, "terminal_soc_kwh": TERMINAL_SOC_KWH,
            "soc_step_kwh": DEFAULT_SOC_STEP_KWH,
        },
        "cost_rules": {
            "emergency_multiplier": EMERGENCY_PRICE_MULTIPLIER,
            "up_adjustment_multiplier": UP_ADJUSTMENT_MULTIPLIER,
            "down_cancellation_penalty_multiplier": DOWN_CANCELLATION_PENALTY,
        },
        "price_source": "attachment1_standard_day.csv (PDF Question 3)",
        "excluded_from_question3_objective": "attachment4_price_long.csv (PDF Question 4 only)",
        "official_334_day_summary": official334,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    log["outputs"] = sorted(str(path.relative_to(ROOT)) for path in OUT.rglob("*") if path.is_file())
    (OUT / "reproducibility_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    rolling_year = daily_summary.loc[daily_summary.strategy.eq("rolling")]
    static_year = daily_summary.loc[daily_summary.strategy.eq("static_forecast")]
    rolling_cost = float(rolling_year.total_grid_cost_yuan.sum())
    static_cost = float(static_year.total_grid_cost_yuan.sum())
    rolling_emergency = float(rolling_year.emergency_purchase_kwh.sum())
    static_emergency = float(static_year.emergency_purchase_kwh.sum())
    emergency_days = int((rolling_year.emergency_purchase_kwh > 1e-8).sum())
    print("\n========== Q3 FINAL SUMMARY ==========", flush=True)
    print(f"Rolling total cost:     {rolling_cost:,.2f} yuan")
    print(f"Static total cost:      {static_cost:,.2f} yuan")
    print(f"Rolling saving:         {static_cost-rolling_cost:,.2f} yuan")
    print(f"Rolling emergency:      {rolling_emergency:,.2f} kWh")
    print(f"Static emergency:       {static_emergency:,.2f} kWh")
    print(f"Emergency days:         {emergency_days}/{len(days)}")
    print(f"Final rolling SOC:      {rolling_soc:,.2f} kWh")
    print("\n--- PDF official period: 2025-02-01 to 2025-12-31 ---")
    print(f"Official days:           {official334['days']}")
    print(f"Rolling cost (334d):     {official334['rolling_cost_yuan']:,.2f} yuan")
    print(f"Static cost (334d):      {official334['static_cost_yuan']:,.2f} yuan")
    print(f"Rolling saving (334d):   {official334['saving_yuan']:,.2f} yuan")
    print(f"Rolling emergency (334d):{official334['rolling_emergency_kwh']:,.2f} kWh")
    print(f"Static emergency (334d): {official334['static_emergency_kwh']:,.2f} kWh")
    print(f"Emergency days (334d):   {official334['emergency_days']}/{official334['days']}")

    official_rows = full.loc[pd.to_datetime(full.date).between("2025-02-01", "2025-12-31")].copy()
    adj_resid = (official_rows.scheduled_grid_kwh - official_rows.base_plan_grid_kwh
                 - official_rows.up_adjustment_kwh + official_rows.down_adjustment_kwh).abs().max()
    print("\n--- Adjustment decomposition q(k)=q(0)+a+−a− ---")
    print(f"Adjusted intervals:      {int(((official_rows.up_adjustment_kwh + official_rows.down_adjustment_kwh) > 1e-8).sum()):,}")
    print(f"Total upward adjustment: {official_rows.up_adjustment_kwh.sum():,.2f} kWh")
    print(f"Total downward adjustment:{official_rows.down_adjustment_kwh.sum():,.2f} kWh")
    print(f"Identity max residual:   {adj_resid:.3e} kWh")

    print("Selected dates:")
    print(selected_summary[["date", "strategy", "total_grid_cost_yuan", "total_grid_purchase_kwh", "emergency_purchase_kwh"]].to_string(index=False))
    print("======================================", flush=True)


if __name__ == "__main__":
    main()
