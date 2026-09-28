"""The Excel export: the field on screen, as a workbook with charts.

A CSV cannot hold a chart -- it is plain text. An .xlsx can hold the same table
*and* native charts, so this is the export for reading rather than scripting.
The CSV and GeoJSON exports stay as they are for code.

Three sheets:

  * Summary -- headline numbers, a pie of how the view splits across value
    bands, and a line of the same box across every downloaded day.
  * Data    -- every cell on screen: latitude, longitude, value.
  * About   -- where the numbers came from, and how each chart was made.

Design choices worth knowing:

  * Headline numbers and band counts are **formulas over the Data sheet**, so
    they stay right if someone filters or edits the table. XlsxWriter also
    stores each formula's result, so the *numbers* show even in previewers that
    never recalculate. The *charts* need a spreadsheet app -- Excel, Numbers,
    LibreOffice, Google Sheets. Quick previewers (macOS Quick Look among them)
    draw the cells and leave the charts out entirely; checked, not assumed.
  * The pie is **part-to-whole, at most six slices**, on a one-hue ordinal ramp
    -- the bands are ordered, so they read light to dark. A two-slice pie is a
    known anti-pattern; that is why chlorophyll's gap-filled share is a headline
    number here, not a chart.
  * The line is the job a line exists for: **change over time**. One series,
    one axis, no legend, and only the exported day's point is labelled.
  * The daily values are computed at export time from the same box, depth and
    stride, but from *other days* -- they are not in the Data sheet, and the
    About sheet says so.
"""

from __future__ import annotations

import io
import math
from datetime import datetime, timezone

import numpy as np
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

from app.services import ocean_model

# The validated ordinal ramp (dataviz reference palette, blue steps 250-650).
# Checked with validate_palette.js --ordinal against the chart surface: monotone
# lightness, visible adjacent steps, one hue, lightest step 2.06:1.
BAND_COLOURS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
LINE_COLOUR = "#2a78d6"          # categorical slot 1
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "Arial"

BAND_COUNT = 5
MAX_SLICES = 6
# Fields the app already draws on a log ramp: equal-width bands would put
# nearly every chlorophyll cell in the bottom slice, so bands follow the map.
LOG_BANDS = {"chl", "chl_model", "par"}

LONG_NAMES = {
    "sst": "Sea temperature", "salinity": "Salinity", "mld": "Mixed layer depth",
    "sla": "Sea level anomaly", "tchp": "Tropical cyclone heat potential",
    "o2": "Dissolved oxygen", "no3": "Nitrate", "chl": "Chlorophyll-a (satellite)",
    "po4": "Phosphate", "si": "Silicate", "chl_model": "Chlorophyll-a (model)",
    "zos": "Sea surface height", "d26": "Depth of the 26 °C isotherm",
    "d20": "Depth of the 20 °C isotherm", "wave": "Significant wave height",
    "par": "Photosynthetically active radiation (derived)",
}


# The source files use CF unit strings; a person reads the symbols the map shows.
DISPLAY_UNITS = {"degrees_C": "°C", "mmol/m3": "mmol/m³", "mg/m3": "mg/m³",
                 "kJ/cm2": "kJ/cm²", "1e-3": "PSU", "mol/m2/day": "mol photons/m²/day"}


def display_units(units: str) -> str:
    return DISPLAY_UNITS.get(units, units)


def describe_depth(actual_depth) -> str:
    if actual_depth is None:
        return "at the sea surface (a 2D field)"
    if actual_depth < 1:
        return f"at the model's top level ({actual_depth:.1f} m)"
    return f"at {actual_depth:,.0f} m"


def friendly_date(date: str) -> str:
    return datetime.strptime(date, "%Y-%m-%d").strftime("%-d %b %Y")


def _decimals_for(step: float) -> int:
    """Enough decimals to tell band edges apart, and no more."""
    if step <= 0:
        return 3
    return int(min(4, max(0, 2 - math.floor(math.log10(step)))))


def band_edges(values: np.ndarray, log_scale: bool) -> list[float]:
    """Edges of up to five ordered bands that together cover every value."""
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return [low, high]
    if log_scale and low > 0:
        raw = np.geomspace(low, high, BAND_COUNT + 1)
        steps = np.diff(raw)
        decimals = [_decimals_for(s) for s in steps]
        edges = [round(float(e), d) for e, d in zip(raw, decimals + decimals[-1:])]
    else:
        raw = np.linspace(low, high, BAND_COUNT + 1)
        decimals = _decimals_for((high - low) / BAND_COUNT)
        edges = [round(float(e), decimals) for e in raw]

    # Rounding must never push the extremes inside the edges, or a value would
    # fall outside every band and the slices would stop summing to the whole.
    places = _decimals_for((high - low) / BAND_COUNT)
    factor = 10 ** places
    edges[0] = math.floor(low * factor) / factor
    edges[-1] = math.ceil(high * factor) / factor
    # Collapse any edges rounding made equal, keeping the order.
    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    return deduped[: MAX_SLICES + 1]


def band_counts(values: np.ndarray, edges: list[float]) -> list[int]:
    """Counts per band: [a, b) for every band but the last, which is [a, b].

    Exactly the rule the COUNTIFS formulas in the sheet use, so the cached
    values and a recalculated workbook can never disagree.
    """
    counts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (values >= lo) & ((values <= hi) if last else (values < hi))
        counts.append(int(mask.sum()))
    return counts


def daily_series(variable: str, depth: float, stride: int, bbox) -> list[dict]:
    """The same box, depth and stride on every downloaded day."""
    rows = []
    for day in ocean_model.available_days(variable):
        points, _ = ocean_model.get_grid(variable, str(day), depth, stride=stride, bbox=bbox)
        values = np.array([p["value"] for p in points if p["value"] is not None], dtype=float)
        if values.size == 0:
            continue
        rows.append({
            "date": str(day), "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "cells": int(values.size),
        })
    return rows


def export_filename(variable: str, date: str, actual_depth) -> str:
    depth = "" if actual_depth is None else f"_{round(actual_depth)}m"
    return f"oceanao_{variable}_{date}{depth}.xlsx"


def build_workbook(variable: str, date: str, depth: float, stride: int, bbox) -> tuple[bytes, str, dict]:
    """(xlsx bytes, filename, summary) for the field exactly as the map fetched it."""
    points, actual_depth = ocean_model.get_grid(variable, date, depth, stride=stride, bbox=bbox)
    points = [p for p in points if p["value"] is not None]
    if not points:
        raise ValueError("no cells in view to export")

    values = np.array([p["value"] for p in points], dtype=float)
    units = ocean_model.VARIABLE_UNITS.get(variable, "")
    source, dataset_id = ocean_model.provenance(variable)
    long_name = LONG_NAMES.get(variable, variable)
    interpolated = (ocean_model.interpolated_fraction(variable, date, stride=stride, bbox=bbox)
                    if hasattr(ocean_model, "interpolated_fraction") else None)
    log_scale = variable in LOG_BANDS
    edges = band_edges(values, log_scale)
    counts = band_counts(values, edges)
    days = daily_series(variable, depth, stride, bbox)
    filename = export_filename(variable, date, actual_depth)

    buffer = io.BytesIO()
    wb = xlsxwriter.Workbook(buffer, {
        "in_memory": True,
        "default_format_properties": {"font_name": FONT, "font_size": 10},
    })
    wb.set_properties({
        "title": f"OCEANAO export - {long_name}, {date}",
        "subject": f"{variable} from {dataset_id or 'a derived field'}",
        "author": "OCEANAO",
        "comments": "Values are model, satellite or derived output as labelled on the About sheet.",
    })

    f = {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": INK}),
        "subtitle": wb.add_format({"font_size": 10, "font_color": INK_SECONDARY}),
        "kpi_label": wb.add_format({"font_size": 9, "font_color": INK_MUTED, "bottom": 1,
                                    "bottom_color": GRIDLINE}),
        "kpi": wb.add_format({"bold": True, "font_size": 16, "font_color": INK,
                              "num_format": "0.00", "align": "left"}),
        "kpi_int": wb.add_format({"bold": True, "font_size": 16, "font_color": INK,
                                  "num_format": "#,##0", "align": "left"}),
        "kpi_pct": wb.add_format({"bold": True, "font_size": 16, "font_color": INK,
                                  "num_format": "0.0%", "align": "left"}),
        "unit": wb.add_format({"font_size": 9, "font_color": INK_SECONDARY}),
        "header": wb.add_format({"bold": True, "font_color": INK, "bottom": 1,
                                 "bottom_color": AXIS, "bg_color": "#f4f3ef"}),
        "cell": wb.add_format({"font_color": INK}),
        "num": wb.add_format({"font_color": INK, "num_format": "0.000"}),
        "num2": wb.add_format({"font_color": INK, "num_format": "0.00"}),
        "int": wb.add_format({"font_color": INK, "num_format": "#,##0"}),
        "pct": wb.add_format({"font_color": INK, "num_format": "0.0%"}),
        "note": wb.add_format({"font_size": 9, "font_color": INK_SECONDARY, "text_wrap": True,
                               "valign": "top"}),
        "key": wb.add_format({"bold": True, "font_color": INK, "valign": "top"}),
        "value": wb.add_format({"font_color": INK, "text_wrap": True, "valign": "top"}),
        "section": wb.add_format({"bold": True, "font_size": 11, "font_color": INK}),
    }

    summary = wb.add_worksheet("Summary")
    data = wb.add_worksheet("Data")
    about = wb.add_worksheet("About")

    # ---------------------------------------------------------------- Data
    data.write_row(0, 0, ["latitude", "longitude", f"value ({units})"], f["header"])
    for i, p in enumerate(points, start=1):
        data.write_number(i, 0, p["lat"], f["num"])
        data.write_number(i, 1, p["lon"], f["num"])
        data.write_number(i, 2, p["value"], f["num"])
    last_row = len(points)                         # 0-based index of the last data row
    data.freeze_panes(1, 0)
    data.autofilter(0, 0, last_row, 2)
    data.set_column(0, 1, 12)
    data.set_column(2, 2, 16)
    value_range = f"Data!$C$2:$C${last_row + 1}"

    # ------------------------------------------------------------- Summary
    # The charts lead: they are the reason to open this file rather than the CSV.
    # Twelve even columns, B-G for the left chart and its table, H-M for the
    # right, so both charts sit side by side in the first screenful and each
    # table sits directly under the chart it backs.
    show_units = display_units(units)
    summary.hide_gridlines(2)
    summary.set_column(0, 0, 2)
    summary.set_column(1, 12, 11)
    summary.write(0, 1, f"{long_name} — {friendly_date(date)}", f["title"])
    summary.write(1, 1, f"{source.capitalize()} data {describe_depth(actual_depth)} · "
                        f"{len(points):,} cells · {show_units} · sources and method on the About sheet",
                  f["subtitle"])

    # Headline numbers: formulas over the Data sheet, results stored alongside,
    # each tile two columns wide with label, value and unit left-aligned.
    kpis = [
        ("Mean", f"=AVERAGE({value_range})", float(values.mean()), f["kpi"], show_units),
        ("Minimum", f"=MIN({value_range})", float(values.min()), f["kpi"], show_units),
        ("Maximum", f"=MAX({value_range})", float(values.max()), f["kpi"], show_units),
        ("Cells", f"=COUNT({value_range})", int(values.size), f["kpi_int"], "in view"),
    ]
    if interpolated is not None:
        # A single share of a whole is a number, not a two-slice pie.
        kpis.append(("Gap-filled", None, float(interpolated), f["kpi_pct"], "filled under cloud"))
    for i, (label, formula, cached, fmt, caption) in enumerate(kpis):
        c0, c1 = 1 + 2 * i, 2 + 2 * i
        summary.merge_range(3, c0, 3, c1, label, f["kpi_label"])
        if formula:
            summary.merge_range(4, c0, 4, c1, "", fmt)
            summary.write_formula(4, c0, formula, fmt, cached)
        else:
            summary.merge_range(4, c0, 4, c1, cached, fmt)
        summary.merge_range(5, c0, 5, c1, caption, f["unit"])

    # Chart titles live in cells above the charts, not inside them: an in-chart
    # title competes with the pie's outside labels for the top edge, and where
    # the big slice lands depends on the data -- chlorophyll's 32% label sat
    # directly on the title. A cell heading cannot collide with anything.
    summary.write(7, 1, "Share of the view by value band", f["section"])
    summary.write(7, 7, f"Mean {long_name.lower()} across the week", f["section"])
    # A spacer row between each heading and its chart: Numbers grows a pie's
    # frame upward to fit outside labels, and a big top-left slice pushed it
    # over the heading. The explicit plot area below is the other half.
    chart_row = 9
    # The charts are 320 px: about 16 rows in Excel, nearer 19 in Numbers, whose
    # rows render shorter. Leave room for the shorter rows, or a chart covers
    # the heading of the table beneath it.
    table_top = chart_row + 21

    # Band table -- the pie's table twin, under the pie.
    summary.write(table_top - 1, 1, "Value bands", f["section"])
    summary.write(table_top, 1, "From", f["header"])
    summary.write(table_top, 2, "To", f["header"])
    summary.merge_range(table_top, 3, table_top, 4, "Band", f["header"])
    summary.write(table_top, 5, "Cells", f["header"])
    summary.write(table_top, 6, "Share", f["header"])
    band_first = table_top + 1
    band_last = band_first + len(counts) - 1
    total = int(values.size)
    counts_range = f"$F${band_first + 1}:$F${band_last + 1}"
    for i, count in enumerate(counts):
        row = band_first + i
        lo, hi = edges[i], edges[i + 1]
        last = i == len(counts) - 1
        summary.write_number(row, 1, lo, f["num2"])
        summary.write_number(row, 2, hi, f["num2"])
        summary.merge_range(row, 3, row, 4, f"{lo:g}–{hi:g} {show_units}", f["cell"])
        lo_cell = xl_rowcol_to_cell(row, 1)
        hi_cell = xl_rowcol_to_cell(row, 2)
        summary.write_formula(
            row, 5,
            f'=COUNTIFS({value_range},">="&{lo_cell},{value_range},"{"<=" if last else "<"}"&{hi_cell})',
            f["int"], count)
        count_cell = xl_rowcol_to_cell(row, 5)
        summary.write_formula(
            row, 6, f"=IF(SUM({counts_range})=0,0,{count_cell}/SUM({counts_range}))",
            f["pct"], count / total if total else 0)
    band_note = (f"Bands are spaced on a log scale, as the map draws {long_name.lower()}."
                 if variable in LOG_BANDS else
                 "Five equal-width bands spanning this view's own minimum to maximum.")
    summary.merge_range(band_last + 1, 1, band_last + 2, 6,
                        band_note + " Counts are live formulas over the Data sheet.", f["note"])

    pie = wb.add_chart({"type": "pie"})
    pie.add_series({
        "name": "Share of cells",
        "categories": ["Summary", band_first, 3, band_last, 3],
        "values": ["Summary", band_first, 5, band_last, 5],
        # A surface-coloured border is the 2px gap between slices.
        "points": [{"fill": {"color": BAND_COLOURS[i % len(BAND_COLOURS)]},
                    "border": {"color": SURFACE, "width": 1.5}} for i in range(len(counts))],
        # A slice under 3% has no room for a label, and neighbouring tiny slices
        # stack their labels on top of each other. Drop those; the band table
        # directly beneath still carries every value.
        "data_labels": {"percentage": True, "position": "outside_end", "leader_lines": True,
                        "num_format": "0%",
                        "font": {"name": FONT, "size": 9, "color": INK_SECONDARY},
                        "custom": [({"delete": True} if total and counts[i] / total < 0.03 else None)
                                   for i in range(len(counts))]},
    })
    # Start the lowest band at three o'clock, away from the title at the top.
    pie.set_rotation(90)
    pie.set_title({"none": True})     # titled by the cell heading above it
    pie.set_legend({"position": "right", "font": {"name": FONT, "size": 9, "color": INK_SECONDARY}})
    pie.set_chartarea({"fill": {"color": SURFACE}, "border": {"none": True}})
    # Reserve a margin inside the frame for the outside labels, so no renderer
    # has to resize the chart to fit them.
    pie.set_plotarea({"layout": {"x": 0.08, "y": 0.1, "width": 0.52, "height": 0.8}})
    pie.set_size({"width": 490, "height": 320})
    summary.insert_chart(chart_row, 1, pie, {"y_offset": 4})

    # Daily table -- the line's table twin, under the line.
    summary.write(table_top - 1, 7, "Every downloaded day", f["section"])
    summary.write_row(table_top, 7, ["Date", "Mean", "Min", "Max", "Cells"], f["header"])
    for i, row in enumerate(days):
        r = table_top + 1 + i
        summary.write(r, 7, row["date"], f["cell"])
        summary.write_number(r, 8, row["mean"], f["num2"])
        summary.write_number(r, 9, row["min"], f["num2"])
        summary.write_number(r, 10, row["max"], f["num2"])
        summary.write_number(r, 11, row["cells"], f["int"])
    day_last = table_top + len(days)
    summary.merge_range(day_last + 1, 7, day_last + 3, 12,
                        "Computed at export time from the same box, depth and stride on each day. "
                        "Only the exported day's cells are also in the Data sheet.", f["note"])

    exported_index = next((i for i, row in enumerate(days) if row["date"] == date), None)
    line = wb.add_chart({"type": "line"})
    line.add_series({
        "name": f"Mean {long_name.lower()}",
        "categories": ["Summary", table_top + 1, 7, day_last, 7],
        "values": ["Summary", table_top + 1, 8, day_last, 8],
        "line": {"color": LINE_COLOUR, "width": 2.0},
        "marker": {"type": "circle", "size": 7, "fill": {"color": LINE_COLOUR},
                   "border": {"color": SURFACE, "width": 1.5}},
        # Label the exported day only: a number on every point goes unread.
        "data_labels": {
            "value": True, "num_format": "0.00",
            "font": {"name": FONT, "size": 9, "bold": True, "color": INK},
            "custom": [({"delete": True} if i != exported_index else None)
                       for i in range(len(days))],
        } if exported_index is not None else {},
    })
    line.set_title({"none": True})    # titled by the cell heading above it
    line.set_legend({"none": True})     # one series: the title names it
    line.set_x_axis({"num_font": {"name": FONT, "size": 9, "color": INK_MUTED},
                     "line": {"color": AXIS, "width": 0.75}})
    line.set_y_axis({"name": show_units, "name_font": {"name": FONT, "size": 9,
                                                       "color": INK_MUTED, "bold": False},
                     "num_font": {"name": FONT, "size": 9, "color": INK_MUTED},
                     "num_format": "0.00", "line": {"none": True},
                     "major_gridlines": {"visible": True, "line": {"color": GRIDLINE, "width": 0.75}}})
    line.set_chartarea({"fill": {"color": SURFACE}, "border": {"none": True}})
    line.set_plotarea({"fill": {"color": SURFACE}})
    line.set_size({"width": 490, "height": 320})
    summary.insert_chart(chart_row, 7, line, {"x_offset": 8})

    # --------------------------------------------------------------- About
    about.hide_gridlines(2)
    about.set_column(0, 0, 26)
    about.set_column(1, 1, 90)
    about.write(0, 0, "Where these numbers come from", f["title"])
    rows = [
        ("Field", f"{long_name} ({variable})"),
        ("Units", f"{display_units(units)} (written '{units}' in the source file)" if units else "none"),
        ("Source", {"model": "Model output (Copernicus Marine)",
                    "satellite": "Satellite observation (Copernicus Marine)",
                    "derived": "Derived — calculated by OCEANAO from model temperature, not downloaded"
                    }.get(source, source)),
        ("Dataset", dataset_id or "n/a"),
        ("Date", date),
        ("Depth", "not applicable (2D field)" if actual_depth is None
         else f"{actual_depth:.3f} m — the model level nearest the {depth:g} m requested"),
        ("Box", f"{bbox[0]}–{bbox[2]}°E, {bbox[1]}–{bbox[3]}°N" if bbox else "whole downloaded domain"),
        ("Grid", f"every {stride} cell(s) per axis — the same thinning the map draws"),
        ("Cells", f"{len(points):,}; land and cells with no value are not included"),
        ("Exported", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
    ]
    if interpolated is not None:
        rows.append(("Gap-filled", f"{interpolated:.1%} of these cells were interpolated under "
                                   f"cloud rather than observed (the product's own flags)."))
    if variable == "wave":
        rows.append(("Time", "An instantaneous snapshot at 12:00 UTC, not a daily mean — the wave "
                             "model is published every three hours."))
    rows += [
        ("Pie chart", band_note + " Counts and shares are formulas over the Data sheet."),
        ("Line chart", "Mean of the same box, depth and stride on each downloaded day, computed "
                       "at export time. Values for days other than the exported one are not in "
                       "the Data sheet."),
        ("Charts", "Open in Excel, Numbers, LibreOffice or Google Sheets to see the charts. "
                   "Quick previewers (such as macOS Quick Look) show the tables but not the "
                   "charts. Formula results are stored with the file, so the numbers show "
                   "everywhere, and a spreadsheet app recalculates them on open."),
    ]
    for i, (key, value) in enumerate(rows, start=2):
        about.write(i, 0, key, f["key"])
        about.write(i, 1, value, f["value"])

    wb.close()
    summary_info = {
        "cells": int(values.size), "mean": float(values.mean()), "edges": edges,
        "counts": counts, "days": days, "exported_index": exported_index,
        "interpolated": interpolated, "actual_depth": actual_depth,
    }
    return buffer.getvalue(), filename, summary_info
