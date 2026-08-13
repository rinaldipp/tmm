"""
Spreadsheet and CSV export helpers for :class:`tmm.tmm.TMM`.

This module keeps the file-format and Excel-chart details out of ``tmm.py``.
The public API remains ``TMM.save2sheet()``; functions here expect a computed
``TMM`` instance and use its method-aware layer-report helpers for metadata.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter


CHART_X_SCALE = 1.55
CHART_Y_SCALE = 1.18
PLOT_LAYOUT = {"x": 0.11, "y": 0.12, "width": 0.78, "height": 0.56}
X_AXIS_TITLE_LAYOUT = {"x": 0.43, "y": 0.82}
ABS_LEGEND_LAYOUT = {"x": 0.39, "y": 0.89, "width": 0.22, "height": 0.06}
Z_LEGEND_LAYOUT = {"x": 0.28, "y": 0.89, "width": 0.44, "height": 0.06}


def save2sheet(
    treatment,
    timestamp=False,
    conversion=None,
    ext=".xlsx",
    chart_styles=None,
    n_oct=3,
    metadata=True,
    export_all=False,
):
    """
    Export selected TMM results to XLSX or CSV.

    XLSX exports contain ``Data``, ``Bands`` and ``Setup`` sheets.  The
    ``Setup`` sheet stores the TMM setup and method-aware layer report, so the
    workbook is self-contained.  CSV exports keep numeric data and metadata in
    separate files: the selected-method data are written to ``.csv`` and, when
    ``metadata=True``, the setup/layer report is written to
    ``*_metadata.csv``.

    When ``export_all=True`` the CSV export is diagnostic rather than
    selected-method-only: it contains angle-wise impedance and absorption,
    field-incidence diffuse impedance/absorption, and Paris diffuse absorption.
    ``export_all`` is CSV-only.
    """
    if chart_styles is None:
        chart_styles = [35, 36]
    if conversion is None:
        conversion = [0.0393701, "[inches]"]

    output_dir = _treatments_folder(treatment)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M_') if timestamp else ''}{treatment.filename}"

    if export_all:
        if ext != ".csv":
            raise ValueError("export_all=True is only available for ext='.csv'.")
        paths = _write_export_all_csv(treatment, output_dir, stem, metadata=metadata, conversion=conversion)
    elif ext == ".xlsx":
        paths = _write_xlsx(
            treatment,
            output_dir,
            stem,
            chart_styles=chart_styles,
            n_oct=n_oct,
            metadata=metadata,
            conversion=conversion,
        )
    elif ext == ".csv":
        paths = _write_selected_csv(treatment, output_dir, stem, n_oct=n_oct, metadata=metadata, conversion=conversion)
    else:
        raise NameError("Unidentified extension. Available extensions: ['.xlsx', '.csv']")

    print("Sheet saved to ", paths.get(ext.lstrip("."), next(iter(paths.values()))))
    return paths


def _treatments_folder(treatment):
    return Path(treatment.project_folder) / "Treatments"


def _backing_data(treatment):
    for key in reversed(list(treatment.matrix.keys())):
        value = treatment.matrix[key]
        if isinstance(value, dict) and value.get("type") == "backing":
            return value
    return {}


def _selected_export_labels_and_note(treatment):
    if treatment.incidence == "diffuse" and treatment.diffuse_method == "paris":
        return (
            None,
            None,
            "diffuse_method='paris': absorption is the angular average of alpha(theta). "
            "This selected-method export intentionally omits impedance columns because the Paris "
            "formula does not define a unique diffuse complex impedance.",
        )
    if treatment.incidence == "diffuse":
        return (
            "Real Z_norm field average",
            "Imag Z_norm field average",
            "diffuse_method='field': absorption is computed from the field-admittance averaged impedance.",
        )
    if treatment.incidence == "angle":
        angle = float(treatment.incidence_angle[0])
        return (
            f"Real Z_norm at {angle:g} deg",
            f"Imag Z_norm at {angle:g} deg",
            f"Single-angle incidence at {angle:g} deg.",
        )
    return (
        "Real Z_norm normal",
        "Imag Z_norm normal",
        "Normal incidence.",
    )


def _setup_rows(treatment, conversion):
    backing = _backing_data(treatment)
    raw_angle = getattr(treatment, "_incidence_angle", None)
    effective_angles = treatment.incidence_angle
    rows = [
        ("TMM setup", "", ""),
        ("filename", treatment.filename, ""),
    ]
    if treatment.display_name is not None:
        rows.append(("display_name", treatment.display_name, ""))
    if treatment.color is not None:
        rows.append(("color", treatment.color, ""))
    rows.extend(
        [
            ("fmin [Hz]", treatment.fmin, ""),
            ("fmax [Hz]", treatment.fmax, ""),
            ("df [Hz]", treatment.df, ""),
            ("x_scale", treatment.x_scale, ""),
            ("frequency lines", len(treatment.freq), ""),
            ("frequency span [Hz]", f"{treatment.freq[0]:g} - {treatment.freq[-1]:g}", ""),
            ("incidence", treatment.incidence, ""),
            ("incidence_angle input", raw_angle, ""),
            ("effective angle span [deg]", f"{np.min(effective_angles):g} - {np.max(effective_angles):g}", ""),
            ("effective angle lines", len(effective_angles), ""),
            ("diffuse_method", treatment.diffuse_method, ""),
            ("s0 reference area [m2]", treatment.s0, ""),
            ("srad termination/radiation area [m2]", treatment.srad, ""),
            ("backing", backing.get("backing", ""), ""),
            ("rigid_backing", backing.get("rigid_backing", ""), ""),
            ("impedance_conjugate", backing.get("impedance_conjugate", ""), ""),
        ]
    )
    if backing.get("backing") == "radiation":
        radius = np.sqrt(treatment.srad / np.pi)
        ka = treatment.k0 * radius
        rows.extend(
            [
                ("radiation radius [m]", radius, ""),
                ("radiation ka span", f"{np.min(ka):.6g} - {np.max(ka):.6g}", ""),
            ]
        )
    z_real, z_imag, note = _selected_export_labels_and_note(treatment)
    rows.extend(
        [
            ("exported impedance real column", z_real or "omitted", ""),
            ("exported impedance imag column", z_imag or "omitted", ""),
            ("export note", note, ""),
            ("", "", ""),
            ("Layer report", "", ""),
        ]
    )
    total_depth = 0.0
    for layer_index, layer_key in enumerate(treatment._report_layer_keys(), start=1):
        rows.append((f"Layer {layer_index}", "", ""))
        for key, value in treatment._layer_report_items(treatment.matrix[layer_key]):
            rows.append((key, value, ""))
            if treatment._is_report_number(value) and "thickness" in key:
                total_depth += float(value)
                converted_key = key.replace("[mm]", conversion[1])
                rows.append((converted_key, float(value) * conversion[0], ""))
    rows.extend(
        [
            ("", "", ""),
            ("Total", "", ""),
            ("total treatment depth [mm]", total_depth, ""),
            (f"total treatment depth {conversion[1]}", total_depth * conversion[0], ""),
        ]
    )
    return rows


def _selected_data_frame(treatment):
    z_real, z_imag, _ = _selected_export_labels_and_note(treatment)
    data = {"Frequency [Hz]": treatment.freq}
    if z_real is not None and z_imag is not None:
        data[z_real] = np.real(treatment.z_norm)
        data[z_imag] = np.imag(treatment.z_norm)
    data["Absorption [-]"] = treatment.alpha
    return pd.DataFrame(data)


def _bands_frame(treatment, n_oct):
    bands, alpha = treatment.filter_alpha(n_oct=n_oct, view=False)
    return pd.DataFrame(
        {
            f"1/{n_oct} octave band [Hz]": bands,
            f"1/{n_oct} octave absorption [-]": alpha,
        }
    )


def _write_selected_csv(treatment, output_dir, stem, n_oct, metadata, conversion):
    csv_path = output_dir / f"{stem}.csv"
    metadata_csv_path = output_dir / f"{stem}_metadata.csv"
    data = pd.concat([_selected_data_frame(treatment), _bands_frame(treatment, n_oct)], axis=1)
    data.to_csv(csv_path, index=False, float_format="%.6g", sep=";", encoding="utf-8-sig")
    paths = {"csv": csv_path}
    if metadata:
        _write_metadata_csv(metadata_csv_path, _setup_rows(treatment, conversion))
        paths["metadata_csv"] = metadata_csv_path
    return paths


def _write_metadata_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Field", "Value", "Notes"])
        writer.writerows(rows)


def _write_xlsx(treatment, output_dir, stem, chart_styles, n_oct, metadata, conversion):
    xlsx_path = output_dir / f"{stem}.xlsx"
    metadata_csv_path = output_dir / f"{stem}_metadata.csv"

    df_data = _selected_data_frame(treatment)
    df_bands = _bands_frame(treatment, n_oct)
    setup_rows = _setup_rows(treatment, conversion)

    workbook = xlsxwriter.Workbook(xlsx_path)
    data_sheet = workbook.add_worksheet("Data")
    bands_sheet = workbook.add_worksheet("Bands")
    setup_sheet = workbook.add_worksheet("Setup")

    formats = _workbook_formats(workbook)
    _write_frame(data_sheet, df_data, formats["bold"], formats["regular"])
    _write_frame(bands_sheet, df_bands, formats["bold"], formats["regular"])
    _write_setup_sheet(setup_sheet, setup_rows, formats)
    _insert_charts(workbook, data_sheet, treatment, df_data, chart_styles)

    data_sheet.set_column("A:D", 28)
    bands_sheet.set_column("A:B", 30)
    setup_sheet.set_column("A:A", 38)
    setup_sheet.set_column("B:C", 72)
    workbook.close()

    paths = {"xlsx": xlsx_path}
    if metadata:
        _write_metadata_csv(metadata_csv_path, setup_rows)
        paths["metadata_csv"] = metadata_csv_path
    return paths


def _workbook_formats(workbook):
    return {
        "bold": workbook.add_format({"bold": True, "font_color": "black", "align": "center", "border": 2}),
        "regular": workbook.add_format({"bold": False, "font_color": "black", "align": "center", "border": 1}),
        "regular_left": workbook.add_format({"bold": False, "font_color": "black", "align": "left", "border": 1}),
        "regular_left_bold": workbook.add_format(
            {"bold": True, "font_color": "black", "align": "right", "border": 1}
        ),
    }


def _write_frame(worksheet, frame, header_format, data_format):
    for col, column_name in enumerate(frame.columns):
        worksheet.write(0, col, column_name, header_format)
        for row, value in enumerate(frame[column_name], start=1):
            worksheet.write(row, col, value, data_format)


def _write_setup_sheet(worksheet, rows, formats):
    for row, (field, value, notes) in enumerate(rows):
        fmt = formats["bold"] if value == "" and notes == "" else formats["regular_left_bold"]
        worksheet.write(row, 0, field, fmt)
        worksheet.write(row, 1, str(value), formats["regular_left"])
        worksheet.write(row, 2, notes, formats["regular_left"])


def _insert_charts(workbook, data_sheet, treatment, df_data, chart_styles):
    alpha_col = int(df_data.columns.get_loc("Absorption [-]"))
    chart_abs = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    chart_abs.add_series(
        {
            "name": ["Data", 0, alpha_col],
            "categories": ["Data", 1, 0, len(df_data), 0],
            "values": ["Data", 1, alpha_col, len(df_data), alpha_col],
            "line": {"width": 1.25},
        }
    )
    chart_abs.set_title({"name": "Absorption Coefficient"})
    chart_abs.set_x_axis(_chart_x_axis_options(treatment))
    chart_abs.set_y_axis(_chart_y_axis_options("Alpha [-]", df_data["Absorption [-]"].to_numpy(), ymax=1.1))
    _apply_chart_surface(chart_abs)
    chart_abs.set_legend({"position": "bottom", "layout": ABS_LEGEND_LAYOUT})
    chart_abs.set_style(chart_styles[0])
    data_sheet.insert_chart("F2", chart_abs, {"x_scale": CHART_X_SCALE, "y_scale": CHART_Y_SCALE})
    _insert_rotated_log_x_textboxes(data_sheet, treatment, 1, 5)

    z_columns = [col for col in df_data.columns if col.startswith(("Real Z_norm", "Imag Z_norm"))]
    if not z_columns:
        return

    chart_z = workbook.add_chart({"type": "scatter", "subtype": "straight"})
    for column_name in z_columns:
        col = int(df_data.columns.get_loc(column_name))
        line = {"width": 1.25}
        if column_name.startswith("Imag Z_norm"):
            line["color"] = "#70AD47"
        elif column_name.startswith("Real Z_norm"):
            line["color"] = "#C00000"
        chart_z.add_series(
            {
                "name": ["Data", 0, col],
                "categories": ["Data", 1, 0, len(df_data), 0],
                "values": ["Data", 1, col, len(df_data), col],
                "line": line,
            }
        )
    chart_z.set_title({"name": "Normalized Surface Impedance"})
    chart_z.set_x_axis(_chart_x_axis_options(treatment))
    chart_z.set_y_axis(_chart_y_axis_options("Z/Z0 [-]", df_data[z_columns].to_numpy().ravel()))
    _apply_chart_surface(chart_z)
    chart_z.set_legend({"position": "bottom", "layout": Z_LEGEND_LAYOUT})
    chart_z.set_style(chart_styles[1])
    data_sheet.insert_chart("F22", chart_z, {"x_scale": CHART_X_SCALE, "y_scale": CHART_Y_SCALE})
    _insert_rotated_log_x_textboxes(data_sheet, treatment, 21, 5)


def _chart_x_axis_options(treatment):
    is_log = _uses_custom_log_x_labels(treatment)
    options = {
        "name": "Frequency [Hz]",
        "name_layout": X_AXIS_TITLE_LAYOUT,
        "label_position": "none" if is_log else "low",
        "major_tick_mark": "outside",
        "minor_tick_mark": "outside" if is_log else "none",
        "major_gridlines": {"visible": True, "line": {"color": "#D9D9D9", "width": 0.5}},
        "minor_gridlines": {"visible": is_log, "line": {"color": "#EEEEEE", "width": 0.25}},
        "num_format": "0",
        "min": float(np.min(treatment.freq)),
        "max": float(np.max(treatment.freq)),
    }
    if is_log:
        options["log_base"] = 10
    return options


def _uses_custom_log_x_labels(treatment):
    return float(np.min(treatment.freq)) > 0


def _log_tick_values(fmin, fmax):
    if fmin <= 0 or fmax <= 0:
        return []
    decade_start = int(np.floor(np.log10(fmin))) - 1
    decade_stop = int(np.ceil(np.log10(fmax))) + 1
    ticks = []
    for exponent in range(decade_start, decade_stop + 1):
        decade = 10**exponent
        for multiplier in range(1, 10):
            value = multiplier * decade
            if fmin <= value <= fmax:
                ticks.append(float(value))
    return sorted(set(ticks))


def _format_frequency_tick(value):
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _nice_axis_unit(values, target_intervals=16):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    span = float(np.max(finite) - np.min(finite))
    if span <= 0:
        return None
    raw = span / target_intervals
    exponent = np.floor(np.log10(raw))
    fraction = raw / (10**exponent)
    for step in (1, 2, 2.5, 5, 10):
        if fraction <= step:
            return step * (10**exponent)
    return 10 * (10**exponent)


def _nice_axis_bounds(values, target_intervals=16):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None, None, None
    major_unit = _nice_axis_unit(finite, target_intervals=target_intervals)
    if major_unit is None:
        return None, None, None
    y_min = np.floor(float(np.min(finite)) / major_unit) * major_unit
    y_max = np.ceil(float(np.max(finite)) / major_unit) * major_unit
    if np.isclose(y_min, y_max):
        y_min -= major_unit
        y_max += major_unit
    return float(y_min), float(y_max), float(major_unit)


def _chart_y_axis_options(name, values, ymax=None):
    y_min, y_max, major_unit = _nice_axis_bounds(values)
    options = {
        "name": name,
        "major_gridlines": {"visible": True, "line": {"color": "#BFBFBF", "width": 0.5}},
        "minor_gridlines": {"visible": False},
    }
    if major_unit is not None:
        options["min"] = y_min
        options["max"] = ymax if ymax is not None else y_max
        options["major_unit"] = major_unit
    return options


def _apply_chart_surface(chart):
    chart.set_chartarea({"fill": {"color": "#FFFFFF"}, "border": {"color": "#BFBFBF", "width": 0.75}})
    chart.set_plotarea(
        {
            "fill": {"color": "#FFFFFF"},
            "border": {"color": "#FFFFFF"},
            "layout": PLOT_LAYOUT,
        }
    )


def _insert_rotated_log_x_textboxes(data_sheet, treatment, chart_row, chart_col):
    if not _uses_custom_log_x_labels(treatment):
        return None

    ticks = _log_tick_values(float(np.min(treatment.freq)), float(np.max(treatment.freq)))
    if not ticks:
        return None

    chart_width = 480 * CHART_X_SCALE
    chart_height = 288 * CHART_Y_SCALE
    plot_x = PLOT_LAYOUT["x"] * chart_width
    plot_y = PLOT_LAYOUT["y"] * chart_height
    plot_width = PLOT_LAYOUT["width"] * chart_width
    plot_height = PLOT_LAYOUT["height"] * chart_height
    fmin = float(np.min(treatment.freq))
    fmax = float(np.max(treatment.freq))
    log_span = np.log10(fmax) - np.log10(fmin)
    y_offset = int(plot_y + plot_height + 2)
    for tick in ticks:
        position = (np.log10(tick) - np.log10(fmin)) / log_span
        x_offset = int(plot_x + position * plot_width - 8)
        data_sheet.insert_textbox(
            chart_row,
            chart_col,
            _format_frequency_tick(tick),
            {
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": 18,
                "height": 40,
                "text_rotation": 90,
                "font": {"size": 9, "color": "#404040"},
                "align": {"horizontal": "center", "vertical": "middle"},
                "fill": {"color": "#FFFFFF", "transparency": 100},
                "line": {"none": True},
                "object_position": 2,
            },
        )
    return len(ticks)


def _angle_label(angle):
    return f"{float(angle):.6g} deg"


def _export_all_frame(treatment):
    """Return a diagnostic table with all angle-wise and diffuse quantities."""
    treatment._raise_if_partial_z_angle("save2sheet(export_all=True)")
    angles = np.asarray(treatment.incidence_angle, dtype=float)
    z_angle = np.asarray(treatment.z_angle, dtype=complex)
    data = {"Frequency [Hz]": treatment.freq}

    for angle_idx, angle in enumerate(angles):
        label = _angle_label(angle)
        z_norm = z_angle[:, angle_idx] / treatment.z0
        _, alpha = treatment.reflection_and_absorption_coefficient(z_angle[:, angle_idx], angle=angle)
        data[f"Real Z_norm {label}"] = np.real(z_norm)
        data[f"Imag Z_norm {label}"] = np.imag(z_norm)
        data[f"Absorption {label}"] = alpha

    z_field = treatment.field_impedance(z_angle)
    _, alpha_field = treatment.reflection_and_absorption_coefficient(z_field)
    alpha_paris = treatment.diffuse_absorption_coefficient(z_angle, angles=angles)

    data["Real Z_norm diffuse field impedance"] = np.real(z_field / treatment.z0)
    data["Imag Z_norm diffuse field impedance"] = np.imag(z_field / treatment.z0)
    data["Absorption diffuse field impedance"] = alpha_field
    data["Absorption diffuse Paris"] = alpha_paris
    return pd.DataFrame(data)


def _write_export_all_csv(treatment, output_dir, stem, metadata, conversion):
    csv_path = output_dir / f"{stem}_export_all.csv"
    metadata_csv_path = output_dir / f"{stem}_export_all_metadata.csv"
    frame = _export_all_frame(treatment)
    frame.to_csv(csv_path, index=False, float_format="%.6g", sep=";", encoding="utf-8-sig")
    paths = {"csv": csv_path}
    if metadata:
        rows = _setup_rows(treatment, conversion)
        rows.append(
            (
                "export_all note",
                "CSV-only diagnostic export. Includes angle-wise impedance/absorption, field diffuse "
                "impedance/absorption, and Paris diffuse absorption.",
                "",
            )
        )
        _write_metadata_csv(metadata_csv_path, rows)
        paths["metadata_csv"] = metadata_csv_path
    return paths
