#!/usr/bin/env python
"""Hand-written SVG plots for unified Fig4 LightGBM summaries."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path

import pandas as pd


FONT_FAMILY = "Arial"
FONT_SIZE = "7pt"
DATA_STROKE = "0.7pt"
AXIS_STROKE = "1pt"
CELL_STROKE = "0.5pt"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
CONDITION_ORDER = (
    "all_non_target_features",
    "non_cde_background",
    "other_cde_modules_only",
)
CONDITION_LABELS = {
    "all_non_target_features": "All non-target",
    "non_cde_background": "Non-CDE bg",
    "other_cde_modules_only": "Other CDE",
}
CONDITION_COLORS = {
    "all_non_target_features": "#8CBAE4",
    "non_cde_background": "#F2CC8F",
    "other_cde_modules_only": "#F1958A",
}
MODULE_ORDER = ("Replication", "Transcription", "Translation")
MODULE_COLORS = {
    "Replication": "#D9EEE7",
    "Transcription": "#DCE8F3",
    "Translation": "#F6DED6",
}
COG_ORDER = ("L", "K", "J")
COG_TO_MODULE = {"L": "Replication", "K": "Transcription", "J": "Translation"}
SUBCATEGORY_CONDITION = "all_minus_target_subcategory_adfu"
SUMMARY_COLUMNS = [
    "analysis",
    "target",
    "condition",
    "module",
    "subcategory",
    "run_n",
    "AUPRC_n",
    "AUPRC_mean",
    "AUPRC_sd",
    "AUROC_n",
    "AUROC_mean",
    "AUROC_sd",
    "R2_n",
    "R2_mean",
    "R2_sd",
    "RMSE_n",
    "RMSE_mean",
    "RMSE_sd",
    "MAE_n",
    "MAE_mean",
    "MAE_sd",
    "positive_rate_test_n",
    "positive_rate_test_mean",
    "positive_rate_test_sd",
]
SUMMARY_BASELINE_RATE_COLUMNS = (
    "target_positive_rate",
    "positive_rate",
    "prevalence",
)
SUMMARY_BASELINE_SD_COLUMNS = (
    "target_positive_rate_sd",
    "positive_rate_sd",
    "prevalence_sd",
)
PREVALENCE_COLUMNS = [
    "analysis",
    "target",
    "module",
    "subcategory",
    "target_positive_n",
    "target_positive_rate",
    "target_count_mean",
    "target_count_max",
]
METRIC_STYLES = {
    "AUPRC": {
        "mean": "AUPRC_mean",
        "sd": "AUPRC_sd",
        "label": "AUPRC",
        "palette": "auprc_light_blue",
        "point_color": "#B8D4E8",
    },
    "R2": {
        "mean": "R2_mean",
        "sd": "R2_sd",
        "label": "R2",
        "palette": "r2_light_coral",
        "point_color": "#F1B6A5",
    },
    "Positive_rate": {
        "mean": "Positive_rate_mean",
        "sd": "Positive_rate_sd",
        "label": "Positive-class prevalence",
        "palette": "positive_rate_gray",
        "point_color": "#D9D9D9",
    },
}


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def _esc(text: object) -> str:
    return escape(str(text), quote=True)


def _svg_root(width_mm: float, height_mm: float) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.0f}mm" height="{height_mm:.0f}mm" viewBox="0 0 {_fmt(width_mm)} {_fmt(height_mm)}">',
        f'<rect x="0" y="0" width="{_fmt(width_mm)}" height="{_fmt(height_mm)}" fill="white"/>',
    ]


def _close_svg(parts: list[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts)


def _text(
    x: float,
    y: float,
    text: object,
    *,
    anchor: str = "start",
    role: str = "label",
    fill: str = TEXT_COLOR,
    extra: str = "",
) -> str:
    return (
        f'<text data-role="{role}" x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY}" font-size="{FONT_SIZE}" fill="{fill}"{extra}>{_esc(text)}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    role: str,
    stroke: str,
    width: str,
    extra: str = "",
) -> str:
    return (
        f'<line data-role="{role}" x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    role: str,
    fill: str,
    stroke: str,
    stroke_width: str,
    extra: str = "",
) -> str:
    return (
        f'<rect data-role="{role}" x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" height="{_fmt(height)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{extra}/>'
    )


def _circle(
    x: float,
    y: float,
    radius: float,
    *,
    role: str,
    fill: str,
    stroke: str,
    stroke_width: str,
    extra: str = "",
) -> str:
    return (
        f'<circle data-role="{role}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(radius)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{extra}/>'
    )


def _validate_columns(frame: pd.DataFrame, required_columns: list[str], context: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {', '.join(missing)}")


def _require_numeric(
    frame: pd.DataFrame,
    columns: list[str],
    context: str,
    *,
    allow_missing: set[str] | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    nullable = allow_missing or set()
    for column in columns:
        source = data[column]
        numeric = pd.to_numeric(source, errors="coerce")
        blank = source.isna() | source.astype(str).str.strip().eq("")
        if ((~blank) & numeric.isna()).any() or (column not in nullable and numeric.isna().any()):
            raise ValueError(f"{context} column {column} must be numeric")
        data[column] = numeric.astype(float)
    return data


def _validate_non_negative(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    for column in columns:
        if (frame[column].astype(float) < 0).any():
            raise ValueError(f"{context} column {column} must be >= 0")


def _unique_value(frame: pd.DataFrame, column: str, context: str) -> str:
    values = frame[column].astype(str).drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{context} column {column} must have exactly one value")
    return values[0]


def _ordered_values(frame: pd.DataFrame, column: str, allowed: tuple[str, ...], context: str) -> None:
    values = set(frame[column].astype(str).tolist())
    if values != set(allowed):
        raise ValueError(f"{context} column {column} contains unknown or incomplete values")


def _validate_summary_counts(data: pd.DataFrame, context: str) -> None:
    count_columns = ("run_n", "AUPRC_n", "R2_n")
    for column in count_columns:
        values = data[column].astype(float)
        if (values % 1 != 0).any():
            raise ValueError(f"{context} column {column} must contain integer counts")
    if not data["run_n"].eq(3).all():
        raise ValueError(f"{context} column run_n must equal 3")
    if not data["AUPRC_n"].eq(3).all():
        raise ValueError(f"{context} column AUPRC_n must equal 3")
    for column in ("AUPRC_mean", "AUPRC_sd"):
        if not data[column].map(lambda value: math.isfinite(float(value))).all():
            raise ValueError(f"{context} column {column} must contain finite values")
    for prefix in ("R2",):
        count_column = f"{prefix}_n"
        mean_column = f"{prefix}_mean"
        sd_column = f"{prefix}_sd"
        counts = data[count_column]
        if (~counts.between(0, 3, inclusive="both")).any():
            raise ValueError(f"{context} column {count_column} must be between 0 and 3")
        means = data[mean_column]
        mean_is_finite = means.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        if ((counts.eq(0) & means.notna()) | (counts.gt(0) & ~mean_is_finite)).any():
            raise ValueError(
                f"{context} column {mean_column} must be finite exactly when {count_column} is greater than 0"
            )
        sds = data[sd_column]
        sd_is_finite = sds.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        if ((counts.le(1) & sds.notna()) | (counts.ge(2) & ~sd_is_finite)).any():
            raise ValueError(
                f"{context} column {sd_column} must be missing for {count_column} <= 1 and finite for {count_column} >= 2"
            )
def _summary_baseline_rate_column(frame: pd.DataFrame) -> str | None:
    for column in SUMMARY_BASELINE_RATE_COLUMNS:
        if column in frame.columns:
            return column
    return None


def _summary_baseline_sd_column(frame: pd.DataFrame) -> str | None:
    for column in SUMMARY_BASELINE_SD_COLUMNS:
        if column in frame.columns:
            return column
    return None


def _validate_prevalence_table(frame: pd.DataFrame) -> pd.DataFrame:
    context = "target_prevalence.csv"
    _validate_columns(frame, PREVALENCE_COLUMNS, context)
    data = _require_numeric(
        frame,
        [
            "target_positive_n",
            "target_positive_rate",
            "target_count_mean",
            "target_count_max",
        ],
        context,
    )
    _validate_non_negative(
        data,
        [
            "target_positive_n",
            "target_count_mean",
            "target_count_max",
        ],
        context,
    )
    positive_rate = data["target_positive_rate"].astype(float)
    if (~positive_rate.between(0.0, 1.0, inclusive="both")).any():
        raise ValueError("target_prevalence.csv column target_positive_rate must be between 0 and 1")
    if data.duplicated(["analysis", "target"], keep=False).any():
        raise ValueError("target_prevalence.csv contains duplicate analysis/target rows")
    return data.copy()


def _add_positive_rate_baseline(
    frame: pd.DataFrame,
    *,
    prevalence: pd.DataFrame | None,
    context: str,
) -> pd.DataFrame:
    data = frame.copy()
    rate_column = _summary_baseline_rate_column(data)
    if rate_column is not None:
        numeric_columns = [rate_column]
        sd_column = _summary_baseline_sd_column(data)
        allow_missing: set[str] = set()
        if sd_column is not None:
            numeric_columns.append(sd_column)
            allow_missing.add(sd_column)
        data = _require_numeric(data, numeric_columns, context, allow_missing=allow_missing)
        data["Positive_rate_mean"] = data[rate_column].astype(float)
        if sd_column is None:
            data["Positive_rate_sd"] = float("nan")
        else:
            data["Positive_rate_sd"] = data[sd_column].astype(float)
    else:
        if prevalence is None:
            raise ValueError(
                f"{context} lacks a baseline positive-rate column; provide the corresponding target_prevalence.csv"
            )
        prevalence_data = _validate_prevalence_table(prevalence)
        merged = data.merge(
            prevalence_data.loc[:, ["analysis", "target", "target_positive_rate", "module", "subcategory"]],
            on=["analysis", "target"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_prevalence"),
        )
        if merged["target_positive_rate"].isna().any():
            missing_targets = sorted(
                merged.loc[merged["target_positive_rate"].isna(), "target"].astype(str).unique().tolist()
            )
            raise ValueError(
                f"{context} is missing baseline positive-rate rows in target_prevalence.csv for: {', '.join(missing_targets)}"
            )
        for column in ("module", "subcategory"):
            prevalence_column = f"{column}_prevalence"
            if prevalence_column in merged.columns:
                left = merged[column].fillna("").astype(str)
                right = merged[prevalence_column].fillna("").astype(str)
                if left.ne(right).any():
                    raise ValueError(
                        f"{context} does not match target_prevalence.csv for column {column}"
                    )
                merged = merged.drop(columns=[prevalence_column])
        data = merged
        data["Positive_rate_mean"] = data["target_positive_rate"].astype(float)
        data["Positive_rate_sd"] = float("nan")
    positive_rate = data["Positive_rate_mean"].astype(float)
    if (~positive_rate.between(0.0, 1.0, inclusive="both")).any():
        raise ValueError(f"{context} baseline positive rate must be between 0 and 1")
    positive_rate_sd = pd.to_numeric(data["Positive_rate_sd"], errors="coerce")
    if (positive_rate_sd.dropna() < 0).any():
        raise ValueError(f"{context} baseline positive-rate sd must be >= 0")
    return data


def _validate_grouped_bar_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    context = "grouped summary"
    _validate_columns(frame, SUMMARY_COLUMNS, context)
    data = _require_numeric(
        frame,
        [
            "run_n",
            "AUPRC_n",
            "AUPRC_mean",
            "AUPRC_sd",
            "R2_n",
            "R2_mean",
            "R2_sd",
        ],
        context,
        allow_missing={
            "R2_mean",
            "R2_sd",
        },
    )
    _validate_non_negative(
        data,
        [
            "run_n",
            "AUPRC_n",
            "AUPRC_sd",
            "R2_n",
            "R2_sd",
            "positive_rate_test_n",
        ],
        context,
    )
    _validate_summary_counts(data, context)
    if data.duplicated(["target", "condition"], keep=False).any():
        raise ValueError("grouped summary contains duplicate target/condition rows")
    analysis = _unique_value(data, "analysis", context)
    if analysis == "cde_module":
        target_order = MODULE_ORDER
        _ordered_values(data, "target", MODULE_ORDER, context)
        if not data["module"].astype(str).equals(data["target"].astype(str)):
            raise ValueError("grouped summary target and module must match for cde_module")
    elif analysis == "cog_lkj":
        target_order = COG_ORDER
        _ordered_values(data, "target", COG_ORDER, context)
        expected_modules = data["target"].astype(str).map(COG_TO_MODULE)
        if not expected_modules.equals(data["module"].astype(str)):
            raise ValueError("grouped summary target/module mapping is invalid for cog_lkj")
    else:
        raise ValueError("grouped summary analysis must be cde_module or cog_lkj")
    if set(data["condition"].astype(str)) != set(CONDITION_ORDER):
        raise ValueError("grouped summary condition values are invalid")
    if len(data) != len(target_order) * len(CONDITION_ORDER):
        raise ValueError("grouped summary must contain exactly three targets by three conditions")
    return data, analysis, target_order


def _validate_subcategory_summary(
    frame: pd.DataFrame,
    prevalence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    context = "subcategory summary"
    _validate_columns(frame, SUMMARY_COLUMNS, context)
    data = _require_numeric(
        frame,
        [
            "run_n",
            "AUPRC_n",
            "AUPRC_mean",
            "AUPRC_sd",
            "R2_n",
            "R2_mean",
            "R2_sd",
        ],
        context,
        allow_missing={
            "R2_mean",
            "R2_sd",
        },
    )
    _validate_non_negative(
        data,
        [
            "run_n",
            "AUPRC_n",
            "AUPRC_sd",
            "R2_n",
            "R2_sd",
            "positive_rate_test_n",
        ],
        context,
    )
    _validate_summary_counts(data, context)
    data = _add_positive_rate_baseline(data, prevalence=prevalence, context=context)
    if _unique_value(data, "analysis", context) != "cde_subcategory":
        raise ValueError("subcategory summary analysis must be cde_subcategory")
    if set(data["condition"].astype(str)) != {SUBCATEGORY_CONDITION}:
        raise ValueError("subcategory summary condition values are invalid")
    if data.duplicated(["module", "subcategory"], keep=False).any():
        raise ValueError("subcategory summary contains duplicate module/subcategory rows")
    if len(data) != 14:
        raise ValueError("subcategory summary must contain exactly 14 rows")
    if set(data["module"].astype(str)) != set(MODULE_ORDER):
        raise ValueError("subcategory summary module values are invalid")
    expected_targets = data.apply(
        lambda row: f"{row['module']} | {row['subcategory']}",
        axis=1,
    )
    if not expected_targets.astype(str).equals(data["target"].astype(str)):
        raise ValueError("subcategory summary target labels are invalid")
    return data


def _metric_value_bounds(frame: pd.DataFrame, metric_key: str) -> tuple[float, float]:
    metric = METRIC_STYLES[metric_key]
    means = frame[metric["mean"]].astype(float)
    sds = frame[metric["sd"]].astype(float)
    if metric_key in {"AUPRC", "Positive_rate"}:
        return 0.0, 1.0
    usable_sds = sds.fillna(0.0)
    lower_values = (means - usable_sds).dropna()
    upper_values = (means + usable_sds).dropna()
    if lower_values.empty or upper_values.empty:
        return -0.1, 0.1
    lower = float(lower_values.min())
    upper = float(upper_values.max())
    min_bound = min(0.0, math.floor(lower * 10.0) / 10.0)
    max_bound = max(0.0, math.ceil(upper * 10.0) / 10.0)
    if min_bound == max_bound:
        min_bound -= 0.1
        max_bound += 0.1
    return min_bound, max_bound


def _scaled_y(value: float, y_min: float, y_max: float, top: float, height: float) -> float:
    clipped = min(max(float(value), y_min), y_max)
    return top + (y_max - clipped) * height / (y_max - y_min)


def _tick_values(y_min: float, y_max: float, n: int = 5) -> list[float]:
    if y_min == y_max:
        return [y_min]
    return [y_min + (y_max - y_min) * idx / n for idx in range(n + 1)]


def _errorbar_parts(
    x: float,
    mean: float,
    sd: float,
    *,
    y_min: float,
    y_max: float,
    top: float,
    height: float,
    cap_half: float,
    metric_name: str,
) -> list[str]:
    upper_y = _scaled_y(mean + sd, y_min, y_max, top, height)
    lower_y = _scaled_y(mean - sd, y_min, y_max, top, height)
    stem = _line(
        x,
        upper_y,
        x,
        lower_y,
        role="error-stem",
        stroke=AXIS_COLOR,
        width=DATA_STROKE,
        extra=f' data-metric="{metric_name}"',
    )
    upper = _line(
        x - cap_half,
        upper_y,
        x + cap_half,
        upper_y,
        role="error-cap",
        stroke=AXIS_COLOR,
        width=DATA_STROKE,
        extra=f' data-cap="upper" data-metric="{metric_name}"',
    )
    lower = _line(
        x - cap_half,
        lower_y,
        x + cap_half,
        lower_y,
        role="error-cap",
        stroke=AXIS_COLOR,
        width=DATA_STROKE,
        extra=f' data-cap="lower" data-metric="{metric_name}"',
    )
    return [stem, upper, lower]


def _draw_legend(parts: list[str], x: float, y: float, item_gap: float) -> None:
    cursor_x = x
    for condition in CONDITION_ORDER:
        parts.append(
            _rect(
                cursor_x,
                y - 3.6,
                3.2,
                3.2,
                role="legend-swatch",
                fill=CONDITION_COLORS[condition],
                stroke=AXIS_COLOR,
                stroke_width=DATA_STROKE,
                extra=f' data-condition="{condition}"',
            )
        )
        parts.append(_text(cursor_x + 4.4, y, CONDITION_LABELS[condition], role="legend"))
        cursor_x += item_gap


def _render_grouped_bar_panel(
    summary: pd.DataFrame,
    *,
    metric_key: str,
    panel_letter: str,
    panel_x: float,
    panel_y: float,
    panel_width: float,
    panel_height: float,
    title: str,
    include_legend: bool,
) -> list[str]:
    data, analysis, target_order = _validate_grouped_bar_summary(summary)
    metric = METRIC_STYLES[metric_key]
    y_min, y_max = _metric_value_bounds(data, metric_key)
    left = panel_x + 11.0
    right = panel_x + panel_width - 3.0
    top = panel_y + 7.0
    bottom = panel_y + panel_height - 12.0
    plot_width = right - left
    plot_height = bottom - top
    group_width = plot_width / len(target_order)
    bar_width = min(6.0, group_width * 0.18)
    offsets = {
        CONDITION_ORDER[0]: -bar_width * 1.4,
        CONDITION_ORDER[1]: 0.0,
        CONDITION_ORDER[2]: bar_width * 1.4,
    }
    parts = [
        _text(panel_x + 2.0, panel_y + 5.0, panel_letter, role="panel-letter"),
        _text(panel_x + panel_width / 2.0, panel_y + 5.0, title, anchor="middle", role="title"),
        _line(left, top, left, bottom, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE),
        _line(left, bottom, right, bottom, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE),
        _text(panel_x + 2.5, panel_y + panel_height / 2.0, metric["label"], role="axis-label", extra=f' transform="rotate(-90 {_fmt(panel_x + 2.5)} {_fmt(panel_y + panel_height / 2.0)})"'),
    ]
    for tick in _tick_values(y_min, y_max):
        y = _scaled_y(tick, y_min, y_max, top, plot_height)
        parts.append(_line(left, y, right, y, role="grid", stroke=GRID_COLOR, width=DATA_STROKE))
        parts.append(_text(left - 1.2, y + 2.3, f"{tick:.1f}", anchor="end", role="tick"))
    zero_y = _scaled_y(0.0, y_min, y_max, top, plot_height)
    if metric_key == "R2" and y_min < 0.0:
        parts.append(
            _line(
                left,
                zero_y,
                right,
                zero_y,
                role="zero-reference",
                stroke=AXIS_COLOR,
                width=DATA_STROKE,
                extra=' data-metric="R2"',
            )
        )
    for group_index, target in enumerate(target_order):
        center = left + group_width * (group_index + 0.5)
        label_y = bottom + 8.0
        parts.append(_text(center, label_y, target, anchor="middle", role="tick"))
        frame = data[data["target"].astype(str).eq(target)]
        for condition in CONDITION_ORDER:
            row = frame[frame["condition"].astype(str).eq(condition)].iloc[0]
            mean = float(row[metric["mean"]])
            sd = float(row[metric["sd"]])
            if math.isnan(mean):
                continue
            bar_center = center + offsets[condition]
            bar_left = bar_center - bar_width / 2.0
            mean_y = _scaled_y(mean, y_min, y_max, top, plot_height)
            bar_y = min(mean_y, zero_y)
            bar_height = abs(mean_y - zero_y)
            parts.append(
                _rect(
                    bar_left,
                    bar_y,
                    bar_width,
                    bar_height,
                    role="bar",
                    fill=CONDITION_COLORS[condition],
                    stroke=AXIS_COLOR,
                    stroke_width=DATA_STROKE,
                    extra=f' data-condition="{condition}" data-target="{target}" data-metric="{metric_key}"',
                )
            )
            if not math.isnan(sd):
                parts.extend(
                    _errorbar_parts(
                        bar_center,
                        mean,
                        sd,
                        y_min=y_min,
                        y_max=y_max,
                        top=top,
                        height=plot_height,
                        cap_half=1.8,
                        metric_name=metric_key,
                    )
                )
    if include_legend:
        legend_x = max(panel_x + 24.0, right - 44.0)
        _draw_legend(parts, legend_x, panel_y + 5.0, 17.0 if analysis == "cde_module" else 17.0)
    return parts


def write_grouped_bar_svg(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    metric: str,
    panel_letter: str,
) -> Path:
    if metric not in {"AUPRC", "R2"}:
        raise ValueError("metric must be AUPRC or R2")
    data, analysis, _ = _validate_grouped_bar_summary(summary)
    title = "CDE module prediction" if analysis == "cde_module" else "COG L/K/J sensitivity"
    if metric == "R2":
        title = f"{title} raw-count"
    parts = _svg_root(88.0, 58.0)
    parts.extend(
        _render_grouped_bar_panel(
            data,
            metric_key=metric,
            panel_letter=panel_letter,
            panel_x=0.0,
            panel_y=0.0,
            panel_width=88.0,
            panel_height=58.0,
            title=title,
            include_legend=True,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_close_svg(parts), encoding="utf-8")
    return out_path


def _subcategory_order(data: pd.DataFrame) -> list[tuple[str, str]]:
    order = []
    for module_name in MODULE_ORDER:
        subset = data[data["module"].astype(str).eq(module_name)].sort_values("subcategory", kind="mergesort")
        order.extend((module_name, value) for value in subset["subcategory"].astype(str).tolist())
    return order


def _palette_fill(metric_key: str, value: float, v_min: float, v_max: float) -> str:
    if math.isnan(float(value)):
        return "#E5E5E5"
    if metric_key == "AUPRC":
        norm = (float(value) - v_min) / (v_max - v_min) if v_max > v_min else 0.5
        norm = min(max(norm, 0.0), 1.0)
        start = (247, 251, 253)
        end = (184, 212, 232)
        rgb = tuple(round(start[idx] + (end[idx] - start[idx]) * norm) for idx in range(3))
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    if metric_key == "Positive_rate":
        norm = (float(value) - v_min) / (v_max - v_min) if v_max > v_min else 0.5
        norm = min(max(norm, 0.0), 1.0)
        start = (247, 247, 247)
        end = (217, 217, 217)
        rgb = tuple(round(start[idx] + (end[idx] - start[idx]) * norm) for idx in range(3))
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    limit = max(abs(v_min), abs(v_max), 1e-12)
    strength = min(abs(float(value)) / limit, 1.0)
    neutral = (247, 247, 247)
    if metric_key == "R2":
        endpoint = (241, 182, 165)
    else:
        endpoint = (140, 109, 49) if value < 0 else (77, 77, 77)
    rgb = tuple(round(neutral[idx] + (endpoint[idx] - neutral[idx]) * strength) for idx in range(3))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def plot_fig4e(
    summary: pd.DataFrame,
    out_dir: Path,
    prevalence: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    data = _validate_subcategory_summary(summary, prevalence=prevalence)
    order = _subcategory_order(data)
    dot_path = out_dir / "Fig4E_subcategory_dot_whisker.svg"
    heatmap_path = out_dir / "Fig4E_subcategory_heatmap.svg"
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_parts = _svg_root(180.0, 112.0)
    left = 13.0
    right = 177.0
    plot_width = right - left
    x_step = plot_width / len(order)
    dot_parts.extend(
        [
            _text(2.5, 5.0, "E", role="panel-letter"),
            _text(90.0, 5.0, "CDE subcategory prediction", anchor="middle", role="title"),
        ]
    )
    metric_keys = ("AUPRC", "R2", "Positive_rate")
    facet_tops = (12.0, 43.0, 74.0)
    facet_height = 19.0
    for facet_index, metric_key in enumerate(metric_keys):
        style = METRIC_STYLES[metric_key]
        top = facet_tops[facet_index]
        bottom = top + facet_height
        y_min, y_max = _metric_value_bounds(data, metric_key)
        dot_parts.extend(
            [
                _text((left + right) / 2.0, top - 2.0, style["label"], anchor="middle", role="axis-label"),
                _line(left, top, left, bottom, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE),
                _line(left, bottom, right, bottom, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE),
            ]
        )
        for tick in _tick_values(y_min, y_max, n=4):
            y = _scaled_y(tick, y_min, y_max, top, facet_height)
            dot_parts.append(_line(left, y, right, y, role="grid", stroke=GRID_COLOR, width=DATA_STROKE))
            dot_parts.append(_text(left - 1.2, y + 2.3, f"{tick:.1f}", anchor="end", role="tick"))
        if metric_key == "R2":
            zero_y = _scaled_y(0.0, y_min, y_max, top, facet_height)
            dot_parts.append(
                _line(
                    left,
                    zero_y,
                    right,
                    zero_y,
                    role="zero-reference",
                    stroke=AXIS_COLOR,
                    width=DATA_STROKE,
                    extra=f' data-metric="{metric_key}"',
                )
            )
        for idx, (module_name, subcategory) in enumerate(order):
            row = data[
                data["module"].astype(str).eq(module_name)
                & data["subcategory"].astype(str).eq(subcategory)
            ].iloc[0]
            center = left + x_step * (idx + 0.5)
            if facet_index == len(metric_keys) - 1:
                label_y = bottom + 8.0
                dot_parts.append(
                    _text(
                        center,
                        label_y,
                        subcategory,
                        anchor="start",
                        role="tick",
                        extra=f' transform="rotate(45 {_fmt(center)} {_fmt(label_y)})"',
                    )
                )
            if idx in {5, 9}:
                separator_x = left + x_step * idx
                dot_parts.append(
                    _line(separator_x, top, separator_x, bottom, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE)
                )
            mean = float(row[style["mean"]])
            sd = float(row[style["sd"]])
            if math.isnan(mean):
                continue
            y = _scaled_y(mean, y_min, y_max, top, facet_height)
            if not math.isnan(sd):
                dot_parts.extend(
                    _errorbar_parts(
                        center,
                        mean,
                        sd,
                        y_min=y_min,
                        y_max=y_max,
                        top=top,
                        height=facet_height,
                        cap_half=1.0,
                        metric_name=metric_key,
                    )
                )
            dot_parts.append(
                _circle(
                    center,
                    y,
                    1.0,
                    role="point",
                    fill=style["point_color"],
                    stroke=AXIS_COLOR,
                    stroke_width=DATA_STROKE,
                    extra=(
                        f' data-metric="{metric_key}" data-palette="{style["palette"]}"'
                        f' data-value="{_fmt(mean)}"'
                    ),
                )
            )
    dot_path.write_text(_close_svg(dot_parts), encoding="utf-8")

    heat_parts = _svg_root(180.0, 88.0)
    heat_parts.extend(
        [
            _text(2.5, 5.0, "E", role="panel-letter"),
            _text(90.0, 5.0, "CDE subcategory prediction heatmap", anchor="middle", role="title"),
        ]
    )
    cell_left = 48.0
    cell_top = 11.0
    row_height = 4.9
    col_width = 32.0
    metric_keys = ("AUPRC", "R2", "Positive_rate")
    for metric_index, metric_key in enumerate(metric_keys):
        x = cell_left + metric_index * col_width
        heat_parts.append(_text(x + col_width / 2.0, 8.5, METRIC_STYLES[metric_key]["label"], anchor="middle", role="column-label"))
    metric_ranges = {
        metric_key: _metric_value_bounds(data, metric_key)
        for metric_key in metric_keys
    }
    for row_index, (module_name, subcategory) in enumerate(order):
        row = data[
            data["module"].astype(str).eq(module_name)
            & data["subcategory"].astype(str).eq(subcategory)
        ].iloc[0]
        y = cell_top + row_index * row_height
        heat_parts.append(_text(3.0, y + 3.4, subcategory, role="row-label"))
        heat_parts.append(
            _rect(
                43.0,
                y,
                3.0,
                row_height - 0.6,
                role="module-strip",
                fill=MODULE_COLORS[module_name],
                stroke="#555555",
                stroke_width=CELL_STROKE,
                extra=f' data-module="{_esc(module_name)}"',
            )
        )
        if row_index in {5, 9}:
            heat_parts.append(_line(3.0, y - 1.0, 176.0, y - 1.0, role="axis", stroke=AXIS_COLOR, width=AXIS_STROKE))
        for metric_index, metric_key in enumerate(metric_keys):
            x = cell_left + metric_index * col_width
            mean_column = METRIC_STYLES[metric_key]["mean"]
            value = float(row[mean_column])
            value_min, value_max = metric_ranges[metric_key]
            fill = _palette_fill(metric_key, value, value_min, value_max)
            scale = "diverging-zero" if metric_key == "R2" else "sequential"
            heat_parts.append(
                _rect(
                    x,
                    y,
                    col_width - 1.4,
                    row_height - 0.6,
                    role="heatmap-cell",
                    fill=fill,
                    stroke="#555555",
                    stroke_width=CELL_STROKE,
                    extra=(
                        f' data-metric="{metric_key}" data-palette="{METRIC_STYLES[metric_key]["palette"]}"'
                        f' data-scale="{scale}" data-value="{_fmt(value) if not math.isnan(value) else "nan"}"'
                    ),
                )
            )
            heat_parts.append(
                _text(
                    x + (col_width - 1.4) / 2.0,
                    y + 3.4,
                    f"{value:.2f}" if not math.isnan(value) else "NA",
                    anchor="middle",
                    role="cell-value",
                    fill="#111111",
                )
            )
    heatmap_path.write_text(_close_svg(heat_parts), encoding="utf-8")
    return dot_path, heatmap_path


def _load_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_fig4cd(summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    c_path = out_dir / "Fig4C_CDE_module_AUPRC.svg"
    d_path = out_dir / "Fig4D_CDE_module_raw_count_R2.svg"
    write_grouped_bar_svg(summary, c_path, metric="AUPRC", panel_letter="C")
    write_grouped_bar_svg(summary, d_path, metric="R2", panel_letter="D")
    return c_path, d_path


def _supp_panel(summary: pd.DataFrame, metric_key: str, panel_letter: str, panel_x: float, panel_width: float) -> list[str]:
    title = "AUPRC" if metric_key == "AUPRC" else "raw-count R2"
    return _render_grouped_bar_panel(
        summary,
        metric_key=metric_key,
        panel_letter=panel_letter,
        panel_x=panel_x,
        panel_y=0.0,
        panel_width=panel_width,
        panel_height=72.0,
        title=title,
        include_legend=False,
    )


def plot_figs5(summary: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    a_path = out_dir / "FigS5A_COG_LKJ_AUPRC.svg"
    b_path = out_dir / "FigS5B_COG_LKJ_raw_count_R2.svg"
    combined_path = out_dir / "FigS5_COG_LKJ_combined.svg"
    write_grouped_bar_svg(summary, a_path, metric="AUPRC", panel_letter="A")
    write_grouped_bar_svg(summary, b_path, metric="R2", panel_letter="B")

    data, _, _ = _validate_grouped_bar_summary(summary)
    combined_parts = _svg_root(180.0, 72.0)
    combined_parts.append(_text(90.0, 5.0, "COG L/K/J sensitivity", anchor="middle", role="title"))
    _draw_legend(combined_parts, 60.0, 10.5, 19.0)
    combined_parts.extend(_supp_panel(data, "AUPRC", "A", 0.0, 89.0))
    combined_parts.extend(_supp_panel(data, "R2", "B", 91.0, 89.0))
    combined_path.write_text(_close_svg(combined_parts), encoding="utf-8")
    return a_path, b_path, combined_path


def plot_all_from_results(
    results_dir: Path,
    plots_dir: Path,
    prevalence_path: Path | None = None,
) -> list[Path]:
    module_summary = _load_summary(results_dir / "fig4cd_cde_module_summary.csv")
    subcategory_summary = _load_summary(results_dir / "fig4e_subcategory_summary.csv")
    cog_summary = _load_summary(results_dir / "figs5_cog_lkj_summary.csv")
    if prevalence_path is None:
        prevalence_path = results_dir.parent / "qc" / "target_prevalence.csv"
    prevalence = _load_summary(prevalence_path) if prevalence_path.is_file() else None
    plots = [
        *plot_fig4cd(module_summary, plots_dir),
        *plot_fig4e(subcategory_summary, plots_dir, prevalence=prevalence),
        *plot_figs5(cog_summary, plots_dir),
    ]
    return plots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--prevalence-path", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.results_dir is None:
        args.results_dir = args.out_dir / "results"
    if args.plots_dir is None:
        args.plots_dir = args.out_dir / "plots"
    if args.prevalence_path is None:
        args.prevalence_path = args.results_dir.parent / "qc" / "target_prevalence.csv"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plot_all_from_results(args.results_dir, args.plots_dir, args.prevalence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
