#!/usr/bin/env python
"""Render the downstream Fig. 3 package as direct UTF-8 SVG files."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PACKAGE_ROOT / "tables"

FONT_PT = 7
AXIS_PT = 1.0
LINE_PT = 0.7
MEDIAN_PT = 1.0
A70_FACTOR = 0.90
CHROM = "Chromosome genes"
TRNA_POS = "tRNA+ plasmid genes"
TRNA_NEG = "tRNA- plasmid genes"
GROUP_ORDER = [CHROM, TRNA_POS, TRNA_NEG]
GROUP_LABELS = {
    CHROM: "Chromosomal genes",
    TRNA_POS: "Genes on tRNA-positive plasmids",
    TRNA_NEG: "Genes on tRNA-negative plasmids",
}
B_COLORS = {CHROM: "#E3E3E3", TRNA_POS: "#94C0DF", TRNA_NEG: "#F8E0D7"}
VIOLIN_FILL = "#94C0DF"
COG_COLORS = {
    "Information storage and processing": "#94C0DF",
    "Metabolism": "#C1DEA4",
    "Cellular processes and signaling": "#F0A386",
}
# Point diameters are 3.5 mm at 100 products and 6 mm at 1,000 products.
# The SVG viewBox is 900 units wide for a 90 mm physical width (10 units/mm).
SIZE_RADII = {"<100": 17.5, "100–1000": 17.5, ">1000": 30.0}
SIGNIFICANCE_LABELS = {"ns", "*", "**", "***"}
CCI_AXIS_MAX = 3.0
FIG3B_AXIS_MAX = 2.0


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _require_nonblank(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.{column} must not be blank or missing")


def _numeric(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.{column} must be numeric and finite") from error
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"{name}.{column} must be numeric and finite")
    return values.astype(float)


def a70_sigma_bins(n: int, std: float, bin_width: float) -> float:
    """Return Scott's bandwidth times 0.90 (A-90%) in histogram-bin units."""
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("A-90% KDE requires at least two observations")
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("A-90% KDE standard deviation must be finite and positive")
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("A-90% KDE bin width must be finite and positive")
    return A70_FACTOR * float(std) * int(n) ** (-0.2) / float(bin_width)


def count_scott_sigma_bins(counts: np.ndarray, std: float, bin_width: float) -> float:
    """Return Scott bandwidth using the total CDS count represented by histogram bins."""
    weights = np.asarray(counts, dtype=float)
    if weights.ndim != 1 or weights.size < 2:
        raise ValueError("count-based Scott KDE requires at least two histogram bins")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("count-based Scott KDE counts must be finite, non-negative, and nonempty")
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("count-based Scott KDE standard deviation must be finite and positive")
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("count-based Scott KDE bin width must be finite and positive")
    sample_n = float(weights.sum())
    if not np.isfinite(sample_n) or sample_n <= 1.0:
        raise ValueError("count-based Scott KDE sample size must exceed one")
    return float(std) * sample_n ** (-0.2) / float(bin_width)


def _gaussian_smooth(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("KDE histogram must contain at least two bins")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0) or counts.sum() <= 0.0:
        raise ValueError("KDE histogram counts must be finite, non-negative, and nonempty")
    if not np.isfinite(sigma_bins) or sigma_bins <= 0.0:
        raise ValueError("KDE Gaussian sigma must be finite and positive")
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(counts.astype(float), radius, mode="constant")
    density = np.convolve(padded, kernel, mode="valid")
    peak = float(density.max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("KDE density has no finite positive support")
    return density / peak


def _padded_kde(
    centers: np.ndarray, counts: np.ndarray, bin_width: float, sigma_bins: float
) -> tuple[np.ndarray, np.ndarray]:
    """Add four bandwidths of zero-count support before Gaussian smoothing."""
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    left = centers[0] - bin_width * np.arange(radius, 0, -1, dtype=float)
    right = centers[-1] + bin_width * np.arange(1, radius + 1, dtype=float)
    extended_centers = np.concatenate([left, centers, right])
    extended_counts = np.pad(counts.astype(float), radius, mode="constant")
    return extended_centers, _gaussian_smooth(extended_counts, sigma_bins)


def tukey_summary(values: np.ndarray) -> dict[str, float]:
    """Calculate quartiles and observed 1.5-IQR whisker endpoints."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("violin observations must be a nonempty finite vector")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = array[(array >= q1 - 1.5 * iqr) & (array <= q3 + 1.5 * iqr)]
    if inside.size == 0:
        raise ValueError("Tukey whiskers have no observed endpoints")
    return {
        "n": int(array.size),
        "std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": float(inside.min()),
        "whisker_high": float(inside.max()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def raw_distribution(values: np.ndarray, bin_n: int = 100) -> dict[str, object]:
    """Build an A-90% binned KDE from every supplied finite observation."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("violin observations must contain at least two finite values")
    summary = tukey_summary(array)
    lower, upper = float(array.min()), float(array.max())
    if lower == upper:
        raise ValueError("A-90% KDE bandwidth is zero for constant observations")
    edges = np.linspace(lower, upper, int(bin_n) + 1)
    counts, _ = np.histogram(array, bins=edges)
    width = float(edges[1] - edges[0])
    sigma = a70_sigma_bins(int(array.size), float(summary["std"]), width)
    centers, density = _padded_kde(
        (edges[:-1] + edges[1:]) / 2.0, counts.astype(float), width, sigma
    )
    return {
        "centers": centers,
        "density": density,
        "summary": summary,
        "support_min": float(centers[0]),
        "support_max": float(centers[-1]),
        "sigma_bins": sigma,
    }


def histogram_distribution(histogram: pd.DataFrame, stats_row: pd.Series) -> dict[str, object]:
    """Build a count-based Scott KDE from a complete precomputed gene histogram."""
    required = ["bin_left", "bin_right", "bin_center", "count"]
    _require_columns(histogram, required, "all-gene histogram")
    ordered = histogram.sort_values("bin_left", kind="stable").copy()
    for column in required:
        ordered[column] = _numeric(ordered, column, "all-gene histogram")
    widths = ordered["bin_right"] - ordered["bin_left"]
    if np.any(widths <= 0.0) or not np.allclose(widths, widths.iloc[0]):
        raise ValueError("all-gene histogram must use uniform positive bin widths")
    counts = ordered["count"].to_numpy(float)
    if np.any(counts < 0.0) or np.any(counts != np.floor(counts)):
        raise ValueError("all-gene histogram count must be a non-negative integer")
    summary_columns = [
        "n",
        "std",
        "q1",
        "median",
        "q3",
        "whisker_low",
        "whisker_high",
        "min",
        "max",
    ]
    summary = {}
    for column in summary_columns:
        try:
            value = float(stats_row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"all-gene stats.{column} must be numeric and finite") from error
        if not np.isfinite(value):
            raise ValueError(f"all-gene stats.{column} must be numeric and finite")
        summary[column] = int(value) if column == "n" else value
    if summary["n"] != int(counts.sum()):
        raise ValueError("all-gene histogram counts do not match stats.n")
    bin_width = float(widths.iloc[0])
    sigma = count_scott_sigma_bins(counts, summary["std"], bin_width)
    centers, density = _padded_kde(
        ordered["bin_center"].to_numpy(float), counts, bin_width, sigma
    )
    return {
        "centers": centers,
        "density": density,
        "summary": summary,
        "support_min": float(centers[0]),
        "support_max": float(centers[-1]),
        "sigma_bins": sigma,
    }


def svg_doc(width: int, height: int, body: list[str], physical_size: tuple[str, str] | None = None) -> str:
    style = (
        f"<style>text{{font-family:Arial,sans-serif;font-size:{FONT_PT}pt;fill:#111}}"
        f".axis{{stroke:#111;stroke-width:{AXIS_PT}pt;fill:none}}"
        f".line,.tick,.violin,.iqr-box,.significance-bracket,.reference-line,.point{{"
        f"stroke:#222;stroke-width:{LINE_PT}pt}}"
        ".line,.tick,.significance-bracket,.reference-line{fill:none}"
        f".median-line{{stroke:black;stroke-width:{MEDIAN_PT}pt}}"
        ".median-dot{fill:#1F4E79;stroke:none}"
        ".iqr-box{fill:white}</style>"
    )
    width_attr, height_attr = physical_size or (str(width), str(height))
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_attr}" height="{height_attr}" viewBox="0 0 {width} {height}">',
            style,
            *body,
            "</svg>",
        ]
    )


def write_svg(path: Path, width: int, height: int, body: list[str], physical_size: tuple[str, str] | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = svg_doc(width, height, body, physical_size)
    path.write_text(document, encoding="utf-8")
    return document


def _scale(value: float, source: tuple[float, float], target: tuple[float, float]) -> float:
    lo, hi = source
    if not np.isfinite(value) or hi <= lo or value < lo - 1e-10 or value > hi + 1e-10:
        raise ValueError("plot value lies outside a valid finite scale")
    return target[0] + (value - lo) / (hi - lo) * (target[1] - target[0])


def _title_label(value: object) -> str:
    text = " ".join(str(value).strip().split())
    return text[:1].upper() + text[1:] if text else ""


def _label_text(x: float, y: float, label: str, rotate: bool, angle: float = 35.0) -> str:
    lines = textwrap.wrap(label, width=20, break_long_words=False) or [label]
    if rotate:
        tspans = "".join(
            f'<tspan x="0" dy="{0 if index == 0 else 9}">{esc(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        estimated_width = max(len(line) for line in lines) * 4.2
        radians = math.radians(angle)
        estimated_right = x + len(lines) * 9.0 * math.sin(radians)
        estimated_bottom = y + estimated_width * math.sin(radians) + len(lines) * 9.0
        return (
            f'<text data-role="category-label" data-right="{estimated_right:.2f}" '
            f'data-bottom="{estimated_bottom:.2f}" '
            f'transform="translate({x:.2f} {y:.2f}) rotate(-{angle:g})" '
            f'text-anchor="end">{tspans}</text>'
        )
    tspans = "".join(
        f'<tspan x="{x:.2f}" dy="{0 if index == 0 else 9}">{esc(line)}</tspan>'
        for index, line in enumerate(lines)
    )


def _smooth_path(points: list[tuple[float, float]], move: bool = True) -> str:
    """Render a polyline as a shape-preserving cubic Bezier path."""
    if len(points) < 2:
        raise ValueError("smooth path requires at least two points")
    values = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("smooth path points must be finite")
    tangents = np.empty_like(values)
    tangents[0] = (values[1] - values[0]) / 6.0
    tangents[-1] = (values[-1] - values[-2]) / 6.0
    if len(values) > 2:
        tangents[1:-1] = (values[2:] - values[:-2]) / 6.0
    commands = [
        ("M" if move else "L") + f" {values[0, 0]:.2f},{values[0, 1]:.2f}"
    ]
    for index in range(len(values) - 1):
        start = values[index]
        end = values[index + 1]
        control_1 = start + tangents[index]
        control_2 = end - tangents[index + 1]
        low, high = sorted((start[1], end[1]))
        control_1[1] = np.clip(control_1[1], low, high)
        control_2[1] = np.clip(control_2[1], low, high)
        commands.append(
            "C "
            f"{control_1[0]:.2f},{control_1[1]:.2f} "
            f"{control_2[0]:.2f},{control_2[1]:.2f} "
            f"{end[0]:.2f},{end[1]:.2f}"
        )
    return " ".join(commands)


def _reduce_curve(centers: np.ndarray, density: np.ndarray, max_points: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Average adjacent KDE samples only for the final SVG rendering path."""
    if centers.size <= max_points:
        return centers, density
    groups = np.array_split(np.arange(centers.size), max_points)
    return (
        np.asarray([centers[group].mean() for group in groups], dtype=float),
        np.asarray([density[group].mean() for group in groups], dtype=float),
    )
    estimated_right = x + max(len(line) for line in lines) * 2.1
    estimated_bottom = y + len(lines) * 9.0
    return (
        f'<text data-role="category-label" data-right="{estimated_right:.2f}" '
        f'data-bottom="{estimated_bottom:.2f}" x="{x:.2f}" y="{y:.2f}" '
        f'text-anchor="middle">{tspans}</text>'
    )


def _auto_ylim(distributions: dict[str, dict[str, object]], base_hi: float) -> tuple[float, float]:
    lower = min(float(item["support_min"]) for item in distributions.values())
    upper = max(float(item["support_max"]) for item in distributions.values())
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("violin support must be finite")
    lo = 0.0
    # Use the requested fixed upper bound; density and summary geometry are
    # clipped to this visible range by the renderers.
    hi = float(base_hi)
    if not np.isfinite(hi) or hi <= 0.0:
        raise ValueError("CCI violin upper bound must be finite and positive")
    if hi <= lo:
        raise ValueError("violin y range must be positive")
    return lo, hi


def _axis_ticks(lo: float, hi: float) -> list[float]:
    if lo != 0.0 or hi <= 0.0:
        raise ValueError("CCI violin axis must start at zero with a positive range")
    ticks = [float(value) for value in range(0, int(math.floor(hi)) + 1)]
    if not math.isclose(hi, ticks[-1]):
        ticks.append(float(hi))
    return ticks


def _significance_label(q_value: float) -> str:
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    return "ns"


def _validate_pairwise(pairwise: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    required = ["group_1", "group_2", "q", "label"]
    _require_columns(pairwise, required, "pairwise Mann-Whitney BH")
    _require_nonblank(pairwise, ["group_1", "group_2"], "pairwise Mann-Whitney BH")
    validated = pairwise.copy()
    validated["q"] = _numeric(validated, "q", "pairwise Mann-Whitney BH")
    if np.any((validated["q"] < 0.0) | (validated["q"] > 1.0)):
        raise ValueError("pairwise Mann-Whitney BH.q must be within [0, 1]")
    expected_pairs = {frozenset(pair) for pair in ((order[0], order[1]), (order[0], order[2]), (order[1], order[2]))}
    actual_pairs = {
        frozenset((str(row.group_1), str(row.group_2)))
        for row in validated.itertuples(index=False)
    }
    if len(validated) != 3 or actual_pairs != expected_pairs:
        raise ValueError("pairwise Mann-Whitney BH must contain the three group pairs")
    validated["label"] = validated["q"].map(_significance_label)
    return validated


def violin_box_svg(
    distributions: dict[str, dict[str, object]],
    order: list[str],
    labels: list[str],
    out_file: Path,
    colors: dict[str, str],
    base_hi: float,
    width: int,
    height: int,
    rotate_labels: bool,
    pairwise: pd.DataFrame | None = None,
    note: str | None = None,
    label_angle: float = 35.0,
    axis_label: str = "CCI",
) -> str:
    """Render A-90% violins, white Tukey boxes, and optional paired brackets."""
    if len(order) == 0 or len(order) != len(labels) or set(order) != set(distributions):
        raise ValueError("violin order, labels, and distributions must align")
    top = 62.0 if pairwise is not None else 14.0
    wrapped = [textwrap.wrap(label, width=20, break_long_words=False) or [label] for label in labels]
    max_line_length = max(len(line) for lines in wrapped for line in lines)
    max_line_n = max(len(lines) for lines in wrapped)
    if rotate_labels:
        radians = math.radians(label_angle)
        label_height = max_line_length * 4.2 * math.sin(radians) + max_line_n * 9.0
        bottom = max(92.0, 22.0 + label_height)
        left = max(44.0, max_line_length * 4.2 * math.cos(radians) + 8.0)
    else:
        bottom = max(34.0, 22.0 + max_line_n * 9.0)
        left = 44.0
    right = 18.0
    plot_w, plot_h = width - left - right, height - top - bottom
    if plot_w <= 0 or plot_h <= 0:
        raise ValueError("SVG canvas is too small")
    ylim = _auto_ylim(distributions, base_hi)
    slot = plot_w / len(order)
    body: list[str] = []
    bottom_y = top + plot_h
    body.append(f'<line class="axis" x1="{left}" y1="{bottom_y}" x2="{left + plot_w}" y2="{bottom_y}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}"/>')
    for tick in _axis_ticks(*ylim):
        y = _scale(tick, ylim, (bottom_y, top))
        body.append(f'<line class="tick" x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}"/>')
        body.append(f'<text x="{left - 7}" y="{y + 3:.2f}" text-anchor="end">{tick:g}</text>')
    body.append(
        f'<text transform="translate(12 {top + plot_h / 2:.2f}) rotate(-90)" '
        f'text-anchor="middle">{esc(axis_label)}</text>'
    )
    centers_by_group = {}
    for index, group in enumerate(order):
        center_x = left + slot * (index + 0.5)
        centers_by_group[group] = center_x
        item = distributions[group]
        centers = np.asarray(item["centers"], dtype=float)
        density = np.asarray(item["density"], dtype=float)
        visible = np.isfinite(centers) & np.isfinite(density) & (centers >= ylim[0]) & (centers <= ylim[1])
        centers, density = centers[visible], density[visible]
        if centers.size < 2 or density.max(initial=0.0) <= 0.0:
            raise ValueError(f"{group} has no visible finite KDE support")
        centers, density = _reduce_curve(centers, density)
        half_widths = density * slot * 0.40
        left_points = [
            (center_x - half, _scale(value, ylim, (bottom_y, top)))
            for value, half in zip(centers, half_widths)
        ]
        right_points = [
            (center_x + half, _scale(value, ylim, (bottom_y, top)))
            for value, half in zip(centers[::-1], half_widths[::-1])
        ]
        body.append(
            f'<path class="violin" d="{_smooth_path(left_points)} '
            f'{_smooth_path(right_points, move=False)} Z" '
            f'fill="{colors[group]}" fill-opacity="0.62"/>'
        )
        summary = item["summary"]
        q1, median, q3 = (min(ylim[1], max(ylim[0], float(summary[key]))) for key in ("q1", "median", "q3"))
        whisker_low, whisker_high = (
            min(ylim[1], max(ylim[0], float(summary[key]))) for key in ("whisker_low", "whisker_high")
        )
        y_q1, y_median, y_q3, y_low, y_high = (
            _scale(value, ylim, (bottom_y, top))
            for value in (q1, median, q3, whisker_low, whisker_high)
        )
        box_half = min(slot * 0.09, 9.0)
        box_body_half = box_half
        body.append(f'<line class="line" x1="{center_x:.2f}" y1="{y_high:.2f}" x2="{center_x:.2f}" y2="{y_low:.2f}"/>')
        for y in (y_high, y_low):
            body.append(f'<line class="line" x1="{center_x - box_half:.2f}" y1="{y:.2f}" x2="{center_x + box_half:.2f}" y2="{y:.2f}"/>')
        body.append(
            f'<rect class="iqr-box" x="{center_x - box_body_half:.2f}" y="{y_q3:.2f}" '
            f'width="{2 * box_body_half:.2f}" height="{max(0.5, y_q1 - y_q3):.2f}" fill="white"/>'
        )
        body.append(f'<line class="median-line" x1="{center_x - box_half:.2f}" y1="{y_median:.2f}" x2="{center_x + box_half:.2f}" y2="{y_median:.2f}"/>')
        body.append(_label_text(center_x, bottom_y + 14, labels[index], rotate_labels, label_angle))

    if pairwise is not None:
        validated = _validate_pairwise(pairwise, order)
        pair_levels = {
            frozenset((order[0], order[1])): 48.0,
            frozenset((order[0], order[2])): 19.0,
            frozenset((order[1], order[2])): 34.0,
        }
        for row in validated.itertuples(index=False):
            x1, x2 = sorted((centers_by_group[str(row.group_1)], centers_by_group[str(row.group_2)]))
            y = pair_levels[frozenset((str(row.group_1), str(row.group_2)))]
            body.append(
                f'<path class="significance-bracket" d="M {x1:.2f},{y + 4:.2f} L {x1:.2f},{y:.2f} '
                f'L {x2:.2f},{y:.2f} L {x2:.2f},{y + 4:.2f}"/>'
            )
            body.append(f'<text x="{(x1 + x2) / 2:.2f}" y="{y - 3:.2f}" text-anchor="middle">{esc(row.label)}</text>')
    if note:
        body.append(f'<text x="{width - right:.2f}" y="10" text-anchor="end">{esc(note)}</text>')
    return write_svg(out_file, width, height, body)


def horizontal_violin_box_svg(
    distributions: dict[str, dict[str, object]],
    order: list[str],
    labels: list[str],
    out_file: Path,
    colors: dict[str, str],
    base_hi: float,
    width: int,
    height: int,
    physical_size: tuple[str, str] | None = None,
    median_marker: str = "line",
    axis_label: str = "CCI",
    category_axis_label: str = "Product",
) -> str:
    """Render A-90% horizontal violins with products on the y axis."""
    if len(order) == 0 or len(order) != len(labels) or set(order) != set(distributions):
        raise ValueError("horizontal violin order, labels, and distributions must align")
    if median_marker not in {"line", "long_line", "dot"}:
        raise ValueError("median_marker must be line, long_line, or dot")
    wrapped = [
        textwrap.wrap(label, width=28, break_long_words=False) or [label]
        for label in labels
    ]
    max_line_length = max(len(line) for lines in wrapped for line in lines)
    left = max(120.0, max_line_length * 4.2 + 18.0)
    right, top, bottom = 18.0, 14.0, 38.0
    plot_w, plot_h = width - left - right, height - top - bottom
    if plot_w <= 0 or plot_h <= 0:
        raise ValueError("horizontal violin SVG canvas is too small")
    xlim = _auto_ylim(distributions, base_hi)
    slot = plot_h / len(order)
    right_x = left + plot_w
    bottom_y = top + plot_h
    body: list[str] = [
        f'<line class="axis" x1="{left}" y1="{bottom_y}" x2="{right_x}" y2="{bottom_y}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}"/>',
    ]
    for tick in _axis_ticks(*xlim):
        x = _scale(tick, xlim, (left, right_x))
        body.append(
            f'<line class="tick" x1="{x:.2f}" y1="{bottom_y}" x2="{x:.2f}" y2="{bottom_y + 4}"/>'
        )
        body.append(
            f'<text x="{x:.2f}" y="{bottom_y + 15}" text-anchor="middle">{int(tick)}</text>'
        )
    body.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 5}" '
        f'text-anchor="middle">{esc(axis_label)}</text>'
    )
    if category_axis_label:
        body.append(
            f'<text transform="translate(12 {top + plot_h / 2:.2f}) rotate(-90)" '
            f'text-anchor="middle">{esc(category_axis_label)}</text>'
        )
    for index, group in enumerate(order):
        center_y = top + slot * (index + 0.5)
        item = distributions[group]
        centers = np.asarray(item["centers"], dtype=float)
        density = np.asarray(item["density"], dtype=float)
        visible = (
            np.isfinite(centers)
            & np.isfinite(density)
            & (centers >= xlim[0])
            & (centers <= xlim[1])
        )
        centers, density = centers[visible], density[visible]
        if centers.size < 2 or density.max(initial=0.0) <= 0.0:
            raise ValueError(f"{group} has no visible finite KDE support")
        half_heights = density * slot * 0.40
        upper_points = [
            f"{_scale(value, xlim, (left, right_x)):.2f},{center_y - half:.2f}"
            for value, half in zip(centers, half_heights)
        ]
        lower_points = [
            f"{_scale(value, xlim, (left, right_x)):.2f},{center_y + half:.2f}"
            for value, half in zip(centers[::-1], half_heights[::-1])
        ]
        body.append(
            f'<path class="violin" d="M {" L ".join(upper_points + lower_points)} Z" '
            f'fill="{colors[group]}" fill-opacity="0.62"/>'
        )
        summary = item["summary"]
        q1, median, q3 = (min(xlim[1], max(xlim[0], float(summary[key]))) for key in ("q1", "median", "q3"))
        whisker_low, whisker_high = (
            min(xlim[1], max(xlim[0], float(summary[key]))) for key in ("whisker_low", "whisker_high")
        )
        x_q1, x_median, x_q3, x_low, x_high = (
            _scale(value, xlim, (left, right_x))
            for value in (q1, median, q3, whisker_low, whisker_high)
        )
        box_half = min(slot * 0.09, 9.0)
        box_body_half = box_half
        body.append(
            f'<line class="line" x1="{x_low:.2f}" y1="{center_y:.2f}" '
            f'x2="{x_high:.2f}" y2="{center_y:.2f}"/>'
        )
        for x in (x_low, x_high):
            body.append(
                f'<line class="line" x1="{x:.2f}" y1="{center_y - box_half:.2f}" '
                f'x2="{x:.2f}" y2="{center_y + box_half:.2f}"/>'
            )
        body.append(
            f'<rect class="iqr-box" x="{x_q1:.2f}" y="{center_y - box_body_half:.2f}" '
            f'width="{max(0.5, x_q3 - x_q1):.2f}" height="{2 * box_body_half:.2f}" fill="white"/>'
        )
        if median_marker == "dot":
            body.append(
                f'<circle class="median-dot" cx="{x_median:.2f}" cy="{center_y:.2f}" r="3.5"/>'
            )
        else:
            median_half = min(slot * 0.22, 16.0) if median_marker == "long_line" else box_half
            body.append(
                f'<line class="median-line" x1="{x_median:.2f}" y1="{center_y - median_half:.2f}" '
                f'x2="{x_median:.2f}" y2="{center_y + median_half:.2f}"/>'
            )
        lines = wrapped[index]
        first_y = center_y - (len(lines) - 1) * 4.5
        tspans = "".join(
            f'<tspan x="{left - 8:.2f}" dy="{0 if line_index == 0 else 9}">{esc(line)}</tspan>'
            for line_index, line in enumerate(lines)
        )
        body.append(
            f'<text data-role="category-label" data-right="{left - 8:.2f}" '
            f'data-bottom="{first_y + (len(lines) - 1) * 9 + 3:.2f}" '
            f'x="{left - 8:.2f}" y="{first_y:.2f}" text-anchor="end">{tspans}</text>'
        )
    return write_svg(out_file, width, height, body, physical_size)


def all_gene_distributions(
    histogram: pd.DataFrame, stats_frame: pd.DataFrame
) -> dict[str, dict[str, object]]:
    """Validate shared continuous bins and construct all-gene KDE inputs."""
    _require_columns(
        histogram,
        ["Gene_group", "bin_left", "bin_right", "bin_center", "count"],
        "all-gene histogram",
    )
    _require_columns(stats_frame, ["Gene_group"], "all-gene stats")
    _require_nonblank(histogram, ["Gene_group"], "all-gene histogram")
    _require_nonblank(stats_frame, ["Gene_group"], "all-gene stats")
    if set(histogram["Gene_group"]) != set(GROUP_ORDER) or set(stats_frame["Gene_group"]) != set(GROUP_ORDER):
        raise ValueError("all-gene histogram and stats group sets must match GROUP_ORDER")
    if stats_frame["Gene_group"].duplicated().any():
        raise ValueError("all-gene stats must contain one row per Gene_group")
    output = {}
    shared_edges = None
    for group in GROUP_ORDER:
        group_hist = histogram.loc[histogram["Gene_group"].eq(group)].sort_values(
            "bin_left", kind="stable"
        )
        numeric = group_hist.copy()
        for column in ["bin_left", "bin_right", "bin_center", "count"]:
            numeric[column] = _numeric(numeric, column, "all-gene histogram")
        left = numeric["bin_left"].to_numpy(float)
        right = numeric["bin_right"].to_numpy(float)
        center = numeric["bin_center"].to_numpy(float)
        if len(numeric) < 2 or not np.allclose(right[:-1], left[1:]):
            raise ValueError("all-gene histogram bins must be continuous")
        if not np.allclose(center, (left + right) / 2.0):
            raise ValueError("all-gene histogram bin_center must be the midpoint")
        edges = np.concatenate([left[:1], right])
        if shared_edges is None:
            shared_edges = edges
        elif not np.allclose(edges, shared_edges):
            raise ValueError("all-gene histogram groups must use shared edges")
        stats_row = stats_frame.loc[stats_frame["Gene_group"].eq(group)].iloc[0]
        output[group] = histogram_distribution(numeric, stats_row)
    return output


def load_thresholds(package: Path) -> tuple[int, int, int]:
    metadata_path = package / "Fig3_downstream_tables_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = metadata["thresholds"]
        minimum_product = values["min_product_gene_n"]
        top_product = values["top_product_n"]
        minimum_phylum = values["min_phylum_plasmid_n"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("metadata thresholds are missing or invalid") from error
    numbers = [minimum_product, top_product, minimum_phylum]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in numbers):
        raise ValueError("metadata thresholds must be integers")
    if minimum_product < 0 or top_product <= 0 or minimum_phylum <= 0:
        raise ValueError("metadata thresholds are outside their valid ranges")
    return minimum_product, top_product, minimum_phylum


def preflight_inputs(
    package: Path,
    minimum_product: int,
    top_product: int,
    minimum_phylum: int,
) -> None:
    """Prove every input exists before creating either figure directory."""
    package_names = [
        "Fig3_downstream_tables_metadata.json",
        "Fig3B_all_gene_distribution_histogram.csv",
        "Fig3B_all_gene_distribution_stats.csv",
        "FigS11C_CDS_pairwise_Mann_Whitney_BH.csv",
        f"Fig3C_product_median_CCI_gene_n_gt{minimum_product}.csv",
        f"Fig3C_top{top_product}_product_gene_CCI_distribution.csv",
        "Fig3D_COG_performance_vs_product_median_CCI.csv",
        "Fig3E_plasmid_level_CCI_for_phylum.csv",
        f"Fig3E_phylum_plasmid_level_CCI_plasmid_n_ge{minimum_phylum}.csv",
    ]
    missing = [str(package / name) for name in package_names if not (package / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required plotting inputs: " + "; ".join(missing))


def plot_b(package: Path, supplement: Path, fig_dir: Path, supplement_fig: Path) -> None:
    histogram = pd.read_csv(package / "Fig3B_all_gene_distribution_histogram.csv", keep_default_na=False)
    stats_frame = pd.read_csv(package / "Fig3B_all_gene_distribution_stats.csv", keep_default_na=False)
    cds_pairwise = pd.read_csv(
        package / "FigS11C_CDS_pairwise_Mann_Whitney_BH.csv",
        keep_default_na=False,
    )
    all_gene = all_gene_distributions(histogram, stats_frame)
    all_gene_path = fig_dir / "Fig3B_gene_level_CCI_chr_trnap_trnam.svg"
    violin_box_svg(
        all_gene,
        GROUP_ORDER,
        [GROUP_LABELS[group] for group in GROUP_ORDER],
        all_gene_path,
        B_COLORS,
        CCI_AXIS_MAX,
        500,
        320,
        True,
        cds_pairwise,
        "CDS-level tests",
        label_angle=20.0,
        axis_label="CDS-level CCI",
    )


def plot_c(package: Path, fig_dir: Path, minimum: int, top_n: int) -> None:
    summary = pd.read_csv(
        package / f"Fig3C_product_median_CCI_gene_n_gt{minimum}.csv",
        keep_default_na=False,
    ).head(top_n)
    distribution = pd.read_csv(
        package / f"Fig3C_top{top_n}_product_gene_CCI_distribution.csv",
        keep_default_na=False,
    )
    _require_columns(summary, ["Product", "Product_norm", "gene_n"], "strict product summary")
    _require_nonblank(summary, ["Product", "Product_norm"], "strict product summary")
    summary["gene_n"] = _numeric(summary, "gene_n", "strict product summary")
    if (
        summary.empty
        or np.any(summary["gene_n"] <= minimum)
        or np.any(summary["gene_n"] != np.floor(summary["gene_n"]))
        or summary["Product_norm"].duplicated().any()
    ):
        raise ValueError(f"strict product summary must contain unique products with integer gene_n > {minimum}")
    _require_columns(distribution, ["Product_norm", "CCI_gene"], "top product distribution")
    _require_nonblank(distribution, ["Product_norm"], "top product distribution")
    order = summary["Product_norm"].astype(str).tolist()
    observed_products = set(distribution["Product_norm"].astype(str))
    if observed_products != set(order):
        raise ValueError("top product distribution Product_norm set does not match summary")
    items = {}
    for product, expected_n in zip(order, summary["gene_n"].astype(int)):
        rows = distribution.loc[distribution["Product_norm"].astype(str).eq(product)]
        if len(rows) != expected_n:
            raise ValueError(f"top product distribution gene_n mismatch for {product}")
        items[product] = raw_distribution(_numeric(rows, "CCI_gene", "top product distribution").to_numpy(float))
    horizontal_violin_box_svg(
        items,
        order,
        [_title_label(value) for value in summary["Product"]],
        fig_dir / f"Fig3C_top{top_n}_product_CCI_violin_box.svg",
        {product: VIOLIN_FILL for product in order},
        6.0,
        600,
        700,
        ("6cm", "7cm"),
    )


def plot_f(package: Path, fig_dir: Path, minimum: int) -> None:
    summary = pd.read_csv(
        package / f"Fig3E_phylum_plasmid_level_CCI_plasmid_n_ge{minimum}.csv",
        keep_default_na=False,
    )
    plasmids = pd.read_csv(
        package / "Fig3E_plasmid_level_CCI_for_phylum.csv", keep_default_na=False
    )
    _require_columns(summary, ["phylum", "plasmid_n"], "phylum summary")
    _require_nonblank(summary, ["phylum"], "phylum summary")
    summary["plasmid_n"] = _numeric(summary, "plasmid_n", "phylum summary")
    if (
        summary.empty
        or np.any(summary["plasmid_n"] < minimum)
        or np.any(summary["plasmid_n"] != np.floor(summary["plasmid_n"]))
        or summary["phylum"].duplicated().any()
    ):
        raise ValueError(f"phylum summary must contain unique phyla with integer plasmid_n >= {minimum}")
    _require_columns(plasmids, ["phylum", "plasmid_CCI_big_gene"], "plasmid-level CCI")
    _require_nonblank(plasmids, ["phylum"], "plasmid-level CCI")
    order = summary["phylum"].astype(str).tolist()
    observed_phyla = set(plasmids["phylum"].astype(str))
    if not set(order).issubset(observed_phyla):
        raise ValueError("phylum summary is not a subset of plasmid median phyla")
    plasmids = plasmids.loc[plasmids["phylum"].astype(str).isin(order)].copy()
    items = {}
    for phylum, expected_n in zip(order, summary["plasmid_n"].astype(int)):
        rows = plasmids.loc[plasmids["phylum"].astype(str).eq(phylum)]
        if len(rows) != expected_n:
            raise ValueError(f"plasmid-level CCI plasmid_n mismatch for {phylum}")
        items[phylum] = raw_distribution(
            _numeric(rows, "plasmid_CCI_big_gene", "plasmid-level CCI").to_numpy(float),
            bin_n=400,
        )
    horizontal_violin_box_svg(
        items,
        order,
        [_title_label(value) for value in order],
        fig_dir / "Fig3E_phylum_CCI.svg",
        {phylum: VIOLIN_FILL for phylum in order},
        2.0,
        500,
        620,
        axis_label="Plasmid-level CCI",
        category_axis_label="",
    )


def _product_count_bin(value: float) -> str:
    if not np.isfinite(value) or value < 0.0 or not float(value).is_integer():
        raise ValueError("product_n must be a finite non-negative integer")
    if value < 100:
        return "<100"
    if value <= 1000:
        return "100–1000"
    return ">1000"


def cog_scatter_svg(
    frame: pd.DataFrame,
    out_file: Path,
    metric_column: str,
    metric_label: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    width: int = 600,
    height: int = 1000,
    x_tick_count: int = 5,
    y_tick_count: int = 3,
) -> str:
    """Render COG-level CCI against held-out predictive performance."""
    required = [
        "COG_category",
        "COG_super_category",
        metric_column,
        "CCI_median",
        "product_n",
        "size_bin",
    ]
    _require_columns(frame, required, "COG scatter")
    _require_nonblank(frame, ["COG_category", "COG_super_category", "size_bin"], "COG scatter")
    if frame.empty or frame["COG_category"].duplicated().any():
        raise ValueError("COG scatter categories must be nonempty and unique")
    plot = frame.copy()
    for column in [metric_column, "CCI_median", "product_n"]:
        plot[column] = _numeric(plot, column, "COG scatter")
    expected_bins = [_product_count_bin(value) for value in plot["product_n"]]
    if expected_bins != plot["size_bin"].astype(str).tolist():
        raise ValueError("size_bin does not match product_n")
    unknown = sorted(set(plot["COG_super_category"]) - set(COG_COLORS))
    if unknown:
        raise ValueError(f"unknown COG_super_category values: {unknown}")

    left, right, top, bottom = 62.0, 170.0, 24.0, 50.0
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = plot[metric_column].to_numpy(float)
    y_values = plot["CCI_median"].to_numpy(float)
    xlim = tuple(float(value) for value in x_bounds)
    if len(xlim) != 2 or not np.all(np.isfinite(xlim)) or xlim[1] <= xlim[0]:
        raise ValueError("COG scatter x-axis bounds must be finite and increasing")
    ylim = tuple(float(value) for value in y_bounds)
    if len(ylim) != 2 or not np.all(np.isfinite(ylim)) or ylim[1] <= ylim[0]:
        raise ValueError("COG scatter y-axis bounds must be finite and increasing")
    if x_tick_count < 2 or y_tick_count < 2:
        raise ValueError("COG scatter tick counts must be at least two")
    x0, x1 = left, left + plot_w
    y0, y1 = top + plot_h, top
    body = []
    body.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    body.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}"/>')
    for tick in np.linspace(*xlim, x_tick_count):
        x = _scale(float(tick), xlim, (x0, x1))
        body.append(f'<line class="tick" x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + 4}"/>')
        body.append(f'<text x="{x:.2f}" y="{y0 + 15}" text-anchor="middle">{tick:.4g}</text>')
    for tick in np.linspace(*ylim, y_tick_count):
        y = _scale(float(tick), ylim, (y0, y1))
        body.append(f'<line class="tick" x1="{x0 - 4}" y1="{y:.2f}" x2="{x0}" y2="{y:.2f}"/>')
        body.append(f'<text x="{x0 - 7}" y="{y + 3:.2f}" text-anchor="end">{tick:.4g}</text>')
    body.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 7}" '
        f'text-anchor="middle">{esc(metric_label)}</text>'
    )
    body.append(f'<text transform="translate(13 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">COG category median product-level CCI</text>')
    cci_median = float(np.median(y_values))
    cci_y = _scale(cci_median, ylim, (y0, y1))
    median_x_value = float(np.median(x_values))
    median_x = _scale(median_x_value, xlim, (x0, x1))
    body.append(f'<line class="reference-line" x1="{x0}" y1="{cci_y:.2f}" x2="{x1}" y2="{cci_y:.2f}" stroke-dasharray="4 3"/>')
    body.append(f'<text x="{x1 - 3}" y="{cci_y - 3:.2f}" text-anchor="end">COG CCI median = {cci_median:.2f}</text>')
    body.append(f'<line class="reference-line" x1="{median_x:.2f}" y1="{y1}" x2="{median_x:.2f}" y2="{y0}" stroke-dasharray="4 3"/>')
    body.append(
        f'<text x="{median_x + 3:.2f}" y="{y1 + 9}" '
        f'text-anchor="start">Median {esc(metric_label)}</text>'
    )
    body.append(
        f'<defs><clipPath id="cog-plot-clip"><rect x="{x0}" y="{y1}" '
        f'width="{plot_w}" height="{plot_h}"/></clipPath></defs>'
    )
    body.append('<g clip-path="url(#cog-plot-clip)">')
    for row in plot.itertuples(index=False):
        x = x0 + (float(getattr(row, metric_column)) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        y = _scale(float(row.CCI_median), ylim, (y0, y1))
        radius = SIZE_RADII[str(row.size_bin)]
        color = COG_COLORS[str(row.COG_super_category)]
        body.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.86"/>')
        body.append(f'<text x="{x:.2f}" y="{y + 2.5:.2f}" text-anchor="middle">{esc(row.COG_category)}</text>')
    body.append('</g>')
    legend_x, legend_y = x1 + 22.0, y1 + 18.0
    body.append(
        f'<g data-role="legend" data-plot-right="{x1:.2f}" '
        f'data-legend-left="{legend_x:.2f}">'
    )
    body.append(f'<text x="{legend_x}" y="{legend_y - 8}">Product count</text>')
    present_size_bins = [
        label for label in SIZE_RADII if label in set(plot["size_bin"].astype(str))
    ]
    for index, label in enumerate(present_size_bins):
        radius = SIZE_RADII[label]
        cy = legend_y + index * 22.0
        body.append(f'<circle class="point" cx="{legend_x + 8}" cy="{cy}" r="{radius}" fill="white"/>')
        body.append(f'<text x="{legend_x + 23}" y="{cy + 3}">{esc(label)}</text>')
    color_y = legend_y + len(present_size_bins) * 22.0 + 16.0
    body.append(f'<text x="{legend_x}" y="{color_y}">COG super-category</text>')
    for index, (label, color) in enumerate(COG_COLORS.items()):
        cy = color_y + 14.0 + index * 27.0
        body.append(f'<circle class="point" cx="{legend_x + 6}" cy="{cy}" r="4" fill="{color}"/>')
        wrapped = textwrap.wrap(label, width=22, break_long_words=False) or [label]
        body.append(
            f'<text x="{legend_x + 17}" y="{cy - 2}">'
            + "".join(
                f'<tspan x="{legend_x + 17}" dy="{0 if line_index == 0 else 8}">{esc(line)}</tspan>'
                for line_index, line in enumerate(wrapped)
            )
            + "</text>"
        )
    body.append("</g>")
    return write_svg(out_file, width, height, body, physical_size=("90mm", "56mm"))


def plot_d(package: Path, fig_dir: Path, supplement_fig: Path) -> None:
    frame = pd.read_csv(
        package / "Fig3D_COG_performance_vs_product_median_CCI.csv",
        keep_default_na=False,
    )
    _require_columns(frame, ["product_n"], "COG scatter")
    product_n = _numeric(frame, "product_n", "COG scatter")
    frame = frame.loc[product_n.gt(50)].copy()
    if frame.empty:
        raise ValueError("COG scatter has no categories with product_n > 50")
    cog_scatter_svg(
        frame,
        fig_dir / "Fig3D_COG_AUPRC_vs_median_CCI.svg",
        "AUPRC_test_mean",
        "COG-category classification AUPRC",
        (0.70, 0.90),
        (1.10, 1.25),
        900,
        560,
        5,
        3,
    )
    cog_scatter_svg(
        frame,
        supplement_fig / "S12_COG_R2_vs_median_CCI.svg",
        "R2_test_mean",
        "COG-category regression R²",
        (0.50, 0.80),
        (1.10, 1.25),
        900,
        560,
        5,
        3,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    args = parse_args()
    minimum_product, top_product, minimum_phylum = load_thresholds(args.package_dir)
    preflight_inputs(
        args.package_dir,
        minimum_product,
        top_product,
        minimum_phylum,
    )
    fig_dir = PACKAGE_ROOT / "fig"
    supplement_fig = PACKAGE_ROOT / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    supplement_fig.mkdir(parents=True, exist_ok=True)
    plot_b(args.package_dir, PACKAGE_ROOT, fig_dir, supplement_fig)
    plot_c(args.package_dir, fig_dir, minimum_product, top_product)
    plot_d(args.package_dir, fig_dir, supplement_fig)
    plot_f(args.package_dir, fig_dir, minimum_phylum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
