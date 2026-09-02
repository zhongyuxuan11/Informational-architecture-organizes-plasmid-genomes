"""Generate the manuscript-style Fig. 2F as deterministic vector SVG."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path


BLUE = "#8EBAD7"
CORAL = "#EDB29B"
INK = "#202020"
FONT = "Arial"
FONT_SIZE = 7
OBJECT_LW = 0.7
AXIS_LW = 1.0
WIDTH = 987.28332
HEIGHT = 384.568513
WIDTH_MM = 170.0
HEIGHT_MM = 45.0
PT_PER_MM = 72.0 / 25.4


def load_metrics(path: Path) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"AUPRC": [], "R2": []}
    )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            code = row["COG_category"]
            task = row["task"]
            if task == "classification":
                values[code]["AUPRC"].append(float(row["AUPRC"]))
            elif task.startswith("regression"):
                values[code]["R2"].append(float(row["R2"]))
    if not values:
        raise ValueError("No COG metrics were loaded")
    for code, metrics in values.items():
        for metric, runs in metrics.items():
            if len(runs) != 3:
                raise ValueError(f"{code} {metric} must contain exactly three runs")
    return dict(values)


def load_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        labels = {row["COG"]: row["COG_description"] for row in csv.DictReader(handle)}
    if not labels:
        raise ValueError("No COG descriptions were loaded")
    return labels


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def line(x1: float, y1: float, x2: float, y2: float, width: float = OBJECT_LW) -> str:
    return (
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{INK}" stroke-width="{width}"/>'
    )


def text(
    x: float,
    y: float,
    value: object,
    *,
    anchor: str = "middle",
    rotate: float | None = None,
    weight: str = "normal",
) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.3f} {y:.3f})"'
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}"{transform} '
        f'font-family="{FONT}" font-size="{FONT_SIZE}" font-weight="{weight}" '
        f'fill="{INK}">{esc(value)}</text>'
    )


def build_svg(metrics_path: Path, labels_path: Path, output: Path) -> None:
    metrics = load_metrics(metrics_path)
    labels = load_labels(labels_path)
    if not set(metrics).issubset(labels):
        missing = sorted(set(metrics).difference(labels))
        raise ValueError(f"Missing COG descriptions: {missing}")

    order = sorted(metrics, key=lambda code: statistics.mean(metrics[code]["AUPRC"]), reverse=True)
    width = WIDTH_MM * PT_PER_MM
    height = HEIGHT_MM * PT_PER_MM
    axis_left, data_left, right, top, bottom = 25.0, 31.0, width - 37.0, 8.0, 98.0
    plot_width, plot_height = right - data_left, bottom - top
    group_width = plot_width / len(order)
    bar_width = group_width * 0.36
    metric_specs = (
        ("AUPRC", BLUE, -bar_width / 2),
        ("R2", CORAL, bar_width / 2),
    )

    def y_pos(value: float) -> float:
        return bottom - value / 1.05 * plot_height

    svg: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM}mm" height="{HEIGHT_MM}mm" '
        f'viewBox="0 0 {width:.6f} {height:.6f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_pos(tick)
        svg.append(line(axis_left - 3.5, y, axis_left, y))
        svg.append(text(axis_left - 7.0, y + 2.4, f"{tick:.1f}", anchor="end"))
    svg.extend(
        [
            line(axis_left, top, axis_left, bottom, AXIS_LW),
            line(axis_left, bottom, right, bottom, AXIS_LW),
            text(8.2, (top + bottom) / 2, "Model performance", rotate=-90),
        ]
    )

    for index, code in enumerate(order):
        center = data_left + group_width * (index + 0.5)
        svg.append(line(center, bottom, center, bottom + 3.0))
        svg.append(
            text(
                center,
                bottom + 10.0,
                code,
            )
        )
        for metric, color, shift in metric_specs:
            runs = metrics[code][metric]
            mean = statistics.mean(runs)
            sd = statistics.stdev(runs)
            x_center = center + shift
            y_mean = y_pos(mean)
            y_zero = y_pos(0)
            svg.append(
                f'<rect x="{x_center - bar_width / 2:.3f}" y="{y_mean:.3f}" '
                f'width="{bar_width:.3f}" height="{y_zero - y_mean:.3f}" fill="{color}"/>'
            )
            y_high = y_pos(min(1.05, mean + sd))
            y_low = y_pos(max(0.0, mean - sd))
            svg.extend(
                [
                    line(x_center, y_high, x_center, y_low),
                    line(x_center - 2.0, y_high, x_center + 2.0, y_high),
                    line(x_center - 2.0, y_low, x_center + 2.0, y_low),
                ]
            )
            for run_index, value in enumerate(runs):
                point_x = x_center + (-2.1, 0.0, 2.1)[run_index]
                point_y = y_pos(value)
                svg.append(f'<circle cx="{point_x:.3f}" cy="{point_y:.3f}" r="0.825" fill="black"/>')
            label_y = max(top + 4.0, y_high - 2.5)
            svg.append(text(x_center, label_y, f"{mean:.2f}", rotate=-90))

    legend_x, legend_y = right + 7.0, top + 8.0
    for index, (label, color) in enumerate((("AUPRC", BLUE), ("R²", CORAL))):
        y = legend_y + index * 11.5
        svg.append(f'<rect x="{legend_x:.3f}" y="{y - 6.0:.3f}" width="9" height="5" fill="{color}"/>')
        svg.append(text(legend_x + 13.0, y - 1.5, label, anchor="start"))

    svg.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".svg":
        raise ValueError("Output must be an SVG file")
    build_svg(args.metrics, args.labels, args.output)


if __name__ == "__main__":
    main()
