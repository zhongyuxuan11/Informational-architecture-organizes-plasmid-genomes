"""Redraw Figure S9 as deterministic SVG without Matplotlib."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path


WIDTH = 367.242756
HEIGHT = 173.227717
BLUE = "#8EBAD7"
CORAL = "#EDB29B"
GREY = "#BDBDBD"
PURPLE = "#C6C0D8"
INK = "#202020"
FONT = "Arial"
FONT_SIZE = 7
OBJECT_LW = 0.7
AXIS_LW = 1.0


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


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


def line(x1: float, y1: float, x2: float, y2: float, width: float = OBJECT_LW) -> str:
    return (
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{INK}" stroke-width="{width}"/>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none") -> str:
    return (
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{OBJECT_LW}"/>'
    )


def load_r2(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            values[row["subset"]].append(float(row["R2"]))
    required = {
        "complete_RefSeq_test",
        "PlasFlow_supported_plasmid_like",
        "Other_replicons_PlasFlow",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing PlasFlow subsets: {missing}")
    for subset in required:
        if len(values[subset]) != 3:
            raise ValueError(f"{subset} must contain exactly three runs")
    return dict(values)


def y_from_value(value: float, top: float, bottom: float, ymax: float) -> float:
    return bottom - value / ymax * (bottom - top)


def draw_axes(svg: list[str], left: float, top: float, right: float, bottom: float) -> None:
    svg.append(line(left, top, left, bottom, AXIS_LW))
    svg.append(line(left, bottom, right, bottom, AXIS_LW))


def draw_ticks(
    svg: list[str],
    left: float,
    top: float,
    bottom: float,
    ticks: tuple[float, ...],
    ymax: float,
    fmt: str,
) -> None:
    for tick in ticks:
        y = y_from_value(tick, top, bottom, ymax)
        svg.append(line(left - 3.0, y, left, y))
        svg.append(text(left - 5.0, y + 2.2, fmt.format(tick), anchor="end"))


def draw_panel_a(svg: list[str]) -> None:
    left, top, right, bottom = 44.069, 15.590, 149.799, 135.118
    draw_axes(svg, left, top, right, bottom)
    draw_ticks(svg, left, top, bottom, (0, 20, 40, 60, 80, 100), 100, "{:.0f}")
    svg.append(text(12.0, (top + bottom) / 2, "Proportion of tRNA-bearing", rotate=-90))
    svg.append(text(21.0, (top + bottom) / 2, "plasmid records (%)", rotate=-90))
    center, bar_w = (left + right) / 2, 30.0
    current = bottom
    for label, value, fill in (
        ("Plasmid", 61.2, CORAL),
        ("Unclassified", 23.9, GREY),
        ("Chromosome", 14.9, PURPLE),
    ):
        height = value / 100.0 * (bottom - top)
        current -= height
        svg.append(rect(center - bar_w / 2, current, bar_w, height, fill, "white"))
    svg.append(text(center, bottom + 11.0, "PlasFlow"))
    svg.append(text(left - 28.0, top - 4.0, "A", anchor="start", weight="bold"))
    legend_x, legend_y = left + 6.0, top + 8.0
    for idx, (label, fill) in enumerate((("Plasmid", CORAL), ("Unclassified", GREY), ("Chromosome", PURPLE))):
        y = legend_y + idx * 10.0
        svg.append(rect(legend_x, y - 5.0, 8.0, 5.0, fill, fill))
        svg.append(text(legend_x + 11.0, y - 1.0, label, anchor="start"))


def draw_panel_b(svg: list[str], values: dict[str, list[float]]) -> None:
    left, top, right, bottom = 210.217, 15.590, 356.225, 135.118
    draw_axes(svg, left, top, right, bottom)
    draw_ticks(svg, left, top, bottom, (0, 0.2, 0.4, 0.6, 0.8, 1.0), 1.02, "{:.1f}")
    svg.append(text(left - 25.0, (top + bottom) / 2, "R²", rotate=-90))
    svg.append(text(left - 31.0, top - 4.0, "B", anchor="start", weight="bold"))

    order = (
        ("complete_RefSeq_test", ("Full baseline",)),
        ("PlasFlow_supported_plasmid_like", ("Plasmid", "(PlasFlow)")),
        ("Other_replicons_PlasFlow", ("Other replicons", "(PlasFlow)")),
    )
    group_w = (right - left) / len(order)
    bar_w = 26.0
    for idx, (subset, label_lines) in enumerate(order):
        runs = values[subset]
        mean = statistics.mean(runs)
        sd = statistics.stdev(runs)
        x = left + group_w * (idx + 0.5)
        y_mean = y_from_value(mean, top, bottom, 1.02)
        y_zero = y_from_value(0, top, bottom, 1.02)
        svg.append(rect(x - bar_w / 2, y_mean, bar_w, y_zero - y_mean, CORAL, "none"))
        y_high = y_from_value(min(1.02, mean + sd), top, bottom, 1.02)
        y_low = y_from_value(max(0.0, mean - sd), top, bottom, 1.02)
        svg.append(line(x, y_high, x, y_low, AXIS_LW))
        svg.append(line(x - 5.0, y_high, x + 5.0, y_high, AXIS_LW))
        svg.append(line(x - 5.0, y_low, x + 5.0, y_low, AXIS_LW))
        for run_idx, value in enumerate(runs):
            px = x + (-2.6, 0.0, 2.6)[run_idx]
            py = y_from_value(value, top, bottom, 1.02)
            svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="1.2" fill="black"/>')
        for line_idx, label in enumerate(label_lines):
            svg.append(text(x, bottom + 10.0 + line_idx * 8.0, label))


def build_svg(input_csv: Path, output_svg: Path) -> None:
    values = load_r2(input_csv)
    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH:.6f}pt" height="{HEIGHT:.6f}pt" viewBox="0 0 {WIDTH:.6f} {HEIGHT:.6f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    draw_panel_a(svg)
    draw_panel_b(svg, values)
    svg.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    build_svg(args.input_csv, args.output_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

