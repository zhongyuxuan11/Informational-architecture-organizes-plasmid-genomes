"""Redraw S16 and Fig4 central-dogma ML panels as deterministic SVG."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path


FONT = "Arial"
FONT_SIZE = 7
INK = "#202020"
BLUE = "#8EBAD7"
CORAL = "#EDB29B"
CREAM = "#F4ECD2"
GREY = "#D9D9D9"
OBJECT_LW = 0.7
AXIS_LW = 1.0


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x, y, value, *, anchor="middle", rotate=None, weight="normal"):
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.3f} {y:.3f})"'
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}"{transform} '
        f'font-family="{FONT}" font-size="{FONT_SIZE}" font-weight="{weight}" fill="{INK}">{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, width=OBJECT_LW):
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{INK}" stroke-width="{width}"/>'


def rect(x, y, w, h, fill, stroke=INK):
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}" stroke="{stroke}" stroke-width="{OBJECT_LW}"/>'


def circle(x, y, r=1.15):
    return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" fill="black"/>'


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No central-dogma rows loaded")
    return rows


def metric_values(rows, analysis, target, setting, metric):
    values = [
        float(row[metric])
        for row in rows
        if row["analysis"] == analysis and row["target"] == target and row["feature_setting"] == setting
    ]
    if len(values) != 3:
        raise ValueError(f"Expected three runs for {analysis} / {target} / {setting} / {metric}")
    return values


def mean_metric(rows, analysis, target, setting, metric):
    return statistics.mean(metric_values(rows, analysis, target, setting, metric))


def y_pos(value, top, bottom, ymax=1.03):
    return bottom - value / ymax * (bottom - top)


def draw_axes(svg, left, top, right, bottom, ylabel, ymax=1.03):
    svg.append(line(left, top, left, bottom, AXIS_LW))
    svg.append(line(left, bottom, right, bottom, AXIS_LW))
    for tick in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_pos(tick, top, bottom, ymax)
        svg.append(line(left - 3, y, left, y))
        svg.append(text(left - 5, y + 2.3, f"{tick:.1f}".rstrip("0").rstrip("."), anchor="end"))
    svg.append(text(left - 24, (top + bottom) / 2, ylabel, rotate=-90))


def draw_grouped(svg, rows, x0, y0, w, h, analysis, targets, settings, metrics, colors, ylabel, labels):
    left, top, right, bottom = x0 + 29, y0 + 6, x0 + w - 4, y0 + h - 32
    draw_axes(svg, left, top, right, bottom, ylabel)
    group_w = (right - left) / len(targets)
    bar_w = group_w / (len(settings) * len(metrics) + 1.0)
    for gi, target in enumerate(targets):
        cx0 = left + group_w * gi
        base = cx0 + group_w * 0.5 - bar_w * (len(settings) * len(metrics) - 1) / 2
        for si, setting in enumerate(settings):
            for mi, metric in enumerate(metrics):
                runs = metric_values(rows, analysis, target, setting, metric)
                mean = statistics.mean(runs)
                sd = statistics.stdev(runs)
                x = base + (si * len(metrics) + mi) * bar_w
                ym = y_pos(mean, top, bottom)
                yz = y_pos(0, top, bottom)
                svg.append(rect(x - bar_w * 0.38, ym, bar_w * 0.76, yz - ym, colors[si * len(metrics) + mi], "none"))
                yh = y_pos(min(1.03, mean + sd), top, bottom)
                yl = y_pos(max(0, mean - sd), top, bottom)
                svg.append(line(x, yh, x, yl))
                svg.append(line(x - 2.5, yh, x + 2.5, yh))
                svg.append(line(x - 2.5, yl, x + 2.5, yl))
                for ri, value in enumerate(runs):
                    svg.append(circle(x + (-1.6, 0, 1.6)[ri], y_pos(value, top, bottom)))
        for li, label in enumerate(labels.get(target, target).split("\n")):
            svg.append(text(cx0 + group_w * 0.5, bottom + 10 + li * 8, label))


def build_s16(rows, output: Path):
    width, height = 279.001654, 216.712913
    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.6f}pt" height="{height:.6f}pt" viewBox="0 0 {width:.6f} {height:.6f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    targets = ["L", "K", "J"]
    labels = {
        "L": "L: Replication,\nrecombination and repair",
        "K": "K: Transcription",
        "J": "J: Translation, ribosomal\nstructure and biogenesis",
    }
    draw_grouped(svg, rows, 15, 26, 250, 175, "cog_lkj", targets, ["all_non_target_features"], ["AUPRC", "R2"], [BLUE, CORAL], "Performance", labels)
    svg.append(rect(119, 14, 9, 5, BLUE, "none"))
    svg.append(text(132, 18, "AUPRC", anchor="start"))
    svg.append(rect(176, 14, 9, 5, CORAL, "none"))
    svg.append(text(189, 18, "R²", anchor="start"))
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


def blend(fill, value):
    value = max(0.0, min(1.0, value))
    if fill == "blue":
        hi = (142, 186, 215)
    elif fill == "coral":
        hi = (237, 178, 155)
    else:
        hi = (217, 217, 217)
    lo = (244, 244, 244)
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def build_fig4(rows, output: Path):
    width, height = 450.75, 295.2
    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.6f}pt" height="{height:.6f}pt" viewBox="0 0 {width:.6f} {height:.6f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    targets = ["Replication", "Transcription", "Translation"]
    settings = ["all_non_target_features", "non_central_dogma_background", "other_central_dogma_modules_only"]
    setting_labels = ["All non-target features", "Non-central dogma background", "Other central dogma modules only"]
    draw_grouped(svg, rows, 6, 15, 200, 112, "cde_module", targets, settings, ["AUPRC"], [BLUE, CREAM, CORAL], "AUPRC", {})
    draw_grouped(svg, rows, 6, 162, 200, 112, "cde_module", targets, settings, ["R2"], [BLUE, CREAM, CORAL], "R²", {})
    svg.append(text(4, 14, "C", anchor="start", weight="bold"))
    svg.append(text(4, 161, "D", anchor="start", weight="bold"))
    for i, (label, color) in enumerate(zip(setting_labels, [BLUE, CREAM, CORAL])):
        x = 42
        y = 134 + i * 10
        svg.append(rect(x, y - 5, 8, 5, color, "none"))
        svg.append(text(x + 11, y - 1, label, anchor="start"))

    heat_targets = [
        "Replication | DNA repair & recombination",
        "Replication | Machinery / Elongation",
        "Replication | Maintenance / Partition",
        "Replication | Replication Initiator",
        "Transcription | Core transcription machinery",
        "Transcription | Initiation & promoter recognition",
        "Transcription | Termination & anti-termination",
        "Transcription | Transcription regulation",
        "Translation | Regulation & Control",
        "Translation | Ribosomal RNA",
        "Translation | Ribosomal proteins",
        "Translation | Ribosome assembly & maturation",
        "Translation | aaRS",
        "Translation | tRNA",
    ]
    heat_labels = [
        "DNA repair /\nrecombination",
        "Machinery / elongation",
        "Maintenance / partition",
        "Replication initiator",
        "Core transcription\nmachinery",
        "Initiation / promoter\nrecognition",
        "Termination /\nanti-termination",
        "Transcription regulation",
        "Regulation / control",
        "rRNA",
        "Ribosomal proteins",
        "Ribosome assembly /\nmaturation",
        "aaRS",
        "tRNA",
    ]
    x0, y0, cell = 280, 35, 28
    svg.append(text(216, 28, "E", anchor="start", weight="bold"))
    for col, label in enumerate(["AUPRC", "R²", "Positive-class\nprevalence"]):
        for li, part in enumerate(label.split("\n")):
            svg.append(text(x0 + col * cell, 20 + li * 8, part))
    for row, target in enumerate(heat_targets):
        y = y0 + row * 15
        for li, part in enumerate(heat_labels[row].split("\n")):
            svg.append(text(265, y + 4 + li * 7, part, anchor="end"))
        vals = [
            mean_metric(rows, "cde_subcategory", target, "all_non_target_features", "AUPRC"),
            mean_metric(rows, "cde_subcategory", target, "all_non_target_features", "R2"),
            mean_metric(rows, "cde_subcategory", target, "all_non_target_features", "test_positive_prevalence"),
        ]
        for col, val in enumerate(vals):
            svg.append(rect(x0 + col * cell - 11, y - 6, 22, 12, blend(["blue", "coral", "grey"][col], val), "white"))
            svg.append(text(x0 + col * cell, y + 2.5, f"{val:.2f}"))
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics)
    build_s16(rows, args.out_dir / "Figure_S16_COG_L_K_J_predictability.svg")
    build_fig4(rows, args.out_dir / "Figure_4_CDE_machine_learning_panels_C_D_E.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

