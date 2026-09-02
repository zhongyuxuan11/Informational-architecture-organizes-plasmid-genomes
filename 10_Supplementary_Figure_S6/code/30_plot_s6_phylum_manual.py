"""Draw Figure S6 without Matplotlib bar patches."""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BLUE_DARK = "#80AACC"
CORAL_DARK = "#DB8B82"
INK = "#202020"
OBJECT_LW = 0.7
AXIS_LW = 1.0
FONT_SIZE = 7.0

WIDTH_PT = 413.0
HEIGHT_PT = 173.0
SCALE = 600.0 / 72.0

ARIAL = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
if not ARIAL.exists():
    raise FileNotFoundError(ARIAL)


def num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def stats(data: pd.DataFrame, category: str, metric: str, order: list[str]) -> pd.DataFrame:
    q = data[[category, "run", metric]].copy()
    q[category] = q[category].astype(str)
    q[metric] = num(q[metric])
    q = q.dropna(subset=[metric])
    grouped = q.groupby(category)[metric]
    out = pd.DataFrame({
        category: order,
        "mean": grouped.mean().reindex(order).to_numpy(dtype=float),
        "sd": grouped.std(ddof=1).reindex(order).fillna(0).to_numpy(dtype=float),
    })
    return out


def bounded_error(mean: float, sd: float, metric: str) -> tuple[float, float]:
    if metric in {"AUPRC", "AUROC", "F1", "Precision", "Recall"}:
        return min(sd, mean), min(sd, 1.0 - mean)
    return sd, sd


def svg_line(x1, y1, x2, y2, width=OBJECT_LW, color=INK) -> str:
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width:.3f}"/>'


def svg_text(x, y, text, anchor="middle", rotate=None, bold=False) -> str:
    weight = "bold" if bold else "normal"
    transform = f' transform="rotate({rotate:.3f} {x:.3f} {y:.3f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial" font-size="{FONT_SIZE:.3f}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{INK}"{transform}>'
        f'{html.escape(str(text))}</text>'
    )


def draw_panel_svg(parts: list[str], data: pd.DataFrame, raw: pd.DataFrame, category: str, metric: str,
                   order: list[str], x0: float, y0: float, w: float, h: float,
                   color: str, ylabel: str, ylim: tuple[float, float], letter: str) -> None:
    ymin, ymax = ylim
    x_positions = [x0 + (i + 0.5) * w / len(order) for i in range(len(order))]
    bar_w = 0.70 * w / len(order)

    def ymap(v: float) -> float:
        return y0 + h - (v - ymin) / (ymax - ymin) * h

    parts.append(svg_line(x0, y0, x0, y0 + h, AXIS_LW))
    parts.append(svg_line(x0, y0 + h, x0 + w, y0 + h, AXIS_LW))
    if ymin < 0 < ymax:
        parts.append(svg_line(x0, ymap(0), x0 + w, ymap(0), OBJECT_LW))
    for tick in [v for v in [-1.0, -0.5, 0.0, 0.5, 1.0] if ymin <= v <= ymax]:
        y = ymap(tick)
        parts.append(svg_line(x0 - 2.5, y, x0, y, OBJECT_LW))
        parts.append(svg_text(x0 - 4.0, y + 2.2, f"{tick:g}", "end"))
    parts.append(svg_text(x0 - 30.0, y0 + h / 2, ylabel, "middle", -90))
    parts.append(svg_text(x0 - 18.0, y0 - 8.0, letter, "start", bold=True))

    run_values = sorted(raw["run"].dropna().unique())
    offsets = [0.0] if len(run_values) == 1 else [
        -0.12 * w / len(order) + i * (0.24 * w / len(order)) / (len(run_values) - 1)
        for i in range(len(run_values))
    ]
    raw_indexed = {run: raw[raw["run"].eq(run)].set_index(category)[metric] for run in run_values}
    for i, name in enumerate(order):
        mean = float(data.loc[data[category].eq(name), "mean"].iloc[0])
        sd = float(data.loc[data[category].eq(name), "sd"].iloc[0])
        low, high = bounded_error(mean, sd, metric)
        xc = x_positions[i]
        x_left = xc - bar_w / 2
        y_mean = ymap(mean)
        y_base = ymap(0 if ymin < 0 else ymin)
        y_rect = min(y_mean, y_base)
        rect_h = abs(y_base - y_mean)
        parts.append(
            f'<rect x="{x_left:.3f}" y="{y_rect:.3f}" width="{bar_w:.3f}" height="{rect_h:.3f}" '
            f'fill="{color}" stroke="{INK}" stroke-width="{OBJECT_LW:.3f}"/>'
        )
        y_low = ymap(mean - low)
        y_high = ymap(mean + high)
        parts.append(svg_line(xc, y_low, xc, y_high, OBJECT_LW))
        parts.append(svg_line(xc - 2.0, y_low, xc + 2.0, y_low, OBJECT_LW))
        parts.append(svg_line(xc - 2.0, y_high, xc + 2.0, y_high, OBJECT_LW))
        for offset, run in zip(offsets, run_values):
            value = raw_indexed[run].reindex(order).iloc[i]
            if pd.isna(value):
                continue
            parts.append(f'<circle cx="{(xc + offset):.3f}" cy="{ymap(float(value)):.3f}" r="1.15" fill="black"/>')
        parts.append(svg_text(xc - 1.0, y0 + h + 8.0, name, "end", -35))


def save_svg(path: Path, class_raw: pd.DataFrame, reg_raw: pd.DataFrame,
             cls_order: list[str], reg_order: list[str]) -> None:
    class_stats = stats(class_raw, "phylum", "AUPRC", cls_order)
    reg_stats = stats(reg_raw, "phylum", "R2", reg_order)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PT:.3f}pt" height="{HEIGHT_PT:.3f}pt" '
        f'viewBox="0 0 {WIDTH_PT:.3f} {HEIGHT_PT:.3f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    draw_panel_svg(parts, class_stats, class_raw, "phylum", "AUPRC", cls_order,
                   42.0, 15.0, 150.0, 105.0, BLUE_DARK, "AUPRC", (0.0, 1.05), "A")
    draw_panel_svg(parts, reg_stats, reg_raw, "phylum", "R2", reg_order,
                   253.0, 15.0, 125.0, 105.0, CORAL_DARK, "R²", (-1.4, 1.05), "B")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_rotated_text(image: Image.Image, xy: tuple[float, float], text: str,
                      font: ImageFont.FreeTypeFont, angle: float, fill: str) -> None:
    layer = Image.new("RGBA", (500, 70), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((490, 4), text, anchor="ra", font=font, fill=fill)
    rotated = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (int(xy[0] - rotated.width + 5), int(xy[1] - 6)))


def save_png(path: Path, class_raw: pd.DataFrame, reg_raw: pd.DataFrame,
             cls_order: list[str], reg_order: list[str]) -> None:
    w_px, h_px = int(round(WIDTH_PT * SCALE)), int(round(HEIGHT_PT * SCALE))
    image = Image.new("RGBA", (w_px, h_px), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ARIAL), int(round(FONT_SIZE * SCALE)))
    bold = ImageFont.truetype(str(ARIAL_BOLD if ARIAL_BOLD.exists() else ARIAL), int(round(FONT_SIZE * SCALE)))

    def pt(v: float) -> int:
        return int(round(v * SCALE))

    def panel(raw, order, metric, x0, y0, ww, hh, color, ylabel, ylim, panel_letter):
        st = stats(raw, "phylum", metric, order)
        ymin, ymax = ylim
        def ymap(v): return y0 + hh - (v - ymin) / (ymax - ymin) * hh
        draw.line([(pt(x0), pt(y0)), (pt(x0), pt(y0 + hh))], fill=INK, width=pt(AXIS_LW))
        draw.line([(pt(x0), pt(y0 + hh)), (pt(x0 + ww), pt(y0 + hh))], fill=INK, width=pt(AXIS_LW))
        if ymin < 0 < ymax:
            draw.line([(pt(x0), pt(ymap(0))), (pt(x0 + ww), pt(ymap(0)))], fill=INK, width=pt(OBJECT_LW))
        for tick in [v for v in [-1.0, -0.5, 0.0, 0.5, 1.0] if ymin <= v <= ymax]:
            y = ymap(tick)
            draw.line([(pt(x0 - 2.5), pt(y)), (pt(x0), pt(y))], fill=INK, width=pt(OBJECT_LW))
            draw.text((pt(x0 - 4), pt(y)), f"{tick:g}", anchor="rm", font=font, fill=INK)
        draw_rotated_text(image, (pt(x0 - 30), pt(y0 + hh / 2)), ylabel, font, 90, INK)
        draw.text((pt(x0 - 18), pt(y0 - 8)), panel_letter, anchor="la", font=bold, fill=INK)
        runs = sorted(raw["run"].dropna().unique())
        offsets = [0.0] if len(runs) == 1 else [-0.12 * ww / len(order) + i * (0.24 * ww / len(order)) / (len(runs) - 1) for i in range(len(runs))]
        raw_indexed = {run: raw[raw["run"].eq(run)].set_index("phylum")[metric] for run in runs}
        bar_w = 0.70 * ww / len(order)
        for i, name in enumerate(order):
            xc = x0 + (i + 0.5) * ww / len(order)
            mean = float(st.loc[st.phylum.eq(name), "mean"].iloc[0])
            sd = float(st.loc[st.phylum.eq(name), "sd"].iloc[0])
            low, high = bounded_error(mean, sd, metric)
            y_mean, y_base = ymap(mean), ymap(0 if ymin < 0 else ymin)
            y_top, y_bottom = sorted([y_mean, y_base])
            draw.rectangle([pt(xc - bar_w / 2), pt(y_top), pt(xc + bar_w / 2), pt(y_bottom)], fill=color, outline=INK, width=pt(OBJECT_LW))
            y_low, y_high = ymap(mean - low), ymap(mean + high)
            draw.line([(pt(xc), pt(y_low)), (pt(xc), pt(y_high))], fill=INK, width=pt(OBJECT_LW))
            draw.line([(pt(xc - 2), pt(y_low)), (pt(xc + 2), pt(y_low))], fill=INK, width=pt(OBJECT_LW))
            draw.line([(pt(xc - 2), pt(y_high)), (pt(xc + 2), pt(y_high))], fill=INK, width=pt(OBJECT_LW))
            for offset, run in zip(offsets, runs):
                value = raw_indexed[run].reindex(order).iloc[i]
                if pd.isna(value):
                    continue
                cx, cy, rr = pt(xc + offset), pt(ymap(float(value))), max(1, pt(1.15))
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill="black")
            draw_rotated_text(image, (pt(xc - 1), pt(y0 + hh + 8)), name, font, 35, INK)

    panel(class_raw, cls_order, "AUPRC", 42.0, 15.0, 150.0, 105.0, BLUE_DARK, "AUPRC", (0.0, 1.05), "A")
    panel(reg_raw, reg_order, "R2", 253.0, 15.0, 125.0, 105.0, CORAL_DARK, "R²", (-1.4, 1.05), "B")
    image.convert("RGB").save(path, dpi=(600, 600))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root20", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.root20 / "results/phylum_stratified_validation/phylum_stratified_metrics_each_run_n_gt10.csv")
    cls_order = ["Pseudomonadota", "Bacillota", "Actinomycetota", "Methanobacteriota", "Bacteroidota", "Spirochaetota", "Cyanobacteriota"]
    reg_order = ["Pseudomonadota", "Bacillota", "Actinomycetota", "Methanobacteriota"]
    class_raw = data[data.task.eq("classification")].copy()
    reg_raw = data[data.task.str.startswith("regression")].copy()
    stem = args.out / "Figure_S6_phylum_stratified_validation_n_gt10"
    save_svg(stem.with_suffix(".svg"), class_raw, reg_raw, cls_order, reg_order)
    save_png(stem.with_suffix(".png"), class_raw, reg_raw, cls_order, reg_order)


if __name__ == "__main__":
    main()
