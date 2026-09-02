"""Draw the size-effect sensitivity comparison in manuscript style without Matplotlib."""

from __future__ import annotations

import argparse
import html
import statistics
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BLUE_DARK = "#8EBAD7"
CORAL_DARK = "#EDB29B"
INK = "#202020"
OBJECT_LW = 0.7
AXIS_LW = 1.0
FONT_SIZE = 7.0
WIDTH_PT = 291.5
HEIGHT_PT = 169.2
SCALE = 600.0 / 72.0
ARIAL = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, *, anchor: str = "middle", rotate: float | None = None,
         bold: bool = False, size: float = FONT_SIZE) -> str:
    transform = "" if rotate is None else f' transform="rotate({rotate:.3f} {x:.3f} {y:.3f})"'
    weight = "bold" if bold else "normal"
    return (f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}"{transform} '
            f'font-family="Arial" font-size="{size:.3f}pt" font-weight="{weight}" fill="{INK}">{esc(value)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, width: float = OBJECT_LW, color: str = INK) -> str:
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width:.3f}"/>'


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = INK,
         stroke_width: float = OBJECT_LW) -> str:
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.3f}"/>'


def circle(x: float, y: float, radius: float = 1.15) -> str:
    return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" fill="black"/>'


def metric_values(data: pd.DataFrame, task: str, metric: str) -> dict[str, list[float]]:
    q = data.loc[data.task.eq(task), ["feature_setting", "run", metric]].copy()
    q[metric] = pd.to_numeric(q[metric], errors="coerce")
    q = q.dropna(subset=[metric])
    order = ["raw_product_counts", "product_abundance_per_100kb"]
    if set(q.feature_setting) != set(order) or q.run.nunique() != 3:
        raise ValueError(f"Expected two feature settings and three replicates for {task}")
    return {key: [float(v) for v in q.loc[q.feature_setting.eq(key)].sort_values("run")[metric]] for key in order}


def ymap(value: float, top: float, bottom: float) -> float:
    return bottom - value / 1.03 * (bottom - top)


def panel_svg(parts: list[str], values: dict[str, list[float]], left: float, top: float,
              right: float, bottom: float, ylabel: str, panel: str) -> None:
    parts.extend([line(left, top, left, bottom, AXIS_LW), line(left, bottom, right, bottom, AXIS_LW)])
    for tick in (0.0, 0.5, 1.0):
        y = ymap(tick, top, bottom)
        parts.extend([line(left - 3.0, y, left, y), text(left - 5.0, y + 2.2, f"{tick:.1f}", anchor="end")])
    parts.extend([text(left - 25.0, (top + bottom) / 2, ylabel, rotate=-90),
                  text(left - 17.0, top - 7.0, panel, anchor="start", bold=True)])
    order = ["raw_product_counts", "product_abundance_per_100kb"]
    fills = [BLUE_DARK, CORAL_DARK]
    centers = [left + (i + 0.5) * (right - left) / 2 for i in range(2)]
    bar_width = 0.52 * (right - left) / 2
    for center, key, fill in zip(centers, order, fills):
        run_values = values[key]
        mean = statistics.mean(run_values)
        sd = statistics.stdev(run_values)
        y_mean, y_zero = ymap(mean, top, bottom), ymap(0.0, top, bottom)
        parts.append(rect(center - bar_width / 2, y_mean, bar_width, y_zero - y_mean, fill))
        y_low, y_high = ymap(max(0.0, mean - sd), top, bottom), ymap(min(1.03, mean + sd), top, bottom)
        parts.extend([line(center, y_low, center, y_high), line(center - 2.0, y_low, center + 2.0, y_low), line(center - 2.0, y_high, center + 2.0, y_high)])
        for offset, value in zip((-2.5, 0.0, 2.5), run_values):
            parts.append(circle(center + offset, ymap(value, top, bottom)))
    for center, label in zip(centers, ("original model", "Density (per 100 kb)")):
        parts.append(text(center, bottom + 10.0, label))


def build_svg(data: pd.DataFrame) -> str:
    class_values = metric_values(data, "classification", "AUPRC")
    reg_values = metric_values(data, "regression_tRNA_positive_only", "R2")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PT:.3f}pt" height="{HEIGHT_PT:.3f}pt" viewBox="0 0 {WIDTH_PT:.3f} {HEIGHT_PT:.3f}">', '<rect width="100%" height="100%" fill="white"/>']
    panel_svg(parts, class_values, 39.0, 25.0, 139.0, 139.0, "AUPRC", "A")
    panel_svg(parts, reg_values, 164.0, 25.0, 264.0, 139.0, "R²", "B")
    parts.extend([rect(60.0, 7.0, 8.0, 5.0, BLUE_DARK), text(71.0, 11.0, "original model", anchor="start"),
                  rect(156.0, 7.0, 8.0, 5.0, CORAL_DARK), text(167.0, 11.0, "Density (per 100 kb)", anchor="start"), '</svg>'])
    return "\n".join(parts)


def font(path: Path, size_pt: float) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise FileNotFoundError(path)
    return ImageFont.truetype(str(path), max(1, int(round(size_pt * SCALE))))


def draw_rotated(image: Image.Image, xy: tuple[int, int], value: str, angle: float, fnt: ImageFont.FreeTypeFont) -> None:
    layer = Image.new("RGBA", (int(60 * SCALE), int(16 * SCALE)), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((layer.width - 2, 2), value, anchor="ra", font=fnt, fill=INK)
    rotated = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2))


def build_png(data: pd.DataFrame, path: Path) -> None:
    class_values = metric_values(data, "classification", "AUPRC")
    reg_values = metric_values(data, "regression_tRNA_positive_only", "R2")
    image = Image.new("RGBA", (round(WIDTH_PT * SCALE), round(HEIGHT_PT * SCALE)), "white")
    draw = ImageDraw.Draw(image)
    fnt = font(ARIAL, FONT_SIZE)
    bold = font(ARIAL_BOLD, FONT_SIZE)

    def px(value: float) -> int:
        return round(value * SCALE)

    def yv(value: float, top: float, bottom: float) -> int:
        return px(ymap(value, top, bottom))

    def panel(values: dict[str, list[float]], left: float, top: float, right: float, bottom: float, ylabel: str, panel_letter: str) -> None:
        draw.line((px(left), px(top), px(left), px(bottom)), fill=INK, width=max(1, px(AXIS_LW)))
        draw.line((px(left), px(bottom), px(right), px(bottom)), fill=INK, width=max(1, px(AXIS_LW)))
        for tick in (0.0, 0.5, 1.0):
            y = yv(tick, top, bottom)
            draw.line((px(left - 3), y, px(left), y), fill=INK, width=max(1, px(OBJECT_LW)))
            draw.text((px(left - 5), y), f"{tick:.1f}", anchor="rm", font=fnt, fill=INK)
        draw.text((px(left - 17), px(top - 7)), panel_letter, anchor="la", font=bold, fill=INK)
        draw_rotated(image, (px(left - 25), px((top + bottom) / 2)), ylabel, 90, fnt)
        order = ["raw_product_counts", "product_abundance_per_100kb"]
        colors = [(142, 186, 215), (237, 178, 155)]
        centers = [left + (i + 0.5) * (right - left) / 2 for i in range(2)]
        bar_width = 0.52 * (right - left) / 2
        for center, key, color in zip(centers, order, colors):
            vals = values[key]
            mean, sd = statistics.mean(vals), statistics.stdev(vals)
            y_mean, y_zero = yv(mean, top, bottom), yv(0.0, top, bottom)
            draw.rectangle((px(center - bar_width / 2), y_mean, px(center + bar_width / 2), y_zero), fill=color, outline=INK, width=max(1, px(OBJECT_LW)))
            y_low, y_high = yv(max(0, mean - sd), top, bottom), yv(min(1.03, mean + sd), top, bottom)
            draw.line((px(center), y_low, px(center), y_high), fill=INK, width=max(1, px(OBJECT_LW)))
            draw.line((px(center - 2), y_low, px(center + 2), y_low), fill=INK, width=max(1, px(OBJECT_LW)))
            draw.line((px(center - 2), y_high, px(center + 2), y_high), fill=INK, width=max(1, px(OBJECT_LW)))
            for offset, value in zip((-2.5, 0.0, 2.5), vals):
                cx, cy, radius = px(center + offset), yv(value, top, bottom), max(1, px(1.15))
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="black")
        for center, label in zip(centers, ("original model", "Density (per 100 kb)")):
            draw.text((px(center), px(bottom + 10)), label, anchor="ma", font=fnt, fill=INK)

    panel(class_values, 39, 25, 139, 139, "AUPRC", "A")
    panel(reg_values, 164, 25, 264, 139, "R²", "B")
    draw.rectangle((px(60), px(7), px(68), px(12)), fill=(142, 186, 215))
    draw.text((px(71), px(9)), "original model", anchor="lm", font=fnt, fill=INK)
    draw.rectangle((px(156), px(7), px(164), px(12)), fill=(237, 178, 155))
    draw.text((px(167), px(9)), "Density (per 100 kb)", anchor="lm", font=fnt, fill=INK)
    image.convert("RGB").save(path, dpi=(600, 600))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.metrics, keep_default_na=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".svg").write_text(build_svg(data), encoding="utf-8")
    build_png(data, args.out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
