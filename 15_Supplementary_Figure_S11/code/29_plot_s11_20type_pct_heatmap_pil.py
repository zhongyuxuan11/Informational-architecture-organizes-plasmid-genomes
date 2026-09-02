"""Plot S11 20-type tRNA feature-importance percent heatmaps with PIL."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TARGET_ORDER = [
    "tRNA-Ala", "tRNA-Arg", "tRNA-Asn", "tRNA-Asp", "tRNA-Cys",
    "tRNA-Gln", "tRNA-Glu", "tRNA-Gly", "tRNA-His", "tRNA-Ile",
    "tRNA-Leu", "tRNA-Lys", "tRNA-Met", "tRNA-Phe", "tRNA-Pro",
    "tRNA-Ser", "tRNA-Thr", "tRNA-Trp", "tRNA-Tyr", "tRNA-Val",
]
TASKS = {
    "classification": ("Classification", (33, 102, 172), "classification"),
    "regression_type_positive_only": (
        "Regression (tRNA-positive)", (215, 48, 39), "regression"
    ),
}
BASE_DPI = 96
OUT_DPI = 600
WIDTH_CM = 18.0
HEIGHT_CM = 22.0
FONT_PT = 7.0
LABEL_PT = 0.7
AXIS_PT = 1.0
ROW_GAP_PX = 2


def cm_to_px(value: float, dpi: int) -> int:
    return int(round(value / 2.54 * dpi))


def pt_to_px(value: float) -> int:
    return max(1, int(round(value * BASE_DPI / 72.0)))


def product_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def display_product_name(value: str) -> str:
    name = " ".join(str(value).strip().split())
    if not name:
        return name
    return name[0].upper() + name[1:]


def font(size_pt: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = pt_to_px(size_pt)
    path = Path("C:/Windows/Fonts/arial.ttf")
    if not path.is_file():
        raise FileNotFoundError("Arial is required: C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(path, size=size)


def read_input(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    required = {"target", "task", "run", "Product", "normalized_gain_pct"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data["normalized_gain_pct"] = pd.to_numeric(
        data["normalized_gain_pct"], errors="raise"
    )
    data["Product_Key"] = data["Product"].map(product_key)
    data["Product"] = data["Product"].map(display_product_name)
    if data["Product"].eq("").any():
        raise ValueError("Empty Product values are not allowed")
    return data


def make_matrix(
    data: pd.DataFrame,
    task: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = data.loc[
        data["task"].eq(task) & data["target"].isin(TARGET_ORDER)
    ].copy()
    runs = sorted(current["run"].unique())
    if len(runs) != 3:
        raise ValueError(f"Expected three runs for {task}, found {runs}")
    display = (
        current[["Product_Key", "Product"]]
        .drop_duplicates()
        .sort_values(["Product_Key", "Product"], kind="stable")
        .drop_duplicates("Product_Key")
        .set_index("Product_Key")["Product"]
    )
    full_index = pd.MultiIndex.from_product(
        [TARGET_ORDER, sorted(current["Product_Key"].unique()), runs],
        names=["target", "Product_Key", "run"],
    )
    complete = (
        current.groupby(["target", "Product_Key", "run"])["normalized_gain_pct"]
        .sum()
        .reindex(full_index, fill_value=0.0)
        .rename("normalized_gain_pct")
        .reset_index()
    )
    means = complete.groupby(["target", "Product_Key"], as_index=False)[
        "normalized_gain_pct"
    ].mean()
    means["Product"] = means["Product_Key"].map(display)
    means["task"] = task
    selected = (
        means.sort_values(
            ["target", "normalized_gain_pct", "Product"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("target", sort=False)
        .head(top_k)
        .copy()
    )
    union_keys = selected["Product_Key"].drop_duplicates().tolist()
    matrix = means.pivot(
        index="Product_Key", columns="target", values="normalized_gain_pct"
    ).reindex(index=union_keys, columns=TARGET_ORDER).fillna(0.0)
    order = matrix.max(axis=1).sort_values(ascending=False, kind="stable").index
    matrix = matrix.reindex(order)
    matrix.index = matrix.index.map(display)
    if matrix.index.duplicated().any():
        duplicated = matrix.index[matrix.index.duplicated()].tolist()
        raise ValueError(f"Duplicated display Product names: {duplicated}")
    return matrix, selected[["target", "task", "Product_Key", "Product", "normalized_gain_pct"]]


def interpolate_color(value: float, endpoint: tuple[int, int, int]) -> tuple[int, int, int]:
    fraction = min(max(value, 0.0), 40.0) / 40.0
    return tuple(int(round(255 + fraction * (channel - 255))) for channel in endpoint)


def text_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=text_font)
    return box[2] - box[0], box[3] - box[1]


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    angle: int,
) -> None:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    d = ImageDraw.Draw(dummy)
    w, h = text_size(d, text, text_font)
    layer = Image.new("RGBA", (w + 4, h + 4), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((2, 2), text, fill=(32, 32, 32, 255), font=text_font)
    rotated = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (xy[0], xy[1]))


def rgb_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def svg_text(
    x: float,
    y: float,
    text: str,
    anchor: str = "start",
    extra: str = "",
) -> str:
    escaped = html.escape(str(text), quote=False)
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial" font-size="{FONT_PT}pt" '
        f'fill="#202020" text-anchor="{anchor}" dominant-baseline="middle"{extra}>'
        f"{escaped}</text>"
    )


def write_vector_svg(
    matrix: pd.DataFrame,
    title: str,
    endpoint: tuple[int, int, int],
    stem: Path,
    width: int,
    height: int,
    height_cm: float,
    layout: dict[str, float],
    colorbar_cm: float | None,
) -> None:
    rows, cols = matrix.shape
    left = layout["left"]
    right = layout["right"]
    top = layout["top"]
    bottom = layout["bottom"]
    heat_w = width - left - right
    heat_h = height - top - bottom
    cell_w = heat_w / cols
    cell_h = heat_h / rows
    axis_w = AXIS_PT
    label_w = LABEL_PT
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_CM}cm" '
            f'height="{height_cm:.4f}cm" viewBox="0 0 {width} {height}">'
        ),
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 14, title, anchor="middle"),
    ]
    for r, product in enumerate(matrix.index):
        y0 = top + r * cell_h
        for c, target in enumerate(TARGET_ORDER):
            x0 = left + c * cell_w
            color = rgb_hex(interpolate_color(float(matrix.loc[product, target]), endpoint))
            parts.append(
                f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{cell_w:.3f}" '
                f'height="{cell_h:.3f}" fill="{color}"/>'
            )
    parts.append(
        f'<line x1="{left:.3f}" y1="{top:.3f}" x2="{left:.3f}" '
        f'y2="{top + heat_h:.3f}" stroke="#202020" stroke-width="{axis_w:.3f}"/>'
    )
    parts.append(
        f'<line x1="{left:.3f}" y1="{top + heat_h:.3f}" x2="{left + heat_w:.3f}" '
        f'y2="{top + heat_h:.3f}" stroke="#202020" stroke-width="{axis_w:.3f}"/>'
    )
    for r, product in enumerate(matrix.index):
        y = top + (r + 0.5) * cell_h
        parts.append(
            f'<line x1="{left - 4:.3f}" y1="{y:.3f}" x2="{left:.3f}" y2="{y:.3f}" '
            f'stroke="#202020" stroke-width="{label_w:.3f}"/>'
        )
        parts.append(svg_text(8, y, product))
    for c, target in enumerate(TARGET_ORDER):
        x = left + (c + 0.5) * cell_w
        parts.append(
            f'<line x1="{x:.3f}" y1="{top + heat_h:.3f}" x2="{x:.3f}" '
            f'y2="{top + heat_h + 4:.3f}" stroke="#202020" stroke-width="{label_w:.3f}"/>'
        )
        label = html.escape(target.replace("tRNA-", ""), quote=False)
        parts.append(
            f'<text x="{x - 2:.3f}" y="{top + heat_h + 12:.3f}" font-family="Arial" '
            f'font-size="{FONT_PT}pt" fill="#202020" text-anchor="end" '
            f'transform="rotate(-60 {x - 2:.3f} {top + heat_h + 12:.3f})">{label}</text>'
        )
    bar_x = width - 70
    bar_y = top + 12
    bar_w = 12
    bar_h = cm_to_px(colorbar_cm, BASE_DPI) if colorbar_cm else int(heat_h * 0.38)
    for i in range(bar_h):
        value = 40.0 * (1.0 - i / max(1, bar_h - 1))
        color = rgb_hex(interpolate_color(value, endpoint))
        parts.append(
            f'<line x1="{bar_x:.3f}" y1="{bar_y + i:.3f}" x2="{bar_x + bar_w:.3f}" '
            f'y2="{bar_y + i:.3f}" stroke="{color}" stroke-width="1"/>'
        )
    parts.append(
        f'<rect x="{bar_x:.3f}" y="{bar_y:.3f}" width="{bar_w:.3f}" height="{bar_h:.3f}" '
        f'fill="none" stroke="#202020" stroke-width="{label_w:.3f}"/>'
    )
    for value in (0, 10, 20, 30, 40):
        y = bar_y + bar_h * (1 - value / 40.0)
        parts.append(
            f'<line x1="{bar_x + bar_w:.3f}" y1="{y:.3f}" x2="{bar_x + bar_w + 3:.3f}" '
            f'y2="{y:.3f}" stroke="#202020" stroke-width="{label_w:.3f}"/>'
        )
        parts.append(svg_text(bar_x + bar_w + 5, y, value))
    label_x = bar_x + bar_w + 26
    label_y = bar_y + 120
    parts.append(
        f'<text x="{label_x:.3f}" y="{label_y:.3f}" font-family="Arial" '
        f'font-size="{FONT_PT}pt" fill="#202020" text-anchor="middle" '
        f'transform="rotate(-90 {label_x:.3f} {label_y:.3f})">'
        "Mean normalized gain (%)</text>"
    )
    parts.append("</svg>")
    stem.with_suffix(".svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    endpoint: tuple[int, int, int],
    stem: Path,
    colorbar_cm: float | None,
) -> float:
    width = cm_to_px(WIDTH_CM, BASE_DPI)
    rows, cols = matrix.shape
    minimum_row_height = pt_to_px(FONT_PT) + ROW_GAP_PX
    fixed_height = 34 + 72
    colorbar_min_height = 0
    if colorbar_cm:
        colorbar_min_height = 34 + cm_to_px(colorbar_cm, BASE_DPI) + 72
    height = max(
        cm_to_px(HEIGHT_CM, BASE_DPI),
        fixed_height + rows * minimum_row_height,
        colorbar_min_height,
    )
    height_cm = height / BASE_DPI * 2.54
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    f = font(FONT_PT)
    axis_w = pt_to_px(AXIS_PT)
    label_w = pt_to_px(LABEL_PT)
    left, right, top, bottom = 250, 86, 34, 72
    heat_w = width - left - right
    heat_h = height - top - bottom
    cell_w = heat_w / cols
    cell_h = heat_h / rows

    draw.text(((width - text_size(draw, title, f)[0]) / 2, 8), title, fill=(32, 32, 32), font=f)
    for r, product in enumerate(matrix.index):
        y0 = top + r * cell_h
        y1 = top + (r + 1) * cell_h
        for c, target in enumerate(TARGET_ORDER):
            x0 = left + c * cell_w
            x1 = left + (c + 1) * cell_w
            draw.rectangle(
                [round(x0), round(y0), round(x1), round(y1)],
                fill=interpolate_color(float(matrix.loc[product, target]), endpoint),
            )
    draw.line([(left, top), (left, top + heat_h)], fill=(32, 32, 32), width=axis_w)
    draw.line([(left, top + heat_h), (left + heat_w, top + heat_h)], fill=(32, 32, 32), width=axis_w)

    for r in range(rows):
        y = int(round(top + (r + 0.5) * cell_h))
        draw.line([(left - 4, y), (left, y)], fill=(32, 32, 32), width=label_w)
        draw.text((8, y - text_size(draw, matrix.index[r], f)[1] / 2), matrix.index[r], fill=(32, 32, 32), font=f)
    for c, target in enumerate(TARGET_ORDER):
        x = int(round(left + (c + 0.5) * cell_w))
        draw.line([(x, top + heat_h), (x, top + heat_h + 4)], fill=(32, 32, 32), width=label_w)
        draw_rotated_text(image, (x - 12, top + heat_h + 8), target.replace("tRNA-", ""), f, -60)

    bar_x, bar_y, bar_w = width - 70, top + 12, 12
    bar_h = cm_to_px(colorbar_cm, BASE_DPI) if colorbar_cm else int(heat_h * 0.38)
    for i in range(bar_h):
        value = 40.0 * (1.0 - i / max(1, bar_h - 1))
        draw.line(
            [(bar_x, bar_y + i), (bar_x + bar_w, bar_y + i)],
            fill=interpolate_color(value, endpoint),
            width=1,
        )
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(32, 32, 32), width=label_w)
    for value in (0, 10, 20, 30, 40):
        y = int(round(bar_y + bar_h * (1 - value / 40.0)))
        draw.line([(bar_x + bar_w, y), (bar_x + bar_w + 3, y)], fill=(32, 32, 32), width=label_w)
        draw.text((bar_x + bar_w + 5, y - text_size(draw, str(value), f)[1] / 2), str(value), fill=(32, 32, 32), font=f)
    draw_rotated_text(image, (bar_x + bar_w + 22, bar_y + 30), "Mean normalized gain (%)", f, -90)

    out_size = (cm_to_px(WIDTH_CM, OUT_DPI), cm_to_px(height_cm, OUT_DPI))
    png = image.convert("RGB").resize(out_size, Image.Resampling.LANCZOS)
    png_path = stem.with_suffix(".png")
    png.save(png_path, dpi=(OUT_DPI, OUT_DPI))
    write_vector_svg(
        matrix,
        title,
        endpoint,
        stem,
        width,
        height,
        height_cm,
        {"left": left, "right": right, "top": top, "bottom": bottom},
        colorbar_cm,
    )
    return height_cm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--colorbar-cm", type=float, default=None)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.colorbar_cm is not None and args.colorbar_cm <= 0:
        raise ValueError("--colorbar-cm must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = read_input(args.input)
    metadata = []
    for task, (title, endpoint, stem_name) in TASKS.items():
        matrix, selected = make_matrix(data, task, args.top_k)
        matrix.to_csv(args.out_dir / f"S9_20type_{stem_name}_union_values_pct.csv")
        selected.to_csv(
            args.out_dir / f"S9_20type_{stem_name}_selected_top{args.top_k}_pct.csv",
            index=False,
        )
        height_cm = plot_heatmap(
            matrix,
            title,
            endpoint,
            args.out_dir / f"S9_20type_{stem_name}_top{args.top_k}_union_heatmap_pct",
            args.colorbar_cm,
        )
        metadata.append(
            {
                "task": task,
                "value": "mean normalized gain percent",
                "color_scale": "continuous 0-40 percent, clipped only for display",
                "union_rule": f"union of per-tRNA top {args.top_k} products after product-name normalization",
                "tRNA_type_n": len(TARGET_ORDER),
                "union_product_n": len(matrix),
                "width_cm": WIDTH_CM,
                "height_cm": height_cm,
                "colorbar_cm": args.colorbar_cm,
            }
        )
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

