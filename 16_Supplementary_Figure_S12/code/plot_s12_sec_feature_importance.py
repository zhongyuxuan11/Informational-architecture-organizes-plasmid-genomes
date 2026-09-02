"""Plot mean tRNA-Sec feature importance from three locked classifier runs."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PACKAGE_DIR / "tables" / "trna_type_top10_feature_importance_for_plot.csv"
PLOT_DATA_CSV = PACKAGE_DIR / "tables" / "Supplementary_Figure_S12_plot_data.csv"
FIGURE_DIR = ROOT / "GitHub_upload_V5_figures_20260831_figures_only" / PACKAGE_DIR.name
OUTPUT_PNG = FIGURE_DIR / "Supplementary_Figure_S12_generated.png"
OUTPUT_SVG = FIGURE_DIR / "Supplementary_Figure_S12_generated.svg"

FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_SIZE_PT = 7
PX_PER_PT = 96 / 72
WIDTH = 690
HEIGHT = 370
LEFT = 500
RIGHT = 680
TOP = 7
BOTTOM = 313
AXIS_WIDTH = 1.0
MARK_WIDTH = 0.7
BAR_HEIGHT = 20
BAR_COLOR = "#8fbad1"
TARGET = "tRNA-Sec"
TASK = "classification"
RUNS = (1, 2, 3)


def require_arial() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError("Arial is required: C:/Windows/Fonts/arial.ttf")


def load_plot_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_CSV, keep_default_na=False)
    required = {
        "target", "task", "run", "seed", "Code", "Product", "normalized_gain_pct",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing S12 columns: {sorted(missing)}")
    selected = data.loc[data["target"].eq(TARGET) & data["task"].eq(TASK)].copy()
    selected["run"] = pd.to_numeric(selected["run"], errors="raise").astype(int)
    selected["seed"] = pd.to_numeric(selected["seed"], errors="raise").astype(int)
    selected["normalized_gain_pct"] = pd.to_numeric(
        selected["normalized_gain_pct"], errors="raise"
    )
    observed_runs = tuple(sorted(selected["run"].unique()))
    if observed_runs != RUNS:
        raise ValueError(f"Expected runs {RUNS}, found {observed_runs}")

    records = []
    for (code, product), group in selected.groupby(["Code", "Product"], sort=False):
        values_by_run = {
            int(row.run): float(row.normalized_gain_pct)
            for row in group.itertuples(index=False)
        }
        values = [values_by_run.get(run, 0.0) for run in RUNS]
        records.append(
            {
                "Code": code,
                "Product": product,
                "run1_seed42_pct": values[0],
                "run2_seed43_pct": values[1],
                "run3_seed44_pct": values[2],
                "mean_normalized_gain_pct": sum(values) / len(values),
            }
        )
    output = (
        pd.DataFrame(records)
        .sort_values("mean_normalized_gain_pct", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    output.insert(0, "rank", range(1, len(output) + 1))
    return output


def svg_text(x: float, y: float, value: str, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Arial" font-size="{FONT_SIZE_PT}pt" fill="black">'
        f'{html.escape(value)}</text>'
    )


def write_svg(data: pd.DataFrame) -> None:
    plot_width = RIGHT - LEFT
    step = 28
    centers = [18 + index * step for index in range(len(data))]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for center, row in zip(centers, data.itertuples(index=False)):
        parts.append(svg_text(LEFT - 10, center + 3, str(row.Product), "end"))
        bar_width = plot_width * float(row.mean_normalized_gain_pct) / 100.0
        parts.append(
            f'<rect x="{LEFT}" y="{center - BAR_HEIGHT / 2:.2f}" width="{bar_width:.4f}" '
            f'height="{BAR_HEIGHT}" fill="{BAR_COLOR}" stroke="black" stroke-width="{MARK_WIDTH}"/>'
        )
    parts.append(
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="black" stroke-width="{AXIS_WIDTH}"/>'
    )
    parts.append(
        f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="black" stroke-width="{AXIS_WIDTH}"/>'
    )
    for tick in (0, 50, 100):
        x = LEFT + plot_width * tick / 100.0
        parts.append(
            f'<line x1="{x:.2f}" y1="{BOTTOM}" x2="{x:.2f}" y2="{BOTTOM + 6}" '
            f'stroke="black" stroke-width="{MARK_WIDTH}"/>'
        )
        parts.append(svg_text(x, BOTTOM + 22, str(tick), "middle"))
    parts.append(svg_text((LEFT + RIGHT) / 2, HEIGHT - 8, "Percentage of total feature importance", "middle"))
    parts.append("</svg>")
    OUTPUT_SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_png(data: pd.DataFrame) -> None:
    scale = 4
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), round(FONT_SIZE_PT * PX_PER_PT * scale))

    def text(x: float, y: float, value: str, anchor: str) -> None:
        draw.text((x * scale, y * scale), value, fill="black", font=font, anchor=anchor)

    plot_width = RIGHT - LEFT
    for index, row in enumerate(data.itertuples(index=False)):
        center = 18 + index * 28
        text(LEFT - 10, center, str(row.Product), "rm")
        bar_width = plot_width * float(row.mean_normalized_gain_pct) / 100.0
        draw.rectangle(
            (
                LEFT * scale,
                (center - BAR_HEIGHT / 2) * scale,
                (LEFT + bar_width) * scale,
                (center + BAR_HEIGHT / 2) * scale,
            ),
            fill=BAR_COLOR,
            outline="black",
            width=round(MARK_WIDTH * scale),
        )
    draw.line((LEFT * scale, TOP * scale, LEFT * scale, BOTTOM * scale), fill="black", width=round(AXIS_WIDTH * scale))
    draw.line((LEFT * scale, BOTTOM * scale, RIGHT * scale, BOTTOM * scale), fill="black", width=round(AXIS_WIDTH * scale))
    for tick in (0, 50, 100):
        x = LEFT + plot_width * tick / 100.0
        draw.line((x * scale, BOTTOM * scale, x * scale, (BOTTOM + 6) * scale), fill="black", width=round(MARK_WIDTH * scale))
        text(x, BOTTOM + 17, str(tick), "mm")
    text((LEFT + RIGHT) / 2, HEIGHT - 5, "Percentage of total feature importance", "ms")
    image.save(OUTPUT_PNG, dpi=(600, 600))


def main() -> int:
    require_arial()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_plot_data()
    data.to_csv(PLOT_DATA_CSV, index=False)
    write_svg(data)
    write_png(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
