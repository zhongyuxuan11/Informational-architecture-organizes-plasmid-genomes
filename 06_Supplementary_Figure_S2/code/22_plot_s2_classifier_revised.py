"""Replot S2 classifier confusion matrices and raw replicate metrics without Matplotlib."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


MODELS = [
    "ComplementNB",
    "LightGBM",
    "LinearSVC",
    "XGBoost",
    "LogisticRegression",
    "RandomForest",
    "SGDClassifier",
]
LABELS = {
    "ComplementNB": "Complement Naive Bayes",
    "LightGBM": "LightGBM",
    "LinearSVC": "Linear SVM",
    "XGBoost": "XGBoost",
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "SGDClassifier": "SGD Classifier",
}
TABLE_METRICS = ["AUPRC", "AUROC", "Precision", "Recall", "F1"]
PLOT_METRICS = ["AUROC", "Precision", "Recall", "F1"]
BAR_COLORS = {
    "AUROC": "#729fc1",
    "Precision": "#f2be2d",
    "Recall": "#b7d5e8",
    "F1": "#999999",
}


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(data_dir / "classifier_test_metrics_each_run.csv")
    predictions = pd.read_csv(data_dir / "classifier_test_predictions.csv")
    if set(metrics["model"]) != set(MODELS):
        raise ValueError("S2 metrics must contain the seven manuscript classifiers")
    if set(predictions["model"]) != set(MODELS):
        raise ValueError("S2 predictions must contain the seven manuscript classifiers")
    return metrics, predictions


def build_per_run_table(metrics: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, run), group in pred.groupby(["model", "run"], sort=False):
        y_true = group.y_true.astype(int).to_numpy()
        y_pred = group.y_pred.astype(int).to_numpy()
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        metric_row = metrics.loc[
            metrics.model.eq(model) & metrics.run.astype(int).eq(int(run))
        ].iloc[0]
        rows.append(
            {
                "model": model,
                "run": int(run),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "AUPRC": float(metric_row.AUPRC),
                "AUROC": float(metric_row.AUROC),
                "Precision": tp / (tp + fp) if tp + fp else 0.0,
                "Recall": tp / (tp + fn) if tp + fn else 0.0,
                "F1": float(metric_row.F1),
            }
        )
    table = pd.DataFrame(rows)
    if set(table.model) != set(MODELS) or not table.groupby("model").run.nunique().eq(3).all():
        raise ValueError("Expected seven models and three independent runs")
    return table


def build_summary(conf: pd.DataFrame) -> pd.DataFrame:
    summary = conf.groupby("model")[TABLE_METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary["Model"] = summary.model.map(LABELS)
    return summary.sort_values("AUPRC_mean", ascending=False).reset_index(drop=True)


def raw_confusion_sum(conf: pd.DataFrame, model: str) -> np.ndarray:
    return conf.loc[conf.model.eq(model), ["TN", "FP", "FN", "TP"]].sum().to_numpy(int).reshape(2, 2)


def heat_color(value: float) -> str:
    light = np.array([238, 245, 250])
    dark = np.array([22, 57, 109])
    rgb = np.rint(light * (1.0 - value) + dark * value).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def txt(x: float, y: float, text: str, size: float = 7.0, anchor: str = "start", extra: str = "") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial" font-size="{size:.2f}pt" '
        f'text-anchor="{anchor}" {extra}>{html.escape(text)}</text>'
    )


def build_svg(conf: pd.DataFrame, summary: pd.DataFrame) -> str:
    width = 453.54
    height = 532.8
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="16cm" height="18.79cm" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        txt(8, 16, "A", 7, extra='font-weight="bold"'),
    ]

    x0, y0 = 36, 34
    panel_w, panel_h = 102, 112
    cell = 30
    for idx, model in enumerate(MODELS):
        px = x0 + (idx % 4) * panel_w
        py = y0 + (idx // 4) * panel_h
        parts.append(txt(px + 63, py - 8, LABELS[model], 7, "middle"))
        count_mat = raw_confusion_sum(conf, model)
        for i in range(2):
            for j in range(2):
                count = int(count_mat[i, j])
                color_value = count / float(count_mat.max())
                cx = px + 32 + j * cell
                cy = py + i * cell
                fill = "white" if color_value > 0.58 else "black"
                parts.append(
                    f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cell:.2f}" height="{cell:.2f}" '
                    f'fill="{heat_color(color_value)}" stroke="black" stroke-width="0.7"/>'
                )
                parts.append(txt(cx + cell / 2, cy + cell / 2 + 2.6, f"{count}", 7, "middle", f'fill="{fill}"'))
        parts.append(txt(px + 47, py + 70, "Absent", 7, "middle"))
        parts.append(txt(px + 77, py + 70, "Present", 7, "middle"))
        parts.append(txt(px + 18, py + 18, "Absent", 7, "end"))
        parts.append(txt(px + 18, py + 48, "Present", 7, "end"))

    table_y = 244
    parts.append(txt(8, table_y - 17, "Mean ± SD, n = 3 independent 80:20 splits", 7))
    cols = [8, 135, 196, 257, 318, 379]
    for x, header in zip(cols, ["Model"] + TABLE_METRICS):
        parts.append(txt(x, table_y, header, 7))
    parts.append(f'<line x1="8" y1="{table_y + 8}" x2="443" y2="{table_y + 8}" stroke="black" stroke-width="0.7"/>')
    for row_idx, row in summary.iterrows():
        y = table_y + 16 + row_idx * 15
        parts.append(txt(cols[0], y, str(row.Model), 7))
        for metric_idx, metric in enumerate(TABLE_METRICS):
            value = f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            parts.append(txt(cols[metric_idx + 1], y, value, 7))

    parts.append(txt(8, 365, "B", 7, extra='font-weight="bold"'))
    plot_x, plot_y, plot_w, plot_h = 50, 380, 322, 90
    parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="black" stroke-width="1"/>')
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="black" stroke-width="1"/>')
    for tick in np.linspace(0, 1, 6):
        y = plot_y + plot_h * (1 - tick)
        parts.append(f'<line x1="{plot_x - 2.5}" y1="{y:.2f}" x2="{plot_x}" y2="{y:.2f}" stroke="black" stroke-width="0.7"/>')
        parts.append(txt(plot_x - 5, y + 2.5, f"{tick:.1f}", 7, "end"))
    parts.append(txt(17, plot_y + plot_h / 2, "Performance", 7, "middle", f'transform="rotate(-90 17 {plot_y + plot_h / 2:.2f})"'))

    order = MODELS
    group_w = plot_w / len(order)
    bar_w = 5.4
    offsets = np.linspace(-9.0, 9.0, len(PLOT_METRICS))
    jitter = np.array([-0.03, 0.0, 0.03]) * group_w
    for model_idx, model in enumerate(order):
        gx = plot_x + group_w * (model_idx + 0.5)
        label_y = plot_y + plot_h + 8
        parts.append(txt(gx, label_y, LABELS[model], 7, "start", f'transform="rotate(40 {gx:.2f} {label_y:.2f})"'))
        for metric_idx, metric in enumerate(PLOT_METRICS):
            mean = float(summary.loc[summary.model.eq(model), f"{metric}_mean"].iloc[0])
            sd = float(summary.loc[summary.model.eq(model), f"{metric}_std"].iloc[0])
            bx = gx + offsets[metric_idx] - bar_w / 2
            by = plot_y + plot_h * (1 - mean)
            bh = plot_h * mean
            err_top = plot_y + plot_h * (1 - min(1.0, mean + sd))
            err_bottom = plot_y + plot_h * (1 - max(0.0, mean - sd))
            mid = bx + bar_w / 2
            parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="{BAR_COLORS[metric]}" stroke="black" stroke-width="0.7"/>')
            parts.append(f'<line x1="{mid:.2f}" y1="{err_top:.2f}" x2="{mid:.2f}" y2="{err_bottom:.2f}" stroke="black" stroke-width="0.7"/>')
            parts.append(f'<line x1="{mid - 2:.2f}" y1="{err_top:.2f}" x2="{mid + 2:.2f}" y2="{err_top:.2f}" stroke="black" stroke-width="0.7"/>')
            parts.append(f'<line x1="{mid - 2:.2f}" y1="{err_bottom:.2f}" x2="{mid + 2:.2f}" y2="{err_bottom:.2f}" stroke="black" stroke-width="0.7"/>')
            values = conf.loc[conf.model.eq(model)].sort_values("run")[metric].to_numpy()
            for point_idx, value in enumerate(values):
                cy = plot_y + plot_h * (1 - value)
                cx = mid + jitter[point_idx]
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="1.25" fill="black"/>')

    legend_x = 382
    for idx, metric in enumerate(PLOT_METRICS):
        ly = plot_y + 5 + idx * 15
        parts.append(f'<rect x="{legend_x:.2f}" y="{ly:.2f}" width="6.50" height="6.50" fill="{BAR_COLORS[metric]}" stroke="black" stroke-width="0.7"/>')
        parts.append(txt(legend_x + 9, ly + 5.7, metric, 7))

    parts.append("</svg>")
    return "\n".join(parts)


def font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    if not arial.is_file():
        raise FileNotFoundError("Arial is required: C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(arial, size_px)


def write_png(conf: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    # The PNG mirrors the SVG layout but uses direct Pillow drawing.
    scale = 600 / 72
    im = Image.new("RGB", (round(453.54 * scale), round(532.8 * scale)), "white")
    draw = ImageDraw.Draw(im)

    def sx(value: float) -> int:
        return round(value * scale)

    def sy(value: float) -> int:
        return round(value * scale)

    def fpt(value: float, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return font(max(1, round(value * scale)), bold)

    def draw_text(x: float, y: float, text: str, size: float = 7.0, fill: str = "black", anchor: str = "la", bold: bool = False) -> None:
        draw.text((sx(x), sy(y)), text, fill=fill, font=fpt(size, bold), anchor=anchor)

    def draw_rotated(x: float, y: float, text: str, angle: float, size: float = 7) -> None:
        local = Image.new("RGBA", (sx(105), sy(18)), (255, 255, 255, 0))
        local_draw = ImageDraw.Draw(local)
        local_draw.text((sx(52), sy(5)), text, fill="black", font=fpt(size), anchor="ma")
        rotated = local.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        im.paste(rotated, (sx(x) - rotated.width // 2, sy(y) - rotated.height // 2), rotated)

    draw_text(8, 16, "A", 7, bold=True)
    x0, y0 = 36, 34
    panel_w, panel_h = 102, 112
    cell = 30
    for idx, model in enumerate(MODELS):
        px = x0 + (idx % 4) * panel_w
        py = y0 + (idx // 4) * panel_h
        draw_text(px + 63, py - 8, LABELS[model], 7, anchor="ma")
        count_mat = raw_confusion_sum(conf, model)
        for i in range(2):
            for j in range(2):
                count = int(count_mat[i, j])
                color_value = count / float(count_mat.max())
                cx = px + 32 + j * cell
                cy = py + i * cell
                fill = "white" if color_value > 0.58 else "black"
                draw.rectangle((sx(cx), sy(cy), sx(cx + cell), sy(cy + cell)), fill=heat_color(color_value), outline="black", width=sx(0.7))
                draw_text(cx + cell / 2, cy + cell / 2, f"{count}", 7, fill, "mm")
        draw_text(px + 47, py + 70, "Absent", 7, anchor="ma")
        draw_text(px + 77, py + 70, "Present", 7, anchor="ma")
        draw_text(px + 18, py + 18, "Absent", 7, anchor="ra")
        draw_text(px + 18, py + 48, "Present", 7, anchor="ra")

    table_y = 244
    draw_text(8, table_y - 17, "Mean ± SD, n = 3 independent 80:20 splits", 7)
    cols = [8, 135, 196, 257, 318, 379]
    for x, header in zip(cols, ["Model"] + TABLE_METRICS):
        draw_text(x, table_y, header, 7)
    draw.line((sx(8), sy(table_y + 8), sx(443), sy(table_y + 8)), fill="black", width=sx(0.7))
    for row_idx, row in summary.iterrows():
        y = table_y + 16 + row_idx * 15
        draw_text(cols[0], y, str(row.Model), 7)
        for metric_idx, metric in enumerate(TABLE_METRICS):
            value = f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            draw_text(cols[metric_idx + 1], y, value, 7)

    draw_text(8, 365, "B", 7, bold=True)
    plot_x, plot_y, plot_w, plot_h = 50, 380, 322, 90
    draw.line((sx(plot_x), sy(plot_y), sx(plot_x), sy(plot_y + plot_h)), fill="black", width=sx(1))
    draw.line((sx(plot_x), sy(plot_y + plot_h), sx(plot_x + plot_w), sy(plot_y + plot_h)), fill="black", width=sx(1))
    for tick in np.linspace(0, 1, 6):
        y = plot_y + plot_h * (1 - tick)
        draw.line((sx(plot_x - 2.5), sy(y), sx(plot_x), sy(y)), fill="black", width=sx(0.7))
        draw_text(plot_x - 5, y, f"{tick:.1f}", 7, anchor="rm")
    draw_rotated(17, plot_y + plot_h / 2, "Performance", 90, 7)

    order = MODELS
    group_w = plot_w / len(order)
    bar_w = 5.4
    offsets = np.linspace(-9.0, 9.0, len(PLOT_METRICS))
    jitter = np.array([-0.03, 0.0, 0.03]) * group_w
    for model_idx, model in enumerate(order):
        gx = plot_x + group_w * (model_idx + 0.5)
        draw_rotated(gx + 10, 500, LABELS[model], -40, 7)
        for metric_idx, metric in enumerate(PLOT_METRICS):
            mean = float(summary.loc[summary.model.eq(model), f"{metric}_mean"].iloc[0])
            sd = float(summary.loc[summary.model.eq(model), f"{metric}_std"].iloc[0])
            bx = gx + offsets[metric_idx] - bar_w / 2
            by = plot_y + plot_h * (1 - mean)
            bh = plot_h * mean
            err_top = plot_y + plot_h * (1 - min(1.0, mean + sd))
            err_bottom = plot_y + plot_h * (1 - max(0.0, mean - sd))
            mid = bx + bar_w / 2
            draw.rectangle((sx(bx), sy(by), sx(bx + bar_w), sy(by + bh)), fill=BAR_COLORS[metric], outline="black", width=sx(0.7))
            draw.line((sx(mid), sy(err_top), sx(mid), sy(err_bottom)), fill="black", width=sx(0.7))
            draw.line((sx(mid - 2), sy(err_top), sx(mid + 2), sy(err_top)), fill="black", width=sx(0.7))
            draw.line((sx(mid - 2), sy(err_bottom), sx(mid + 2), sy(err_bottom)), fill="black", width=sx(0.7))
            values = conf.loc[conf.model.eq(model)].sort_values("run")[metric].to_numpy()
            for point_idx, value in enumerate(values):
                cy = plot_y + plot_h * (1 - value)
                cx = mid + jitter[point_idx]
                r = sx(1.25)
                draw.ellipse((sx(cx) - r, sy(cy) - r, sx(cx) + r, sy(cy) + r), fill="black")

    legend_x = 382
    for idx, metric in enumerate(PLOT_METRICS):
        ly = plot_y + 5 + idx * 15
        draw.rectangle((sx(legend_x), sy(ly), sx(legend_x + 6.5), sy(ly + 6.5)), fill=BAR_COLORS[metric], outline="black", width=sx(0.7))
        draw_text(legend_x + 9, ly + 5.7, metric, 7)
    im.save(out_png, dpi=(600, 600))


def main() -> None:
    parser = argparse.ArgumentParser()
    package = Path(__file__).resolve().parents[1]
    figure_root = Path(__file__).resolve().parents[3] / "GitHub_upload_V5_figures_20260831_figures_only" / package.name
    parser.add_argument("--data-dir", type=Path, default=package / "tables")
    parser.add_argument("--out-dir", type=Path, default=figure_root)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics, pred = read_inputs(args.data_dir)
    conf = build_per_run_table(metrics, pred)
    summary = build_summary(conf)
    conf.to_csv(args.out_dir / "seven_classifier_confusion_metrics_each_run.csv", index=False)
    summary.rename(columns={"model": "model_id"}).to_csv(
        args.out_dir / "seven_classifier_confusion_metrics_summary.csv", index=False
    )

    svg = build_svg(conf, summary)
    (args.out_dir / "Supplementary_Figure_S2_generated.svg").write_text(svg, encoding="utf-8")
    write_png(conf, summary, args.out_dir / "Supplementary_Figure_S2_generated.png")


if __name__ == "__main__":
    main()

