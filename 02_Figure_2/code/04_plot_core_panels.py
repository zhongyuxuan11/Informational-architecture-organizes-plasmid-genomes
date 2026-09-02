"""Plot V6 core ML panels in original-summary and raw-run-point versions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, precision_recall_curve

COLORS = {
    "LightGBM": "#E76F51",
    "XGBoost": "#F4A261",
    "LogisticRegression": "#2A9D8F",
    "LinearSVC": "#457B9D",
    "RandomForest": "#8E6C8A",
    "SGDClassifier": "#7A8F55",
    "ComplementNB": "#8D6E63",
}
MODEL_ORDER = (
    "LightGBM",
    "XGBoost",
    "SGDClassifier",
    "LinearSVC",
    "RandomForest",
    "LogisticRegression",
    "ComplementNB",
)
BLUE = "#8EBAD7"
CORAL = "#EDB29B"
DOTS = ("#1B6CA8", "#D1495B", "#2A9D8F")
DISPLAY_NAMES = {
    "LinearSVC": "Linear SVM",
    "RandomForest": "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "SGDClassifier": "SGD Classifier",
    "ComplementNB": "Complement Naive Bayes",
}


def configure_plot() -> None:
    arial_path = Path("C:/Windows/Fonts/arial.ttf")
    if not arial_path.is_file():
        raise FileNotFoundError("Arial is required: C:/Windows/Fonts/arial.ttf")
    font_manager.fontManager.addfont(str(arial_path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 0.7,
            "patch.linewidth": 0.7,
            "svg.fonttype": "none",
        }
    )


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    print(f"saved {path}", flush=True)
    fig.savefig(path.with_suffix(".png"), dpi=600)
    print(f"saved {path.with_suffix('.png')}", flush=True)
    plt.close(fig)


def pooled_pr_curves(predictions: pd.DataFrame, output: Path, show_runs: bool) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for model_name in MODEL_ORDER:
        current = predictions.loc[predictions["model"].eq(model_name)]
        if show_runs:
            for run in (1, 2, 3):
                run_data = current.loc[current["run"].eq(run)]
                precision, recall, _ = precision_recall_curve(
                    run_data["y_true"], run_data["y_score"]
                )
                ax.plot(recall, precision, color=COLORS[model_name], alpha=0.22, linewidth=0.7)
        precision, recall, _ = precision_recall_curve(current["y_true"], current["y_score"])
        auprc = average_precision_score(current["y_true"], current["y_score"])
        ax.plot(
            recall,
            precision,
            color=COLORS[model_name],
            linewidth=0.7,
            label=f"{DISPLAY_NAMES.get(model_name, model_name)} ({auprc:.3f})",
        )
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, handlelength=1.3, handletextpad=0.35)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.97)
    finish(fig, output)


def model_metric_panel(detail: pd.DataFrame, output: Path, show_runs: bool) -> None:
    metrics = ("AUPRC", "AUROC", "F1")
    summary = detail.groupby("model", as_index=False).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in metrics
            for stat in ("mean", "std")
        }
    ).set_index("model").loc[list(MODEL_ORDER)]
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.09, 2.55))
    for metric_index, (metric, color) in enumerate(
        zip(metrics, (BLUE, "#B8D8BA", CORAL))
    ):
        positions = x + (metric_index - 1) * width
        ax.bar(
            positions,
            summary[f"{metric}_mean"],
            width,
            yerr=summary[f"{metric}_std"],
            color=color,
            edgecolor="black",
            label=metric,
            error_kw={"elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7},
        )
        if show_runs:
            for run_index, run in enumerate((1, 2, 3)):
                values = (
                    detail.loc[detail["run"].eq(run)]
                    .set_index("model")
                    .loc[list(MODEL_ORDER), metric]
                    .to_numpy()
                )
                ax.scatter(
                    positions + (run_index - 1) * 0.028,
                    values,
                    s=8,
                    facecolor=DOTS[run_index],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(model, model) for model in MODEL_ORDER],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Performance")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.97)
    finish(fig, output)


def regression_scatter(predictions: pd.DataFrame, output: Path, show_runs: bool) -> None:
    print(f"starting regression panel {output.name}", flush=True)
    observed = predictions["y_true_tRNA_count"].to_numpy(dtype=float)
    predicted = predictions["y_pred_tRNA_count"].to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression panel requires finite observed and predicted counts")
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    print(f"axes created for {output.name}", flush=True)
    if show_runs:
        for run_index, run in enumerate((1, 2, 3)):
            mask = predictions["run"].eq(run).to_numpy()
            ax.scatter(
                observed[mask],
                predicted[mask],
                s=7,
                alpha=0.24,
                color=DOTS[run_index],
                edgecolor="none",
                label=f"Run {run}",
            )
        ax.legend(frameon=False, loc="upper left")
    else:
        collection = ax.hexbin(
            observed, predicted, gridsize=35, mincnt=1, cmap="Blues", linewidths=0
        )
        collection.set_rasterized(True)
    print(f"points drawn for {output.name}", flush=True)
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#666666", linewidth=0.7)
    ax.set(
        xlabel="Observed tRNA count",
        ylabel="Predicted tRNA count",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.97)
    print(f"layout complete for {output.name}", flush=True)
    finish(fig, output)


def feature_importance_panel(
    detail: pd.DataFrame,
    task: str,
    output: Path,
    show_runs: bool,
) -> None:
    current = detail.loc[detail["task"].eq(task)].copy()
    summary = (
        current.groupby(["Code", "Product"], as_index=False)
        .agg(
            mean=("normalized_gain_pct", "mean"),
            SD=("normalized_gain_pct", "std"),
        )
        .nlargest(20, "mean")
        .sort_values("mean", ascending=True)
    )
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(3.35, 4.0))
    ax.barh(
        y,
        summary["mean"],
        xerr=summary["SD"],
        color=BLUE if task == "classification" else CORAL,
        edgecolor="black",
        error_kw={"elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7},
    )
    if show_runs:
        key = summary[["Code"]].copy()
        for run_index, run in enumerate((1, 2, 3)):
            values = key.merge(
                current.loc[current["run"].eq(run), ["Code", "normalized_gain_pct"]],
                on="Code",
                validate="one_to_one",
            )["normalized_gain_pct"].to_numpy()
            ax.scatter(
                values,
                y + (run_index - 1) * 0.10,
                s=8,
                color=DOTS[run_index],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(summary["Product"])
    ax.set_xlabel("Percentage of total feature importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.52, right=0.97, bottom=0.12, top=0.98)
    finish(fig, output)


def topk_panel(detail: pd.DataFrame, task: str, output: Path, show_runs: bool) -> None:
    metric = "AUPRC" if task == "classification" else "R2"
    current = detail.loc[detail["task"].eq(task)].copy()
    summary = current.groupby("K", as_index=False)[metric].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("K")
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.errorbar(
        x,
        summary["mean"],
        yerr=summary["std"],
        color="#D84A3A" if task == "classification" else "#4C78A8",
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.7,
        capsize=1.5,
        linewidth=0.7,
    )
    if show_runs:
        for run_index, run in enumerate((1, 2, 3)):
            values = (
                current.loc[current["run"].eq(run)]
                .set_index("K")
                .loc[summary["K"], metric]
                .to_numpy()
            )
            ax.scatter(
                x + (run_index - 1) * 0.07,
                values,
                s=9,
                color=DOTS[run_index],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["K"].astype(int))
    ax.set_xlabel("Number of top features")
    ax.set_ylabel(metric.replace("R2", r"R$^2$"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.97)
    finish(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--skip-classifier-comparison", action="store_true")
    args = parser.parse_args()
    configure_plot()
    original = args.figure_root / "original_style"
    raw = args.figure_root / "with_raw_points"
    original.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    print("figure directories created", flush=True)

    reg_predictions = pd.read_csv(
        args.regressor_results / "regressor_test_predictions.csv"
    )
    print("regression predictions loaded", flush=True)
    fi_detail = pd.read_csv(args.primary_results / "feature_importance_each_run.csv")
    print("feature importance loaded", flush=True)
    topk_detail = pd.read_csv(args.primary_results / "topk_metrics_each_run.csv")
    print("Top-K metrics loaded", flush=True)

    if not args.skip_classifier_comparison:
        class_detail = pd.read_csv(
            args.classifier_results / "classifier_test_metrics_each_run.csv"
        )
        class_predictions = pd.read_csv(
            args.classifier_results / "classifier_test_predictions.csv"
        )
        pooled_pr_curves(class_predictions, original / "Fig2B_7model_PR_curves.svg", False)
        pooled_pr_curves(class_predictions, raw / "Fig2B_7model_PR_curves_raw_runs.svg", True)
        model_metric_panel(class_detail, original / "S2_7model_AUPRC_AUROC_F1.svg", False)
        model_metric_panel(class_detail, raw / "S2_7model_AUPRC_AUROC_F1_raw_points.svg", True)
    regression_scatter(reg_predictions, original / "Fig2C_total_tRNA_regression.svg", False)
    regression_scatter(reg_predictions, raw / "Fig2C_total_tRNA_regression_raw_points.svg", True)
    for task, name in (("classification", "Fig2D"), ("regression", "S10A")):
        feature_importance_panel(fi_detail, task, original / f"{name}_feature_importance.svg", False)
        feature_importance_panel(fi_detail, task, raw / f"{name}_feature_importance_raw_points.svg", True)
    for task, name in (("classification", "Fig2E"), ("regression", "S10B")):
        topk_panel(topk_detail, task, original / f"{name}_topK_performance.svg", False)
        topk_panel(topk_detail, task, raw / f"{name}_topK_performance_raw_points.svg", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
