"""Redraw manuscript machine-learning figures with Matplotlib and DOCX geometry."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

# Exact mean-bar colors used by the manuscript Fig. 2F reference panels.
BLUE = BLUE_DARK = "#8EBAD7"
CORAL = CORAL_DARK = "#EDB29B"
CREAM, GREY, PURPLE, INK = "#F4ECD2", "#BDBDBD", "#BBB5D1", "#202020"
OBJECT_LW, AXIS_LW, FONT_SIZE = 0.7, 1.0, 7.0

ARIAL_PATHS = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/mnt/c/Windows/Fonts/arial.ttf"),
]
arial_path = next((path for path in ARIAL_PATHS if path.exists()), None)
if arial_path is None:
    raise FileNotFoundError("Arial font file was not found")
font_manager.fontManager.addfont(str(arial_path))

mpl.rcParams.update({
    "font.family": "Arial", "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
    "axes.linewidth": AXIS_LW, "xtick.major.width": OBJECT_LW,
    "ytick.major.width": OBJECT_LW, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.edgecolor": "black",
    "axes.labelcolor": "black", "xtick.color": "black",
    "ytick.color": "black", "text.color": "black",
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def num(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def save(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=600, facecolor="white")
    plt.close(fig)


def style(ax: mpl.axes.Axes, zero: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["bottom"].set_linewidth(AXIS_LW)
    ax.tick_params(direction="out", width=OBJECT_LW, length=2.5, pad=2)
    ax.grid(False)
    if zero:
        ax.axhline(0, color=INK, linewidth=OBJECT_LW, zorder=1)


def letter(ax: mpl.axes.Axes, text: str, x: float = -0.14, y: float = 1.07) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FONT_SIZE,
            fontweight="bold", va="top", ha="left")


def bar_panel(ax, df, category, metric, color, order=None, labels=None,
              ylabel=None, ylim=None, rotation=0, italic=False):
    data = df[[category, "run", metric]].copy()
    data[category] = data[category].astype(str)
    data[metric] = num(data[metric])
    data = data.dropna(subset=[metric])
    if data.empty:
        raise ValueError(f"No values for {metric}")
    order = order or data[category].drop_duplicates().tolist()
    data = data[data[category].isin(order)]
    x = np.arange(len(order), dtype=float)
    means = data.groupby(category)[metric].mean().reindex(order)
    sds = data.groupby(category)[metric].std(ddof=1).reindex(order).fillna(0)
    mean_values = means.to_numpy(dtype=float)
    sd_values = sds.to_numpy(dtype=float)
    yerr = sd_values
    if metric in {"AUPRC", "AUROC", "F1", "Precision", "Recall"}:
        yerr = np.vstack([
            np.minimum(sd_values, mean_values),
            np.minimum(sd_values, 1.0 - mean_values),
        ])
    ax.bar(x, mean_values, width=0.7, color=color, edgecolor=INK,
           linewidth=OBJECT_LW, zorder=2)
    ax.errorbar(x, mean_values, yerr=yerr, fmt="none", ecolor=INK,
                elinewidth=OBJECT_LW, capsize=2, capthick=OBJECT_LW, zorder=3)
    runs = sorted(data.run.dropna().unique())
    offsets = np.linspace(-0.12, 0.12, len(runs)) if len(runs) > 1 else [0]
    for offset, run in zip(offsets, runs):
        values = data[data.run.eq(run)].set_index(category)[metric].reindex(order)
        ax.scatter(x + offset, values, c="black", s=7, linewidths=0, zorder=4)
    texts = [labels.get(v, v) if labels else v for v in order]
    texts = [v.replace("<br>", "\n") for v in texts]
    if italic:
        texts = [v.replace("tRNA-", "") for v in texts]
    ax.set_xticks(x, texts, rotation=rotation, ha="right" if rotation else "center")
    if italic:
        for tick in ax.get_xticklabels():
            tick.set_fontstyle("italic")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    style(ax)


def grouped_panel(ax, data, category, value_col, group_col, groups, colors,
                  order, labels, ylabel, ylim, legend=False):
    x = np.arange(len(order), dtype=float)
    width = 0.34 if len(groups) == 2 else 0.22
    shifts = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width
    for group, color, shift in zip(groups, colors, shifts):
        q = data[data[group_col].eq(group)].copy()
        q[category] = q[category].astype(str)
        q[value_col] = num(q[value_col])
        q = q.dropna(subset=[value_col])
        means = q.groupby(category)[value_col].mean().reindex(order)
        sds = q.groupby(category)[value_col].std(ddof=1).reindex(order).fillna(0)
        ax.bar(x + shift, means, width=width, color=color, edgecolor=INK,
               linewidth=OBJECT_LW, yerr=sds,
               error_kw={"ecolor": INK, "elinewidth": OBJECT_LW,
                          "capsize": 1.8, "capthick": OBJECT_LW},
               label=group, zorder=2)
        runs = sorted(q.run.dropna().unique())
        offsets = np.linspace(-0.055, 0.055, len(runs)) if len(runs) > 1 else [0]
        for offset, run in zip(offsets, runs):
            values = q[q.run.eq(run)].set_index(category)[value_col].reindex(order)
            ax.scatter(x + shift + offset, values, c="black", s=7,
                       linewidths=0, zorder=4)
    ax.set_xticks(x, [labels.get(v, v).replace("<br>", "\n") for v in order])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(frameon=False)
    style(ax)


def build_s2(root20: Path, root16: Path, out: Path) -> None:
    path = root16 / "figures_available_without_sgd/seven_classifier_confusion_metrics_raw_final/seven_classifier_confusion_metrics_each_run.csv"
    data = read_csv(path)
    order = ["ComplementNB", "LightGBM", "LinearSVC", "XGBoost",
             "LogisticRegression", "RandomForest", "SGDClassifier"]
    fig = plt.figure(figsize=(6.2638888889, 5.5625))
    gs = fig.add_gridspec(3, 4, height_ratios=[.31, .31, .38], hspace=.55,
                          wspace=.45, left=.08, right=.80, bottom=.12, top=.94)
    cmap = LinearSegmentedColormap.from_list("confusion", ["#F4F7FB", "#173B6C"])
    for i, model in enumerate(order):
        row, col = (0, i) if i < 4 else (1, i - 4)
        ax = fig.add_subplot(gs[row, col])
        q = data[data.model.eq(model)]
        matrix = np.array([[q.TN.sum(), q.FP.sum()], [q.FN.sum(), q.TP.sum()]])
        ax.imshow(matrix, cmap=cmap, aspect="equal")
        threshold = matrix.max() / 2
        for yy in range(2):
            for xx in range(2):
                ax.text(xx, yy, str(matrix[yy, xx]), ha="center", va="center",
                        color="white" if matrix[yy, xx] > threshold else INK)
        ax.set_xticks([0, 1], ["Absent", "Present"])
        ax.set_yticks([0, 1], ["Absent", "Present"])
        ax.set_title(model, pad=4)
        ax.tick_params(width=OBJECT_LW, length=2.5, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_LW)
    fig.add_subplot(gs[1, 3]).axis("off")
    ax = fig.add_subplot(gs[2, :])
    metrics = ["AUROC", "Precision", "Recall", "F1"]
    colors = ["#6FA1C7", "#F3BE35", "#A9CFE7", GREY]
    x, width = np.arange(len(order), dtype=float), .18
    for index, (metric, color) in enumerate(zip(metrics, colors)):
        shift = (index - 1.5) * width
        means = data.groupby("model")[metric].mean().reindex(order)
        sds = data.groupby("model")[metric].std(ddof=1).reindex(order)
        ax.bar(x + shift, means, width=width, color=color, edgecolor=INK,
               linewidth=OBJECT_LW, yerr=sds,
               error_kw={"ecolor": INK, "elinewidth": OBJECT_LW,
                          "capsize": 1.5, "capthick": OBJECT_LW}, label=metric)
        for ri, run in enumerate(sorted(data.run.unique())):
            values = data[data.run.eq(run)].set_index("model")[metric].reindex(order)
            ax.scatter(x + shift + (ri - 1) * .035, values, c="black", s=6,
                       linewidths=0, zorder=4)
    ax.set_xticks(x, order, rotation=45, ha="right")
    ax.set_ylabel("Performance")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, .5), loc="center left")
    style(ax)
    fig.text(.02, .975, "A", fontsize=FONT_SIZE, fontweight="bold", va="top")
    fig.text(.02, .39, "B", fontsize=FONT_SIZE, fontweight="bold", va="top")
    save(fig, out / "Figure_S2_seven_classifier_benchmark")


def build_s3(root: Path, out: Path) -> None:
    data = read_csv(root / "results/trna_types/trna_type_metrics_each_run.csv")
    cls = data[data.task.eq("classification")].copy()
    reg = data[data.task.str.startswith("regression") & ~data.target.eq("tRNA-Sec")].copy()
    cls_order = cls.assign(v=num(cls.AUPRC)).groupby("target").v.mean().sort_values(ascending=False).index.tolist()
    reg_order = reg.assign(v=num(reg.R2)).groupby("target").v.mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(4.9791666667, 3.5625), gridspec_kw={"hspace": .52})
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.95)
    bar_panel(axes[0], cls, "target", "AUPRC", BLUE, cls_order, ylabel="AUPRC", ylim=(0, 1.03), italic=True)
    bar_panel(axes[1], reg, "target", "R2", CORAL, reg_order, ylabel="R²", ylim=(0, 1.03), italic=True)
    letter(axes[0], "A", -.09, 1.08); letter(axes[1], "B", -.09, 1.08)
    save(fig, out / "Figure_S3_tRNA_type_prediction")


def build_s4(root20: Path, root16: Path, out: Path) -> None:
    knn = read_csv(root20 / "results/knn_baseline/knn_metrics_each_run.csv")
    host = read_csv(root16 / "results/host_chromosome_common_lightgbm_20260820/host_chromosome_metrics_each_run.csv")
    ext = read_csv(root16 / "results/temporal_external_validation_common_lightgbm_20260820_run2/temporal_external_metrics.csv")
    fig, axes = plt.subplots(1, 4, figsize=(5.8680555556, 1.8819444444), gridspec_kw={"wspace": .72})
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.90)
    bar_panel(axes[0], knn[knn.task.eq("classification")], "model", "AUPRC", BLUE_DARK, ["1-NN", "5-NN"], ylabel="AUPRC", ylim=(0, 1.02))
    bar_panel(axes[1], knn[knn.task.str.startswith("regression")], "model", "R2", "#CFE3E8", ["1-NN", "5-NN"], ylabel="R²", ylim=(0, .5))
    host_long = pd.concat([
        host[host.task.eq("classification")].assign(metric="AUPRC", value=num(host[host.task.eq("classification")].AUPRC)),
        host[host.task.str.startswith("regression")].assign(metric="R²", value=num(host[host.task.str.startswith("regression")].R2)),
    ])
    bar_panel(axes[2], host_long, "metric", "value", BLUE_DARK, ["AUPRC", "R²"], ylabel="Performance", ylim=(-.1, .5))
    rows = []
    for i, row in ext.reset_index(drop=True).iterrows():
        task = str(row.get("task", ""))
        if task == "classification": rows.append({"metric": "AUPRC", "run": i + 1, "value": row.AUPRC})
        elif task.startswith("regression"): rows.append({"metric": "R²", "run": i + 1, "value": row.R2})
    bar_panel(axes[3], pd.DataFrame(rows), "metric", "value", CORAL_DARK, ["AUPRC", "R²"], ylabel="Performance", ylim=(0, 1.02))
    for ax, panel in zip(axes, "ABCD"): letter(ax, panel, -.28, 1.08)
    save(fig, out / "Figure_S4_alternative_baselines_external_validation")


def build_s5(root: Path, out: Path) -> None:
    data = read_csv(root / "results/generalization/generalization_metrics_each_run.csv")
    labels = {"random_80_20": "Random split", "Assembly_ID_blocked": "Genome-blocked",
              "species_blocked": "Species-blocked", "genus_blocked": "Genus-blocked",
              "small_lt_100kb": "<100 kb", "large_ge_100kb": "≥100 kb",
              "small_to_large": "<100 →\n≥100 kb", "large_to_small": "≥100 →\n<100 kb"}
    order = ["random_80_20", "Assembly_ID_blocked", "species_blocked", "genus_blocked"]
    colors = [BLUE, "#45A7AD", "#DE8E87", GREY]
    fig, axes = plt.subplots(2, 2, figsize=(4.0486111111, 3.6736111111), gridspec_kw={"hspace": .72, "wspace": .52})
    fig.subplots_adjust(left=.12, right=.74, bottom=.14, top=.94)
    internal = data[data.experiment.eq("internal_generalization")]
    for ax, subset, metric, ylabel in [(axes[0, 0], internal[internal.task.eq("classification")], "AUPRC", "AUPRC"), (axes[0, 1], internal[internal.task.str.startswith("regression")], "R2", "R²")]:
        subset = subset.copy(); subset[metric] = num(subset[metric]); x = np.arange(len(order))
        means = subset.groupby("split_strategy")[metric].mean().reindex(order)
        sds = subset.groupby("split_strategy")[metric].std(ddof=1).reindex(order).fillna(0)
        ax.bar(x, means, color=colors, edgecolor=INK, linewidth=OBJECT_LW, yerr=sds,
               error_kw={"ecolor": INK, "elinewidth": OBJECT_LW, "capsize": 2, "capthick": OBJECT_LW})
        for xi, key in enumerate(order):
            values = subset[subset.split_strategy.eq(key)].sort_values("run")[metric]
            ax.scatter(xi + np.linspace(-.12, .12, len(values)), values, c="black", s=7, linewidths=0, zorder=4)
        ax.set_xticks(x, [labels[k] for k in order], rotation=35, ha="right")
        ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); style(ax)
    within = data[data.experiment.eq("within_size")].copy()
    within["value"] = np.where(within.task.eq("classification"), num(within.AUPRC), num(within.R2))
    within["metric"] = np.where(within.task.eq("classification"), "AUPRC", "R²")
    grouped_panel(axes[1, 0], within, "split_strategy", "value", "metric", ["AUPRC", "R²"], [BLUE_DARK, CORAL_DARK], ["small_lt_100kb", "large_ge_100kb"], labels, "Performance", (0, 1.05))
    transfer = data[data.experiment.eq("directional_size_transfer")].copy()
    transfer["value"] = np.where(transfer.task.eq("classification"), num(transfer.AUPRC), num(transfer.R2))
    transfer["metric"] = np.where(transfer.task.eq("classification"), "AUPRC", "R²")
    grouped_panel(axes[1, 1], transfer, "split_strategy", "value", "metric", ["AUPRC", "R²"], [BLUE_DARK, CORAL_DARK], ["large_to_small", "small_to_large"], labels, "Performance", (-.45, 1.05))
    style(axes[1, 1], zero=True)
    for ax, panel in zip(axes.flat, "ABCD"): letter(ax, panel, -.20, 1.10)
    legend_labels = ["Random split", "Genome-\nblocked", "Species-\nblocked", "Genus-\nblocked"]
    fig.legend(handles=[Patch(facecolor=c, edgecolor=INK, linewidth=OBJECT_LW, label=v) for c, v in zip(colors, legend_labels)], title="Split strategy", frameon=False, bbox_to_anchor=(.995, .96), loc="upper right")
    axes[1, 0].legend(handles=[Patch(facecolor=BLUE_DARK, edgecolor=INK, label="AUPRC"), Patch(facecolor=CORAL_DARK, edgecolor=INK, label="R²")], frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.28), ncol=2)
    save(fig, out / "Figure_S5_split_size_robustness")


def build_s6(root: Path, out: Path) -> None:
    data = read_csv(root / "results/phylum_stratified_validation/phylum_stratified_metrics_each_run_n_gt10.csv")
    cls = ["Pseudomonadota", "Bacillota", "Actinomycetota", "Methanobacteriota", "Bacteroidota", "Spirochaetota", "Cyanobacteriota"]
    reg = ["Pseudomonadota", "Bacillota", "Actinomycetota", "Methanobacteriota"]
    fig, axes = plt.subplots(1, 2, figsize=(5.7361111111, 2.4027777778), gridspec_kw={"wspace": .45})
    fig.subplots_adjust(left=.09, right=.98, bottom=.34, top=.91)
    bar_panel(axes[0], data[data.task.eq("classification")], "phylum", "AUPRC", BLUE_DARK, cls, ylabel="AUPRC", ylim=(0, 1.05), rotation=35)
    bar_panel(axes[1], data[data.task.str.startswith("regression")], "phylum", "R2", CORAL_DARK, reg, ylabel="R²", ylim=(-1.4, 1.05), rotation=35)
    style(axes[1], zero=True); letter(axes[0], "A", -.16, 1.10); letter(axes[1], "B", -.16, 1.10)
    save(fig, out / "Figure_S6_phylum_stratified_validation_n_gt10")


def build_s7(root20: Path, root16: Path, out: Path) -> None:
    no_hypo = read_csv(root20 / "results/primary_fi_topk/no_hypothetical_ADFU_metrics_each_run.csv")
    full_cls = read_csv(root16 / "results/lightgbm_common_from_run_winners_20260820/classifier_common_test_metrics_each_run.csv")
    full_reg = read_csv(root16 / "results/lightgbm_common_from_run_winners_20260820/regressor_common_test_metrics_each_run.csv")
    rows = []
    for _, r in full_cls.iterrows(): rows.append({"feature_set": "All products", "metric": "AUPRC", "run": r.run, "value": r.AUPRC})
    for _, r in full_reg.iterrows(): rows.append({"feature_set": "All products", "metric": "R²", "run": r.run, "value": r.R2})
    for _, r in no_hypo.iterrows(): rows.append({"feature_set": "No hypothetical\nproteins", "metric": "AUPRC" if r.task == "classification" else "R²", "run": r.run, "value": r.AUPRC if r.task == "classification" else r.R2})
    fig, ax = plt.subplots(figsize=(2.7371281715, 1.9207917760)); fig.subplots_adjust(left=.17, right=.96, bottom=.23, top=.82)
    grouped_panel(ax, pd.DataFrame(rows), "feature_set", "value", "metric", ["AUPRC", "R²"], [BLUE_DARK, CORAL_DARK], ["All products", "No hypothetical\nproteins"], {}, "Performance", (0, 1.03), legend=True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2)
    save(fig, out / "Figure_S7_hypothetical_feature_sensitivity")


def build_s10_feature_importance(root: Path, out: Path, task: str, metric: str, color: str) -> None:
    fi = read_csv(root / "results/primary_fi_topk/feature_importance_each_run.csv")
    topk = read_csv(root / "results/primary_fi_topk/topk_metrics_each_run.csv")
    comp = read_csv(root / "results/primary_fi_topk/topk_category_composition_each_run.csv")
    fi = fi[fi.task.eq(task)].copy(); fi["normalized_gain_pct"] = num(fi.normalized_gain_pct)
    order = fi.groupby(["Code", "Product"])["normalized_gain_pct"].mean().sort_values(ascending=False).head(20).index.tolist()
    labels = [(p if isinstance(p, str) and p.strip() else c) for c, p in order]
    labels = [v if len(v) <= 27 else v[:25] + "…" for v in labels]
    keys = [f"{c}|||{p}" for c, p in order]; fi["key"] = fi.Code.astype(str) + "|||" + fi.Product.fillna("").astype(str)
    fig = plt.figure(figsize=(6.2638888889, 2.6458333333))
    gs = fig.add_gridspec(2, 2, width_ratios=[.54, .46], height_ratios=[.48, .52], left=.31, right=.80, bottom=.14, top=.94, wspace=.38, hspace=.55)
    ax_fi, ax_top, ax_comp = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    q = fi[fi.key.isin(keys)]; means = q.groupby("key").normalized_gain_pct.mean().reindex(keys); sds = q.groupby("key").normalized_gain_pct.std(ddof=1).reindex(keys).fillna(0); y = np.arange(len(keys))[::-1]
    ax_fi.barh(y, means, color=PURPLE, edgecolor=INK, linewidth=OBJECT_LW, xerr=sds, error_kw={"ecolor": INK, "elinewidth": OBJECT_LW, "capsize": 1.5, "capthick": OBJECT_LW})
    for ri, run in enumerate(sorted(q.run.unique())):
        values = q[q.run.eq(run)].set_index("key").normalized_gain_pct.reindex(keys)
        ax_fi.scatter(values, y + (ri - 1) * .08, c="black", s=6, linewidths=0, zorder=4)
    ax_fi.set_yticks(y, labels); ax_fi.set_xlabel("Total feature importance (%)"); style(ax_fi); letter(ax_fi, "A", -.60, 1.04)
    tk = topk[topk.task.eq(task)].copy(); tk[metric] = num(tk[metric]); ks = sorted(tk.K.unique()); x = np.arange(len(ks)); means = tk.groupby("K")[metric].mean().reindex(ks); sds = tk.groupby("K")[metric].std(ddof=1).reindex(ks).fillna(0)
    ax_top.errorbar(x, means, yerr=sds, color=color, linewidth=OBJECT_LW, marker="o", markersize=3.5, markerfacecolor="white", markeredgewidth=OBJECT_LW, ecolor=INK, elinewidth=OBJECT_LW, capsize=1.5)
    for ri, run in enumerate(sorted(tk.run.unique())):
        values = tk[tk.run.eq(run)].set_index("K")[metric].reindex(ks)
        ax_top.scatter(x + (ri - 1) * .06, values, c="black", s=6, linewidths=0, zorder=4)
    shown = [k for k in ks if k in {5, 20, 100, 1000}]; shown_x = [ks.index(k) for k in shown]
    ax_top.set_xticks(shown_x, [str(k) for k in shown]); ax_top.set_ylabel("AUPRC" if metric == "AUPRC" else "R²"); ax_top.set_ylim(0, 1.02); style(ax_top); letter(ax_top, "B", -.28, 1.08)
    ct = comp[comp.task.eq(task)].copy(); cat = next(c for c in ["Category", "category", "KEGG_category"] if c in ct.columns); val = next(c for c in ["fraction_pct", "composition_pct", "percentage"] if c in ct.columns); ct[val] = num(ct[val])
    means_cat = ct.groupby(cat)[val].mean().sort_values(ascending=False); categories = [c for c in means_cat.index if str(c) != "Others"][:6]; ct["display"] = np.where(ct[cat].isin(categories), ct[cat], "Others"); categories += ["Others"]
    bottoms = np.zeros(len(ks)); palette = ["#F17C1E", "#FDBA72", "#2E77A8", "#A8C4DC", "#329338", "#90C482", "#9B6C5B"]
    category_labels = {
        "DNA repair and recombination proteins": "DNA repair / recombination",
        "Membrane trafficking": "Membrane\ntrafficking",
        "Non-coding RNAs": "Non-coding RNAs",
        "Translation factors": "Translation factors",
        "Viral proteins": "Viral proteins",
        "Transporters": "Transporters",
        "Ribosome": "Ribosome",
        "Enzymes": "Enzymes",
        "Others": "Others",
    }
    for category, fill in zip(categories, palette):
        values = ct[ct.display.eq(category)].groupby(["run", "K"])[val].sum().groupby("K").mean().reindex(ks).fillna(0).to_numpy()
        ax_comp.bar(x, values, bottom=bottoms, color=fill, edgecolor=INK, linewidth=OBJECT_LW, label=category_labels.get(str(category), str(category))); bottoms += values
    ax_comp.set_xticks(shown_x, [str(k) for k in shown]); ax_comp.set_xlabel("Top features (K)"); ax_comp.set_ylabel("Composition (%)"); ax_comp.set_ylim(0, 100); style(ax_comp)
    ax_comp.legend(frameon=False, bbox_to_anchor=(1.02, .5), loc="center left")
    label = "classification" if task == "classification" else "regression"
    save(fig, out / f"Figure_S10_{label}_feature_importance_topK")


def build_s9(root: Path, out: Path) -> None:
    data = read_csv(root / "results/plasflow_sensitivity/plasflow_regression_sensitivity_each_run.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.1005938320, 2.4059405074), gridspec_kw={"width_ratios": [.42, .58], "wspace": .48})
    fig.subplots_adjust(left=.12, right=.97, bottom=.22, top=.91)
    bottom = 0
    for label, value, fill in [("Plasmid", 61.2, CORAL_DARK), ("Unclassified", 23.9, GREY), ("Chromosome", 14.9, "#C6C0D8")]:
        axes[0].bar([0], [value], bottom=bottom, width=.48, color=fill, edgecolor="white", linewidth=OBJECT_LW, label=label); bottom += value
    axes[0].set_xticks([0], ["PlasFlow"]); axes[0].set_ylabel("Proportion of tRNA-bearing\nplasmid records (%)"); axes[0].set_ylim(0, 100); axes[0].legend(frameon=False, loc="upper left"); style(axes[0])
    order = ["complete_RefSeq_test", "PlasFlow_supported_plasmid_like", "Other_replicons_PlasFlow"]
    labels = {"complete_RefSeq_test": "Full baseline", "PlasFlow_supported_plasmid_like": "Plasmid\n(PlasFlow)", "Other_replicons_PlasFlow": "Other replicons\n(PlasFlow)"}
    bar_panel(axes[1], data, "subset", "R2", CORAL_DARK, order, labels, "R²", (0, 1.02))
    letter(axes[0], "A", -.30, 1.08); letter(axes[1], "B", -.22, 1.08)
    save(fig, out / "Figure_S9_PlasFlow_sensitivity")


def build_s16(root: Path, out: Path) -> None:
    data = read_csv(root / "results/central_dogma/central_dogma_metrics_each_run.csv"); data = data[data.analysis.eq("cog_lkj")]
    rows = []
    for _, r in data.iterrows(): rows += [{"target": r.target, "metric": "AUPRC", "run": r.run, "value": r.AUPRC}, {"target": r.target, "metric": "R²", "run": r.run, "value": r.R2}]
    labels = {"L": "L: Replication,\nrecombination and repair", "K": "K: Transcription", "J": "J: Translation, ribosomal\nstructure and biogenesis"}
    fig, ax = plt.subplots(figsize=(3.8750229659, 3.0099015748)); fig.subplots_adjust(left=.14, right=.97, bottom=.31, top=.82)
    grouped_panel(ax, pd.DataFrame(rows), "target", "value", "metric", ["AUPRC", "R²"], [BLUE, CORAL], ["L", "K", "J"], labels, "Performance", (0, 1.03), legend=True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2)
    save(fig, out / "Figure_S16_COG_L_K_J_predictability")


def build_fig4(root: Path, out: Path) -> None:
    data = read_csv(root / "results/central_dogma/central_dogma_metrics_each_run.csv"); modules = data[data.analysis.eq("cde_module")].copy()
    settings = {"all_non_target_features": "All non-target features", "non_central_dogma_background": "Non-central dogma background", "other_central_dogma_modules_only": "Other central dogma modules only"}
    modules["setting"] = modules.feature_setting.map(settings); groups = list(settings.values()); colors = [BLUE, CREAM, "#EAA99E"]
    fig = plt.figure(figsize=(6.2604166667, 4.10)); gs = fig.add_gridspec(2, 2, width_ratios=[.50, .50], left=.09, right=.97, bottom=.13, top=.94, wspace=.48, hspace=.68)
    ax_c, ax_d, ax_e = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    targets = ["Replication", "Transcription", "Translation"]
    grouped_panel(ax_c, modules, "target", "AUPRC", "setting", groups, colors, targets, {}, "AUPRC", (0, 1.03))
    grouped_panel(ax_d, modules, "target", "R2", "setting", groups, colors, targets, {}, "R²", (0, 1.03))
    fig.legend(handles=[Patch(facecolor=c, edgecolor=INK, linewidth=OBJECT_LW, label=g) for c, g in zip(colors, groups)], frameon=False, loc="center left", bbox_to_anchor=(.09, .535))
    sub = data[data.analysis.eq("cde_subcategory") & data.feature_setting.eq("all_non_target_features")]
    order = sub.target.drop_duplicates().tolist(); metrics = ["AUPRC", "R2", "test_positive_prevalence"]
    values = np.array([[num(sub[sub.target.eq(t)][m]).mean() for m in metrics] for t in order])
    labels = ["DNA repair /\nrecombination", "Machinery / elongation", "Maintenance / partition", "Replication initiator", "Core transcription\nmachinery", "Initiation / promoter\nrecognition", "Termination /\nanti-termination", "Transcription regulation", "Regulation / control", "rRNA", "Ribosomal proteins", "Ribosome assembly /\nmaturation", "aaRS", "tRNA"]
    if len(labels) != len(order): raise ValueError("Unexpected central-dogma subcategory count")
    cmaps = [LinearSegmentedColormap.from_list("cde_blue", ["#F4F4F4", BLUE]), LinearSegmentedColormap.from_list("cde_coral", ["#F4F4F4", CORAL]), LinearSegmentedColormap.from_list("cde_grey", ["#F4F4F4", "#D9D9D9"])]
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax_e.add_patch(Rectangle((col - .48, row - .48), .96, .96, facecolor=cmaps[col](np.clip(values[row, col], 0, 1)), edgecolor="white", linewidth=OBJECT_LW))
            ax_e.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center")
    ax_e.set_xlim(-.5, 2.5); ax_e.set_ylim(len(labels) - .5, -.5); ax_e.set_xticks([0, 1, 2], ["AUPRC", "R²", "Positive-class\nprevalence"]); ax_e.set_yticks(np.arange(len(labels)), labels); ax_e.tick_params(length=0)
    for spine in ax_e.spines.values(): spine.set_visible(False)
    letter(ax_c, "C", -.18, 1.08); letter(ax_d, "D", -.18, 1.08); letter(ax_e, "E", -.40, 1.02)
    save(fig, out / "Figure_4_CDE_machine_learning_panels_C_D_E")


def copy_s11(root: Path, out: Path) -> None:
    source = root / "figures/manuscript_style_redraw/Figure_S11_type_specific_feature_importance"
    if not source.exists(): raise FileNotFoundError(source)
    target = out / "Figure_S11_type_specific_feature_importance"; target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix.lower() in {".png", ".svg", ".csv"}: shutil.copy2(path, target / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root20", type=Path, required=True)
    parser.add_argument("--root16", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    build_s2(args.root20, args.root16, args.out)
    build_s3(args.root20, args.out)
    build_s4(args.root20, args.root16, args.out)
    build_s5(args.root20, args.out)
    build_s6(args.root20, args.out)
    build_s7(args.root20, args.root16, args.out)
    build_s10_feature_importance(args.root20, args.out, "classification", "AUPRC", BLUE_DARK)
    build_s10_feature_importance(args.root20, args.out, "regression", "R2", CORAL_DARK)
    build_s9(args.root20, args.out)
    build_s16(args.root20, args.out)
    build_fig4(args.root20, args.out)
    copy_s11(args.root20, args.out)


if __name__ == "__main__":
    main()

