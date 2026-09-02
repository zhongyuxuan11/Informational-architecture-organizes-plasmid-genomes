#!/usr/bin/env python
"""Generate Figure 1F/G and size-stratified Figure S1F/G mobility analyses."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from _svg import bar_chart, box_chart, grouped_bar, grouped_box_chart


ORDER = ["conjugative", "mobilizable", "non-mobilizable"]
LABELS = ["Conjugative", "Mobilizable", "Non-mobilizable"]
AXIS_LABELS = ["Conjugative", "Mobilizable", "Non-\nmobilizable"]
COLORS = ["#EF7869", "#85B4D3", "#9E9E9E"]


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def prevalence_stats(data: pd.DataFrame, stratum: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    tests = []
    for group in ORDER:
        subset = data[data["predicted_mobility"].eq(group)]
        summary.append({"stratum": stratum, "predicted_mobility": group, "n_plasmids": len(subset),
                        "tRNA_positive_n": int(subset["has_tRNA"].sum()),
                        "tRNA_positive_percent": 100 * subset["has_tRNA"].mean()})
    raw = []
    pairs = list(combinations(ORDER, 2))
    for first, second in pairs:
        subset = data[data["predicted_mobility"].isin([first, second])]
        table = pd.crosstab(subset["predicted_mobility"], subset["has_tRNA"]).reindex(index=[first, second], columns=[False, True], fill_value=0)
        odds_ratio, p_value = fisher_exact(table.to_numpy(), alternative="two-sided")
        raw.append(p_value)
        tests.append({"stratum": stratum, "group_1": first, "group_2": second,
                      "test": "two-sided Fisher exact", "odds_ratio": odds_ratio, "p_raw": p_value})
    adjusted = multipletests(raw, method="fdr_bh")[1]
    for row, p_value in zip(tests, adjusted):
        row["p_bh"] = p_value
    return pd.DataFrame(summary), pd.DataFrame(tests)


def abundance_stats(data: pd.DataFrame, stratum: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    carriers = data[data["has_tRNA"]].copy()
    arrays = [carriers.loc[carriers["predicted_mobility"].eq(group), "Total_tRNA"] for group in ORDER]
    statistic, p_kw = kruskal(*arrays)
    summary = []
    for group, values in zip(ORDER, arrays):
        summary.append({"stratum": stratum, "predicted_mobility": group, "n_tRNA_positive": len(values),
                        "median_Total_tRNA": values.median(), "mean_Total_tRNA": values.mean(),
                        "kruskal_statistic": statistic, "kruskal_p": p_kw})
    tests = []
    raw = []
    for first, second in combinations(ORDER, 2):
        first_values = carriers.loc[carriers["predicted_mobility"].eq(first), "Total_tRNA"]
        second_values = carriers.loc[carriers["predicted_mobility"].eq(second), "Total_tRNA"]
        statistic_u, p_value = mannwhitneyu(first_values, second_values, alternative="two-sided")
        raw.append(p_value)
        tests.append({"stratum": stratum, "group_1": first, "group_2": second,
                      "test": "two-sided Mann-Whitney U", "U": statistic_u, "p_raw": p_value})
    adjusted = multipletests(raw, method="fdr_bh")[1]
    for row, p_value in zip(tests, adjusted):
        row["p_bh"] = p_value
    return pd.DataFrame(summary), pd.DataFrame(tests)


def plot_prevalence(summary: pd.DataFrame, tests: pd.DataFrame, directory: Path, stem: str) -> None:
    values = summary.set_index("predicted_mobility").loc[ORDER, "tRNA_positive_percent"]
    comparisons = [(ORDER.index(row.group_1), ORDER.index(row.group_2), stars(row.p_bh)) for row in tests.itertuples(index=False)]
    bar_chart(directory / f"{stem}.svg", AXIS_LABELS, values.tolist(), COLORS, "tRNA-positive rate (%)", comparisons, width=250)


def plot_abundance(data: pd.DataFrame, tests: pd.DataFrame, directory: Path, stem: str) -> None:
    carriers = data[data["has_tRNA"]].copy()
    carriers["log10_tRNA_count"] = np.log10(1 + carriers["Total_tRNA"])
    arrays = [carriers.loc[carriers["predicted_mobility"].eq(group), "log10_tRNA_count"] for group in ORDER]
    comparisons = [(ORDER.index(row.group_1), ORDER.index(row.group_2), stars(row.p_bh)) for row in tests.itertuples(index=False)]
    box_chart(directory / f"{stem}.svg", AXIS_LABELS, arrays, COLORS,
              "tRNA count per plasmid\n(log10(1+count))", comparisons, width=250,
              y_ticks=[0, 1, 2], y_min=0, y_max=2.65)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility-table", required=True, type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.mobility_table)
    if len(data) != 61_961 or data["Plasmid_ID"].nunique() != 61_961:
        raise ValueError("Mobility table must contain exactly 61,961 unique target plasmids")
    missing = data["mobility_status"].eq("missing")
    if int(missing.sum()) != 38:
        raise ValueError(f"Expected 38 explicitly missing mobility results, found {int(missing.sum())}")
    data = data[~missing].copy()
    if not set(data["predicted_mobility"]).issubset(set(ORDER)):
        raise ValueError("Unexpected mobility class")
    data["has_tRNA"] = data["Total_tRNA"].gt(0)
    data["size_group"] = np.where(data["Length"].lt(100_000), "<100 kb", "≥100 kb")

    main_prev, main_prev_tests = prevalence_stats(data, "all")
    main_ab, main_ab_tests = abundance_stats(data, "all")
    main_prev.to_csv(args.table_dir / "Fig1F_mobility_trna_prevalence.csv", index=False)
    main_prev_tests.to_csv(args.table_dir / "Fig1F_mobility_trna_prevalence_tests.csv", index=False)
    carriers = data[data["has_tRNA"]].copy()
    carriers.to_csv(args.table_dir / "Fig1G_mobility_trna_abundance_per_plasmid.csv", index=False)
    main_ab.to_csv(args.table_dir / "Fig1G_mobility_trna_abundance_summary.csv", index=False)
    main_ab_tests.to_csv(args.table_dir / "Fig1G_mobility_trna_abundance_tests.csv", index=False)
    plot_prevalence(main_prev, main_prev_tests, args.figure_dir, "Fig1F_mobility_trna_prevalence")
    plot_abundance(data, main_ab_tests, args.figure_dir, "Fig1G_mobility_trna_abundance")

    prev_summaries, prev_tests, ab_summaries, ab_tests = [], [], [], []
    for size_group in ["<100 kb", "≥100 kb"]:
        subset = data[data["size_group"].eq(size_group)]
        summary_p, tests_p = prevalence_stats(subset, size_group)
        summary_a, tests_a = abundance_stats(subset, size_group)
        prev_summaries.append(summary_p); prev_tests.append(tests_p)
        ab_summaries.append(summary_a); ab_tests.append(tests_a)
    s1_prev = pd.concat(prev_summaries, ignore_index=True)
    s1_prev_tests = pd.concat(prev_tests, ignore_index=True)
    s1_ab = pd.concat(ab_summaries, ignore_index=True)
    s1_ab_tests = pd.concat(ab_tests, ignore_index=True)
    s1_prev.to_csv(args.table_dir / "FigS1F_mobility_trna_prevalence_by_size.csv", index=False)
    s1_prev_tests.to_csv(args.table_dir / "FigS1F_mobility_trna_prevalence_by_size_tests.csv", index=False)
    carriers.to_csv(args.table_dir / "FigS1G_mobility_trna_abundance_per_plasmid.csv", index=False)
    s1_ab.to_csv(args.table_dir / "FigS1G_mobility_trna_abundance_by_size_summary.csv", index=False)
    s1_ab_tests.to_csv(args.table_dir / "FigS1G_mobility_trna_abundance_by_size_tests.csv", index=False)

    matrix = [[float(s1_prev[(s1_prev["stratum"].eq(size)) & s1_prev["predicted_mobility"].eq(group)]["tRNA_positive_percent"].iloc[0])
               for group in ORDER] for size in ["<100 kb", "≥100 kb"]]
    grouped_bar(args.figure_dir / "FigS1F_mobility_trna_prevalence_by_size.svg",
                ["<100 kb", "≥100 kb"], LABELS, matrix, COLORS, "tRNA-positive rate (%)", width=280,
                y_ticks=[0, 10, 20, 30], y_max=38,
                series_comparisons=[
                    (size_index, ORDER.index(row.group_1), ORDER.index(row.group_2), stars(row.p_bh))
                    for size_index,size in enumerate(["<100 kb", "≥100 kb"])
                    for row in s1_prev_tests[s1_prev_tests["stratum"].eq(size)].itertuples(index=False)
                ])
    arrays=[]; labels=[]; colors=[]
    for size in ["<100 kb", "≥100 kb"]:
        for group,label,color in zip(ORDER,LABELS,COLORS):
            arrays.append(np.log10(1 + data.loc[data["size_group"].eq(size) & data["predicted_mobility"].eq(group) & data["has_tRNA"], "Total_tRNA"]))
            labels.append(f"{size} {label}"); colors.append(color)
    comparisons=[]
    for size_index,size in enumerate(["<100 kb", "≥100 kb"]):
        tests_size=s1_ab_tests[s1_ab_tests["stratum"].eq(size)]
        for row in tests_size.itertuples(index=False):
            comparisons.append((size_index,ORDER.index(row.group_1),ORDER.index(row.group_2),stars(row.p_bh)))
    grouped_box_chart(args.figure_dir / "FigS1G_mobility_trna_abundance_by_size.svg",
              ["<100 kb", "≥100 kb"], LABELS, arrays, colors,
              "tRNA count per plasmid\n(log10(1+count))", comparisons, width=320,
              y_ticks=[0, 1, 2], y_min=0, y_max=2.65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
