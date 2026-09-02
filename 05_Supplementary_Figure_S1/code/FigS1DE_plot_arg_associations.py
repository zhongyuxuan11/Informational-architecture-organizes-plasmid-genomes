#!/usr/bin/env python
"""Generate Figure 1D/E and Figure S1D/E ARG tables and panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from _svg import bar_chart, box_chart, grouped_bar, grouped_box_chart


COLORS = {"tRNA+": "#DDB08F", "tRNA-": "#B8D0E0"}


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def box_summary(values: pd.Series) -> dict[str, float]:
    array = values.dropna().to_numpy(dtype=float)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    iqr = q3 - q1
    return {
        "n": len(array), "q1": q1, "median": median, "q3": q3,
        "whisker_low": array[array >= q1 - 1.5 * iqr].min(),
        "whisker_high": array[array <= q3 + 1.5 * iqr].max(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arg-table", required=True, type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.arg_table)
    required = {"Plasmid_ID", "Size_bp", "Total_tRNA", "Total_ARG_Count", "Resistant_Class_Count", "Is_MDR", "ARG_data_source"}
    if missing := required.difference(data.columns):
        raise ValueError(f"ARG table missing columns: {sorted(missing)}")
    if len(data) != 61_961 or data["Plasmid_ID"].nunique() != 61_961:
        raise ValueError("ARG table must contain exactly 61,961 unique plasmids")
    data = data[data["ARG_data_source"].ne("no_ARG_data_available")].copy()
    if len(data) != 61_958:
        raise ValueError("Expected 61,958 plasmids with ARG data")
    data["tRNA_Group"] = data["Total_tRNA"].gt(0).map({True: "tRNA+", False: "tRNA-"})
    data["ARG_positive"] = data["Total_ARG_Count"].gt(0)
    data["size_group"] = np.where(data["Size_bp"].lt(100_000), "<100 kb", "≥100 kb")
    data["Is_MDR"] = data["Resistant_Class_Count"].ge(3)

    all_card = data.copy()
    all_card["log10_1p_ARG_count"] = np.log10(1 + all_card["Total_ARG_Count"])
    positive = all_card[all_card["ARG_positive"]].copy()
    positive[["Plasmid_ID", "Size_bp", "size_group", "tRNA_Group", "Total_ARG_Count", "log10_1p_ARG_count"]].to_csv(
        args.table_dir / "Fig1D_arg_count_arg_positive.csv", index=False
    )
    raw_groups = [positive.loc[positive["tRNA_Group"].eq(group), "Total_ARG_Count"] for group in ["tRNA+", "tRNA-"]]
    display_groups = [positive.loc[positive["tRNA_Group"].eq(group), "log10_1p_ARG_count"] for group in ["tRNA+", "tRNA-"]]
    statistic_arg, p_arg = mannwhitneyu(raw_groups[0], raw_groups[1], alternative="two-sided")
    main_d_stats = []
    for group, values in zip(["tRNA+", "tRNA-"], display_groups):
        main_d_stats.append({"tRNA_Group": group, **box_summary(values), "mannwhitney_U": statistic_arg,
                             "p_raw": p_arg, "display_transform": "log10(1 + Total_ARG_Count)"})
    pd.DataFrame(main_d_stats).to_csv(args.table_dir / "Fig1D_arg_count_arg_positive_summary_tests.csv", index=False)
    box_chart(args.figure_dir / "Fig1D_arg_count_arg_positive.svg",
              ["tRNA+\nplasmid", "tRNA-\nplasmid"], display_groups, [COLORS["tRNA+"], COLORS["tRNA-"]],
              "ARG count per plasmid\n(log10(1+count))", [(0, 1, stars(p_arg))],
              width=215, y_ticks=[0, 1, 2, 3], y_min=0, y_max=3.35)
    richness_rows = []
    p_values = []
    for threshold in range(1, 6):
        event = data["Resistant_Class_Count"].ge(threshold)
        table = pd.crosstab(data["tRNA_Group"], event).reindex(index=["tRNA+", "tRNA-"], columns=[False, True], fill_value=0)
        odds_ratio, p_value = fisher_exact(table.to_numpy(), alternative="two-sided")
        p_values.append(p_value)
        for group in ["tRNA+", "tRNA-"]:
            subset = data[data["tRNA_Group"].eq(group)]
            count = int(subset["Resistant_Class_Count"].ge(threshold).sum())
            richness_rows.append({"threshold": threshold, "tRNA_Group": group, "n": len(subset),
                                  "positive_n": count, "prevalence_percent": 100 * count / len(subset),
                                  "odds_ratio": odds_ratio, "fisher_p": p_value})
    adjusted = multipletests(p_values, method="fdr_bh")[1]
    richness = pd.DataFrame(richness_rows)
    richness["fisher_p_bh"] = richness["threshold"].map(dict(zip(range(1, 6), adjusted)))
    richness.to_csv(args.table_dir / "Fig1E_resistance_class_richness.csv", index=False)
    matrix = [[float(richness[(richness["threshold"].eq(t)) & richness["tRNA_Group"].eq(g)]["prevalence_percent"].iloc[0])
               for g in ["tRNA+", "tRNA-"]] for t in range(1, 6)]
    grouped_bar(args.figure_dir / "Fig1E_resistance_class_richness.svg",
                [f"≥ {value}" for value in range(1, 6)], ["tRNA+", "tRNA-"], matrix,
                [COLORS["tRNA+"], COLORS["tRNA-"]], "Prevalence of plasmids (%)",
                [(index, stars(p)) for index, p in enumerate(adjusted)], width=255,
                y_ticks=[0, 5, 10, 15, 20, 25, 30], y_max=33, series_comparisons=(), group_span=2/3,
                xlabel="Number of ARG classes")

    s1d_tests = []
    s1d_summaries = []
    s1d_arrays = []
    for size_group in ["<100 kb", "≥100 kb"]:
        subset = all_card[all_card["ARG_positive"] & all_card["size_group"].eq(size_group)]
        raw = [subset.loc[subset["tRNA_Group"].eq(group), "Total_ARG_Count"] for group in ["tRNA+", "tRNA-"]]
        display = [subset.loc[subset["tRNA_Group"].eq(group), "log10_1p_ARG_count"] for group in ["tRNA+", "tRNA-"]]
        statistic, p_value = mannwhitneyu(raw[0], raw[1], alternative="two-sided")
        s1d_tests.append({"size_group": size_group, "group_1": "tRNA+", "group_2": "tRNA-",
                          "test": "two-sided Mann-Whitney U on raw ARG counts", "U": statistic, "p_raw": p_value})
        for group, values in zip(["tRNA+", "tRNA-"], display):
            s1d_summaries.append({"size_group": size_group, "tRNA_Group": group, **box_summary(values),
                                  "display_transform": "log10(1 + Total_ARG_Count)"})
        s1d_arrays.extend(display)
    s1d_adjusted = multipletests([row["p_raw"] for row in s1d_tests], method="fdr_bh")[1]
    for row, q_value in zip(s1d_tests, s1d_adjusted):
        row["p_bh"] = q_value
    positive[["Plasmid_ID", "Size_bp", "size_group", "tRNA_Group", "Total_ARG_Count", "log10_1p_ARG_count"]].to_csv(
        args.table_dir / "FigS1D_arg_abundance_by_size_raw.csv", index=False
    )
    pd.DataFrame(s1d_summaries).to_csv(args.table_dir / "FigS1D_arg_abundance_by_size_summary.csv", index=False)
    pd.DataFrame(s1d_tests).to_csv(args.table_dir / "FigS1D_arg_abundance_by_size_tests.csv", index=False)
    grouped_box_chart(args.figure_dir / "FigS1D_arg_abundance_by_size.svg",
                      ["<100 kb", "≥100 kb"], ["tRNA+", "tRNA-"], s1d_arrays,
                      [COLORS["tRNA+"], COLORS["tRNA-"], COLORS["tRNA+"], COLORS["tRNA-"]],
                      "ARG count per plasmid\n(log10(1+count))",
                      [(index, 0, 1, stars(q_value)) for index, q_value in enumerate(s1d_adjusted)],
                      width=255, y_ticks=[0, 1, 2, 3], y_min=0, y_max=3.35)

    mdr_rows = []
    mdr_p = []
    for size_group in ["<100 kb", "≥100 kb"]:
        subset = data[data["size_group"].eq(size_group)].copy()
        table = pd.crosstab(subset["tRNA_Group"], subset["Is_MDR"]).reindex(index=["tRNA+", "tRNA-"], columns=[False, True], fill_value=0)
        odds_ratio, p_value = fisher_exact(table.to_numpy(), alternative="two-sided")
        mdr_p.append(p_value)
        for group in ["tRNA+", "tRNA-"]:
            group_data = subset[subset["tRNA_Group"].eq(group)]
            mdr_rows.append({"size_group": size_group, "tRNA_Group": group, "n_with_ARG_data": len(group_data),
                             "MDR_n": int(group_data["Is_MDR"].sum()), "MDR_percent": 100 * group_data["Is_MDR"].mean(),
                             "odds_ratio": odds_ratio, "fisher_p": p_value})
    mdr_adjusted = multipletests(mdr_p, method="fdr_bh")[1]
    mdr = pd.DataFrame(mdr_rows)
    mdr["fisher_p_bh"] = mdr["size_group"].map(dict(zip(["<100 kb", "≥100 kb"], mdr_adjusted)))
    mdr.to_csv(args.table_dir / "FigS1E_mdr_prevalence_by_size.csv", index=False)
    matrix = [[float(mdr[(mdr["size_group"].eq(size)) & mdr["tRNA_Group"].eq(group)]["MDR_percent"].iloc[0])
               for group in ["tRNA+", "tRNA-"]] for size in ["<100 kb", "≥100 kb"]]
    grouped_bar(args.figure_dir / "FigS1E_mdr_prevalence_by_size.svg",
                ["<100 kb", "≥100 kb"], ["tRNA+", "tRNA-"], matrix,
                [COLORS["tRNA+"], COLORS["tRNA-"]], "MDR prevalence (≥3 ARG classes, %)",
                [(index, stars(p)) for index, p in enumerate(mdr_adjusted)], width=240,
                y_ticks=[0, 10, 20, 30, 40], y_max=46)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
