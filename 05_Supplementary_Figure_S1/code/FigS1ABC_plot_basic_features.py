#!/usr/bin/env python
"""Generate Figure S1A-C basic plasmid-feature comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

from _svg import box_chart


COLORS = ["#DDB08F", "#B8D0E0"]


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", required=True, type=Path)
    parser.add_argument("--coding-density-table", required=True, type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args(); args.table_dir.mkdir(parents=True, exist_ok=True); args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.feature_table)
    if len(data) != 61_961 or data["Replicon_Acc"].nunique() != 61_961:
        raise ValueError("Basic-feature table must contain 61,961 unique plasmids")
    density = pd.read_csv(args.coding_density_table, usecols=["Replicon_Acc", "Coding_Density"])
    if len(density) != 61_961 or density["Replicon_Acc"].nunique() != 61_961:
        raise ValueError("Coding-density table must contain 61,961 unique plasmids")
    data = data.drop(columns=["Coding_Density"], errors="ignore").merge(density, on="Replicon_Acc", how="left")
    if data["Coding_Density"].isna().any() or data["Coding_Density"].gt(100).any():
        raise ValueError("Merged coding-density values must be complete and <= 100%")
    data["tRNA_Group"] = data["Total_tRNA"].gt(0).map({True: "tRNA+", False: "tRNA-"})
    data.to_csv(args.table_dir / "FigS1ABC_basic_features.csv", index=False)
    definitions = [
        ("A", "log10_Size_bp", "Plasmid size (log10 bp)", [0, 2, 4, 6, 8], 8),
        ("B", "GC_Content", "GC content (%)", [0, 20, 40, 60, 80, 100], 100),
        ("C", "Coding_Density", "Coding density (%)", [0, 20, 40, 60, 80, 100], 100),
    ]
    stats_rows = []
    for panel, metric, ylabel, ticks, maximum in definitions:
        valid = data.dropna(subset=[metric])
        arrays = [valid.loc[valid["tRNA_Group"].eq(group), metric] for group in ["tRNA+", "tRNA-"]]
        statistic, p_value = mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        stats_rows.append({"panel": panel, "metric": metric, "tRNA_plus_n": len(arrays[0]),
                           "tRNA_minus_n": len(arrays[1]), "U": statistic, "p_mannwhitney": p_value})
        box_chart(args.figure_dir / f"FigS1{panel}_{metric}.svg",
                  ["tRNA+\nplasmid", "tRNA-\nplasmid"], arrays, COLORS, ylabel,
                  [(0, 1, stars(p_value))], width=215, y_ticks=ticks, y_min=0, y_max=maximum)
    pd.DataFrame(stats_rows).to_csv(args.table_dir / "FigS1ABC_basic_feature_stats.csv", index=False)
    return 0


if __name__ == "__main__": raise SystemExit(main())
