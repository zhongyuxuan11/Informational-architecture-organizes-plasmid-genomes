#!/usr/bin/env python
"""Generate the taxonomy-tree and genus-by-tRNA heatmap for Figure 1C."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from _svg import esc, line, rect, text, write


AA_TYPES = [
    "Asn", "Ile", "Thr", "Ser", "Arg", "Leu", "Ala", "Met", "Tyr", "Gly", "Pro",
    "Val", "Lys", "Glu", "Asp", "Gln", "Cys", "Trp", "Phe", "His", "Sec",
]
TAXONOMY_LEVELS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]
UNKNOWN_GENUS_PATTERN = r"unknown|uncultured"


def build_retained_input(plasmids: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    tax = taxonomy.rename(columns={
        "kingdom": "Domain",
        "phylum": "Phylum",
        "class": "Class",
        "order": "Order",
        "family": "Family",
        "genus": "Genus",
    })
    missing_tax = ["GCF_ID", *TAXONOMY_LEVELS]
    missing = [column for column in missing_tax if column not in tax.columns]
    if missing:
        raise ValueError(f"Taxonomy table lacks columns: {missing}")
    missing_plasmid = ["GCF_ID", "Replicon_Acc", "Length", "Total_tRNA", *AA_TYPES]
    missing = [column for column in missing_plasmid if column not in plasmids.columns]
    if missing:
        raise ValueError(f"Plasmid tRNA table lacks columns: {missing}")
    positive = plasmids.loc[plasmids["Total_tRNA"].gt(0)].merge(
        tax[["GCF_ID", *TAXONOMY_LEVELS]], on="GCF_ID", how="inner", validate="many_to_one"
    )
    positive = positive.loc[
        positive["Genus"].notna()
        & positive["Genus"].astype(str).str.strip().ne("")
        & ~positive["Genus"].astype(str).str.contains(UNKNOWN_GENUS_PATTERN, case=False, regex=True)
    ].copy()
    genus_counts = positive.groupby("Genus").size()
    return positive.loc[positive["Genus"].isin(genus_counts[genus_counts.ge(5)].index)].copy()


def enrichment_table(retained: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genus in sorted(retained["Genus"].unique()):
        focal = retained["Genus"].eq(genus)
        for aa in AA_TYPES:
            present = retained[aa].gt(0)
            a = int((focal & present).sum())
            b = int((focal & ~present).sum())
            c = int((~focal & present).sum())
            d = int((~focal & ~present).sum())
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            focal_rate = (a + 0.5) / (a + b + 1)
            background_rate = (c + 0.5) / (c + d + 1)
            rows.append({
                "Genus": genus,
                "tRNA_type": aa,
                "n_in": a + b,
                "n_out": c + d,
                "trna_in": a,
                "trna_out": c,
                "odds_ratio": odds_ratio,
                "enrichment_ratio": focal_rate / background_rate,
                "log2_enrichment": np.log2(focal_rate / background_rate),
                "p_value": p_value,
            })
    results = pd.DataFrame(rows)
    results["p_bh"] = multipletests(results["p_value"], method="fdr_bh")[1]
    return results


def styled_text(x, y, value, anchor="middle", rotate=None, size=7, italic=False, baseline=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    style = ' font-style="italic"' if italic else ""
    baseline_style = f' dominant-baseline="{baseline}" alignment-baseline="{baseline}"' if baseline else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="Arial" font-size="{size}pt"{style}{baseline_style}{transform}>{esc(value)}</text>')


def coolwarm(value: float, limit: float) -> str:
    """Return the exact Matplotlib coolwarm colour for a centred value."""
    scaled = max(-1.0, min(1.0, value / limit)) if limit else 0.0
    rgb = colormaps["coolwarm"]((scaled + 1.0) / 2.0, bytes=True)[:3]
    return "rgb({},{},{})".format(*rgb)


def add_taxonomy_tree(body: list[str], taxonomy: pd.DataFrame, leaf_y: dict[str, float], x_by_level: dict[str, float]) -> None:
    node_y: dict[tuple[str, ...], float] = {}
    for level_index in range(len(TAXONOMY_LEVELS) - 1, -1, -1):
        level = TAXONOMY_LEVELS[level_index]
        prefix_columns = TAXONOMY_LEVELS[: level_index + 1]
        for prefix, group in taxonomy.groupby(prefix_columns, sort=False, dropna=False):
            key = prefix if isinstance(prefix, tuple) else (prefix,)
            if level == "Genus":
                y = leaf_y[str(key[-1])]
            else:
                child_level = TAXONOMY_LEVELS[level_index + 1]
                child_keys = []
                for child in group[child_level].drop_duplicates():
                    child_keys.append((*key, child))
                child_positions = [node_y[child_key] for child_key in child_keys]
                y = float(np.mean(child_positions))
                x = x_by_level[level]
                body.append(line(x, min(child_positions), x, max(child_positions), .7))
                for child_y in child_positions:
                    body.append(line(x, child_y, x_by_level[child_level], child_y, .7))
                if level == "Phylum" and str(key[-1]) != "Unknown":
                    body.append(styled_text(x - 3, y - 2, key[-1], anchor="end", rotate=-90, italic=False))
            node_y[key] = y


def plot_heatmap(results: pd.DataFrame, retained: pd.DataFrame, figure_path: Path) -> None:
    display_genera = results.loc[
        results["p_bh"].lt(0.05) & results["log2_enrichment"].abs().gt(1), "Genus"
    ].unique()
    if len(display_genera) == 0:
        raise ValueError("No genus passes the Figure 1C display threshold")

    tax = retained[TAXONOMY_LEVELS].drop_duplicates("Genus").copy()
    tax[TAXONOMY_LEVELS] = tax[TAXONOMY_LEVELS].fillna("Unknown").astype(str)
    tax = tax.loc[tax["Genus"].isin(display_genera)].sort_values(TAXONOMY_LEVELS).reset_index(drop=True)
    y_order = tax["Genus"].tolist()
    x_order = retained[AA_TYPES].gt(0).sum().sort_values(ascending=False).index.tolist()
    matrix = results.pivot(index="Genus", columns="tRNA_type", values="log2_enrichment").loc[y_order, x_order]
    significance = results.pivot(index="Genus", columns="tRNA_type", values="p_bh").loc[y_order, x_order]

    cell_w, cell_h = 13.0, 8.0
    tree_left, tree_width, heatmap_left = 45.0, 42.0, 91.5
    top, label_right, legend_width, bottom = 12.0, 76.0, 48.0, 30.0
    heatmap_width = cell_w * len(x_order)
    width = heatmap_left + heatmap_width + label_right + legend_width
    height = top + cell_h * len(y_order) + bottom
    body: list[str] = []
    limit = float(np.percentile(np.abs(matrix.to_numpy()), 98))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Heatmap colour limit must be positive and finite")

    leaf_y = {genus: top + (index + 0.5) * cell_h for index, genus in enumerate(y_order)}
    x_by_level = {
        level: tree_left + index * tree_width / (len(TAXONOMY_LEVELS) - 1)
        for index, level in enumerate(TAXONOMY_LEVELS)
    }
    x_by_level["Genus"] = heatmap_left
    add_taxonomy_tree(body, tax, leaf_y, x_by_level)

    for row_index, genus in enumerate(y_order):
        y = top + row_index * cell_h
        for column_index, aa in enumerate(x_order):
            x = heatmap_left + column_index * cell_w
            body.append(rect(x, y, cell_w, cell_h, coolwarm(float(matrix.loc[genus, aa]), limit), stroke="white", line_width=.7))
            p_value = float(significance.loc[genus, aa])
            marker = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            if marker:
                # Use a standard baseline with a font-metric offset for renderer-independent centering.
                body.append(styled_text(x + cell_w / 2, y + cell_h / 2 + 5.0, marker, size=8))
        body.append(styled_text(heatmap_left + heatmap_width + 4, y + cell_h * .72, genus, anchor="start", italic=True))

    label_y = top + cell_h * len(y_order) + 8
    for column_index, aa in enumerate(x_order):
        x = heatmap_left + (column_index + 0.5) * cell_w
        body.append(styled_text(x, label_y, aa, rotate=-45, italic=True))

    legend_x = heatmap_left + heatmap_width + label_right
    legend_top = top + 0.30 * cell_h * len(y_order)
    legend_h = 0.40 * cell_h * len(y_order)
    legend_steps = 80
    for index in range(legend_steps):
        fraction = index / (legend_steps - 1)
        value = limit * (1 - 2 * fraction)
        body.append(rect(legend_x, legend_top + fraction * legend_h, 8, legend_h / legend_steps + .2, coolwarm(value, limit), stroke="none"))
    body.extend([
        text(legend_x + 12, legend_top + 3, f"{limit:.1f}", anchor="start"),
        text(legend_x + 12, legend_top + legend_h / 2 + 2.5, "0", anchor="start"),
        text(legend_x + 12, legend_top + legend_h + 2.5, f"{-limit:.1f}", anchor="start"),
        text(legend_x + 31, legend_top + legend_h / 2, "log2 enrichment", rotate=-90),
    ])
    write(figure_path, width, height, body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plasmid-table", type=Path)
    parser.add_argument("--taxonomy-table", type=Path)
    parser.add_argument("--table-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plasmid_table is None or args.taxonomy_table is None:
        raise ValueError("--plasmid-table and --taxonomy-table are required")
    retained = build_retained_input(pd.read_csv(args.plasmid_table), pd.read_csv(args.taxonomy_table))
    if retained.empty:
        raise ValueError("No tRNA-positive plasmids remain after genus filtering")

    results = enrichment_table(retained)
    retained.to_csv(args.table_dir / "Fig1C_current_taxonomy_filtered_input.csv", index=False)
    results.to_csv(args.table_dir / "Fig1C_genus_trna_enrichment.csv", index=False)
    plot_heatmap(results, retained, args.figure_dir / "Fig1C_genus_trna_enrichment.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
