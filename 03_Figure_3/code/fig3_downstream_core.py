"""Pure downstream statistics for the Fig. 3 CCI workflow."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


CHROM = "Chromosome genes"
TRNA_POS = "tRNA+ plasmid genes"
TRNA_NEG = "tRNA- plasmid genes"
GROUP_ORDER = [CHROM, TRNA_POS, TRNA_NEG]


def _require_columns(frame, required, frame_name):
    """Reject tabular inputs that do not provide the declared interface."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing)}")


def _validated_numeric_column(frame, column, frame_name):
    """Return a numeric column after rejecting non-finite values."""
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{frame_name}.{column} must be numeric and finite") from error
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"{frame_name}.{column} must be numeric and finite")
    return values


def _require_nonblank(frame, columns, frame_name):
    """Reject missing or whitespace-only identifiers."""
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{frame_name}.{column} must not be blank or missing")


def _require_unique_super_category(frame, frame_name):
    """Require each COG category to map to exactly one super-category."""
    super_counts = frame.groupby("COG_category")["COG_super_category"].nunique()
    if super_counts.gt(1).any():
        raise ValueError(
            f"{frame_name} maps a COG_category to multiple COG_super_category values"
        )


def bh_adjust(p_values):
    """Adjust p-values with the Benjamini-Hochberg step-up procedure."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and within [0, 1]")
    if values.size == 0:
        return values.copy()

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scale = values.size / np.arange(1, values.size + 1, dtype=float)
    adjusted_ranked = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def significance_label(q_value):
    """Map an adjusted p-value to the fixed manuscript label set."""
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    return "ns"


def unpaired_cds_tests(arrays):
    """Run CDS-level tests with every CDS treated as an independent observation."""
    if set(arrays) != set(GROUP_ORDER):
        raise ValueError("arrays must contain exactly the three Gene_group values")
    validated = {}
    for group in GROUP_ORDER:
        values = np.asarray(arrays[group], dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{group} must contain at least two finite CDS-level CCI values")
        validated[group] = values

    kruskal = stats.kruskal(*(validated[group] for group in GROUP_ORDER))
    global_p = float(kruskal.pvalue)
    global_test = pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis",
                "CDS_n": int(sum(values.size for values in validated.values())),
                "statistic": float(kruskal.statistic),
                "p": global_p,
                "q": global_p,
                "label": significance_label(global_p),
            }
        ]
    )

    rows = []
    for group_1, group_2 in combinations(GROUP_ORDER, 2):
        result = stats.mannwhitneyu(
            validated[group_1],
            validated[group_2],
            alternative="two-sided",
            method="asymptotic",
        )
        rows.append(
            {
                "test": "Mann-Whitney U",
                "group_1": group_1,
                "group_2": group_2,
                "CDS_n_1": int(validated[group_1].size),
                "CDS_n_2": int(validated[group_2].size),
                "statistic": float(result.statistic),
                "p": float(result.pvalue),
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["q"] = bh_adjust(pairwise["p"])
    pairwise["label"] = pairwise["q"].map(significance_label)
    return global_test, pairwise


def summarize_products(genes):
    """Summarize gene-level CCI values once per normalized product."""
    required = ["Product_norm", "Product", "CCI_gene"]
    _require_columns(genes, required, "genes")

    validated = genes.loc[:, required].copy()
    validated["CCI_gene"] = _validated_numeric_column(validated, "CCI_gene", "genes")

    def product_mode(values):
        return values.mode(dropna=False).iloc[0]

    summary = (
        validated.groupby("Product_norm", as_index=False, dropna=False)
        .agg(
            Product=("Product", product_mode),
            gene_n=("CCI_gene", "count"),
            CCI_median=("CCI_gene", "median"),
        )
        .sort_values(
            ["CCI_median", "gene_n", "Product_norm"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return summary[["Product", "Product_norm", "gene_n", "CCI_median"]]


def aggregate_product_cci_by_cog(product_summary, product_cog):
    """Aggregate product medians by COG without gene-frequency weighting."""
    _require_columns(
        product_summary, ["Product_norm", "CCI_median"], "product_summary"
    )
    mapping_columns = ["Product_norm", "COG_category", "COG_super_category"]
    _require_columns(product_cog, mapping_columns, "product_cog")
    _require_nonblank(product_summary, ["Product_norm"], "product_summary")
    _require_nonblank(product_cog, mapping_columns, "product_cog")
    _require_unique_super_category(product_cog, "product_cog")
    if not product_summary["Product_norm"].is_unique:
        raise ValueError("product_summary must contain one row per Product_norm")

    validated = product_summary.loc[:, ["Product_norm", "CCI_median"]].copy()
    validated["CCI_median"] = _validated_numeric_column(
        validated, "CCI_median", "product_summary"
    )

    non_s = product_cog.loc[
        product_cog["COG_category"].ne("S"), mapping_columns
    ].drop_duplicates()
    if non_s.empty:
        raise ValueError("product_cog has no non-S category mapping")
    mapped = validated.merge(
        non_s[mapping_columns].drop_duplicates(),
        on="Product_norm",
        how="inner",
        validate="one_to_many",
    )
    if mapped.empty:
        raise ValueError("product_summary has no matching non-S product_cog rows")
    result = (
        mapped.groupby(["COG_category", "COG_super_category"], as_index=False)
        .agg(product_n=("Product_norm", "nunique"), CCI_median=("CCI_median", "median"))
        .sort_values(["COG_category", "COG_super_category"], kind="stable")
        .reset_index(drop=True)
    )
    return result[
        ["COG_category", "COG_super_category", "product_n", "CCI_median"]
    ]


def product_count_bin(product_n):
    """Assign the fixed point-size bin from a product count."""
    if isinstance(product_n, (bool, np.bool_)):
        raise ValueError("product_n must be a finite, non-negative integer")
    try:
        count = float(product_n)
    except (TypeError, ValueError) as error:
        raise ValueError("product_n must be a finite, non-negative integer") from error
    if not np.isfinite(count) or count < 0.0 or not count.is_integer():
        raise ValueError("product_n must be a finite, non-negative integer")
    if count < 100:
        return "<100"
    if count <= 1000:
        return "100–1000"
    return ">1000"
