#!/usr/bin/env python
"""Generate strict downstream-only Fig. 3 CCI tables from a gene CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fig3_downstream_core as core


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PACKAGE_ROOT / "tables"
INPUTS_SMALL = PACKAGE_ROOT / "inputs_small"
TRIAD_GENE_CCI = PACKAGE_DIR / "01_gene_level_pooled_CCI_all_proteins_triad_genomes.csv"
COG_MAP = INPUTS_SMALL / "Fig3_COG_multilabel_noS_mapping_minimal.csv"
COG_PERFORMANCE = INPUTS_SMALL / "Fig2F_COG_AUPRC_R2_multilabel_noS.csv"
PLASMID_BIG_GENE_CCI = INPUTS_SMALL / "Fig3F_plasmid_big_gene_CCI.csv"
PHYLUM_BIG_GENE_SUMMARY = (
    INPUTS_SMALL / "Fig3F_phylum_plasmid_balanced_big_gene_CCI_plasmid_n_ge10.csv"
)

CHROM = core.CHROM
TRNA_POS = core.TRNA_POS
TRNA_NEG = core.TRNA_NEG
GROUP_ORDER = core.GROUP_ORDER
EXCLUDED_PRODUCT_NORM = None
GENE_COLUMNS = [
    "Genome_ID",
    "Replicon_ID",
    "Product",
    "Gene_group",
    "plasmid_size_group",
    "Length",
    "Product_norm",
    "CCI_gene",
]


def norm_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _require_nonblank(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.{column} must not be blank or missing")


def read_gene_input(path: Path, chunksize: int) -> dict[str, object]:
    """Read genes once, retaining numeric arrays and only downstream gene rows."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0, keep_default_na=False)
    _require_columns(header, GENE_COLUMNS, "gene CSV")

    arrays: dict[str, list[np.ndarray]] = {group: [] for group in GROUP_ORDER}
    selected_chunks: list[pd.DataFrame] = []
    invalid_cci_row_n = 0
    data_rows_seen = 0
    rows_seen = 0

    reader = pd.read_csv(
        path, usecols=GENE_COLUMNS, chunksize=chunksize, keep_default_na=False
    )
    try:
        chunks = reader
        for chunk in chunks:
            _require_nonblank(
                chunk,
                [
                    "Genome_ID",
                    "Replicon_ID",
                    "Product",
                    "Gene_group",
                    "Product_norm",
                ],
                "gene CSV",
            )
            raw_cci = chunk["CCI_gene"]
            raw_text = raw_cci.astype(str).str.strip()
            blank = raw_cci.isna().to_numpy() | raw_text.eq("").to_numpy()
            nonfinite_tokens = raw_text.str.lower().isin(
                {
                    "nan",
                    "+nan",
                    "-nan",
                    "inf",
                    "+inf",
                    "-inf",
                    "infinity",
                    "+infinity",
                    "-infinity",
                }
            ).to_numpy()
            numeric_cci = pd.to_numeric(raw_cci, errors="coerce")
            numeric_values = numeric_cci.to_numpy(dtype=float)
            non_numeric = np.isnan(numeric_values) & ~blank & ~nonfinite_tokens
            nonfinite = (
                nonfinite_tokens
                | (~np.isfinite(numeric_values) & ~np.isnan(numeric_values))
            )
            negative = np.isfinite(numeric_values) & (numeric_values < 0)
            invalid = blank | non_numeric | nonfinite | negative
            chunk_blank_n = int(blank.sum())
            chunk_non_numeric_n = int(non_numeric.sum())
            chunk_nonfinite_n = int(nonfinite.sum())
            chunk_negative_n = int(negative.sum())
            if invalid.any():
                first_offset = int(np.flatnonzero(invalid)[0])
                absolute_csv_row = data_rows_seen + first_offset + 2
                if blank[first_offset]:
                    invalid_type = "blank"
                elif non_numeric[first_offset]:
                    invalid_type = "non_numeric"
                elif nonfinite[first_offset]:
                    invalid_type = "nonfinite"
                else:
                    invalid_type = "negative"
                raise ValueError(
                    "invalid CCI_gene at absolute CSV row "
                    f"{absolute_csv_row}; type={invalid_type}; "
                    f"chunk_blank_n={chunk_blank_n}; "
                    f"chunk_non_numeric_n={chunk_non_numeric_n}; "
                    f"chunk_nonfinite_n={chunk_nonfinite_n}; "
                    f"chunk_negative_n={chunk_negative_n}"
                )
            chunk["CCI_gene"] = numeric_cci.astype(float)
            if (~chunk["Gene_group"].isin(GROUP_ORDER)).any():
                bad = sorted(chunk.loc[~chunk["Gene_group"].isin(GROUP_ORDER), "Gene_group"].unique())
                raise ValueError(f"gene CSV has unsupported Gene_group values: {bad}")
            plasmid_rows = chunk["Gene_group"].isin([TRNA_POS, TRNA_NEG])
            size_blank = (
                chunk["plasmid_size_group"].isna()
                | chunk["plasmid_size_group"].astype(str).str.strip().eq("")
            )
            if (plasmid_rows & size_blank).any():
                raise ValueError(
                    "plasmid rows require nonblank plasmid_size_group"
                )
            raw_length = chunk["Length"]
            length_blank = raw_length.isna() | raw_length.astype(str).str.strip().eq("")
            if (plasmid_rows & length_blank).any():
                raise ValueError("plasmid rows require nonblank Length")
            lengths = pd.to_numeric(raw_length, errors="coerce")
            nonblank_length = ~length_blank
            invalid_length = nonblank_length & (
                ~np.isfinite(lengths.to_numpy(dtype=float)) | lengths.le(0).to_numpy()
            )
            if invalid_length.any():
                raise ValueError(
                    "gene CSV.Length must be finite and positive when nonblank"
                )
            chunk["Length"] = lengths.astype(float)

            for group in GROUP_ORDER:
                values = chunk.loc[chunk["Gene_group"].eq(group), "CCI_gene"].to_numpy(dtype=float)
                if values.size:
                    arrays[group].append(values)

            normalized = chunk["Product_norm"].map(norm_text)
            selected = chunk.loc[
                chunk["Gene_group"].eq(TRNA_POS) & normalized.ne(""),
                ["Genome_ID", "Replicon_ID", "Product", "Product_norm", "CCI_gene"],
            ].copy()
            selected["Product_norm"] = normalized.loc[selected.index]
            if not selected.empty:
                selected_chunks.append(selected)

            rows_seen += len(chunk)
            data_rows_seen += len(chunk)
    finally:
        reader.close()

    if rows_seen == 0:
        raise ValueError("gene CSV contains no rows")
    compact_arrays = {
        group: np.concatenate(parts) if parts else np.array([], dtype=float)
        for group, parts in arrays.items()
    }
    if any(values.size == 0 for values in compact_arrays.values()):
        raise ValueError("gene CSV must contain every required Gene_group")
    selected_gene = (
        pd.concat(selected_chunks, ignore_index=True)
        if selected_chunks
        else pd.DataFrame(columns=["Genome_ID", "Replicon_ID", "Product", "Product_norm", "CCI_gene"])
    )
    if selected_gene.empty:
        raise ValueError("gene CSV has no tRNA+ plasmid genes with non-empty product")
    return {
        "arrays": compact_arrays,
        "selected_gene": selected_gene,
        "invalid_cci_row_n": invalid_cci_row_n,
    }


def filter_products_strict(summary: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if threshold < 0:
        raise ValueError("min-product-gene-n must be non-negative")
    _require_columns(summary, ["gene_n"], "product summary")
    return summary.loc[summary["gene_n"].gt(threshold)].copy()


def _distribution_stats(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        values = arrays[group]
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
        rows.append(
            {
                "Gene_group": group,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "whisker_low": float(inside.min()),
                "whisker_high": float(inside.max()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _histogram(arrays: dict[str, np.ndarray], bin_n: int = 1200) -> pd.DataFrame:
    lower = min(float(values.min()) for values in arrays.values())
    upper = max(float(values.max()) for values in arrays.values())
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, bin_n + 1)
    frames = []
    for group in GROUP_ORDER:
        counts, _ = np.histogram(arrays[group], bins=edges)
        frames.append(
            pd.DataFrame(
                {
                    "Gene_group": group,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "bin_center": (edges[:-1] + edges[1:]) / 2.0,
                    "count": counts,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _gene_group_stats(arrays: dict[str, np.ndarray], gene: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        values = arrays[group]
        subset = gene.loc[gene["Gene_group"].eq(group)]
        q1, q3 = np.quantile(values, [0.25, 0.75])
        rows.append(
            {
                "Gene_group": group,
                "gene_n": int(values.size),
                "genome_n": int(subset["Genome_ID"].nunique()),
                "CCI_mean": float(values.mean()),
                "CCI_median": float(np.median(values)),
                "q25": float(q1),
                "q75": float(q3),
                "frac_CCI_gt1": float((values > 1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_cog_map(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = ["COG_category", "Code", "Product", "COG_super_category"]
    frame = pd.read_csv(path, keep_default_na=False)
    _require_columns(frame, required, "COG map")
    _require_nonblank(frame, required, "COG map")
    frame["Product_norm"] = frame["Product"].map(norm_text)
    frame["Code"] = frame["Code"].astype(str).str.strip()
    frame["COG_category"] = frame["COG_category"].astype(str).str.strip().str.upper()
    frame["COG_super_category"] = frame["COG_super_category"].astype(str).str.strip()
    frame = frame.loc[frame["COG_category"].ne("S")]
    if frame.empty:
        raise ValueError("COG map has no non-S rows")
    return frame[["Product_norm", "Product", "Code", "COG_category", "COG_super_category"]].drop_duplicates().reset_index(drop=True)


def _load_cog_performance(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    required = ["COG", "AUPRC_test_mean", "AUPRC_test_sd", "R2_test_mean", "R2_test_sd"]
    _require_columns(frame, required, "COG performance")
    _require_nonblank(frame, ["COG"], "COG performance")
    frame["COG"] = frame["COG"].astype(str).str.strip().str.upper()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.all(np.isfinite(frame[column].to_numpy(dtype=float))):
            raise ValueError(f"COG performance.{column} must be finite")
    if frame["COG"].duplicated().any():
        raise ValueError("COG performance must contain one row per COG")
    return frame[required]


def merge_cog_performance_and_cci(
    performance: pd.DataFrame, cog_cci: pd.DataFrame
) -> pd.DataFrame:
    performance_categories = set(performance["COG"].astype(str))
    cci_categories = set(cog_cci["COG_category"].astype(str))
    if performance_categories != cci_categories:
        performance_only = sorted(performance_categories - cci_categories)
        cci_only = sorted(cci_categories - performance_categories)
        raise ValueError(
            f"COG category mismatch: performance_only={performance_only}; cci_only={cci_only}"
        )
    combined = cog_cci.merge(
        performance,
        left_on="COG_category",
        right_on="COG",
        how="inner",
        validate="one_to_one",
    )
    return combined.drop(columns="COG")


def _load_plasmid_big_gene_tables(
    plasmid_path: Path, summary_path: Path, minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plasmid = pd.read_csv(plasmid_path, keep_default_na=False)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    _require_columns(
        plasmid,
        ["phylum", "Genome_ID", "Replicon_ID", "plasmid_CCI_big_gene"],
        "plasmid big-gene CCI",
    )
    _require_columns(
        summary,
        ["phylum", "plasmid_n", "median_plasmid_CCI_big_gene"],
        "phylum plasmid big-gene CCI",
    )
    if np.any(pd.to_numeric(summary["plasmid_n"], errors="raise") < minimum):
        raise ValueError(f"phylum summary includes plasmid_n < {minimum}")
    return plasmid, summary


def _input_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triad-gene-cci", type=Path, default=TRIAD_GENE_CCI)
    parser.add_argument("--cog-map", type=Path, default=COG_MAP)
    parser.add_argument("--cog-performance", type=Path, default=COG_PERFORMANCE)
    parser.add_argument(
        "--plasmid-big-gene-cci",
        type=Path,
        default=PLASMID_BIG_GENE_CCI,
    )
    parser.add_argument(
        "--phylum-big-gene-summary",
        type=Path,
        default=PHYLUM_BIG_GENE_SUMMARY,
    )
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--top-product-n", type=int, default=10)
    parser.add_argument("--min-product-gene-n", type=int, default=50)
    parser.add_argument("--min-phylum-plasmid-n", type=int, default=10)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    args = parse_args()
    input_paths = {
        "triad_gene_cci": args.triad_gene_cci,
        "cog_map": args.cog_map,
        "cog_performance": args.cog_performance,
        "plasmid_big_gene_cci": args.plasmid_big_gene_cci,
        "phylum_big_gene_summary": args.phylum_big_gene_summary,
    }
    for path in input_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.top_product_n <= 0 or args.min_phylum_plasmid_n <= 0:
        raise ValueError("top-product-n and min-phylum-plasmid-n must be positive")

    data = read_gene_input(args.triad_gene_cci, args.chunksize)
    arrays = data["arrays"]
    gene = data["selected_gene"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, int] = {}

    def output_key(path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(args.out_dir.resolve()):
            return "package/" + resolved.relative_to(args.out_dir.resolve()).as_posix()
        raise ValueError(f"output path is outside declared roots: {resolved}")

    def write(frame: pd.DataFrame, path: Path) -> None:
        frame.to_csv(path, index=False)
        output_rows[output_key(path)] = int(len(frame))

    write(_gene_group_stats(arrays, gene), args.out_dir / "FigS11C_CDS_level_CCI_group_stats.csv")
    write(_histogram(arrays), args.out_dir / "Fig3B_all_gene_distribution_histogram.csv")
    write(_distribution_stats(arrays), args.out_dir / "Fig3B_all_gene_distribution_stats.csv")
    cds_global, cds_pairwise = core.unpaired_cds_tests(arrays)
    write(cds_global, args.out_dir / "FigS11C_CDS_Kruskal_Wallis.csv")
    write(cds_pairwise, args.out_dir / "FigS11C_CDS_pairwise_Mann_Whitney_BH.csv")

    write(gene, args.out_dir / "Fig3_gene_level_CCI_tRNA_positive_plasmid_genes_minimal.csv")
    product = core.summarize_products(gene)
    write(product, args.out_dir / "Fig3C_product_median_CCI_all.csv")
    product_gt = filter_products_strict(product, args.min_product_gene_n)
    product_precise_name = f"Fig3C_product_median_CCI_gene_n_gt{args.min_product_gene_n}.csv"
    write(product_gt, args.out_dir / product_precise_name)
    if args.min_product_gene_n == 200:
        write(product_gt, args.out_dir / "Fig3C_product_median_CCI_gene_n_ge200.csv")
    top_products = product_gt.head(args.top_product_n)["Product_norm"]
    top_gene = gene.loc[gene["Product_norm"].isin(top_products)]
    top_product_name = (
        f"Fig3C_top{args.top_product_n}_product_gene_CCI_distribution.csv"
    )
    write(top_gene, args.out_dir / top_product_name)

    cog = _load_cog_map(args.cog_map)
    write(cog, args.out_dir / "Fig3_COG_multilabel_noS_mapping_minimal.csv")
    cog_cci = core.aggregate_product_cci_by_cog(product, cog)
    write(cog_cci, args.out_dir / "Fig3D_COG_median_CCI_noS_multilabel.csv")
    performance = _load_cog_performance(args.cog_performance)
    combined = merge_cog_performance_and_cci(performance, cog_cci)
    combined["size_bin"] = combined["product_n"].map(core.product_count_bin)
    write(combined, args.out_dir / "Fig3D_COG_performance_vs_product_median_CCI.csv")

    plasmid, phylum = _load_plasmid_big_gene_tables(
        args.plasmid_big_gene_cci,
        args.phylum_big_gene_summary,
        args.min_phylum_plasmid_n,
    )
    write(plasmid, args.out_dir / "Fig3E_plasmid_level_CCI_for_phylum.csv")
    phylum_name = (
        "Fig3E_phylum_plasmid_level_CCI_"
        f"plasmid_n_ge{args.min_phylum_plasmid_n}.csv"
    )
    write(phylum, args.out_dir / phylum_name)

    metadata = {
        "scope": "downstream-only",
        "legacy_upstream_constraints": {
            "archived_CCI_filter": "old archived tables used CCI > 0; this downstream run retains all finite CCI >= 0 from the supplied input",
            "initiation_codon": "the supplied upstream CCI input may retain initiation codons; this script does not recalculate gene-level CCI",
        },
        "inputs": {name: _input_metadata(path) for name, path in input_paths.items()},
        "validation": {
            "validated_invalid_cci_row_n": data["invalid_cci_row_n"],
            "invalid_cci_policy": "fail_fast",
            "size_strata": ["Length < 100000", "Length >= 100000"],
        },
        "thresholds": {
            "min_product_gene_n": args.min_product_gene_n,
            "product_gene_n_rule": f"> {args.min_product_gene_n}",
            "min_phylum_plasmid_n": args.min_phylum_plasmid_n,
            "phylum_plasmid_n_rule": f">= {args.min_phylum_plasmid_n}",
            "top_product_n": args.top_product_n,
        },
        "statistics": {
            "CDS_global_test": "Kruskal-Wallis test treating every valid CDS as an independent observation",
            "CDS_pairwise_tests": "three two-sided Mann-Whitney U tests treating every valid CDS as independent, with Benjamini-Hochberg correction",
            "all_gene_histogram": "1200 common equal-width bins spanning the observed global minimum and maximum; every CDS contributes; no clipping",
            "box_statistics": "sample standard deviation and Tukey 1.5-IQR observed whiskers",
            "product_CCI": "gene-level median within normalized product",
            "COG_CCI": "median of product medians, not gene-frequency weighted",
            "COG_performance": "mean held-out classification AUPRC and regression R2 for models trained within each non-S COG category",
            "phylum_CCI": "one CCI calculated directly from codon counts pooled across all retained CDSs per tRNA-positive plasmid, using only that plasmid's tRNAs",
        },
        "output_row_counts": output_rows,
    }
    if args.min_product_gene_n == 200:
        metadata["legacy_filenames"] = {}
        metadata["legacy_filenames"]["Fig3C_product_median_CCI_gene_n_ge200.csv"] = (
            "legacy filename; content uses the strict greater-than threshold"
        )
    metadata_path = args.out_dir / "Fig3_downstream_tables_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
