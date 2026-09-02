#!/usr/bin/env python
"""Recompute Fig3B pooled-supply CDS-level CCI.

The comparison is restricted to genomes that have all three comparison
contexts: chromosome, tRNA-positive plasmid, and tRNA-negative plasmid.
Size-stratified comparisons additionally require both tRNA-positive and
tRNA-negative plasmids in the same size stratum.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODON_COUNTS = PACKAGE_ROOT / "inputs_large" / "3_CDS_64_Codon_Counts.csv"
REPLICON_INFO = PACKAGE_ROOT / "inputs_small" / "1_Replicon_Metadata.csv"
PLASMID_SIZE = PACKAGE_ROOT / "inputs_small" / "01_plasmid_trna_61961.csv"
TRNA_SUPPLY = PACKAGE_ROOT / "inputs_small" / "trna_supply_table_rebuilt_from_all.csv"
DEFAULT_OUT = PACKAGE_ROOT / "recomputed_fig3b_cds_level_cci"

CHROM = "Chromosome genes"
TRNA_POS = "tRNA+ plasmid genes"
TRNA_NEG = "tRNA- plasmid genes"
GROUP_ORDER = [CHROM, TRNA_POS, TRNA_NEG]
GROUP_LABEL = {
    CHROM: "Chromosomal genes",
    TRNA_POS: "Genes on tRNA-positive plasmids",
    TRNA_NEG: "Genes on tRNA-negative plasmids",
}
GROUP_COLOR = {CHROM: "#E3E3E3", TRNA_POS: "#94C0DF", TRNA_NEG: "#F8E0D7"}

STRATUM_ALL = "all_triad_genomes"
SIZE_ORDER = ["small_lt100kb", "large_gt100kb"]
SIZE_LABEL = {
    STRATUM_ALL: "All triad genomes",
    "small_lt100kb": "<100 kb",
    "large_gt100kb": ">100 kb",
}

FONT_PT = 7
LINE_PT = 0.75
EXCLUDED_PRODUCT_NORM = None
RNG_SEED = 20260616

SYNONYMOUS_CODONS = {
    "F": ["TTT", "TTC"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "I": ["ATT", "ATC", "ATA"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "Y": ["TAT", "TAC"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["TGT", "TGC"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
}
CODONS = [codon for codons in SYNONYMOUS_CODONS.values() for codon in codons]
CODON_TO_IDX = {codon: i for i, codon in enumerate(CODONS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--codon-counts", default=str(CODON_COUNTS))
    p.add_argument("--replicon-info", default=str(REPLICON_INFO))
    p.add_argument("--plasmid-size", default=str(PLASMID_SIZE))
    p.add_argument("--trna-supply", default=str(TRNA_SUPPLY))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--mirror-dir", default=None)
    p.add_argument("--chunksize", type=int, default=80_000)
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--plot-sample-per-group", type=int, default=5000)
    return p.parse_args()


def norm_product(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "qc": out_dir / "qc",
        "scripts": out_dir / "scripts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_plasmid_meta(size_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(size_path, usecols=["GCF_ID", "Replicon_Acc", "Length", "Total_tRNA"])
    meta = meta.rename(columns={"GCF_ID": "Genome_ID", "Replicon_Acc": "Replicon_ID"})
    for col in ["Genome_ID", "Replicon_ID"]:
        meta[col] = meta[col].astype(str).str.strip()
    meta["Length"] = pd.to_numeric(meta["Length"], errors="coerce")
    meta["Total_tRNA"] = pd.to_numeric(meta["Total_tRNA"], errors="coerce").fillna(0)
    meta["has_plasmid_tRNA"] = meta["Total_tRNA"] > 0
    meta["plasmid_size_group"] = np.where(meta["Length"] < 100000, "small_lt100kb", "large_gt100kb")
    meta["replicon_key"] = meta["Genome_ID"] + "|" + meta["Replicon_ID"]
    return meta.drop_duplicates("replicon_key")


def load_replicon_info(path: Path, plasmid_meta: pd.DataFrame) -> pd.DataFrame:
    rep = pd.read_csv(path)
    required = {"Genome_ID", "Replicon_ID"}
    missing = required.difference(rep.columns)
    if missing:
        raise ValueError(f"Replicon metadata is missing required columns: {sorted(missing)}")

    if "Replicon_Type" not in rep.columns:
        if "Is_Plasmid" not in rep.columns:
            raise ValueError("Replicon metadata requires Replicon_Type or Is_Plasmid")
        is_plasmid = rep["Is_Plasmid"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        rep["Replicon_Type"] = np.where(is_plasmid, "Plasmid", "Chromosome")
    if "Total_CDS" not in rep.columns:
        if "CDS_Count" not in rep.columns:
            raise ValueError("Replicon metadata requires Total_CDS or CDS_Count")
        rep["Total_CDS"] = pd.to_numeric(rep["CDS_Count"], errors="coerce")

    rep = rep[["Genome_ID", "Replicon_ID", "Replicon_Type", "Total_CDS"]].copy()
    for col in ["Genome_ID", "Replicon_ID", "Replicon_Type"]:
        rep[col] = rep[col].astype(str).str.strip()
    rep["replicon_key"] = rep["Genome_ID"] + "|" + rep["Replicon_ID"]
    rep = rep.merge(
        plasmid_meta[["replicon_key", "Length", "has_plasmid_tRNA", "plasmid_size_group"]],
        on="replicon_key",
        how="left",
    )
    is_chrom = rep["Replicon_Type"].str.lower().eq("chromosome")
    is_plasmid = rep["Replicon_Type"].str.lower().eq("plasmid")
    has_plasmid_trna = rep["has_plasmid_tRNA"].eq(True)
    rep["Gene_group"] = np.select(
        [is_chrom, is_plasmid & has_plasmid_trna, is_plasmid & ~has_plasmid_trna],
        [CHROM, TRNA_POS, TRNA_NEG],
        default="Other",
    )
    return rep


def valid_cci_mask(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.isfinite(arr) & (arr >= 0.0)


def load_pooled_supply(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, list[int]]]]:
    supply = pd.read_csv(path)
    genome_col = "Genome_ID"
    codon_col = "tRNA_codons_set"
    for col in [genome_col, codon_col]:
        supply[col] = supply[col].astype(str).str.strip()

    by_genome: dict[str, set[str]] = defaultdict(set)
    valid = set(CODONS)
    for _, row in supply.iterrows():
        genome = row[genome_col]
        for codon in str(row[codon_col]).split(","):
            codon = codon.strip().upper()
            if codon in valid:
                by_genome[genome].add(codon)

    by_genome_aa: dict[str, dict[str, list[int]]] = {}
    for genome, codons in by_genome.items():
        aa_map: dict[str, list[int]] = {}
        for aa, family in SYNONYMOUS_CODONS.items():
            idx = [CODON_TO_IDX[c] for c in family if c in codons]
            if idx:
                aa_map[aa] = idx
        if aa_map:
            by_genome_aa[genome] = aa_map
    return by_genome, by_genome_aa


def genome_sets(rep: pd.DataFrame) -> tuple[set[str], dict[str, set[str]], pd.DataFrame]:
    has_chr = set(rep.loc[rep["Gene_group"].eq(CHROM), "Genome_ID"])
    has_pos = set(rep.loc[rep["Gene_group"].eq(TRNA_POS), "Genome_ID"])
    has_neg = set(rep.loc[rep["Gene_group"].eq(TRNA_NEG), "Genome_ID"])
    all_triad = has_chr & has_pos & has_neg

    size_eligible: dict[str, set[str]] = {}
    for sg in SIZE_ORDER:
        p = rep[rep["plasmid_size_group"].eq(sg)]
        size_eligible[sg] = (
            has_chr
            & set(p.loc[p["Gene_group"].eq(TRNA_POS), "Genome_ID"])
            & set(p.loc[p["Gene_group"].eq(TRNA_NEG), "Genome_ID"])
        )

    qc_rows = []
    for label, genomes in [(STRATUM_ALL, all_triad), *size_eligible.items()]:
        qc_rows.append(
            {
                "stratum": label,
                "genome_n": len(genomes),
                "definition": "chromosome + tRNA-positive plasmid + tRNA-negative plasmid present",
            }
        )
    return all_triad, size_eligible, pd.DataFrame(qc_rows)


def compute_chunk_cci(chunk: pd.DataFrame, supply_by_genome_aa: dict[str, dict[str, list[int]]]) -> pd.DataFrame:
    counts = chunk[CODONS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    cci = np.full(len(chunk), np.nan, dtype=float)
    raw_s = np.zeros(len(chunk), dtype=float)
    raw_z = np.zeros(len(chunk), dtype=float)
    row_index = np.arange(len(chunk))

    for genome, local_idx in chunk.groupby("Genome_ID", sort=False).indices.items():
        aa_map = supply_by_genome_aa.get(str(genome))
        if not aa_map:
            continue
        local_idx_arr = row_index[list(local_idx)]
        sub_counts = counts[local_idx_arr, :]
        score_sum = np.zeros(len(local_idx_arr), dtype=float)
        score_n = np.zeros(len(local_idx_arr), dtype=float)
        s_sum = np.zeros(len(local_idx_arr), dtype=float)
        z_sum = np.zeros(len(local_idx_arr), dtype=float)
        for aa, supported_idx in aa_map.items():
            family = SYNONYMOUS_CODONS[aa]
            family_idx = [CODON_TO_IDX[c] for c in family]
            total = sub_counts[:, family_idx].sum(axis=1)
            supported = sub_counts[:, supported_idx].sum(axis=1)
            valid = total > 0
            if not np.any(valid):
                continue
            exp = len(supported_idx) / float(len(family_idx))
            score = np.zeros(len(local_idx_arr), dtype=float)
            score[valid] = (supported[valid] / total[valid]) / exp
            score_sum[valid] += score[valid]
            score_n[valid] += 1
            s_sum[valid] += supported[valid]
            z_sum[valid] += total[valid] - supported[valid]
        valid_rows = score_n > 0
        cci[local_idx_arr[valid_rows]] = score_sum[valid_rows] / score_n[valid_rows]
        raw_s[local_idx_arr] = s_sum
        raw_z[local_idx_arr] = z_sum

    out = chunk[["Genome_ID", "Replicon_ID", "Replicon_Type", "Locus_Tag", "Product", "Gene_group", "plasmid_size_group", "Length"]].copy()
    out["Product_norm"] = out["Product"].map(norm_product)
    out["Raw_s_count"] = raw_s
    out["Raw_z_count"] = raw_z
    out["CCI_gene"] = cci
    return out


def append_values(store: dict[tuple[str, str], list[np.ndarray]], stratum: str, group: str, values: pd.Series) -> None:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr):
        store[(stratum, group)].append(arr)


def build_gene_cci(
    codon_path: Path,
    rep: pd.DataFrame,
    all_triad: set[str],
    size_eligible: dict[str, set[str]],
    supply_by_genome_aa: dict[str, dict[str, list[int]]],
    out_gene: Path,
    chunksize: int,
    max_chunks: int,
) -> tuple[dict[tuple[str, str], list[np.ndarray]], dict[str, int]]:
    rep_meta = rep[["replicon_key", "Gene_group", "plasmid_size_group", "Length"]].drop_duplicates("replicon_key")
    values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    usecols = ["Genome_ID", "Replicon_ID", "Replicon_Type", "Locus_Tag", "Product", "Seq_Length"] + CODONS
    first = True
    counters = defaultdict(int)

    if out_gene.exists():
        out_gene.unlink()

    for chunk_i, chunk in enumerate(pd.read_csv(codon_path, usecols=usecols, chunksize=chunksize), start=1):
        counters["chunks_seen"] += 1
        counters["rows_seen"] += len(chunk)
        for col in ["Genome_ID", "Replicon_ID", "Replicon_Type", "Locus_Tag"]:
            chunk[col] = chunk[col].astype(str).str.strip()
        chunk = chunk[chunk["Genome_ID"].isin(all_triad)].copy()
        counters["rows_in_all_triad_genomes"] += len(chunk)
        if chunk.empty:
            if max_chunks and chunk_i >= max_chunks:
                break
            continue

        chunk["replicon_key"] = chunk["Genome_ID"] + "|" + chunk["Replicon_ID"]
        chunk = chunk.merge(rep_meta, on="replicon_key", how="left", suffixes=("", "_meta"))
        chunk = chunk[chunk["Gene_group"].isin(GROUP_ORDER)].copy()
        if chunk.empty:
            if max_chunks and chunk_i >= max_chunks:
                break
            continue

        cci = compute_chunk_cci(chunk, supply_by_genome_aa)
        cci = cci[
            valid_cci_mask(cci["CCI_gene"])
            & cci["Product_norm"].ne("")
            & cci["Product_norm"].ne("nan")
        ].copy()
        counters["rows_written_gene_level"] += len(cci)
        if cci.empty:
            if max_chunks and chunk_i >= max_chunks:
                break
            continue

        cci.to_csv(out_gene, index=False, mode="w" if first else "a", header=first)
        first = False

        for group in GROUP_ORDER:
            sub = cci[cci["Gene_group"].eq(group)]
            append_values(values, STRATUM_ALL, group, sub["CCI_gene"])

        for sg, genomes in size_eligible.items():
            chr_sub = cci[cci["Gene_group"].eq(CHROM) & cci["Genome_ID"].isin(genomes)]
            pos_sub = cci[
                cci["Gene_group"].eq(TRNA_POS)
                & cci["Genome_ID"].isin(genomes)
                & cci["plasmid_size_group"].eq(sg)
            ]
            neg_sub = cci[
                cci["Gene_group"].eq(TRNA_NEG)
                & cci["Genome_ID"].isin(genomes)
                & cci["plasmid_size_group"].eq(sg)
            ]
            append_values(values, sg, CHROM, chr_sub["CCI_gene"])
            append_values(values, sg, TRNA_POS, pos_sub["CCI_gene"])
            append_values(values, sg, TRNA_NEG, neg_sub["CCI_gene"])

        if max_chunks and chunk_i >= max_chunks:
            break

    return values, {k: int(v) for k, v in counters.items()}


def combine_values(values: dict[tuple[str, str], list[np.ndarray]]) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for key, parts in values.items():
        if parts:
            out[key] = np.concatenate(parts)
        else:
            out[key] = np.array([], dtype=float)
    return out


def summarize(values: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for stratum in [STRATUM_ALL, *SIZE_ORDER]:
        for group in GROUP_ORDER:
            arr = values.get((stratum, group), np.array([], dtype=float))
            arr = arr[np.isfinite(arr)]
            rows.append(
                {
                    "stratum": stratum,
                    "stratum_label": SIZE_LABEL[stratum],
                    "Gene_group": group,
                    "gene_n": int(len(arr)),
                    "CCI_mean": float(np.mean(arr)) if len(arr) else np.nan,
                    "CCI_median": float(np.median(arr)) if len(arr) else np.nan,
                    "q25": float(np.quantile(arr, 0.25)) if len(arr) else np.nan,
                    "q75": float(np.quantile(arr, 0.75)) if len(arr) else np.nan,
                    "q95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
                    "frac_CCI_gt1": float(np.mean(arr > 1.0)) if len(arr) else np.nan,
                    "frac_CCI_gt2": float(np.mean(arr > 2.0)) if len(arr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bh_adjust(pvals: list[float]) -> list[float]:
    p = np.array(pvals, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    prev = 1.0
    n = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        real_rank = n - rank + 1
        val = min(prev, p[idx] * n / real_rank)
        ranked[idx] = val
        prev = val
    return ranked.tolist()


def tests(values: dict[tuple[str, str], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    pair_rows = []
    comparisons = [(TRNA_POS, CHROM), (TRNA_POS, TRNA_NEG), (TRNA_NEG, CHROM)]
    for stratum in [STRATUM_ALL, *SIZE_ORDER]:
        arrays = [values.get((stratum, g), np.array([], dtype=float)) for g in GROUP_ORDER]
        arrays = [a[np.isfinite(a)] for a in arrays]
        if all(len(a) > 0 for a in arrays):
            stat, p = scipy_stats.kruskal(*arrays, nan_policy="omit")
        else:
            stat, p = np.nan, np.nan
        global_rows.append(
            {
                "stratum": stratum,
                "stratum_label": SIZE_LABEL[stratum],
                "test": "Kruskal-Wallis",
                "statistic": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }
        )
        stratum_pair_rows = []
        stratum_pvals = []
        for g1, g2 in comparisons:
            a = values.get((stratum, g1), np.array([], dtype=float))
            b = values.get((stratum, g2), np.array([], dtype=float))
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if len(a) and len(b):
                res = scipy_stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
                u = float(res.statistic)
                pval = float(res.pvalue)
                delta = (2.0 * u / (len(a) * len(b))) - 1.0
                med_diff = float(np.median(a) - np.median(b))
            else:
                u, pval, delta, med_diff = np.nan, np.nan, np.nan, np.nan
            stratum_pair_rows.append(
                {
                    "stratum": stratum,
                    "stratum_label": SIZE_LABEL[stratum],
                    "comparison": f"{g1} vs {g2}",
                    "test": "Mann-Whitney U",
                    "group_1": g1,
                    "group_2": g2,
                    "n_group_1": int(len(a)),
                    "n_group_2": int(len(b)),
                    "median_group_1": float(np.median(a)) if len(a) else np.nan,
                    "median_group_2": float(np.median(b)) if len(b) else np.nan,
                    "median_difference_group1_minus_group2": med_diff,
                    "cliffs_delta_approx": delta,
                    "U_statistic": u,
                    "p_value": pval,
                }
            )
            stratum_pvals.append(pval)
        qvals = bh_adjust([p if np.isfinite(p) else 1.0 for p in stratum_pvals])
        for row, q in zip(stratum_pair_rows, qvals):
            row["q_value_BH_within_stratum"] = q
        pair_rows.extend(stratum_pair_rows)
    return pd.DataFrame(global_rows), pd.DataFrame(pair_rows)


def build_plot_sample(values: dict[tuple[str, str], np.ndarray], per_group: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for stratum in [STRATUM_ALL, *SIZE_ORDER]:
        for group in GROUP_ORDER:
            arr = values.get((stratum, group), np.array([], dtype=float))
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            take = min(per_group, len(arr))
            sample = rng.choice(arr, size=take, replace=False)
            rows.extend({"stratum": stratum, "stratum_label": SIZE_LABEL[stratum], "Gene_group": group, "CCI_gene": float(v)} for v in sample)
    return pd.DataFrame(rows)


def text(x: float, y: float, value: object, anchor: str = "middle", weight: str = "normal", rotate: float | None = None, cls: str = "") -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
    cls_attr = f' class="{cls}"' if cls else ""
    return (
        f'<text{cls_attr} x="{x:.2f}" y="{y:.2f}" font-family="Arial" font-size="{FONT_PT}pt" '
        f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle"{transform}>{esc(value)}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:7pt;fill:#111}.axis{stroke:#111;stroke-width:1.0pt;fill:none}.line,.tick,.violin,.iqr-box,.significance-bracket,.reference-line,.point{stroke:#222;stroke-width:0.7pt}.line,.tick,.significance-bracket,.reference-line{fill:none}.median-line{stroke:black;stroke-width:1.0pt}.iqr-box{fill:white}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def plot_panels(stats_df: pd.DataFrame, sample: pd.DataFrame, out_file: Path, strata: list[str], title: str, pairwise_df: pd.DataFrame | None = None) -> None:
    panel_w = 460
    width = panel_w * len(strata)
    height = 400
    top, bottom = 62, 92
    plot_h = height - top - bottom
    y_min, y_max = 0.0, 3.0
    rng = np.random.default_rng(RNG_SEED)
    body = []

    def sy(v: float) -> float:
        return top + plot_h - (float(v) - y_min) / (y_max - y_min) * plot_h

    def category_label(x: float, y: float, group: str) -> str:
        lines = {
            CHROM: ["Chromosomal genes"],
            TRNA_POS: ["Genes on tRNA-", "positive plasmids"],
            TRNA_NEG: ["Genes on tRNA-", "negative plasmids"],
        }[group]
        spans = "".join(
            f'<tspan x="0" dy="{0 if index == 0 else 9}">{esc(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        return f'<text transform="translate({x:.2f} {y:.2f}) rotate(-20)" text-anchor="end">{spans}</text>'

    for p, stratum in enumerate(strata):
        left = 75 + p * panel_w
        plot_w = panel_w - 93
        centers = np.linspace(left + plot_w * 0.18, left + plot_w * 0.82, 3)
        for tick in [0, 1, 2, 3]:
            y = sy(tick)
            body.append(f'<line class="tick" x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}"/>')
            body.append(text(left - 7, y, f"{tick:g}", "end"))
        body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
        body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

        for i, group in enumerate(GROUP_ORDER):
            x0 = float(centers[i])
            vals = sample[(sample["stratum"].eq(stratum)) & (sample["Gene_group"].eq(group))]["CCI_gene"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, y_min, y_max)
            if len(vals):
                hist, edges = np.histogram(vals, bins=55, range=(y_min, y_max), density=True)
                kernel = np.array([1, 2, 3, 2, 1], dtype=float)
                hist = np.convolve(hist, kernel / kernel.sum(), mode="same")
                hist = hist / (hist.max() + 1e-12)
                yc = (edges[:-1] + edges[1:]) / 2
                half = 28 * hist
                right = [f"{x0 + w:.2f},{sy(y):.2f}" for y, w in zip(yc, half)]
                left_pts = [f"{x0 - w:.2f},{sy(y):.2f}" for y, w in zip(yc[::-1], half[::-1])]
                body.append(f'<polygon class="violin" points="{" ".join(right + left_pts)}" fill="{GROUP_COLOR[group]}" fill-opacity="0.62"/>')

            row = stats_df[(stats_df["stratum"].eq(stratum)) & (stats_df["Gene_group"].eq(group))]
            if row.empty:
                continue
            row = row.iloc[0]
            q25, med, q75 = row["q25"], row["CCI_median"], row["q75"]
            iqr = q75 - q25
            in_fence = vals[(vals >= q25 - 1.5 * iqr) & (vals <= q75 + 1.5 * iqr)]
            if in_fence.size == 0:
                raise ValueError(f"no Tukey-whisker observations for {stratum}: {group}")
            whisker_low = float(in_fence.min())
            whisker_high = float(in_fence.max())
            box_w = 24
            body.append(f'<line class="line" x1="{x0:.2f}" y1="{sy(whisker_high):.2f}" x2="{x0:.2f}" y2="{sy(whisker_low):.2f}"/>')
            for whisker in (whisker_low, whisker_high):
                body.append(f'<line class="line" x1="{x0 - box_w / 2:.2f}" y1="{sy(whisker):.2f}" x2="{x0 + box_w / 2:.2f}" y2="{sy(whisker):.2f}"/>')
            body.append(
                f'<rect class="iqr-box" x="{x0 - box_w / 2:.2f}" y="{sy(q75):.2f}" width="{box_w}" height="{sy(q25) - sy(q75):.2f}"/>'
            )
            body.append(f'<line class="median-line" x1="{x0 - box_w / 2:.2f}" y1="{sy(med):.2f}" x2="{x0 + box_w / 2:.2f}" y2="{sy(med):.2f}"/>')
            body.append(category_label(x0, top + plot_h + 14, group))
        if pairwise_df is not None:
            pairs = pairwise_df[pairwise_df["stratum"].eq(stratum)]
            levels = [(0, 1, 48), (0, 2, 19), (1, 2, 34)]
            for i, j, y in levels:
                row = pairs[((pairs["group_1"].eq(GROUP_ORDER[i])) & (pairs["group_2"].eq(GROUP_ORDER[j]))) | ((pairs["group_1"].eq(GROUP_ORDER[j])) & (pairs["group_2"].eq(GROUP_ORDER[i])))]
                if row.empty:
                    continue
                label = "***" if float(row.iloc[0]["q_value_BH_within_stratum"]) <= 0.001 else ("**" if float(row.iloc[0]["q_value_BH_within_stratum"]) <= 0.01 else ("*" if float(row.iloc[0]["q_value_BH_within_stratum"]) <= 0.05 else "ns"))
                x1, x2 = centers[i], centers[j]
                body.append(f'<path class="significance-bracket" d="M {x1:.2f},{y+4:.2f} L {x1:.2f},{y:.2f} L {x2:.2f},{y:.2f} L {x2:.2f},{y+4:.2f}"/>')
                body.append(text((x1+x2)/2, y-7, label, cls="small"))
        panel_label = "All triad genomes" if stratum == STRATUM_ALL else ("<100 kb" if stratum == "small_lt100kb" else "≥100 kb")
        body.append(text(left + plot_w, 10, panel_label, "end", weight="bold"))
        if len(strata) > 1:
            body.append(text(p * panel_w + 4, 10, chr(ord("A") + p), "start", weight="bold"))
        body.append(text(p * panel_w + 12, top + plot_h / 2, "CDS-level CCI", rotate=-90))
    write_svg(out_file, width, height, body)


def write_report(path: Path, metadata: dict[str, object], stats_df: pd.DataFrame) -> None:
    all_stats = stats_df[stats_df["stratum"].eq(STRATUM_ALL)].set_index("Gene_group")
    lines = [
        "# Fig3B CDS-level triad CCI",
        "",
        "Definition: CCI was recomputed from CDS codon counts using the pooled set of codons supported by tRNA-positive plasmids in the same genome.",
        "Hypothetical protein rows were retained before comparison.",
        "The main comparison keeps only genomes with chromosome, tRNA-positive plasmid, and tRNA-negative plasmid contexts.",
        "Size-stratified panels keep genomes with chromosome plus both tRNA-positive and tRNA-negative plasmids in the indicated size stratum.",
        "",
        "## Main all-triad summary",
    ]
    for group in GROUP_ORDER:
        if group in all_stats.index:
            row = all_stats.loc[group]
            lines.append(
                f"- {group}: n={int(row['gene_n']):,}, median={row['CCI_median']:.3f}, "
                f"IQR={row['q25']:.3f}-{row['q75']:.3f}, CCI>1={row['frac_CCI_gt1']*100:.1f}%"
            )
    lines.extend(
        [
            "",
            "## Inputs",
            f"- Codon counts: `{metadata['input_codon_counts']}`",
            f"- Replicon info: `{metadata['input_replicon_info']}`",
            f"- Plasmid size/tRNA labels: `{metadata['input_plasmid_size']}`",
            f"- tRNA supply: `{metadata['input_trna_supply']}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    dirs = ensure_dirs(out_dir)
    mirror = Path(args.mirror_dir) if args.mirror_dir else None
    if mirror is not None:
        mirror.mkdir(parents=True, exist_ok=True)

    plasmid_meta = load_plasmid_meta(Path(args.plasmid_size))
    rep = load_replicon_info(Path(args.replicon_info), plasmid_meta)
    supply, supply_aa = load_pooled_supply(Path(args.trna_supply))
    all_triad, size_eligible, genome_qc = genome_sets(rep)
    all_triad = {g for g in all_triad if g in supply_aa}
    size_eligible = {k: {g for g in v if g in supply_aa} for k, v in size_eligible.items()}
    genome_qc["genome_n_with_pooled_supply"] = genome_qc["stratum"].map(
        {STRATUM_ALL: len(all_triad), **{k: len(v) for k, v in size_eligible.items()}}
    )
    genome_qc.to_csv(dirs["qc"] / "triad_genome_counts.csv", index=False)

    gene_out = dirs["tables"] / "01_gene_level_pooled_CCI_no_hypothetical_triad_genomes.csv"
    raw_values, counters = build_gene_cci(
        Path(args.codon_counts),
        rep,
        all_triad,
        size_eligible,
        supply_aa,
        gene_out,
        args.chunksize,
        args.max_chunks,
    )
    values = combine_values(raw_values)
    stats_df = summarize(values)
    global_df, pairwise_df = tests(values)
    sample_df = build_plot_sample(values, args.plot_sample_per_group)

    stats_path = dirs["tables"] / "02_Fig3B_gene_level_CCI_group_stats_no_hypothetical_triad.csv"
    global_path = dirs["tables"] / "03_Fig3B_gene_level_CCI_global_tests_no_hypothetical_triad.csv"
    pairwise_path = dirs["tables"] / "04_Fig3B_gene_level_CCI_pairwise_tests_no_hypothetical_triad.csv"
    sample_path = dirs["tables"] / "05_Fig3B_gene_level_CCI_plot_sample_no_hypothetical_triad.csv"
    stats_df.to_csv(stats_path, index=False)
    global_df.to_csv(global_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    main_svg = dirs["figures"] / "Fig3B_gene_level_CCI_chr_trnap_trnam.svg"
    size_svg = dirs["figures"] / "Fig3B_gene_level_CCI_chr_trnap_trnam_by_100kb.svg"
    plot_panels(stats_df, sample_df, main_svg, [STRATUM_ALL], "Fig3B CDS-level pooled CCI", pairwise_df)
    plot_panels(stats_df, sample_df, size_svg, SIZE_ORDER, "Fig3B CDS-level pooled CCI by size", pairwise_df)

    if mirror is not None:
        shutil.copy2(main_svg, mirror / "Fig3B_gene_level_CCI_chr_trnap_trnam.svg")
        stats_df.to_csv(mirror / "Fig3B_gene_level_CCI_group_stats_used.csv", index=False)
        global_df.to_csv(mirror / "Fig3B_gene_level_CCI_global_test_used.csv", index=False)
        pairwise_df.to_csv(mirror / "Fig3B_gene_level_CCI_pairwise_tests_used.csv", index=False)
        shutil.copy2(size_svg, mirror / "Fig3B_gene_level_CCI_chr_trnap_trnam_by_100kb.svg")

    metadata = {
        "input_codon_counts": str(Path(args.codon_counts)),
        "input_replicon_info": str(Path(args.replicon_info)),
        "input_plasmid_size": str(Path(args.plasmid_size)),
        "input_trna_supply": str(Path(args.trna_supply)),
        "out_dir": str(out_dir),
        "mirror_dir": str(mirror) if mirror is not None else None,
        "excluded_product_norm": EXCLUDED_PRODUCT_NORM,
        "filter_rule": "finite CCI_gene >= 0; hypothetical proteins retained",
        "pooled_supply_rule": "union of strict plasmid-tRNA-supported codons across tRNA-positive plasmids within the same genome",
        "main_triad_rule": "genome has chromosome, at least one tRNA-positive plasmid, and at least one tRNA-negative plasmid",
        "size_triad_rule": "for each size stratum, genome has chromosome plus tRNA-positive and tRNA-negative plasmids in that stratum",
        "size_threshold_bp": 100000,
        "size_label_note": "The source table has zero plasmids exactly 100000 bp in current verification; large_gt100kb is Length >= 100000 in code but labeled >100 kb.",
        "synonymous_codon_families": SYNONYMOUS_CODONS,
        "rows": counters,
        "triad_genome_n": int(len(all_triad)),
        "small_size_triad_genome_n": int(len(size_eligible["small_lt100kb"])),
        "large_size_triad_genome_n": int(len(size_eligible["large_gt100kb"])),
    }
    (dirs["root"] / "Fig3B_nohypo_triad_CCI_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(dirs["root"] / "Fig3B_nohypo_triad_CCI_report.md", metadata, stats_df)
    shutil.copy2(Path(__file__), dirs["scripts"] / Path(__file__).name)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
