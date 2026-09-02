#!/usr/bin/env python
"""Build plasmid-level CDE subcategory count/percent landscape table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


MODULE_ORDER = {"replication": 1, "transcription": 2, "translation": 3}


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def norm_text(x: object) -> str:
    return clean_text(x).casefold()


def sparse_row_sum(xmat: sparse.csr_matrix, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(xmat.shape[0], dtype=np.int64)
    return np.asarray(xmat[:, cols].sum(axis=1)).ravel().astype(np.int64)


def sparse_row_nnz(xmat: sparse.csr_matrix, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(xmat.shape[0], dtype=np.int64)
    return np.asarray((xmat[:, cols] > 0).sum(axis=1)).ravel().astype(np.int64)


def load_feature_map(matrix_dir: Path, n_cols: int) -> pd.DataFrame:
    feature_map = pd.read_csv(matrix_dir / "feature_code_mapping.csv", dtype=str, encoding="utf-8-sig")
    feature_map.columns = [clean_text(c) for c in feature_map.columns]
    missing = {"Code", "Product"} - set(feature_map.columns)
    if missing:
        raise ValueError(f"feature_code_mapping.csv missing columns: {sorted(missing)}")
    feature_map = feature_map.reset_index(drop=True).copy()
    feature_map["matrix_col"] = np.arange(feature_map.shape[0])
    feature_map = feature_map[feature_map["matrix_col"] < n_cols].copy()
    feature_map["Code"] = feature_map["Code"].map(clean_text)
    feature_map["Product"] = feature_map["Product"].map(clean_text)
    feature_map["product_key"] = feature_map["Product"].map(norm_text)
    feature_map["code_key"] = feature_map["Code"].map(norm_text)
    return feature_map


def match_cde_products(cde: pd.DataFrame, feature_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_to_cols = (
        feature_map.groupby("product_key")["matrix_col"]
        .apply(lambda s: sorted(set(s.astype(int).tolist())))
        .to_dict()
    )
    code_to_cols = (
        feature_map.groupby("code_key")["matrix_col"]
        .apply(lambda s: sorted(set(s.astype(int).tolist())))
        .to_dict()
    )
    col_to_code = dict(zip(feature_map["matrix_col"].astype(int), feature_map["Code"]))
    col_to_product = dict(zip(feature_map["matrix_col"].astype(int), feature_map["Product"]))

    matched: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    for _, row in cde.iterrows():
        module = clean_text(row["module"]).casefold()
        subcategory = clean_text(row["subcategory"])
        product = clean_text(row["product"])
        key = norm_text(product)
        hits: dict[int, str] = {}
        for col in product_to_cols.get(key, []):
            hits[int(col)] = "Product_exact"
        for col in code_to_cols.get(key, []):
            hits[int(col)] = "Code_exact"
        if not hits:
            unmatched.append({
                "module": module,
                "subcategory": subcategory,
                "input_product": product,
                "reason": "not matched in feature_map Product or Code",
            })
            continue
        for col, method in sorted(hits.items()):
            matched.append({
                "module": module,
                "subcategory": subcategory,
                "input_product": product,
                "matched_matrix_col": col,
                "matched_code": col_to_code.get(col, ""),
                "matched_product": col_to_product.get(col, ""),
                "matched_by": method,
            })
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def load_taxonomy(path: Path) -> pd.DataFrame:
    tax = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    tax.columns = [clean_text(c) for c in tax.columns]
    tax_cols = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    missing = {"GCF_ID", *tax_cols} - set(tax.columns)
    if missing:
        raise ValueError(f"taxonomy file missing columns: {sorted(missing)}")
    tax = tax[["GCF_ID", *tax_cols]].drop_duplicates("GCF_ID").copy()
    rename = {
        "kingdom": "Kingdom",
        "phylum": "Phylum",
        "class": "Class",
        "order": "Order",
        "family": "Family",
        "genus": "Genus",
        "species": "Species",
    }
    tax = tax.rename(columns=rename)
    for col in rename.values():
        tax[col] = tax[col].fillna("").map(clean_text)
    return tax


def build_table(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.out)
    audit_dir = Path(args.audit_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    matrix_dir = Path(args.matrix_dir)
    xmat = sparse.load_npz(matrix_dir / "X_plasmids_by_codes.npz").tocsr()
    sample_ids = pd.read_csv(matrix_dir / "sample_ids.csv", dtype=str)
    sample_ids["RowIndex"] = pd.to_numeric(sample_ids["RowIndex"], errors="coerce").astype(int)
    sample_ids = sample_ids[["Assembly_ID", "Replicon_ID", "Sample_ID", "RowIndex"]].copy()
    feature_map = load_feature_map(matrix_dir, xmat.shape[1])

    cde = pd.read_csv(args.cde_product, dtype=str, encoding="utf-8-sig")
    cde.columns = [clean_text(c) for c in cde.columns]
    missing = {"module", "subcategory", "product"} - set(cde.columns)
    if missing:
        raise ValueError(f"CDE_product.csv missing columns: {sorted(missing)}")
    cde = cde[["module", "subcategory", "product"]].copy()
    for col in cde.columns:
        cde[col] = cde[col].map(clean_text)
    cde["module"] = cde["module"].str.casefold()
    cde = cde[cde["module"].isin(MODULE_ORDER)].copy()
    cde["_module_order"] = cde["module"].map(MODULE_ORDER)

    match, unmatched = match_cde_products(cde, feature_map)
    match.to_csv(audit_dir / "CDE_product_to_code_matched_detail_latest.csv", index=False, encoding="utf-8-sig")
    unmatched.to_csv(audit_dir / "CDE_product_to_code_unmatched_latest.csv", index=False, encoding="utf-8-sig")

    tax = load_taxonomy(Path(args.taxonomy))
    meta = sample_ids.merge(tax, left_on="Assembly_ID", right_on="GCF_ID", how="left")
    for col in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]:
        meta[col] = meta[col].fillna("")
    meta = meta.rename(columns={"Assembly_ID": "Genome_ID", "Replicon_ID": "Plasmid_ID"})
    meta_cols = ["Genome_ID", "Plasmid_ID", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    meta = meta[meta_cols + ["RowIndex"]].copy()

    denom = (
        cde.groupby(["module", "subcategory"], sort=False)
        .size()
        .rename("cde_product_n")
        .reset_index()
    )
    matched_n = (
        match.groupby(["module", "subcategory"], sort=False)["matched_matrix_col"]
        .nunique()
        .rename("matched_product_n")
        .reset_index()
    )
    subcats = (
        denom.merge(matched_n, on=["module", "subcategory"], how="left")
        .fillna({"matched_product_n": 0})
    )
    subcats["matched_product_n"] = subcats["matched_product_n"].astype(int)
    subcats["_module_order"] = subcats["module"].map(MODULE_ORDER)
    subcats = subcats.sort_values(["_module_order", "subcategory"]).drop(columns="_module_order")
    subcats.to_csv(audit_dir / "CDE_subcategory_denominators_latest.csv", index=False, encoding="utf-8-sig")

    frames: list[pd.DataFrame] = []
    row_order = meta["RowIndex"].to_numpy()
    base_meta = meta.drop(columns="RowIndex").reset_index(drop=True)
    for _, sub in subcats.iterrows():
        module = sub["module"]
        subcategory = sub["subcategory"]
        product_n = int(sub["cde_product_n"])
        cols = sorted(
            match.loc[
                match["module"].eq(module) & match["subcategory"].eq(subcategory),
                "matched_matrix_col",
            ].astype(int).unique().tolist()
        )
        raw_copy_count_all = sparse_row_sum(xmat, cols)
        detected_product_n_all = sparse_row_nnz(xmat, cols)
        raw_copy_count = raw_copy_count_all[row_order]
        detected_product_n = detected_product_n_all[row_order]
        percent = np.round(detected_product_n / product_n * 100, 4) if product_n else np.zeros(len(meta))

        count_df = base_meta.copy()
        count_df["module"] = module
        count_df["subcategory"] = subcategory
        count_df["metric"] = "Count"
        count_df["value"] = detected_product_n
        count_df["cde_product_n"] = product_n
        count_df["matched_product_n"] = len(cols)
        count_df["raw_copy_count"] = raw_copy_count

        pct_df = base_meta.copy()
        pct_df["module"] = module
        pct_df["subcategory"] = subcategory
        pct_df["metric"] = "Percent"
        pct_df["value"] = percent
        pct_df["cde_product_n"] = product_n
        pct_df["matched_product_n"] = len(cols)
        pct_df["raw_copy_count"] = raw_copy_count
        frames.extend([count_df, pct_df])

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out, index=False, encoding="utf-8")

    metadata = {
        "output": str(out),
        "rows": int(len(result)),
        "sample_rows": int(len(sample_ids)),
        "subcategory_n": int(len(subcats)),
        "expected_rows": int(len(sample_ids) * len(subcats) * 2),
        "metric_rows_per_subcategory": ["Count", "Percent"],
        "count_definition": "number of distinct CDE products from the subcategory detected in the plasmid",
        "percent_definition": "Count / cde_product_n from CDE_product.csv * 100",
        "raw_copy_count_note": "raw summed matrix counts are retained as an audit column but are not the Count metric",
        "cde_product": str(Path(args.cde_product)),
        "matrix_dir": str(matrix_dir),
        "taxonomy": str(Path(args.taxonomy)),
        "unmatched_product_rows": int(len(unmatched)),
    }
    (audit_dir / "Plasmid_CDE_subcategory_landscape_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[2]
    preprocessing = package_root / "supplementary_tables" / "00_preprocessing"
    tables = package_root / "supplementary_tables" / "04_fig4_figS8"
    p = argparse.ArgumentParser()
    p.add_argument("--cde-product", default=preprocessing / "CDE_product.csv")
    p.add_argument("--matrix-dir", default=preprocessing)
    p.add_argument("--taxonomy", default=preprocessing / "gcf_with_full_taxonomy.csv")
    p.add_argument(
        "--out",
        default=preprocessing / "Plasmid_CDE_module_subcategory_landscape_per_plasmid.csv",
    )
    p.add_argument(
        "--audit-dir",
        default=tables / "cde_mapping_audit",
    )
    return p.parse_args()


def main() -> None:
    build_table(parse_args())


if __name__ == "__main__":
    main()
