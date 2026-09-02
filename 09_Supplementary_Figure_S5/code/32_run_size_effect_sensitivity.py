"""Compare locked LightGBM models on raw counts and per-100-kb product densities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error, r2_score, roc_auc_score

from v4_common import SEEDS, build_classifier, build_regressor, continuous_score, f1_threshold


def load_lengths(labels: pd.DataFrame, metadata_path: Path) -> np.ndarray:
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    required = {"GCF_ID", "Replicon_Acc", "Length"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Plasmid metadata is missing columns: {sorted(missing)}")
    joined = labels[["Assembly_ID", "Replicon_ID"]].merge(
        metadata[["GCF_ID", "Replicon_Acc", "Length"]],
        left_on=["Assembly_ID", "Replicon_ID"],
        right_on=["GCF_ID", "Replicon_Acc"],
        how="left",
        validate="one_to_one",
    )
    lengths = pd.to_numeric(joined["Length"], errors="coerce")
    if lengths.isna().any() or (lengths <= 0).any():
        raise ValueError("Plasmid length metadata is incomplete or non-positive")
    return lengths.to_numpy(dtype=np.float64)


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (score >= threshold).astype(int)
    return {
        "AUPRC": float(average_precision_score(y_true, score)),
        "AUROC": float(roc_auc_score(y_true, score)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
    }


def load_params(path: Path, parameter_column: str = "params") -> dict[int, dict]:
    frame = pd.read_csv(path, keep_default_na=False)
    if "run" not in frame.columns or parameter_column not in frame.columns:
        raise ValueError(f"Invalid parameter table: {path}")
    result: dict[int, dict] = {}
    for row in frame.itertuples(index=False):
        run = int(row.run)
        if run in result:
            raise ValueError(f"Duplicate parameter row for run {run}: {path}")
        result[run] = json.loads(getattr(row, parameter_column))
    if set(result) != {1, 2, 3}:
        raise ValueError(f"Expected parameter rows for runs 1, 2, 3: {path}")
    return result


def load_common_params(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Common parameter JSON lacks a params object: {path}")
    return {run: dict(params) for run in (1, 2, 3)}


def classification_rows(
    matrix,
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    params: dict[int, dict],
    setting: str,
    n_jobs: int,
) -> list[dict]:
    y = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    rows: list[dict] = []
    for run, seed in zip((1, 2, 3), SEEDS):
        current = manifest.loc[manifest["run"].astype(int).eq(run)]
        inner = current.loc[current["split"].eq("development_inner_train"), "row_index"].to_numpy(dtype=int)
        validation = current.loc[current["split"].eq("development_validation"), "row_index"].to_numpy(dtype=int)
        test = current.loc[current["split"].eq("untouched_test"), "row_index"].to_numpy(dtype=int)
        if np.intersect1d(inner, validation).size or np.intersect1d(inner, test).size or np.intersect1d(validation, test).size:
            raise ValueError(f"Classification split overlap in run {run}")
        validation_model = build_classifier("LightGBM", params[run], seed, y[inner], n_jobs)
        validation_model.fit(matrix[inner], y[inner])
        threshold = f1_threshold(y[validation], continuous_score(validation_model, matrix[validation]))
        final_train = np.sort(np.concatenate((inner, validation)))
        final_model = build_classifier("LightGBM", params[run], seed, y[final_train], n_jobs)
        final_model.fit(matrix[final_train], y[final_train])
        score = continuous_score(final_model, matrix[test])
        rows.append({
            "feature_setting": setting,
            "task": "classification",
            "run": run,
            "seed": seed,
            "train_n": int(len(final_train)),
            "test_n": int(len(test)),
            "test_positive_n": int(y[test].sum()),
            "test_positive_prevalence": float(y[test].mean()),
            "F1_threshold_from_validation": float(threshold),
            **classification_metrics(y[test], score, threshold),
        })
    return rows


def regression_rows(
    matrix,
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    params: dict[int, dict],
    setting: str,
    n_jobs: int,
) -> list[dict]:
    positive_rows = np.flatnonzero(pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int) == 1)
    positive_labels = labels.iloc[positive_rows].reset_index(drop=True)
    y = pd.to_numeric(positive_labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    rows: list[dict] = []
    for run, seed in zip((1, 2, 3), SEEDS):
        current = manifest.loc[manifest["run"].astype(int).eq(run)]
        inner = current.loc[current["split"].eq("development_inner_train"), "positive_subset_row_index"].to_numpy(dtype=int)
        validation = current.loc[current["split"].eq("development_validation"), "positive_subset_row_index"].to_numpy(dtype=int)
        test = current.loc[current["split"].eq("untouched_test"), "positive_subset_row_index"].to_numpy(dtype=int)
        if np.intersect1d(inner, validation).size or np.intersect1d(inner, test).size or np.intersect1d(validation, test).size:
            raise ValueError(f"Regression split overlap in run {run}")
        train = np.sort(np.concatenate((inner, validation)))
        model = build_regressor(params[run], seed, n_jobs)
        model.fit(matrix[positive_rows[train]], y[train])
        prediction = model.predict(matrix[positive_rows[test]])
        rows.append({
            "feature_setting": setting,
            "task": "regression_tRNA_positive_only",
            "run": run,
            "seed": seed,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            **regression_metrics(y[test], prediction),
        })
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    metric_names = ("AUPRC", "AUROC", "F1", "R2", "RMSE")
    rows: list[dict] = []
    for (setting, task), group in detail.groupby(["feature_setting", "task"], sort=False):
        row = {"feature_setting": setting, "task": task, "run_n": int(group["run"].nunique())}
        for metric in metric_names:
            if metric in group.columns and group[metric].notna().any():
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_SD"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--plasmid-metadata", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--classifier-params-json", type=Path)
    parser.add_argument("--regressor-params-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix = sparse.load_npz(args.matrix_dir / "X_plasmids_by_codes_no_tRNA.npz").tocsr().astype(np.float32)
    labels = pd.read_csv(args.matrix_dir / "derived_labels_rebuilt.csv", keep_default_na=False)
    if matrix.shape[0] != len(labels):
        raise ValueError("Matrix and labels are not aligned")
    lengths_bp = load_lengths(labels, args.plasmid_metadata)
    density_scale = 100_000.0 / lengths_bp
    normalized_matrix = sparse.diags(density_scale).dot(matrix).tocsr().astype(np.float32)
    classifier_manifest = pd.read_csv(args.classifier_results / "classifier_split_manifest.csv", keep_default_na=False)
    regressor_manifest = pd.read_csv(args.regressor_results / "regressor_split_manifest.csv", keep_default_na=False)
    classifier_params = (
        load_common_params(args.classifier_params_json)
        if args.classifier_params_json
        else load_params(args.classifier_results / "classifier_locked_parameters_by_run.csv")
    )
    regressor_params = (
        load_common_params(args.regressor_params_json)
        if args.regressor_params_json
        else load_params(args.regressor_results / "regressor_locked_parameters_by_run.csv")
    )

    detail_rows = []
    for setting, current_matrix in (("raw_product_counts", matrix), ("product_abundance_per_100kb", normalized_matrix)):
        detail_rows.extend(classification_rows(current_matrix, labels, classifier_manifest, classifier_params, setting, args.n_jobs))
        detail_rows.extend(regression_rows(current_matrix, labels, regressor_manifest, regressor_params, setting, args.n_jobs))
    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail)
    detail.to_csv(args.out_dir / "size_effect_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "size_effect_metrics_summary.csv", index=False)
    pd.DataFrame({"Sample_ID": labels["Sample_ID"], "Assembly_ID": labels["Assembly_ID"], "Replicon_ID": labels["Replicon_ID"], "Length_bp": lengths_bp, "scale_per_100kb": density_scale}).to_csv(args.out_dir / "plasmid_length_scaling.csv", index=False)
    (args.out_dir / "run_metadata.json").write_text(json.dumps({
        "analysis": "size-effect sensitivity analysis",
        "matrix": "same no-hypothetical raw product-count matrix as primary analysis",
        "normalized_feature_definition": "100000 * product_count / plasmid_length_bp (product abundance per 100 kb)",
        "regression_target": "original tRNA_count; not length-normalized",
        "classifier": "LightGBM with locked latest parameters; scale_pos_weight recomputed from each training split",
        "regressor": "LightGBMRegressor with locked latest parameters; fitted on tRNA-positive plasmids only",
        "classifier_parameter_source": str(args.classifier_params_json or (args.classifier_results / "classifier_locked_parameters_by_run.csv")),
        "regressor_parameter_source": str(args.regressor_params_json or (args.regressor_results / "regressor_locked_parameters_by_run.csv")),
        "split_policy": "exact existing classifier and regressor split manifests; identical splits for raw and normalized features",
        "seeds": list(SEEDS),
        "replicates": 3,
        "leakage_control": "no test rows used for fitting or threshold selection; only feature representation differs",
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
