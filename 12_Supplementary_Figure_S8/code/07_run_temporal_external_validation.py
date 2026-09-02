"""Fit all historical plasmids and evaluate the temporal cohort once."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from v4_common import (
    build_classifier,
    build_regressor,
    classification_metrics,
    continuous_score,
    load_primary_data,
    regression_metrics,
    write_json,
)


VERSION_RE = re.compile(r"\.\d+$")


def load_common_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "params" not in payload:
        raise ValueError(f"Missing params in {path}")
    return payload


def assembly_lineage(value: object) -> str:
    return VERSION_RE.sub("", str(value).strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-matrix-dir", type=Path, required=True)
    parser.add_argument("--temporal-matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path)
    parser.add_argument("--regressor-results", type=Path)
    parser.add_argument("--classifier-params-json", type=Path)
    parser.add_argument("--regressor-params-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if bool(args.classifier_params_json) != bool(args.regressor_params_json):
        raise ValueError("Common classifier and regressor parameter JSONs must be provided together")
    if bool(args.classifier_results) != bool(args.regressor_results):
        raise ValueError("Locked classifier and regressor result directories must be provided together")
    if bool(args.classifier_params_json) == bool(args.classifier_results):
        raise ValueError("Provide either common parameter JSONs or locked result directories")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    old_x, old_labels, old_names = load_primary_data(args.historical_matrix_dir)
    new_x = sparse.load_npz(
        args.temporal_matrix_dir / "X_temporal_plasmids_by_old_codes_no_tRNA.npz"
    ).tocsr().astype(np.float32)
    new_labels = pd.read_csv(
        args.temporal_matrix_dir / "temporal_labels.csv", keep_default_na=False
    )
    new_names = np.load(
        args.temporal_matrix_dir / "feature_names_no_tRNA.npy", allow_pickle=True
    ).astype(str)
    if new_x.shape != (len(new_labels), len(old_names)):
        raise ValueError("Temporal matrix and label dimensions differ")
    if not np.array_equal(old_names, new_names):
        raise ValueError("Temporal columns do not exactly match the frozen historical vocabulary")

    old_lineages = set(old_labels["Assembly_ID"].map(assembly_lineage))
    temporal_lineages = new_labels["Assembly_ID"].map(assembly_lineage)
    overlap_mask = temporal_lineages.isin(old_lineages)
    overlap = new_labels.loc[
        overlap_mask, ["Sample_ID", "Assembly_ID", "Replicon_ID"]
    ].copy()
    overlap["assembly_lineage_without_version"] = temporal_lineages.loc[overlap_mask]
    overlap.to_csv(args.out_dir / "excluded_temporal_historical_overlap.csv", index=False)
    new_x = new_x[~overlap_mask.to_numpy()].tocsr()
    new_labels = new_labels.loc[~overlap_mask].reset_index(drop=True)
    if new_labels.empty:
        raise ValueError("No non-overlapping temporal plasmids remain")

    if args.classifier_params_json:
        class_payload = load_common_params(args.classifier_params_json)
        reg_payload = load_common_params(args.regressor_params_json)
        class_params = class_payload["params"]
        reg_params = reg_payload["params"]
        threshold = float(class_payload["mean_validation_F1_threshold"])
        parameter_mode = "single common LightGBM parameter set"
        threshold_source = "mean validation F1 threshold from common classifier parameter selection"
    else:
        class_locked = pd.read_csv(
            args.classifier_results / "classifier_locked_parameters_by_run.csv",
            keep_default_na=False,
        )
        class_locked = class_locked.loc[class_locked["model"].eq("LightGBM")]
        reg_locked = pd.read_csv(
            args.regressor_results / "regressor_locked_parameters_by_run.csv",
            keep_default_na=False,
        )
        if len(class_locked) != 3 or len(reg_locked) != 3:
            raise ValueError("Temporal validation requires three locked tuning results")
        class_row = class_locked.loc[class_locked["selected_validation_AUPRC"].idxmax()]
        reg_row = reg_locked.loc[reg_locked["selected_validation_RMSE"].idxmin()]
        class_params = json.loads(class_row["params"])
        reg_params = json.loads(reg_row["params"])
        threshold = float(class_locked["F1_threshold_from_validation"].median())
        parameter_mode = "selected from per-run locked LightGBM parameter rows"
        threshold_source = "median of the three historical validation-derived thresholds"

    old_y_class = pd.to_numeric(old_labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    new_y_class = pd.to_numeric(new_labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    old_y_count = pd.to_numeric(old_labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    new_y_count = pd.to_numeric(new_labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    classifier = build_classifier(
        "LightGBM", class_params, 42, old_y_class, args.n_jobs
    )
    classifier.fit(old_x, old_y_class)
    class_score = continuous_score(classifier, new_x)
    class_metrics = {
        "task": "classification",
        "historical_train_n": int(len(old_y_class)),
        "temporal_test_n": int(len(new_y_class)),
        "temporal_positive_n": int(new_y_class.sum()),
        "temporal_positive_prevalence": float(new_y_class.mean()),
        "F1_threshold_from_historical_validation": threshold,
        **classification_metrics(new_y_class, class_score, threshold),
    }

    old_positive = old_y_class == 1
    new_positive = new_y_class == 1
    if not new_positive.any():
        raise ValueError("Temporal cohort contains no tRNA-positive plasmids for regression")
    regressor = build_regressor(reg_params, 42, args.n_jobs)
    regressor.fit(old_x[old_positive], old_y_count[old_positive])
    reg_prediction = regressor.predict(new_x[new_positive])
    reg_metrics = {
        "task": "regression_tRNA_positive_only",
        "historical_train_n": int(old_positive.sum()),
        "temporal_test_n": int(new_positive.sum()),
        **regression_metrics(new_y_count[new_positive], reg_prediction),
    }
    pd.DataFrame([class_metrics, reg_metrics]).to_csv(
        args.out_dir / "temporal_external_metrics.csv", index=False
    )
    predictions = new_labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
    predictions["y_true_has_tRNA"] = new_y_class
    predictions["classification_score"] = class_score
    predictions["classification_threshold"] = threshold
    predictions["classification_prediction"] = (class_score >= threshold).astype(int)
    predictions["y_true_tRNA_count"] = new_y_count
    predictions["regression_prediction"] = np.nan
    predictions.loc[new_positive, "regression_prediction"] = reg_prediction
    predictions.to_csv(args.out_dir / "temporal_external_predictions.csv", index=False)
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "evaluation_n": 1,
            "historical_training_fraction": 1.0,
            "temporal_role": "external test only",
            "historical_feature_vocabulary_frozen": True,
            "assembly_overlap_rule": "remove accession version suffix before comparison",
            "overlap_excluded_n": int(overlap_mask.sum()),
            "parameter_mode": parameter_mode,
            "classifier_parameter_source": str(args.classifier_params_json)
            if args.classifier_params_json
            else "highest historical validation AUPRC among the three locked runs",
            "regressor_parameter_source": str(args.regressor_params_json)
            if args.regressor_params_json
            else "lowest historical validation RMSE among the three locked runs",
            "F1_threshold_source": threshold_source,
            "external_data_used_for_model_development": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
