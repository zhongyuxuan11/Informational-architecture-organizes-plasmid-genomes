"""Tune the V4 task-specific LightGBM regressor on tRNA-positive plasmids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from v4_common import (
    N_CANDIDATES,
    SEEDS,
    build_regressor,
    file_sha256,
    json_default,
    load_primary_data,
    regression_candidates,
    regression_metrics,
    software_metadata,
    write_json,
)


def stratification_bins(y: np.ndarray) -> np.ndarray:
    value_counts = pd.Series(y).value_counts()
    if value_counts.min() >= 2:
        return y
    return pd.qcut(pd.Series(y).rank(method="first"), q=5, labels=False).to_numpy()


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strata = stratification_bins(y)
    development_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(y)), strata
        )
    )
    inner_strata = stratification_bins(y[development_idx])
    inner_train_rel, validation_rel = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(development_idx)), inner_strata
        )
    )
    return development_idx[inner_train_rel], development_idx[validation_rel], test_idx


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    matrix, labels, _ = load_primary_data(args.matrix_dir)
    positive_mask = pd.to_numeric(labels["has_tRNA"], errors="raise").eq(1).to_numpy()
    positive_rows = np.flatnonzero(positive_mask)
    matrix = matrix[positive_rows].tocsr()
    labels = labels.iloc[positive_rows].reset_index(drop=True)
    y = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    if np.any(y <= 0):
        raise ValueError("Regression data must contain tRNA-positive plasmids only")

    tuning_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    split_frames: list[pd.DataFrame] = []
    for run, seed in enumerate(SEEDS, start=1):
        inner_train_idx, validation_idx, test_idx = split_indices(y, seed)
        split = labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        split.insert(0, "positive_subset_row_index", np.arange(len(split), dtype=int))
        split["original_matrix_row_index"] = positive_rows
        split["run"] = run
        split["seed"] = seed
        split["split"] = "development_inner_train"
        split.loc[validation_idx, "split"] = "development_validation"
        split.loc[test_idx, "split"] = "untouched_test"
        split_frames.append(split)

        stem = checkpoints / f"run{run}_LightGBMRegressor"
        tuning_path = stem.with_suffix(".tuning.csv")
        metrics_path = stem.with_suffix(".metrics.json")
        predictions_path = stem.with_suffix(".predictions.csv")
        selected_path = stem.with_suffix(".selected.json")
        final_paths = (metrics_path, predictions_path, selected_path)
        if any(path.exists() for path in final_paths):
            if not (tuning_path.exists() and all(path.exists() for path in final_paths)):
                raise RuntimeError(f"Incomplete final checkpoint for {stem}")
            tuning_frames.append(pd.read_csv(tuning_path))
            metrics_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            prediction_frames.append(pd.read_csv(predictions_path))
            selected_rows.append(json.loads(selected_path.read_text(encoding="utf-8")))
            continue

        candidates = regression_candidates(seed * 100 + 77)
        completed = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
        completed_ids = set(completed.get("candidate_id", pd.Series(dtype=int)).astype(int))
        for candidate_id, params in enumerate(candidates, start=1):
            if candidate_id in completed_ids:
                continue
            model = build_regressor(params, seed, args.n_jobs)
            model.fit(matrix[inner_train_idx], y[inner_train_idx])
            validation_prediction = model.predict(matrix[validation_idx])
            validation_rmse = regression_metrics(
                y[validation_idx], validation_prediction
            )["RMSE"]
            row = {
                "run": run,
                "seed": seed,
                "model": "LightGBMRegressor",
                "candidate_id": candidate_id,
                "params": json.dumps(params, sort_keys=True, default=json_default),
                "inner_train_n": int(len(inner_train_idx)),
                "validation_n": int(len(validation_idx)),
                "validation_RMSE": validation_rmse,
            }
            append_row(tuning_path, row)
            print(
                f"run={run} regressor candidate={candidate_id}/{N_CANDIDATES} "
                f"validation_RMSE={validation_rmse:.6f}",
                flush=True,
            )

        tuning = pd.read_csv(tuning_path)
        if len(tuning) != N_CANDIDATES or set(tuning["candidate_id"]) != set(range(1, 51)):
            raise RuntimeError(f"Incomplete regression tuning for {stem}")
        best_row = tuning.loc[tuning["validation_RMSE"].idxmin()]
        best_params = json.loads(best_row["params"])
        development_idx = np.sort(np.concatenate((inner_train_idx, validation_idx)))
        model = build_regressor(best_params, seed, args.n_jobs)
        model.fit(matrix[development_idx], y[development_idx])
        test_prediction = model.predict(matrix[test_idx])
        metrics = {
            "run": run,
            "seed": seed,
            "model": "LightGBMRegressor",
            "development_n": int(len(development_idx)),
            "test_n": int(len(test_idx)),
            "selected_candidate_id": int(best_row["candidate_id"]),
            "selected_validation_RMSE": float(best_row["validation_RMSE"]),
            **regression_metrics(y[test_idx], test_prediction),
        }
        predictions = labels.iloc[test_idx][
            ["Sample_ID", "Assembly_ID", "Replicon_ID"]
        ].copy()
        predictions.insert(0, "positive_subset_row_index", test_idx)
        predictions["original_matrix_row_index"] = positive_rows[test_idx]
        predictions["run"] = run
        predictions["seed"] = seed
        predictions["y_true_tRNA_count"] = y[test_idx]
        predictions["y_pred_tRNA_count"] = test_prediction
        selected = {
            "run": run,
            "seed": seed,
            "model": "LightGBMRegressor",
            "selected_candidate_id": int(best_row["candidate_id"]),
            "selected_validation_RMSE": float(best_row["validation_RMSE"]),
            "params": best_params,
        }
        write_json(metrics_path, metrics)
        predictions.to_csv(predictions_path, index=False)
        write_json(selected_path, selected)
        tuning_frames.append(tuning)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        selected_rows.append(selected)

    tuning_detail = pd.concat(tuning_frames, ignore_index=True)
    metrics_detail = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    splits = pd.concat(split_frames, ignore_index=True)
    selected = pd.DataFrame(
        {
            "run": [item["run"] for item in selected_rows],
            "seed": [item["seed"] for item in selected_rows],
            "selected_candidate_id": [item["selected_candidate_id"] for item in selected_rows],
            "selected_validation_RMSE": [item["selected_validation_RMSE"] for item in selected_rows],
            "params": [json.dumps(item["params"], sort_keys=True) for item in selected_rows],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "run_n": int(metrics_detail["run"].nunique()),
                **{
                    f"{metric}_{stat}": float(getattr(metrics_detail[metric], stat)())
                    for metric in ("R2", "RMSE", "Spearman_rho")
                    for stat in ("mean", "std")
                },
            }
        ]
    )
    if int(summary.loc[0, "run_n"]) != 3:
        raise RuntimeError("Regression tuning must complete exactly three runs")

    tuning_detail.to_csv(args.out_dir / "regressor_tuning_detail.csv", index=False)
    metrics_detail.to_csv(args.out_dir / "regressor_test_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "regressor_metrics_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "regressor_test_predictions.csv", index=False)
    splits.to_csv(args.out_dir / "regressor_split_manifest.csv", index=False)
    selected.to_csv(args.out_dir / "regressor_locked_parameters_by_run.csv", index=False)
    write_json(
        args.out_dir / "regressor_run_metadata.json",
        {
            "seeds": list(SEEDS),
            "cohort": "tRNA-positive plasmids only",
            "target": "raw total tRNA count",
            "outer_design": "three independent stratified 80:20 development-test holdouts",
            "inner_design": "stratified 80:20 inner-training-validation split within development",
            "candidate_n_per_run": N_CANDIDATES,
            "candidate_selection_metric": "validation RMSE",
            "test_data_used_for_tuning": False,
            "matrix_sha256": file_sha256(args.matrix_dir / "X_plasmids_by_codes_no_tRNA.npz"),
            "labels_sha256": file_sha256(args.matrix_dir / "derived_labels_rebuilt.csv"),
            "n_jobs_per_fit": args.n_jobs,
            "software": software_metadata(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

