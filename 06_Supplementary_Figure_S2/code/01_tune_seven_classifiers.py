"""Tune seven classifiers with V4 three-repetition 80:20 holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from v4_common import (
    MODEL_ORDER,
    N_CANDIDATES,
    SEEDS,
    build_classifier,
    classification_metrics,
    classifier_candidates,
    continuous_score,
    f1_threshold,
    file_sha256,
    json_default,
    load_primary_data,
    software_metadata,
    write_json,
)


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    development_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )
    inner_train_rel, validation_rel = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(development_idx)), y[development_idx]
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


def run_one(
    model_name: str,
    run: int,
    seed: int,
    matrix,
    labels: pd.DataFrame,
    y: np.ndarray,
    inner_train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    checkpoints: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    stem = checkpoints / f"run{run}_{model_name}"
    tuning_path = stem.with_suffix(".tuning.csv")
    metrics_path = stem.with_suffix(".metrics.json")
    predictions_path = stem.with_suffix(".predictions.csv")
    selected_path = stem.with_suffix(".selected.json")
    completed_paths = (tuning_path, metrics_path, predictions_path, selected_path)
    if any(path.exists() for path in completed_paths[1:]):
        if not all(path.exists() for path in completed_paths):
            raise RuntimeError(f"Incomplete final checkpoint for {stem}")
        return (
            pd.read_csv(tuning_path),
            json.loads(metrics_path.read_text(encoding="utf-8")),
            pd.read_csv(predictions_path),
            json.loads(selected_path.read_text(encoding="utf-8")),
        )

    candidates = classifier_candidates(model_name, seed * 100 + MODEL_ORDER.index(model_name))
    if len(candidates) != N_CANDIDATES:
        raise RuntimeError("Candidate generator did not return exactly 50 configurations")
    completed = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
    completed_ids = set(completed.get("candidate_id", pd.Series(dtype=int)).astype(int))
    for candidate_id, params in enumerate(candidates, start=1):
        if candidate_id in completed_ids:
            continue
        model = build_classifier(model_name, params, seed, y[inner_train_idx], n_jobs)
        model.fit(matrix[inner_train_idx], y[inner_train_idx])
        validation_score = continuous_score(model, matrix[validation_idx])
        threshold = f1_threshold(y[validation_idx], validation_score)
        row = {
            "run": run,
            "seed": seed,
            "model": model_name,
            "candidate_id": candidate_id,
            "params": json.dumps(params, sort_keys=True, default=json_default),
            "inner_train_n": int(len(inner_train_idx)),
            "validation_n": int(len(validation_idx)),
            "validation_AUPRC": classification_metrics(
                y[validation_idx], validation_score, threshold
            )["AUPRC"],
            "validation_F1_threshold": threshold,
        }
        append_row(tuning_path, row)
        print(
            f"run={run} model={model_name} candidate={candidate_id}/{N_CANDIDATES} "
            f"validation_AUPRC={row['validation_AUPRC']:.6f}",
            flush=True,
        )

    tuning = pd.read_csv(tuning_path)
    if len(tuning) != N_CANDIDATES or set(tuning["candidate_id"]) != set(range(1, 51)):
        raise RuntimeError(f"Incomplete tuning results for {stem}")
    best_row = tuning.loc[tuning["validation_AUPRC"].idxmax()]
    best_params = json.loads(best_row["params"])
    threshold = float(best_row["validation_F1_threshold"])
    development_idx = np.sort(np.concatenate((inner_train_idx, validation_idx)))
    final_model = build_classifier(model_name, best_params, seed, y[development_idx], n_jobs)
    final_model.fit(matrix[development_idx], y[development_idx])
    test_score = continuous_score(final_model, matrix[test_idx])
    metrics = {
        "run": run,
        "seed": seed,
        "model": model_name,
        "development_n": int(len(development_idx)),
        "test_n": int(len(test_idx)),
        "test_positive_n": int(y[test_idx].sum()),
        "test_positive_prevalence": float(y[test_idx].mean()),
        "selected_candidate_id": int(best_row["candidate_id"]),
        "selected_validation_AUPRC": float(best_row["validation_AUPRC"]),
        "F1_threshold_from_validation": threshold,
        **classification_metrics(y[test_idx], test_score, threshold),
    }
    predictions = labels.iloc[test_idx][
        ["Sample_ID", "Assembly_ID", "Replicon_ID"]
    ].copy()
    predictions.insert(0, "row_index", test_idx)
    predictions["run"] = run
    predictions["seed"] = seed
    predictions["model"] = model_name
    predictions["y_true"] = y[test_idx]
    predictions["y_score"] = test_score
    predictions["F1_threshold_from_validation"] = threshold
    predictions["y_pred"] = (test_score >= threshold).astype(int)
    selected = {
        "run": run,
        "seed": seed,
        "model": model_name,
        "selected_candidate_id": int(best_row["candidate_id"]),
        "selected_validation_AUPRC": float(best_row["validation_AUPRC"]),
        "F1_threshold_from_validation": threshold,
        "params": best_params,
    }
    write_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)
    write_json(selected_path, selected)
    return tuning, metrics, predictions, selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoint_dir or args.out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    matrix, labels, _ = load_primary_data(args.matrix_dir)
    y = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("has_tRNA must contain both binary classes")

    split_frames: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    all_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_selected: list[dict] = []
    for run, seed in enumerate(SEEDS, start=1):
        inner_train_idx, validation_idx, test_idx = split_indices(y, seed)
        split = labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        split.insert(0, "row_index", np.arange(len(split), dtype=int))
        split["run"] = run
        split["seed"] = seed
        split["split"] = "development_inner_train"
        split.loc[validation_idx, "split"] = "development_validation"
        split.loc[test_idx, "split"] = "untouched_test"
        split_frames.append(split)
        for model_name in args.models:
            tuning, metrics, predictions, selected = run_one(
                model_name,
                run,
                seed,
                matrix,
                labels,
                y,
                inner_train_idx,
                validation_idx,
                test_idx,
                checkpoints,
                args.n_jobs,
            )
            all_tuning.append(tuning)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            all_selected.append(selected)
            print(f"run={run} model={model_name} complete", flush=True)

    tuning_detail = pd.concat(all_tuning, ignore_index=True)
    metrics_detail = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    splits = pd.concat(split_frames, ignore_index=True)
    selected_frame = pd.DataFrame(
        {
            "run": [item["run"] for item in all_selected],
            "seed": [item["seed"] for item in all_selected],
            "model": [item["model"] for item in all_selected],
            "selected_candidate_id": [item["selected_candidate_id"] for item in all_selected],
            "selected_validation_AUPRC": [item["selected_validation_AUPRC"] for item in all_selected],
            "F1_threshold_from_validation": [item["F1_threshold_from_validation"] for item in all_selected],
            "params": [json.dumps(item["params"], sort_keys=True) for item in all_selected],
        }
    )
    summary = metrics_detail.groupby("model", as_index=False).agg(
        run_n=("run", "nunique"),
        validation_AUPRC_mean=("selected_validation_AUPRC", "mean"),
        validation_AUPRC_SD=("selected_validation_AUPRC", "std"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_SD=("AUPRC", "std"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_SD=("AUROC", "std"),
        F1_mean=("F1", "mean"),
        F1_SD=("F1", "std"),
    )
    if not summary["run_n"].eq(3).all():
        raise RuntimeError("Every classifier must have exactly three completed runs")
    winner = summary.loc[summary["validation_AUPRC_mean"].idxmax()]
    winner_record = {
        "selection_rule": "highest mean internal-validation AUPRC across seeds 42, 43, and 44",
        "selected_classifier_family": str(winner["model"]),
        "mean_validation_AUPRC": float(winner["validation_AUPRC_mean"]),
        "test_data_used_for_selection": False,
    }

    tuning_detail.to_csv(args.out_dir / "classifier_tuning_detail.csv", index=False)
    metrics_detail.to_csv(args.out_dir / "classifier_test_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "classifier_metrics_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "classifier_test_predictions.csv", index=False)
    splits.to_csv(args.out_dir / "classifier_split_manifest.csv", index=False)
    selected_frame.to_csv(args.out_dir / "classifier_locked_parameters_by_run.csv", index=False)
    write_json(args.out_dir / "selected_classifier_family.json", winner_record)
    write_json(
        args.out_dir / "classifier_run_metadata.json",
        {
            "seeds": list(SEEDS),
            "outer_design": "three independent stratified 80:20 development-test holdouts",
            "inner_design": "stratified 80:20 inner-training-validation split within development",
            "candidate_n_per_model_per_run": N_CANDIDATES,
            "candidate_selection_metric": "validation AUPRC",
            "F1_threshold_source": "selected candidate validation predictions only",
            "test_data_roles": ["final AUPRC", "final AUROC", "final F1"],
            "test_data_prohibited_roles": [
                "algorithm selection",
                "hyperparameter tuning",
                "feature selection",
                "class weighting",
                "threshold optimization",
                "feature vocabulary construction",
            ],
            "matrix_path": str((args.matrix_dir / "X_plasmids_by_codes_no_tRNA.npz").resolve()),
            "matrix_sha256": file_sha256(args.matrix_dir / "X_plasmids_by_codes_no_tRNA.npz"),
            "labels_sha256": file_sha256(args.matrix_dir / "derived_labels_rebuilt.csv"),
            "n_jobs_per_fit": args.n_jobs,
            "software": software_metadata(),
        },
    )
    print(json.dumps(winner_record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
