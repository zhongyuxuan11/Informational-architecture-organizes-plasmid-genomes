"""Select common LightGBM params from run winners and evaluate them."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from v4_common import (
    SEEDS,
    build_classifier,
    build_regressor,
    classification_metrics,
    continuous_score,
    f1_threshold,
    file_sha256,
    json_default,
    load_primary_data,
    regression_metrics,
    software_metadata,
    write_json,
)


def classifier_split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def stratification_bins(y: np.ndarray) -> np.ndarray:
    value_counts = pd.Series(y).value_counts()
    if value_counts.min() >= 2:
        return y
    return pd.qcut(pd.Series(y).rank(method="first"), q=5, labels=False).to_numpy()


def regressor_split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_unique_candidates(path: Path, model: str) -> list[dict]:
    frame = pd.read_csv(path)
    if "model" in frame.columns:
        frame = frame.loc[frame["model"].eq(model)].copy()
    if len(frame) != 3:
        raise ValueError(f"Expected three {model} locked rows in {path}, found {len(frame)}")
    candidates: list[dict] = []
    seen: set[str] = set()
    for row in frame.sort_values("run").itertuples(index=False):
        params = json.loads(row.params)
        signature = json.dumps(params, sort_keys=True, default=json_default)
        if signature in seen:
            raise ValueError(f"Duplicate {model} winner parameters are not expected")
        seen.add(signature)
        candidates.append(
            {
                "source_run": int(row.run),
                "source_seed": int(row.seed),
                "source_candidate_id": int(row.selected_candidate_id),
                "params": params,
            }
        )
    return candidates


def mean_std_rows(frame: pd.DataFrame, metrics: tuple[str, ...]) -> dict:
    result = {"run_n": int(frame["run"].nunique())}
    for metric in metrics:
        result[f"{metric}_mean"] = float(frame[metric].mean())
        result[f"{metric}_SD"] = float(frame[metric].std())
    return result


def select_classifier_params(matrix, labels: pd.DataFrame, candidates: list[dict], n_jobs: int):
    y = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    rows: list[dict] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        for run, seed in enumerate(SEEDS, start=1):
            inner_train_idx, validation_idx, _ = classifier_split_indices(y, seed)
            start = time.perf_counter()
            model = build_classifier("LightGBM", candidate["params"], seed, y[inner_train_idx], n_jobs)
            model.fit(matrix[inner_train_idx], y[inner_train_idx])
            validation_score = continuous_score(model, matrix[validation_idx])
            threshold = f1_threshold(y[validation_idx], validation_score)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "candidate_index": candidate_index,
                    **{k: v for k, v in candidate.items() if k != "params"},
                    "run": run,
                    "seed": seed,
                    "validation_AUPRC": classification_metrics(
                        y[validation_idx], validation_score, threshold
                    )["AUPRC"],
                    "validation_F1_threshold": threshold,
                    "fit_predict_seconds": elapsed,
                    "params": json.dumps(candidate["params"], sort_keys=True, default=json_default),
                }
            )
            print(
                f"classifier candidate={candidate_index} seed={seed} "
                f"validation_AUPRC={rows[-1]['validation_AUPRC']:.6f}",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby("candidate_index", as_index=False).agg(
        source_run=("source_run", "first"),
        source_seed=("source_seed", "first"),
        source_candidate_id=("source_candidate_id", "first"),
        validation_AUPRC_mean=("validation_AUPRC", "mean"),
        validation_AUPRC_SD=("validation_AUPRC", "std"),
        validation_F1_threshold_mean=("validation_F1_threshold", "mean"),
        fit_predict_seconds_sum=("fit_predict_seconds", "sum"),
        params=("params", "first"),
    )
    best = summary.loc[summary["validation_AUPRC_mean"].idxmax()]
    return y, detail, summary, json.loads(best["params"]), best


def evaluate_classifier(matrix, labels: pd.DataFrame, y: np.ndarray, params: dict, n_jobs: int):
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    for run, seed in enumerate(SEEDS, start=1):
        inner_train_idx, validation_idx, test_idx = classifier_split_indices(y, seed)
        development_idx = np.sort(np.concatenate((inner_train_idx, validation_idx)))
        validation_model = build_classifier("LightGBM", params, seed, y[inner_train_idx], n_jobs)
        validation_model.fit(matrix[inner_train_idx], y[inner_train_idx])
        validation_score = continuous_score(validation_model, matrix[validation_idx])
        threshold = f1_threshold(y[validation_idx], validation_score)
        final_model = build_classifier("LightGBM", params, seed, y[development_idx], n_jobs)
        start = time.perf_counter()
        final_model.fit(matrix[development_idx], y[development_idx])
        test_score = continuous_score(final_model, matrix[test_idx])
        elapsed = time.perf_counter() - start
        metrics = {
            "run": run,
            "seed": seed,
            "model": "LightGBM",
            "development_n": int(len(development_idx)),
            "test_n": int(len(test_idx)),
            "test_positive_n": int(y[test_idx].sum()),
            "test_positive_prevalence": float(y[test_idx].mean()),
            "selected_validation_AUPRC": classification_metrics(
                y[validation_idx], validation_score, threshold
            )["AUPRC"],
            "F1_threshold_from_validation": threshold,
            "final_fit_predict_seconds": elapsed,
            **classification_metrics(y[test_idx], test_score, threshold),
        }
        predictions = labels.iloc[test_idx][["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        predictions.insert(0, "row_index", test_idx)
        predictions["run"] = run
        predictions["seed"] = seed
        predictions["model"] = "LightGBM"
        predictions["y_true"] = y[test_idx]
        predictions["y_score"] = test_score
        predictions["F1_threshold_from_validation"] = threshold
        predictions["y_pred"] = (test_score >= threshold).astype(int)
        split = labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        split.insert(0, "row_index", np.arange(len(split), dtype=int))
        split["run"] = run
        split["seed"] = seed
        split["split"] = "development_inner_train"
        split.loc[validation_idx, "split"] = "development_validation"
        split.loc[test_idx, "split"] = "untouched_test"
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        split_frames.append(split)
        print(f"classifier final run={run} test_AUPRC={metrics['AUPRC']:.6f}", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames), pd.concat(split_frames)


def select_regressor_params(matrix, labels: pd.DataFrame, candidates: list[dict], n_jobs: int):
    positive_mask = pd.to_numeric(labels["has_tRNA"], errors="raise").eq(1).to_numpy()
    positive_rows = np.flatnonzero(positive_mask)
    reg_matrix = matrix[positive_rows].tocsr()
    reg_labels = labels.iloc[positive_rows].reset_index(drop=True)
    y = pd.to_numeric(reg_labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    rows: list[dict] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        for run, seed in enumerate(SEEDS, start=1):
            inner_train_idx, validation_idx, _ = regressor_split_indices(y, seed)
            start = time.perf_counter()
            model = build_regressor(candidate["params"], seed, n_jobs)
            model.fit(reg_matrix[inner_train_idx], y[inner_train_idx])
            validation_prediction = model.predict(reg_matrix[validation_idx])
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "candidate_index": candidate_index,
                    **{k: v for k, v in candidate.items() if k != "params"},
                    "run": run,
                    "seed": seed,
                    "validation_RMSE": regression_metrics(
                        y[validation_idx], validation_prediction
                    )["RMSE"],
                    "fit_predict_seconds": elapsed,
                    "params": json.dumps(candidate["params"], sort_keys=True, default=json_default),
                }
            )
            print(
                f"regressor candidate={candidate_index} seed={seed} "
                f"validation_RMSE={rows[-1]['validation_RMSE']:.6f}",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby("candidate_index", as_index=False).agg(
        source_run=("source_run", "first"),
        source_seed=("source_seed", "first"),
        source_candidate_id=("source_candidate_id", "first"),
        validation_RMSE_mean=("validation_RMSE", "mean"),
        validation_RMSE_SD=("validation_RMSE", "std"),
        fit_predict_seconds_sum=("fit_predict_seconds", "sum"),
        params=("params", "first"),
    )
    best = summary.loc[summary["validation_RMSE_mean"].idxmin()]
    return positive_rows, reg_matrix, reg_labels, y, detail, summary, json.loads(best["params"]), best


def evaluate_regressor(reg_matrix, reg_labels: pd.DataFrame, positive_rows: np.ndarray, y: np.ndarray, params: dict, n_jobs: int):
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    for run, seed in enumerate(SEEDS, start=1):
        inner_train_idx, validation_idx, test_idx = regressor_split_indices(y, seed)
        development_idx = np.sort(np.concatenate((inner_train_idx, validation_idx)))
        model = build_regressor(params, seed, n_jobs)
        start = time.perf_counter()
        model.fit(reg_matrix[development_idx], y[development_idx])
        test_prediction = model.predict(reg_matrix[test_idx])
        elapsed = time.perf_counter() - start
        validation_model = build_regressor(params, seed, n_jobs)
        validation_model.fit(reg_matrix[inner_train_idx], y[inner_train_idx])
        validation_prediction = validation_model.predict(reg_matrix[validation_idx])
        metrics = {
            "run": run,
            "seed": seed,
            "model": "LightGBMRegressor",
            "development_n": int(len(development_idx)),
            "test_n": int(len(test_idx)),
            "selected_validation_RMSE": regression_metrics(
                y[validation_idx], validation_prediction
            )["RMSE"],
            "final_fit_predict_seconds": elapsed,
            **regression_metrics(y[test_idx], test_prediction),
        }
        predictions = reg_labels.iloc[test_idx][["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        predictions.insert(0, "positive_subset_row_index", test_idx)
        predictions["original_matrix_row_index"] = positive_rows[test_idx]
        predictions["run"] = run
        predictions["seed"] = seed
        predictions["y_true_tRNA_count"] = y[test_idx]
        predictions["y_pred_tRNA_count"] = test_prediction
        split = reg_labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        split.insert(0, "positive_subset_row_index", np.arange(len(split), dtype=int))
        split["original_matrix_row_index"] = positive_rows
        split["run"] = run
        split["seed"] = seed
        split["split"] = "development_inner_train"
        split.loc[validation_idx, "split"] = "development_validation"
        split.loc[test_idx, "split"] = "untouched_test"
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        split_frames.append(split)
        print(f"regressor final run={run} test_RMSE={metrics['RMSE']:.6f}", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames), pd.concat(split_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-locked", type=Path, required=True)
    parser.add_argument("--regressor-locked", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")

    matrix, labels, _ = load_primary_data(args.matrix_dir)
    classifier_candidates = load_unique_candidates(args.classifier_locked, "LightGBM")
    regressor_candidates = load_unique_candidates(args.regressor_locked, "LightGBMRegressor")

    y_cls, cls_detail, cls_summary, best_cls_params, best_cls = select_classifier_params(
        matrix, labels, classifier_candidates, args.n_jobs
    )
    cls_metrics, cls_predictions, cls_splits = evaluate_classifier(
        matrix, labels, y_cls, best_cls_params, args.n_jobs
    )

    positive_rows, reg_matrix, reg_labels, y_reg, reg_detail, reg_summary, best_reg_params, best_reg = (
        select_regressor_params(matrix, labels, regressor_candidates, args.n_jobs)
    )
    reg_metrics, reg_predictions, reg_splits = evaluate_regressor(
        reg_matrix, reg_labels, positive_rows, y_reg, best_reg_params, args.n_jobs
    )

    cls_detail.to_csv(args.out_dir / "classifier_common_candidate_validation_detail.csv", index=False)
    cls_summary.to_csv(args.out_dir / "classifier_common_candidate_validation_summary.csv", index=False)
    cls_metrics.to_csv(args.out_dir / "classifier_common_test_metrics_each_run.csv", index=False)
    cls_predictions.to_csv(args.out_dir / "classifier_common_test_predictions.csv", index=False)
    cls_splits.to_csv(args.out_dir / "classifier_common_split_manifest.csv", index=False)
    pd.DataFrame([mean_std_rows(cls_metrics, ("selected_validation_AUPRC", "AUPRC", "AUROC", "F1"))]).to_csv(
        args.out_dir / "classifier_common_test_metrics_summary.csv", index=False
    )

    reg_detail.to_csv(args.out_dir / "regressor_common_candidate_validation_detail.csv", index=False)
    reg_summary.to_csv(args.out_dir / "regressor_common_candidate_validation_summary.csv", index=False)
    reg_metrics.to_csv(args.out_dir / "regressor_common_test_metrics_each_run.csv", index=False)
    reg_predictions.to_csv(args.out_dir / "regressor_common_test_predictions.csv", index=False)
    reg_splits.to_csv(args.out_dir / "regressor_common_split_manifest.csv", index=False)
    pd.DataFrame([mean_std_rows(reg_metrics, ("selected_validation_RMSE", "R2", "RMSE", "Spearman_rho"))]).to_csv(
        args.out_dir / "regressor_common_test_metrics_summary.csv", index=False
    )

    write_json(
        args.out_dir / "best_lightgbm_classifier_params.json",
        {
            "selection_scope": "three run-specific LightGBM classifier winners only",
            "selection_rule": "highest mean validation AUPRC across seeds 42, 43, and 44",
            "source_run": int(best_cls["source_run"]),
            "source_seed": int(best_cls["source_seed"]),
            "source_candidate_id": int(best_cls["source_candidate_id"]),
            "mean_validation_AUPRC": float(best_cls["validation_AUPRC_mean"]),
            "sd_validation_AUPRC": float(best_cls["validation_AUPRC_SD"]),
            "mean_validation_F1_threshold": float(best_cls["validation_F1_threshold_mean"]),
            "params": best_cls_params,
        },
    )
    write_json(
        args.out_dir / "best_lightgbm_regressor_params.json",
        {
            "selection_scope": "three run-specific LightGBMRegressor winners only",
            "selection_rule": "lowest mean validation RMSE across seeds 42, 43, and 44",
            "source_run": int(best_reg["source_run"]),
            "source_seed": int(best_reg["source_seed"]),
            "source_candidate_id": int(best_reg["source_candidate_id"]),
            "mean_validation_RMSE": float(best_reg["validation_RMSE_mean"]),
            "sd_validation_RMSE": float(best_reg["validation_RMSE_SD"]),
            "params": best_reg_params,
        },
    )
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "method": "compressed common-parameter selection from prior per-run winners",
            "seeds": list(SEEDS),
            "classifier_candidate_n": len(classifier_candidates),
            "regressor_candidate_n": len(regressor_candidates),
            "final_classifier_threshold_policy": "per-run F1 threshold from validation predictions under the selected common params",
            "test_data_used_for_parameter_selection": False,
            "matrix_sha256": file_sha256(args.matrix_dir / "X_plasmids_by_codes_no_tRNA.npz"),
            "labels_sha256": file_sha256(args.matrix_dir / "derived_labels_rebuilt.csv"),
            "n_jobs_per_fit": args.n_jobs,
            "software": software_metadata(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
