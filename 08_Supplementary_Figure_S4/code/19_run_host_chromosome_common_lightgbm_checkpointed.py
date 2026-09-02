"""Run host chromosome LightGBM models with Assembly_ID-blocked checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

from v4_common import (
    SEEDS,
    build_classifier,
    build_regressor,
    classification_metrics,
    continuous_score,
    f1_threshold,
    load_primary_data,
    regression_metrics,
    write_json,
)


def load_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "params" not in payload:
        raise ValueError(f"Missing params in {path}")
    return payload["params"]


def group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(groups)), groups=groups
        )
    )
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise ValueError("Assembly_ID leakage across host train and test")
    return train_idx, test_idx


def validation_split(train_idx: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    inner_train_rel, validation_rel = group_split(groups[train_idx], seed)
    return train_idx[inner_train_rel], train_idx[validation_rel]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    return detail.groupby("task", as_index=False).agg(
        run_n=("run", "nunique"),
        AUPRC_mean=("AUPRC", "mean"),
        AUPRC_SD=("AUPRC", "std"),
        AUROC_mean=("AUROC", "mean"),
        AUROC_SD=("AUROC", "std"),
        F1_mean=("F1", "mean"),
        F1_SD=("F1", "std"),
        R2_mean=("R2", "mean"),
        R2_SD=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_SD=("RMSE", "std"),
        Spearman_rho_mean=("Spearman_rho", "mean"),
        Spearman_rho_SD=("Spearman_rho", "std"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--host-matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-params-json", type=Path, required=True)
    parser.add_argument("--regressor-params-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    _, labels, _ = load_primary_data(args.matrix_dir)
    samples = pd.read_csv(args.matrix_dir / "sample_ids.csv", keep_default_na=False)
    if not samples["Sample_ID"].astype(str).equals(labels["Sample_ID"].astype(str)):
        raise ValueError("sample_ids and labels are not row-aligned")
    host_x = sparse.load_npz(
        args.host_matrix_dir / "X_plasmids_by_host_chromosome_codes_no_tRNA.npz"
    ).tocsr().astype(np.float32)
    if host_x.shape[0] != len(labels):
        raise ValueError("Host matrix and labels are not row-aligned")

    host_qc = pd.read_csv(args.host_matrix_dir / "host_chromosome_matrix_qc.csv", keep_default_na=False)
    missing_hosts = set(
        host_qc.loc[
            pd.to_numeric(host_qc["non_plasmid_record_n"], errors="raise").eq(0),
            "Assembly_ID",
        ].astype(str)
    )
    keep = ~samples["Assembly_ID"].astype(str).isin(missing_hosts)
    samples.loc[~keep, ["Sample_ID", "Assembly_ID", "Replicon_ID"]].to_csv(
        args.out_dir / "excluded_plasmids_without_host_chromosome.csv", index=False
    )
    host_x = host_x[keep.to_numpy()].tocsr()
    samples = samples.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)

    groups = samples["Assembly_ID"].astype(str).to_numpy()
    y_class = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    y_count = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    class_params = load_params(args.classifier_params_json)
    reg_params = load_params(args.regressor_params_json)

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    for run, seed in enumerate(SEEDS, start=1):
        metrics_path = checkpoints / f"run{run}.metrics.csv"
        predictions_path = checkpoints / f"run{run}.predictions.csv"
        split_path = checkpoints / f"run{run}.split.csv"
        if metrics_path.exists() and predictions_path.exists() and split_path.exists():
            metric_frames.append(pd.read_csv(metrics_path))
            prediction_frames.append(pd.read_csv(predictions_path))
            split_frames.append(pd.read_csv(split_path))
            print(f"host chromosome run {run}/3 loaded", flush=True)
            continue
        if any(path.exists() for path in (metrics_path, predictions_path, split_path)):
            raise RuntimeError(f"Incomplete checkpoint for run {run}")

        train_idx, test_idx = group_split(groups, seed)
        inner_train_idx, validation_idx = validation_split(train_idx, groups, seed)
        validation_model = build_classifier(
            "LightGBM", class_params, seed, y_class[inner_train_idx], args.n_jobs
        )
        validation_model.fit(host_x[inner_train_idx], y_class[inner_train_idx])
        validation_score = continuous_score(validation_model, host_x[validation_idx])
        threshold = f1_threshold(y_class[validation_idx], validation_score)

        classifier = build_classifier("LightGBM", class_params, seed, y_class[train_idx], args.n_jobs)
        classifier.fit(host_x[train_idx], y_class[train_idx])
        test_score = continuous_score(classifier, host_x[test_idx])
        positive_train = train_idx[y_class[train_idx] == 1]
        positive_test = test_idx[y_class[test_idx] == 1]
        regressor = build_regressor(reg_params, seed, args.n_jobs)
        regressor.fit(host_x[positive_train], y_count[positive_train])
        predicted_count = regressor.predict(host_x[positive_test])

        metric_frame = pd.DataFrame(
            [
                {
                    "task": "classification",
                    "feature_source": "host_chromosome",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_assembly_n": int(pd.Series(groups[train_idx]).nunique()),
                    "test_assembly_n": int(pd.Series(groups[test_idx]).nunique()),
                    "test_positive_n": int(y_class[test_idx].sum()),
                    "test_positive_prevalence": float(y_class[test_idx].mean()),
                    "F1_threshold_from_inner_group_validation": threshold,
                    **classification_metrics(y_class[test_idx], test_score, threshold),
                },
                {
                    "task": "regression_tRNA_positive_only",
                    "feature_source": "host_chromosome",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(positive_train)),
                    "test_n": int(len(positive_test)),
                    **regression_metrics(y_count[positive_test], predicted_count),
                },
            ]
        )
        prediction_frame = samples.iloc[test_idx][["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        prediction_frame["run"] = run
        prediction_frame["seed"] = seed
        prediction_frame["y_true_has_tRNA"] = y_class[test_idx]
        prediction_frame["host_chromosome_classification_score"] = test_score
        prediction_frame["host_chromosome_classification_threshold"] = threshold
        prediction_frame["y_true_tRNA_count"] = y_count[test_idx]
        prediction_frame["host_chromosome_regression_prediction"] = np.nan
        prediction_frame.loc[prediction_frame["y_true_has_tRNA"].eq(1), "host_chromosome_regression_prediction"] = predicted_count
        split_frame = samples[["Sample_ID", "Assembly_ID", "Replicon_ID"]].copy()
        split_frame["run"] = run
        split_frame["seed"] = seed
        split_frame["split"] = "train"
        split_frame.loc[test_idx, "split"] = "test"

        metric_frame.to_csv(metrics_path, index=False)
        prediction_frame.to_csv(predictions_path, index=False)
        split_frame.to_csv(split_path, index=False)
        metric_frames.append(metric_frame)
        prediction_frames.append(prediction_frame)
        split_frames.append(split_frame)
        print(f"host chromosome run {run}/3 complete", flush=True)

    detail = pd.concat(metric_frames, ignore_index=True)
    if not detail.groupby("task")["run"].nunique().eq(3).all():
        raise RuntimeError("Host chromosome analysis did not complete all three seeds")
    detail.to_csv(args.out_dir / "host_chromosome_metrics_each_run.csv", index=False)
    summarize(detail).to_csv(args.out_dir / "host_chromosome_metrics_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.out_dir / "host_chromosome_predictions.csv", index=False
    )
    pd.concat(split_frames, ignore_index=True).to_csv(
        args.out_dir / "host_chromosome_split_manifest.csv", index=False
    )
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "seeds": list(SEEDS),
            "split": "three approximately 80:20 Assembly_ID-group-constrained holdouts",
            "feature_source": "host_chromosome",
            "host_vector": "sum of all non-plasmid replicon product counts per Assembly_ID",
            "regression_cohort": "tRNA-positive target plasmids only",
            "classifier_params_json": str(args.classifier_params_json),
            "regressor_params_json": str(args.regressor_params_json),
            "hyperparameter_tuning": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
