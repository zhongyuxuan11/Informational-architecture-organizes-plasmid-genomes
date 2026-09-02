"""Run cosine 1-NN and 5-NN baselines on the primary split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from v4_common import (
    SEEDS,
    classification_metrics,
    f1_threshold,
    load_primary_data,
    regression_metrics,
    write_json,
)


def parts(manifest: pd.DataFrame, run: int, index_column: str) -> dict[str, np.ndarray]:
    current = manifest.loc[manifest["run"].astype(int).eq(run)]
    return {
        split: current.loc[current["split"].eq(split), index_column].to_numpy(dtype=int)
        for split in current["split"].unique()
    }


def neighbor_scores(train_x, train_y: np.ndarray, query_x, k: int, n_jobs: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    model.fit(train_x)
    indices = model.kneighbors(query_x, return_distance=False)
    return np.asarray(train_y[indices].mean(axis=1), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    matrix, labels, _ = load_primary_data(args.matrix_dir)
    y_class = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    y_count = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    positive_rows = np.flatnonzero(y_class == 1)
    positive_matrix = matrix[positive_rows].tocsr()
    positive_y = y_count[positive_rows]
    class_manifest = pd.read_csv(
        args.classifier_results / "classifier_split_manifest.csv", keep_default_na=False
    )
    reg_manifest = pd.read_csv(
        args.regressor_results / "regressor_split_manifest.csv", keep_default_na=False
    )
    rows: list[dict] = []
    for run, seed in enumerate(SEEDS, start=1):
        class_parts = parts(class_manifest, run, "row_index")
        inner_train = class_parts["development_inner_train"]
        validation = class_parts["development_validation"]
        development = np.sort(np.concatenate((inner_train, validation)))
        test = class_parts["untouched_test"]
        reg_parts = parts(reg_manifest, run, "positive_subset_row_index")
        reg_development = np.sort(
            np.concatenate(
                (
                    reg_parts["development_inner_train"],
                    reg_parts["development_validation"],
                )
            )
        )
        reg_test = reg_parts["untouched_test"]
        for k in (1, 5):
            validation_score = neighbor_scores(
                matrix[inner_train], y_class[inner_train], matrix[validation], k, args.n_jobs
            )
            threshold = f1_threshold(y_class[validation], validation_score)
            test_score = neighbor_scores(
                matrix[development], y_class[development], matrix[test], k, args.n_jobs
            )
            rows.append(
                {
                    "model": f"{k}-NN",
                    "task": "classification",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(development)),
                    "test_n": int(len(test)),
                    "F1_threshold_from_validation": threshold,
                    **classification_metrics(y_class[test], test_score, threshold),
                }
            )
            reg_prediction = neighbor_scores(
                positive_matrix[reg_development],
                positive_y[reg_development],
                positive_matrix[reg_test],
                k,
                args.n_jobs,
            )
            rows.append(
                {
                    "model": f"{k}-NN",
                    "task": "regression_tRNA_positive_only",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(reg_development)),
                    "test_n": int(len(reg_test)),
                    **regression_metrics(positive_y[reg_test], reg_prediction),
                }
            )
        print(f"KNN baseline run {run}/3 complete", flush=True)

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["model", "task"], as_index=False).agg(
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
    if not summary["run_n"].eq(3).all():
        raise RuntimeError("A KNN baseline is missing a run")
    detail.to_csv(args.out_dir / "knn_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "knn_metrics_summary.csv", index=False)
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "metric": "cosine distance in the non-tRNA feature matrix",
            "neighbors": [1, 5],
            "weights": "uniform",
            "shared_primary_splits": True,
            "regression_cohort": "tRNA-positive plasmids only",
            "independent_random_partitions": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

