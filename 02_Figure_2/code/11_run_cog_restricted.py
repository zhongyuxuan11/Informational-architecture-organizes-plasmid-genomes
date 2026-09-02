"""Run locked total-tRNA models using each multilabel COG feature subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def split_parts(manifest: pd.DataFrame, run: int, index_column: str) -> dict[str, np.ndarray]:
    current = manifest.loc[manifest["run"].astype(int).eq(run)]
    result = {
        split: current.loc[current["split"].eq(split), index_column].to_numpy(dtype=int)
        for split in current["split"].unique()
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--cog-mapping", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    matrix, labels, feature_names = load_primary_data(args.matrix_dir)
    code_to_column = {code: index for index, code in enumerate(feature_names)}
    cog = pd.read_csv(args.cog_mapping, keep_default_na=False)
    cog = cog.loc[cog["COG_category"].astype(str).str.len().eq(1)]
    categories = sorted(cog["COG_category"].unique())
    category_columns: dict[str, np.ndarray] = {}
    for category in categories:
        codes = cog.loc[cog["COG_category"].eq(category), "Code"].drop_duplicates()
        columns = sorted({code_to_column[code] for code in codes if code in code_to_column})
        if not columns:
            raise ValueError(f"COG category {category} has no matrix columns")
        category_columns[category] = np.asarray(columns, dtype=int)

    y_class = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    y_count = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    positive_rows = np.flatnonzero(y_class == 1)
    positive_matrix = matrix[positive_rows].tocsr()
    positive_y = y_count[positive_rows]
    class_locked = pd.read_csv(
        args.classifier_results / "classifier_locked_parameters_by_run.csv",
        keep_default_na=False,
    )
    class_locked = class_locked.loc[class_locked["model"].eq("LightGBM")]
    reg_locked = pd.read_csv(
        args.regressor_results / "regressor_locked_parameters_by_run.csv",
        keep_default_na=False,
    )
    class_manifest = pd.read_csv(
        args.classifier_results / "classifier_split_manifest.csv", keep_default_na=False
    )
    reg_manifest = pd.read_csv(
        args.regressor_results / "regressor_split_manifest.csv", keep_default_na=False
    )
    if len(class_locked) != 3 or len(reg_locked) != 3:
        raise ValueError("COG analysis requires three locked parameter rows")

    rows: list[dict] = []
    for category, columns in category_columns.items():
        for run, seed in enumerate(SEEDS, start=1):
            class_parts = split_parts(class_manifest, run, "row_index")
            inner_train = class_parts["development_inner_train"]
            validation = class_parts["development_validation"]
            test = class_parts["untouched_test"]
            development = np.sort(np.concatenate((inner_train, validation)))
            class_params = json.loads(
                class_locked.loc[class_locked["run"].eq(run), "params"].iloc[0]
            )
            threshold_model = build_classifier(
                "LightGBM", class_params, seed, y_class[inner_train], args.n_jobs
            )
            threshold_model.fit(matrix[inner_train][:, columns], y_class[inner_train])
            threshold = f1_threshold(
                y_class[validation],
                continuous_score(threshold_model, matrix[validation][:, columns]),
            )
            classifier = build_classifier(
                "LightGBM", class_params, seed, y_class[development], args.n_jobs
            )
            classifier.fit(matrix[development][:, columns], y_class[development])
            score = continuous_score(classifier, matrix[test][:, columns])
            rows.append(
                {
                    "COG_category": category,
                    "feature_n": int(len(columns)),
                    "task": "classification",
                    "run": run,
                    "seed": seed,
                    "test_n": int(len(test)),
                    "test_positive_n": int(y_class[test].sum()),
                    "test_positive_prevalence": float(y_class[test].mean()),
                    "F1_threshold_from_validation": threshold,
                    **classification_metrics(y_class[test], score, threshold),
                }
            )

            reg_parts = split_parts(reg_manifest, run, "positive_subset_row_index")
            reg_development = np.sort(
                np.concatenate(
                    (
                        reg_parts["development_inner_train"],
                        reg_parts["development_validation"],
                    )
                )
            )
            reg_test = reg_parts["untouched_test"]
            reg_params = json.loads(
                reg_locked.loc[reg_locked["run"].eq(run), "params"].iloc[0]
            )
            regressor = build_regressor(reg_params, seed, args.n_jobs)
            regressor.fit(
                positive_matrix[reg_development][:, columns], positive_y[reg_development]
            )
            prediction = regressor.predict(positive_matrix[reg_test][:, columns])
            rows.append(
                {
                    "COG_category": category,
                    "feature_n": int(len(columns)),
                    "task": "regression_tRNA_positive_only",
                    "run": run,
                    "seed": seed,
                    "test_n": int(len(reg_test)),
                    **regression_metrics(positive_y[reg_test], prediction),
                }
            )
        print(f"COG-restricted category {category} complete", flush=True)

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["COG_category", "feature_n", "task"], as_index=False).agg(
        run_n=("run", "nunique"),
        positive_prevalence_mean=("test_positive_prevalence", "mean"),
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
        raise RuntimeError("A COG-restricted model is missing a run")
    detail.to_csv(args.out_dir / "cog_restricted_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "cog_restricted_metrics_summary.csv", index=False)
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "seeds": list(SEEDS),
            "multilabel_COG_assignments_retained": True,
            "shared_primary_splits": True,
            "hyperparameter_tuning": False,
            "regression_cohort": "tRNA-positive plasmids only",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

