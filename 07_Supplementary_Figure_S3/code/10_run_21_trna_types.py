"""Run locked classification and positive-only copy-number regression for 21 tRNA types."""

from __future__ import annotations

import argparse
import json
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
    load_primary_data,
    regression_metrics,
    write_json,
)


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = y if pd.Series(y).value_counts().min() >= 2 else pd.qcut(
        pd.Series(y).rank(method="first"), q=min(5, len(y)), labels=False
    ).to_numpy()
    return next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(y)), strata
        )
    )


def normalized_gain(model, feature_n: int, allow_zero: bool = False) -> np.ndarray | None:
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    if len(gain) != feature_n:
        raise ValueError("Invalid gain-based feature importance")
    if float(gain.sum()) <= 0:
        if allow_zero:
            return None
        raise ValueError("Invalid gain-based feature importance")
    return 100.0 * gain / gain.sum()


def type_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if np.unique(y_true).size == 1:
        return {
            "R2": np.nan,
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "Spearman_rho": np.nan,
            "regression_metric_status": "R2 and Spearman undefined: constant test target",
        }
    return {
        **regression_metrics(y_true, y_pred),
        "regression_metric_status": "defined",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    matrix, labels, feature_names = load_primary_data(args.matrix_dir)
    feature_mapping = pd.read_csv(
        args.matrix_dir / "feature_code_mapping.csv", keep_default_na=False
    ).set_index("Code").reindex(feature_names)
    if feature_mapping["Product"].isna().any():
        raise ValueError("Feature mapping is incomplete")
    target_columns = [
        column
        for column in labels.columns
        if column.startswith("tRNA-") and column != "tRNA-OTHER"
    ]
    if len(target_columns) != 21:
        raise ValueError(f"Expected 21 tRNA-type targets, found {len(target_columns)}")
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
        raise ValueError("Type-specific analysis requires three locked parameter rows")

    metrics: list[dict] = []
    fi_frames: list[pd.DataFrame] = []
    fi_status_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    for target in target_columns:
        y_count_all = pd.to_numeric(labels[target], errors="raise").to_numpy(dtype=float)
        y_presence = (y_count_all > 0).astype(int)
        if set(np.unique(y_presence)) != {0, 1}:
            raise ValueError(f"Target {target} does not contain both classes")
        positive_rows = np.flatnonzero(y_presence == 1)
        positive_y = y_count_all[positive_rows]
        for run, seed in enumerate(SEEDS, start=1):
            class_train, class_test = stratified_split(y_presence, seed)
            inner_rel, validation_rel = stratified_split(y_presence[class_train], seed)
            inner_train = class_train[inner_rel]
            validation = class_train[validation_rel]
            class_params = json.loads(
                class_locked.loc[class_locked["run"].eq(run), "params"].iloc[0]
            )
            threshold_model = build_classifier(
                "LightGBM", class_params, seed, y_presence[inner_train], args.n_jobs
            )
            threshold_model.fit(matrix[inner_train], y_presence[inner_train])
            threshold = f1_threshold(
                y_presence[validation], continuous_score(threshold_model, matrix[validation])
            )
            classifier = build_classifier(
                "LightGBM", class_params, seed, y_presence[class_train], args.n_jobs
            )
            classifier.fit(matrix[class_train], y_presence[class_train])
            class_score = continuous_score(classifier, matrix[class_test])
            metrics.append(
                {
                    "target": target,
                    "task": "classification",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(class_train)),
                    "test_n": int(len(class_test)),
                    "test_positive_n": int(y_presence[class_test].sum()),
                    "test_positive_prevalence": float(y_presence[class_test].mean()),
                    "F1_threshold_from_validation": threshold,
                    **classification_metrics(y_presence[class_test], class_score, threshold),
                }
            )
            fi_frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "task": "classification",
                        "run": run,
                        "seed": seed,
                        "Code": feature_names,
                        "Product": feature_mapping["Product"].to_numpy(),
                        "normalized_gain_pct": normalized_gain(classifier, matrix.shape[1]),
                    }
                )
            )

            reg_train_rel, reg_test_rel = stratified_split(positive_y, seed)
            reg_train = positive_rows[reg_train_rel]
            reg_test = positive_rows[reg_test_rel]
            reg_params = json.loads(
                reg_locked.loc[reg_locked["run"].eq(run), "params"].iloc[0]
            )
            regressor = build_regressor(reg_params, seed, args.n_jobs)
            regressor.fit(matrix[reg_train], y_count_all[reg_train])
            reg_prediction = regressor.predict(matrix[reg_test])
            metrics.append(
                {
                    "target": target,
                    "task": "regression_type_positive_only",
                    "run": run,
                    "seed": seed,
                    "train_n": int(len(reg_train)),
                    "test_n": int(len(reg_test)),
                    **type_regression_metrics(y_count_all[reg_test], reg_prediction),
                }
            )
            reg_gain = normalized_gain(regressor, matrix.shape[1], allow_zero=True)
            if reg_gain is None:
                fi_status_rows.append(
                    {
                        "target": target,
                        "task": "regression_type_positive_only",
                        "run": run,
                        "seed": seed,
                        "status": "undefined: fitted model produced zero total gain",
                    }
                )
            else:
                fi_frames.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "task": "regression_type_positive_only",
                            "run": run,
                            "seed": seed,
                            "Code": feature_names,
                            "Product": feature_mapping["Product"].to_numpy(),
                            "normalized_gain_pct": reg_gain,
                        }
                    )
                )
            prediction = labels.iloc[class_test][
                ["Sample_ID", "Assembly_ID", "Replicon_ID"]
            ].copy()
            prediction["target"] = target
            prediction["task"] = "classification"
            prediction["run"] = run
            prediction["seed"] = seed
            prediction["y_true"] = y_presence[class_test]
            prediction["y_score"] = class_score
            prediction_frames.append(prediction)
            reg_prediction_frame = labels.iloc[reg_test][
                ["Sample_ID", "Assembly_ID", "Replicon_ID"]
            ].copy()
            reg_prediction_frame["target"] = target
            reg_prediction_frame["task"] = "regression_type_positive_only"
            reg_prediction_frame["run"] = run
            reg_prediction_frame["seed"] = seed
            reg_prediction_frame["y_true"] = y_count_all[reg_test]
            reg_prediction_frame["y_score"] = reg_prediction
            prediction_frames.append(reg_prediction_frame)
            split_frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "task": "classification",
                        "run": run,
                        "seed": seed,
                        "row_index": np.concatenate((class_train, class_test)),
                        "split": ["train"] * len(class_train) + ["test"] * len(class_test),
                    }
                )
            )
            split_frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "task": "regression_type_positive_only",
                        "run": run,
                        "seed": seed,
                        "row_index": np.concatenate((reg_train, reg_test)),
                        "split": ["train"] * len(reg_train) + ["test"] * len(reg_test),
                    }
                )
            )
        print(f"21-type target {target} complete", flush=True)

    metric_detail = pd.DataFrame(metrics)
    summary = metric_detail.groupby(["target", "task"], as_index=False).agg(
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
        raise RuntimeError("A type-specific target is missing a run")
    fi_detail = pd.concat(fi_frames, ignore_index=True)
    top10 = (
        fi_detail.sort_values(
            ["target", "task", "run", "normalized_gain_pct"],
            ascending=[True, True, True, False],
        )
        .groupby(["target", "task", "run"], as_index=False)
        .head(10)
    )
    metric_detail.to_csv(args.out_dir / "trna_type_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "trna_type_metrics_summary.csv", index=False)
    top10.to_csv(args.out_dir / "trna_type_top10_feature_importance_each_run.csv", index=False)
    pd.DataFrame(fi_status_rows).to_csv(
        args.out_dir / "trna_type_feature_importance_undefined.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.out_dir / "trna_type_test_predictions.csv", index=False
    )
    pd.concat(split_frames, ignore_index=True).to_csv(
        args.out_dir / "trna_type_split_manifest.csv", index=False
    )
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "targets": target_columns,
            "seeds": list(SEEDS),
            "classification": "target-specific stratified three repeated 80:20 holdouts",
            "regression": "target-type-positive plasmids only with raw copy number",
            "hyperparameter_tuning": False,
            "class_weight": "recomputed from the corresponding classification training subset",
            "feature_importance": "outer-training gain normalized to 100 percent per fitted model",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
