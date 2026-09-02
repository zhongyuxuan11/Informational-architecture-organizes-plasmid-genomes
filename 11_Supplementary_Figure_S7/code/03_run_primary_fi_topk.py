"""Run locked primary LightGBM models, feature importance, and Top-K analyses."""

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
    load_primary_data,
    regression_metrics,
    write_json,
)


K_VALUES = (5, 10, 20, 50, 100, 200, 500, 1000)


def normalized_gain(model, feature_n: int) -> np.ndarray:
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    if len(gain) != feature_n:
        raise ValueError("LightGBM gain vector does not match the fitted feature count")
    total = float(gain.sum())
    if total <= 0:
        raise ValueError("LightGBM produced non-positive total gain")
    return 100.0 * gain / total


def load_locked(path: Path, model: str | None = None) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    if model is not None:
        data = data.loc[data["model"].eq(model)]
    if len(data) != 3 or set(data["run"].astype(int)) != {1, 2, 3}:
        raise ValueError(f"Expected one locked parameter row per run in {path}")
    return data.sort_values("run", kind="stable").reset_index(drop=True)


def indices_for_run(split_manifest: pd.DataFrame, run: int, index_column: str) -> tuple[np.ndarray, np.ndarray]:
    current = split_manifest.loc[split_manifest["run"].astype(int).eq(run)]
    if current.empty:
        raise ValueError(f"Missing split manifest for run {run}")
    development = current.loc[
        ~current["split"].eq("untouched_test"), index_column
    ].to_numpy(dtype=int)
    test = current.loc[
        current["split"].eq("untouched_test"), index_column
    ].to_numpy(dtype=int)
    if np.intersect1d(development, test).size:
        raise ValueError("Development and test indices overlap")
    return development, test


def category_composition(
    fi_detail: pd.DataFrame,
    category_mapping_path: Path,
    task: str,
) -> pd.DataFrame:
    mapping = pd.read_csv(category_mapping_path, keep_default_na=False)
    mapping = mapping[["Code", "Category"]].drop_duplicates()
    mapping["assignment_n"] = mapping.groupby("Code")["Category"].transform("size")
    mapping["assignment_weight"] = 1.0 / mapping["assignment_n"]
    rows: list[pd.DataFrame] = []
    for run in (1, 2, 3):
        ranked = fi_detail.loc[
            fi_detail["task"].eq(task) & fi_detail["run"].eq(run)
        ].sort_values("normalized_gain_pct", ascending=False, kind="stable")
        for k in K_VALUES:
            selected = ranked.head(k)[["Code", "normalized_gain_pct"]]
            merged = selected.merge(mapping, on="Code", how="left", validate="one_to_many")
            merged["Category"] = merged["Category"].replace("", "Others").fillna("Others")
            merged["weighted_gain"] = (
                merged["normalized_gain_pct"] * merged["assignment_weight"].fillna(1.0)
            )
            grouped = merged.groupby("Category", as_index=False)["weighted_gain"].sum()
            grouped["fraction_pct"] = 100.0 * grouped["weighted_gain"] / grouped["weighted_gain"].sum()
            grouped["task"] = task
            grouped["run"] = run
            grouped["K"] = k
            rows.append(grouped[["task", "run", "K", "Category", "fraction_pct"]])
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--category-mapping", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    matrix, labels, feature_names = load_primary_data(args.matrix_dir)
    y_class = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    y_count = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    class_locked = load_locked(
        args.classifier_results / "classifier_locked_parameters_by_run.csv",
        model="LightGBM",
    )
    reg_locked = load_locked(args.regressor_results / "regressor_locked_parameters_by_run.csv")
    class_splits = pd.read_csv(args.classifier_results / "classifier_split_manifest.csv", keep_default_na=False)
    reg_splits = pd.read_csv(args.regressor_results / "regressor_split_manifest.csv", keep_default_na=False)
    reg_positive_rows = np.flatnonzero(y_class == 1)
    reg_matrix = matrix[reg_positive_rows].tocsr()
    reg_y = y_count[reg_positive_rows]
    if np.any(reg_y <= 0):
        raise ValueError("Regression target contains non-positive values")

    fi_frames: list[pd.DataFrame] = []
    topk_rows: list[dict] = []
    nohypo_rows: list[dict] = []
    mapping = pd.read_csv(args.matrix_dir / "feature_code_mapping.csv", keep_default_na=False)
    mapping = mapping.set_index("Code").reindex(feature_names)
    if mapping["Product"].isna().any():
        raise ValueError("Feature mapping does not cover all no-tRNA matrix columns")
    hypothetical_mask = mapping["Product"].str.casefold().eq("hypothetical protein").to_numpy()
    if int(hypothetical_mask.sum()) > 1 or (int(hypothetical_mask.sum()) == 1 and feature_names[hypothetical_mask][0] != "ADFU"):
        raise ValueError("Expected zero or one ADFU hypothetical-protein feature")

    for run, seed in enumerate(SEEDS, start=1):
        class_development, class_test = indices_for_run(class_splits, run, "row_index")
        class_params = json.loads(class_locked.loc[class_locked["run"].eq(run), "params"].iloc[0])
        threshold = float(
            class_locked.loc[class_locked["run"].eq(run), "F1_threshold_from_validation"].iloc[0]
        )
        classifier = build_classifier(
            "LightGBM", class_params, seed, y_class[class_development], args.n_jobs
        )
        classifier.fit(matrix[class_development], y_class[class_development])
        class_gain = normalized_gain(classifier, matrix.shape[1])
        fi_frames.append(
            pd.DataFrame(
                {
                    "task": "classification",
                    "run": run,
                    "seed": seed,
                    "Code": feature_names,
                    "Product": mapping["Product"].to_numpy(),
                    "normalized_gain_pct": class_gain,
                }
            )
        )

        kept_columns = np.flatnonzero(~hypothetical_mask)
        nohypo_model = build_classifier(
            "LightGBM", class_params, seed, y_class[class_development], args.n_jobs
        )
        nohypo_model.fit(matrix[class_development][:, kept_columns], y_class[class_development])
        nohypo_score = continuous_score(nohypo_model, matrix[class_test][:, kept_columns])
        nohypo_rows.append(
            {
                "task": "classification",
                "run": run,
                "seed": seed,
                "feature_setting": "without_ADFU_hypothetical_protein",
                **classification_metrics(y_class[class_test], nohypo_score, threshold),
            }
        )
        class_order = np.argsort(class_gain)[::-1]
        for k in K_VALUES:
            selected_columns = class_order[:k]
            topk_model = build_classifier(
                "LightGBM", class_params, seed, y_class[class_development], args.n_jobs
            )
            topk_model.fit(matrix[class_development][:, selected_columns], y_class[class_development])
            score = continuous_score(topk_model, matrix[class_test][:, selected_columns])
            topk_rows.append(
                {
                    "task": "classification",
                    "run": run,
                    "seed": seed,
                    "K": k,
                    "rank_source": "same-run outer development only",
                    **classification_metrics(y_class[class_test], score, threshold),
                }
            )

        reg_development, reg_test = indices_for_run(
            reg_splits, run, "positive_subset_row_index"
        )
        reg_params = json.loads(reg_locked.loc[reg_locked["run"].eq(run), "params"].iloc[0])
        regressor = build_regressor(reg_params, seed, args.n_jobs)
        regressor.fit(reg_matrix[reg_development], reg_y[reg_development])
        reg_gain = normalized_gain(regressor, reg_matrix.shape[1])
        fi_frames.append(
            pd.DataFrame(
                {
                    "task": "regression",
                    "run": run,
                    "seed": seed,
                    "Code": feature_names,
                    "Product": mapping["Product"].to_numpy(),
                    "normalized_gain_pct": reg_gain,
                }
            )
        )
        nohypo_regressor = build_regressor(reg_params, seed, args.n_jobs)
        nohypo_regressor.fit(
            reg_matrix[reg_development][:, kept_columns], reg_y[reg_development]
        )
        nohypo_prediction = nohypo_regressor.predict(reg_matrix[reg_test][:, kept_columns])
        nohypo_rows.append(
            {
                "task": "regression",
                "run": run,
                "seed": seed,
                "feature_setting": "without_ADFU_hypothetical_protein",
                **regression_metrics(reg_y[reg_test], nohypo_prediction),
            }
        )
        reg_order = np.argsort(reg_gain)[::-1]
        for k in K_VALUES:
            selected_columns = reg_order[:k]
            topk_regressor = build_regressor(reg_params, seed, args.n_jobs)
            topk_regressor.fit(
                reg_matrix[reg_development][:, selected_columns], reg_y[reg_development]
            )
            prediction = topk_regressor.predict(reg_matrix[reg_test][:, selected_columns])
            topk_rows.append(
                {
                    "task": "regression",
                    "run": run,
                    "seed": seed,
                    "K": k,
                    "rank_source": "same-run outer development only",
                    **regression_metrics(reg_y[reg_test], prediction),
                }
            )
        print(f"primary/FI/Top-K run {run}/3 complete", flush=True)

    fi_detail = pd.concat(fi_frames, ignore_index=True)
    fi_summary = (
        fi_detail.groupby(["task", "Code", "Product"], as_index=False)
        .agg(
            run_n=("run", "nunique"),
            normalized_gain_mean_pct=("normalized_gain_pct", "mean"),
            normalized_gain_SD_pct=("normalized_gain_pct", "std"),
        )
        .sort_values(["task", "normalized_gain_mean_pct"], ascending=[True, False])
    )
    if not fi_summary["run_n"].eq(3).all():
        raise RuntimeError("Feature-importance summary is missing a run")
    topk_detail = pd.DataFrame(topk_rows)
    topk_summary = topk_detail.groupby(["task", "K"], as_index=False).agg(
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
    if not topk_summary["run_n"].eq(3).all():
        raise RuntimeError("Top-K summary is missing a run")
    composition = pd.concat(
        [
            category_composition(fi_detail, args.category_mapping, "classification"),
            category_composition(fi_detail, args.category_mapping, "regression"),
        ],
        ignore_index=True,
    )

    fi_detail.to_csv(args.out_dir / "feature_importance_each_run.csv", index=False)
    fi_summary.to_csv(args.out_dir / "feature_importance_summary.csv", index=False)
    topk_detail.to_csv(args.out_dir / "topk_metrics_each_run.csv", index=False)
    topk_summary.to_csv(args.out_dir / "topk_metrics_summary.csv", index=False)
    composition.to_csv(args.out_dir / "topk_category_composition_each_run.csv", index=False)
    pd.DataFrame(nohypo_rows).to_csv(
        args.out_dir / "no_hypothetical_ADFU_metrics_each_run.csv", index=False
    )
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "seeds": list(SEEDS),
            "locked_classifier_parameters": str(
                args.classifier_results / "classifier_locked_parameters_by_run.csv"
            ),
            "locked_regressor_parameters": str(
                args.regressor_results / "regressor_locked_parameters_by_run.csv"
            ),
            "feature_importance": "gain normalized to 100 percent within each run",
            "feature_importance_data": "corresponding outer development partition only",
            "topk_rank_source": "corresponding outer development partition only",
            "topk_test_role": "evaluation only",
            "hypothetical_feature_removed": "ADFU CDS|hypothetical protein",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
