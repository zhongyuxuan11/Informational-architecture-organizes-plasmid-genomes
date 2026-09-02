"""Recalculate primary regression metrics on PlasFlow-supported test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    ss_res = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot == 0:
        raise ValueError("Regression R2 is undefined for a constant target")
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float).copy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float).copy()
    true_rank -= float(np.mean(true_rank))
    pred_rank -= float(np.mean(pred_rank))
    rank_denominator = float(np.sqrt(np.sum(true_rank * true_rank) * np.sum(pred_rank * pred_rank)))
    if rank_denominator == 0:
        raise ValueError("Spearman rho is undefined for a constant ranked vector")
    return {
        "R2": float(1.0 - ss_res / ss_tot),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "Spearman_rho": float(np.sum(true_rank * pred_rank) / rank_denominator),
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--plasflow-assignments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    predictions = pd.read_csv(
        args.regressor_results / "regressor_test_predictions.csv",
        keep_default_na=False,
    )
    assignments = pd.read_csv(args.plasflow_assignments, sep="\t", keep_default_na=False)
    assignments["Replicon_ID"] = assignments["seq_id"].str.split("|", regex=False).str[0]
    if assignments["Replicon_ID"].duplicated().any():
        raise ValueError("PlasFlow assignments contain duplicate Replicon_ID values")
    merged = predictions.merge(
        assignments[["Replicon_ID", "plasflow_class"]],
        on="Replicon_ID",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict] = []
    for run in (1, 2, 3):
        current = merged.loc[merged["run"].eq(run)]
        if current.empty:
            raise ValueError(f"Missing regression predictions for run {run}")
        supported = current["plasflow_class"].eq("plasmid")
        for subset, mask in (
            ("complete_RefSeq_test", pd.Series(True, index=current.index)),
            ("PlasFlow_supported_plasmid_like", supported),
            ("Other_replicons_PlasFlow", ~supported),
        ):
            selected = current.loc[mask]
            if selected.empty:
                raise ValueError(f"Empty {subset} subset in run {run}")
            rows.append(
                {
                    "run": run,
                    "seed": int(selected["seed"].iloc[0]),
                    "subset": subset,
                    "n": int(len(selected)),
                    "PlasFlow_assignment_missing_n": int(selected["plasflow_class"].eq("").sum()),
                    **regression_metrics(
                        selected["y_true_tRNA_count"].to_numpy(dtype=float),
                        selected["y_pred_tRNA_count"].to_numpy(dtype=float),
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = detail.groupby("subset", as_index=False).agg(
        run_n=("run", "nunique"),
        n_mean=("n", "mean"),
        R2_mean=("R2", "mean"),
        R2_SD=("R2", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_SD=("RMSE", "std"),
        Spearman_rho_mean=("Spearman_rho", "mean"),
        Spearman_rho_SD=("Spearman_rho", "std"),
    )
    if not summary["run_n"].eq(3).all():
        raise RuntimeError("PlasFlow sensitivity requires all three primary runs")
    detail.to_csv(args.out_dir / "plasflow_regression_sensitivity_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "plasflow_regression_sensitivity_summary.csv", index=False)
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "sensitivity_step_model_refit": False,
            "source_predictions_refit": True,
            "new_split_generated": False,
            "source_predictions": str(
                args.regressor_results / "regressor_test_predictions.csv"
            ),
            "source_prediction_rule": "locked LightGBMRegressor was refitted on each run's development partition before untouched-test prediction",
            "subset_rule": "complete RefSeq test, plasflow_class equals plasmid, and all non-plasmid PlasFlow classes",
            "purpose": "annotation sensitivity for RefSeq tRNA-positive plasmid regression",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
