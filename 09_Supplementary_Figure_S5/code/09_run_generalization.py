"""Run locked random, group-blocked, size, and directional-transfer analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

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


def random_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = y if pd.Series(y).value_counts().min() >= 2 else pd.qcut(
        pd.Series(y).rank(method="first"), q=5, labels=False
    ).to_numpy()
    return next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(y)), strata
        )
    )


def group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            np.zeros(len(groups)), groups=groups
        )
    )
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise ValueError("Group leakage across train and test")
    return train_idx, test_idx


def threshold_from_train(
    matrix,
    y: np.ndarray,
    train_idx: np.ndarray,
    params: dict,
    seed: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
) -> float:
    if groups is None:
        inner_rel, validation_rel = random_split(y[train_idx], seed)
    else:
        inner_rel, validation_rel = group_split(groups[train_idx], seed)
    inner_idx = train_idx[inner_rel]
    validation_idx = train_idx[validation_rel]
    model = build_classifier("LightGBM", params, seed, y[inner_idx], n_jobs)
    model.fit(matrix[inner_idx], y[inner_idx])
    return f1_threshold(y[validation_idx], continuous_score(model, matrix[validation_idx]))


def evaluate_pair(
    experiment: str,
    split_strategy: str,
    run: int,
    seed: int,
    matrix,
    y_class: np.ndarray,
    y_count: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    class_params: dict,
    reg_params: dict,
    n_jobs: int,
    threshold_groups: np.ndarray | None = None,
) -> list[dict]:
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Train and test rows overlap")
    threshold = threshold_from_train(
        matrix, y_class, train_idx, class_params, seed, n_jobs, threshold_groups
    )
    classifier = build_classifier(
        "LightGBM", class_params, seed, y_class[train_idx], n_jobs
    )
    classifier.fit(matrix[train_idx], y_class[train_idx])
    score = continuous_score(classifier, matrix[test_idx])
    positive_train = train_idx[y_class[train_idx] == 1]
    positive_test = test_idx[y_class[test_idx] == 1]
    if len(positive_train) < 2 or len(positive_test) < 2:
        raise ValueError(f"Insufficient tRNA-positive rows for {experiment} / {split_strategy}")
    regressor = build_regressor(reg_params, seed, n_jobs)
    regressor.fit(matrix[positive_train], y_count[positive_train])
    prediction = regressor.predict(matrix[positive_test])
    common = {
        "experiment": experiment,
        "split_strategy": split_strategy,
        "run": run,
        "seed": seed,
    }
    return [
        {
            **common,
            "task": "classification",
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "test_positive_n": int(y_class[test_idx].sum()),
            "test_positive_prevalence": float(y_class[test_idx].mean()),
            "F1_threshold_from_validation": threshold,
            **classification_metrics(y_class[test_idx], score, threshold),
        },
        {
            **common,
            "task": "regression_tRNA_positive_only",
            "train_n": int(len(positive_train)),
            "test_n": int(len(positive_test)),
            **regression_metrics(y_count[positive_test], prediction),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--plasmid-metadata", type=Path, required=True)
    parser.add_argument("--classifier-results", type=Path, required=True)
    parser.add_argument("--regressor-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    matrix, labels, _ = load_primary_data(args.matrix_dir)
    taxonomy = pd.read_csv(args.taxonomy, keep_default_na=False)
    metadata = pd.read_csv(args.plasmid_metadata, keep_default_na=False)
    taxa = labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].merge(
        taxonomy[["GCF_ID", "species", "genus", "phylum"]],
        left_on="Assembly_ID",
        right_on="GCF_ID",
        how="left",
        validate="many_to_one",
    )
    lengths = labels[["Sample_ID", "Assembly_ID", "Replicon_ID"]].merge(
        metadata[["GCF_ID", "Replicon_Acc", "Length"]],
        left_on=["Assembly_ID", "Replicon_ID"],
        right_on=["GCF_ID", "Replicon_Acc"],
        how="left",
        validate="one_to_one",
    )
    if lengths["Length"].isna().any():
        raise ValueError("Plasmid length metadata is incomplete")
    y_class = pd.to_numeric(labels["has_tRNA"], errors="raise").to_numpy(dtype=int)
    y_count = pd.to_numeric(labels["tRNA_count"], errors="raise").to_numpy(dtype=float)
    assemblies = labels["Assembly_ID"].astype(str).to_numpy()
    length_values = pd.to_numeric(lengths["Length"], errors="raise").to_numpy(dtype=int)
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
        raise ValueError("Generalization analysis requires three locked parameter rows")

    rows: list[dict] = []
    exclusion_rows: list[dict] = []
    for run, seed in enumerate(SEEDS, start=1):
        class_params = json.loads(
            class_locked.loc[class_locked["run"].eq(run), "params"].iloc[0]
        )
        reg_params = json.loads(reg_locked.loc[reg_locked["run"].eq(run), "params"].iloc[0])
        train_idx, test_idx = random_split(y_class, seed)
        rows.extend(
            evaluate_pair(
                "internal_generalization",
                "random_80_20",
                run,
                seed,
                matrix,
                y_class,
                y_count,
                train_idx,
                test_idx,
                class_params,
                reg_params,
                args.n_jobs,
            )
        )
        for group_name, group_values in (
            ("Assembly_ID_blocked", assemblies),
            ("species_blocked", taxa["species"].to_numpy()),
            ("genus_blocked", taxa["genus"].to_numpy()),
            ("phylum_blocked", taxa["phylum"].to_numpy()),
        ):
            valid = pd.Series(group_values).notna() & pd.Series(group_values).astype(str).str.strip().ne("")
            valid_idx = np.flatnonzero(valid.to_numpy())
            excluded_idx = np.flatnonzero(~valid.to_numpy())
            for index in excluded_idx:
                exclusion_rows.append(
                    {
                        "split_strategy": group_name,
                        "row_index": int(index),
                        "Sample_ID": labels.iloc[index]["Sample_ID"],
                        "reason": "missing grouping value",
                    }
                )
            local_train, local_test = group_split(np.asarray(group_values)[valid_idx], seed)
            blocked_train = valid_idx[local_train]
            blocked_test = valid_idx[local_test]
            rows.extend(
                evaluate_pair(
                    "internal_generalization",
                    group_name,
                    run,
                    seed,
                    matrix,
                    y_class,
                    y_count,
                    blocked_train,
                    blocked_test,
                    class_params,
                    reg_params,
                    args.n_jobs,
                    np.asarray(group_values),
                )
            )
        for size_name, size_mask in (
            ("small_lt_100kb", length_values < 100_000),
            ("large_ge_100kb", length_values >= 100_000),
        ):
            size_idx = np.flatnonzero(size_mask)
            local_train, local_test = random_split(y_class[size_idx], seed)
            rows.extend(
                evaluate_pair(
                    "within_size",
                    size_name,
                    run,
                    seed,
                    matrix,
                    y_class,
                    y_count,
                    size_idx[local_train],
                    size_idx[local_test],
                    class_params,
                    reg_params,
                    args.n_jobs,
                )
            )
        for transfer_name, source_mask, target_mask in (
            ("small_to_large", length_values < 100_000, length_values >= 100_000),
            ("large_to_small", length_values >= 100_000, length_values < 100_000),
        ):
            source_idx = np.flatnonzero(source_mask)
            source_assemblies = set(assemblies[source_idx])
            target_idx = np.flatnonzero(target_mask & ~pd.Series(assemblies).isin(source_assemblies).to_numpy())
            if set(assemblies[source_idx]) & set(assemblies[target_idx]):
                raise ValueError("Assembly leakage in directional size transfer")
            rows.extend(
                evaluate_pair(
                    "directional_size_transfer",
                    transfer_name,
                    run,
                    seed,
                    matrix,
                    y_class,
                    y_count,
                    source_idx,
                    target_idx,
                    class_params,
                    reg_params,
                    args.n_jobs,
                    assemblies,
                )
            )
        print(f"generalization run {run}/3 complete", flush=True)

    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["experiment", "split_strategy", "task"], as_index=False
    ).agg(
        run_n=("run", "nunique"),
        test_n_mean=("test_n", "mean"),
        test_positive_n_mean=("test_positive_n", "mean"),
        test_positive_prevalence_mean=("test_positive_prevalence", "mean"),
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
        raise RuntimeError("A generalization setting is missing a run")
    detail.to_csv(args.out_dir / "generalization_metrics_each_run.csv", index=False)
    summary.to_csv(args.out_dir / "generalization_metrics_summary.csv", index=False)
    pd.DataFrame(exclusion_rows).drop_duplicates().to_csv(
        args.out_dir / "grouping_value_exclusions.csv", index=False
    )
    write_json(
        args.out_dir / "run_metadata.json",
        {
            "seeds": list(SEEDS),
            "resampling": "three approximately 80:20 holdouts",
            "directional_transfer": "fixed source and assembly-disjoint target sets; seeds vary model stochasticity only",
            "small_definition": "length < 100000 bp",
            "large_definition": "length >= 100000 bp",
            "regression_cohort": "tRNA-positive plasmids only",
            "hyperparameter_tuning": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

