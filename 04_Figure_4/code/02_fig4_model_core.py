import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import AbstractSet, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


REQUIRED_FIG2_PARAMS = frozenset(
    {
        "boosting_type",
        "objective",
        "metric",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "colsample_bytree",
        "subsample",
        "subsample_freq",
        "min_child_samples",
        "lambda_l1",
        "lambda_l2",
        "scale_pos_weight",
        "force_col_wise",
        "n_jobs",
        "verbose",
    }
)
INTEGER_FIG2_PARAMS = (
    "n_estimators",
    "num_leaves",
    "max_depth",
    "subsample_freq",
    "min_child_samples",
    "n_jobs",
    "verbose",
)
NUMBER_FIG2_PARAMS = (
    "learning_rate",
    "colsample_bytree",
    "subsample",
    "lambda_l1",
    "lambda_l2",
    "scale_pos_weight",
)
SUMMARY_METRICS = (
    "AUPRC",
    "AUROC",
    "R2",
    "RMSE",
    "MAE",
    "positive_rate_test",
)
IDENTITY_COLUMNS = ("analysis", "target", "condition")
EACH_RUN_COLUMNS = (
    *IDENTITY_COLUMNS,
    "run",
    "seed",
    "train_n",
    "test_n",
    "positive_rate_test",
    "AUPRC",
    "AUROC",
    "R2",
    "RMSE",
    "MAE",
)
PREDICTION_COLUMNS = (
    *IDENTITY_COLUMNS,
    "run",
    "seed",
    "row_index",
    "target_presence",
    "target_count",
    "classification_score",
    "regression_prediction",
)
EACH_RUN_INTEGER_COLUMNS = ("run", "seed", "train_n", "test_n")
EACH_RUN_FLOAT_COLUMNS = (
    "positive_rate_test",
    "AUPRC",
    "AUROC",
    "R2",
    "RMSE",
    "MAE",
)
PREDICTION_INTEGER_COLUMNS = ("run", "seed", "row_index", "target_presence")
PREDICTION_FLOAT_COLUMNS = (
    "target_count",
    "classification_score",
    "regression_prediction",
)
RESERVED_OUTPUT_COLUMNS = frozenset(EACH_RUN_COLUMNS + PREDICTION_COLUMNS)
DEFAULT_PREDICTION_BATCH_SIZE = 4096


@dataclass(frozen=True)
class SplitSpec:
    run: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.run, bool)
            or not isinstance(self.run, Integral)
            or self.run <= 0
        ):
            raise ValueError("run must be a positive non-bool integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative non-bool integer")

        train_input = np.asarray(self.train_idx)
        test_input = np.asarray(self.test_idx)
        for name, values in (
            ("train_idx", train_input),
            ("test_idx", test_input),
        ):
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(
                    f"{name} must be a one-dimensional non-boolean integer array"
                )

        train_source = np.ascontiguousarray(train_input, dtype=int)
        test_source = np.ascontiguousarray(test_input, dtype=int)
        train_idx = np.frombuffer(
            train_source.tobytes(), dtype=train_source.dtype
        ).reshape(train_source.shape)
        test_idx = np.frombuffer(
            test_source.tobytes(), dtype=test_source.dtype
        ).reshape(test_source.shape)
        object.__setattr__(self, "run", int(self.run))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "train_idx", train_idx)
        object.__setattr__(self, "test_idx", test_idx)


def _binary_target(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values)
    if target.ndim != 1 or target.dtype.kind not in "biuf":
        raise ValueError("binary target must be one-dimensional and contain both 0 and 1")
    if not np.isfinite(target).all():
        raise ValueError("binary target must be one-dimensional and contain both 0 and 1")
    if not np.array_equal(np.unique(target), np.array([0, 1])):
        raise ValueError("binary target must be one-dimensional and contain both 0 and 1")
    return target


def _validate_fig2_params(params: dict[str, object]) -> None:
    missing = REQUIRED_FIG2_PARAMS.difference(params)
    if missing:
        raise ValueError(f"Fig2 params missing required keys: {', '.join(sorted(missing))}")

    for key in ("boosting_type", "objective", "metric"):
        if not isinstance(params[key], str) or not params[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(params["force_col_wise"], bool):
        raise ValueError("force_col_wise must be a bool")

    for key in INTEGER_FIG2_PARAMS:
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    for key in NUMBER_FIG2_PARAMS:
        value = params[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")

    range_checks = {
        "n_estimators": params["n_estimators"] > 0,
        "learning_rate": params["learning_rate"] > 0,
        "num_leaves": params["num_leaves"] >= 2,
        "colsample_bytree": 0 < params["colsample_bytree"] <= 1,
        "subsample": 0 < params["subsample"] <= 1,
        "subsample_freq": params["subsample_freq"] >= 0,
        "min_child_samples": params["min_child_samples"] >= 1,
        "lambda_l1": params["lambda_l1"] >= 0,
        "lambda_l2": params["lambda_l2"] >= 0,
        "scale_pos_weight": params["scale_pos_weight"] > 0,
    }
    invalid = [key for key, valid in range_checks.items() if not valid]
    if invalid:
        raise ValueError(f"Fig2 params outside valid range: {', '.join(invalid)}")


def load_fig2_params(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Fig2 parameter file must contain a JSON object")
    if payload.get("best_config") != "small_leaf_dense":
        raise ValueError('best_config must be "small_leaf_dense"')
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("Fig2 parameter file must contain a params object")
    _validate_fig2_params(params)
    return dict(params)


def model_params(
    base: dict[str, object],
    task: Literal["classification", "regression_raw_count"],
    y_train: np.ndarray,
    seed: int,
) -> dict[str, object]:
    params = dict(base)
    params["random_state"] = int(seed)

    if task == "classification":
        target = _binary_target(y_train)
        positive_n = int(np.count_nonzero(target == 1))
        negative_n = int(np.count_nonzero(target == 0))
        params.update(
            objective="binary",
            metric="aucpr",
            scale_pos_weight=negative_n / positive_n,
        )
        return params

    if task == "regression_raw_count":
        params.pop("scale_pos_weight", None)
        params.update(objective="regression", metric="rmse")
        return params

    raise ValueError(f"unsupported task: {task}")


def make_target_splits(
    y_presence: np.ndarray,
    seeds: tuple[int, ...] = (42, 43, 44),
    test_size: float = 0.2,
) -> list[SplitSpec]:
    target = _binary_target(y_presence)
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds
    ):
        raise ValueError("seeds must be nonempty, unique, non-bool integers")
    split_seeds = tuple(int(seed) for seed in seeds)
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("seeds must be nonempty, unique, non-bool integers")

    size = float(test_size)
    if not 0.0 < size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    indices = np.arange(target.size, dtype=int)
    splits = []
    for run, seed in enumerate(split_seeds, start=1):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=size,
            random_state=seed,
            stratify=target,
        )
        splits.append(
            SplitSpec(
                run=run,
                seed=seed,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )
    return splits


def sparse_row_sum(
    x: sparse.csr_matrix,
    cols: Iterable[int],
) -> np.ndarray:
    if not sparse.isspmatrix_csr(x):
        raise ValueError("x must be a CSR matrix")
    try:
        selected = tuple(cols)
    except TypeError as error:
        raise ValueError("cols must be a nonempty iterable of column indices") from error
    if not selected:
        raise ValueError("cols must not be empty")
    if any(isinstance(col, bool) or not isinstance(col, Integral) for col in selected):
        raise ValueError("cols must contain non-bool integer column indices")

    indices = np.asarray(selected, dtype=int)
    if np.any(indices < 0) or np.any(indices >= x.shape[1]):
        raise ValueError("column index is outside the feature matrix")
    return np.asarray(x[:, indices].sum(axis=1)).ravel().astype(float)


def _validate_splits(
    splits: Iterable[SplitSpec],
    n_rows: int,
    *,
    require_partition: bool,
) -> list[SplitSpec]:
    try:
        split_specs = list(splits)
    except TypeError as error:
        raise ValueError("splits must be a nonempty iterable of SplitSpec") from error
    if not split_specs:
        raise ValueError("splits must not be empty")

    seen_runs: set[int] = set()
    for split in split_specs:
        if not isinstance(split, SplitSpec):
            raise ValueError("splits must contain only SplitSpec records")
        if split.run in seen_runs:
            raise ValueError("split run values must be unique")
        seen_runs.add(split.run)

        train_idx = split.train_idx
        test_idx = split.test_idx
        if train_idx.ndim != 1 or test_idx.ndim != 1:
            raise ValueError("split indices must be one-dimensional")
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("train and test indices must both be nonempty")
        all_idx = np.concatenate((train_idx, test_idx))
        if np.any(all_idx < 0) or np.any(all_idx >= n_rows):
            raise ValueError("split index is outside the matrix row range")
        if np.unique(train_idx).size != train_idx.size:
            raise ValueError("train indices must not contain duplicates")
        if np.unique(test_idx).size != test_idx.size:
            raise ValueError("test indices must not contain duplicates")
        if np.intersect1d(train_idx, test_idx, assume_unique=True).size:
            raise ValueError("train and test indices must be disjoint")
        if require_partition and np.unique(all_idx).size != n_rows:
            raise ValueError("each split must partition every row without unused rows")
    return split_specs


def _validate_identity(identity: dict[str, object]) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise ValueError("identity must be a dictionary")
    required = set(IDENTITY_COLUMNS)
    keys = set(identity)
    collisions = (keys - required) & RESERVED_OUTPUT_COLUMNS
    if collisions:
        raise ValueError(
            f"identity keys collide with output columns: {', '.join(sorted(collisions))}"
        )
    if keys != required:
        missing = required - keys
        extra = keys - required
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra {', '.join(sorted(extra))}")
        raise ValueError(f"identity must contain exactly {IDENTITY_COLUMNS}: {'; '.join(details)}")
    normalized = {}
    for column in IDENTITY_COLUMNS:
        value = identity[column]
        if not isinstance(value, str):
            raise ValueError("identity values must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("identity values must not be blank")
        normalized[column] = stripped
    return normalized


def _predict_in_batches(
    model: object,
    x: sparse.csr_matrix,
    test_idx: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, test_idx.size, batch_size):
        batch_idx = test_idx[start : start + batch_size]
        chunks.append(
            np.asarray(model.booster_.predict(x[batch_idx]), dtype=float)
        )
    return np.concatenate(chunks)


def fit_target_condition(
    x: sparse.csr_matrix,
    y_presence: np.ndarray,
    y_count: np.ndarray,
    splits: list[SplitSpec],
    base_params: dict[str, object],
    identity: dict[str, object],
    *,
    prediction_sink: Callable[[pd.DataFrame], None],
    prediction_batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE,
) -> pd.DataFrame:
    if not sparse.isspmatrix_csr(x):
        raise ValueError("x must be a CSR matrix")
    if x.shape[1] == 0:
        raise ValueError("x must contain at least one feature")
    if x.dtype != np.dtype(np.float32):
        raise ValueError("x must have float32 dtype")
    if not callable(prediction_sink):
        raise ValueError("prediction_sink must be callable")
    if (
        isinstance(prediction_batch_size, bool)
        or not isinstance(prediction_batch_size, Integral)
        or prediction_batch_size <= 0
    ):
        raise ValueError("prediction_batch_size must be a positive non-bool integer")

    presence = _binary_target(y_presence)
    counts = np.asarray(y_count)
    if counts.ndim != 1 or counts.dtype.kind not in "biuf":
        raise ValueError("counts must be a one-dimensional numeric array")
    if x.shape[0] != presence.size or x.shape[0] != counts.size:
        raise ValueError("x, y_presence, and y_count must have the same row count")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("counts must be finite and nonnegative")
    expected_presence = (counts > 0).astype(int)
    if not np.array_equal(presence, expected_presence):
        raise ValueError("y_presence must equal (y_count > 0).astype(int)")

    identity_values = _validate_identity(identity)
    batch_size = int(prediction_batch_size)
    presence = presence.astype(np.int64, copy=False)
    counts = counts.astype(float, copy=False)
    split_specs = _validate_splits(splits, x.shape[0], require_partition=True)
    metric_rows: list[dict[str, object]] = []

    for split in split_specs:
        train_idx = split.train_idx
        test_idx = split.test_idx
        classification_params = model_params(
            base_params,
            "classification",
            presence[train_idx],
            split.seed,
        )
        x_train = x[train_idx]
        classifier = LGBMClassifier(**classification_params)
        classifier.fit(x_train, presence[train_idx])
        classification_score = _predict_in_batches(
            classifier,
            x,
            test_idx,
            batch_size,
        )
        del classifier

        presence_test = presence[test_idx]
        count_test = counts[test_idx]
        regression_params = model_params(
            base_params,
            "regression_raw_count",
            counts[train_idx],
            split.seed,
        )
        regressor = LGBMRegressor(**regression_params)
        regressor.fit(x_train, counts[train_idx])
        del x_train
        regression_prediction = np.clip(
            _predict_in_batches(
                regressor,
                x,
                test_idx,
                batch_size,
            ),
            0.0,
            None,
        )
        del regressor

        auprc = (
            float(average_precision_score(presence_test, classification_score))
            if np.any(presence_test == 1)
            else 0.0
        )
        auroc = (
            float(roc_auc_score(presence_test, classification_score))
            if np.unique(presence_test).size == 2
            else float("nan")
        )
        count_r2 = (
            float(r2_score(count_test, regression_prediction, force_finite=False))
            if count_test.size >= 2
            and not np.all(count_test == count_test[0])
            else float("nan")
        )
        metric_rows.append(
            {
                **identity_values,
                "run": int(split.run),
                "seed": int(split.seed),
                "train_n": int(train_idx.size),
                "test_n": int(test_idx.size),
                "positive_rate_test": float(np.mean(presence_test)),
                "AUPRC": auprc,
                "AUROC": auroc,
                "R2": count_r2,
                "RMSE": float(
                    math.sqrt(mean_squared_error(count_test, regression_prediction))
                ),
                "MAE": float(
                    mean_absolute_error(count_test, regression_prediction)
                ),
            }
        )
        prediction_frame = pd.DataFrame(
            {
                **identity_values,
                "run": int(split.run),
                "seed": int(split.seed),
                "row_index": test_idx.astype(np.int64, copy=False),
                "target_presence": presence_test.astype(np.int64, copy=False),
                "target_count": count_test.astype(float, copy=False),
                "classification_score": classification_score,
                "regression_prediction": regression_prediction,
            },
            columns=PREDICTION_COLUMNS,
        )
        prediction_frame = prediction_frame.astype(
            {
                **{column: "int64" for column in PREDICTION_INTEGER_COLUMNS},
                **{column: "float64" for column in PREDICTION_FLOAT_COLUMNS},
            }
        )
        prediction_sink(prediction_frame)

    each_run_df = pd.DataFrame(metric_rows, columns=EACH_RUN_COLUMNS)
    each_run_df = each_run_df.astype(
        {
            **{column: "int64" for column in EACH_RUN_INTEGER_COLUMNS},
            **{column: "float64" for column in EACH_RUN_FLOAT_COLUMNS},
        }
    )
    return each_run_df


def summarize_runs(
    each_run: pd.DataFrame,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    groups = tuple(group_cols)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("group_cols must contain unique column names")

    required = set(groups) | {"run", *SUMMARY_METRICS}
    missing = required.difference(each_run.columns)
    if missing:
        raise ValueError(f"each_run missing required columns: {', '.join(sorted(missing))}")
    if each_run["run"].isna().any():
        raise ValueError("run must not contain missing values")
    duplicate_keys = [*groups, "run"]
    if each_run.duplicated(duplicate_keys, keep=False).any():
        raise ValueError("each group must contain exactly one record per run")

    data = each_run.copy()
    data = data.sort_values([*groups, "run"], kind="mergesort", na_position="last")
    grouper: str | list[str]
    grouper = groups[0] if len(groups) == 1 else list(groups)
    rows: list[dict[str, object]] = []
    for keys, group in data.groupby(grouper, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, key_values))
        row["run_n"] = int(len(group))
        for metric in SUMMARY_METRICS:
            row[f"{metric}_n"] = int(group[metric].notna().sum())
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)

    columns = [*groups, "run_n"]
    columns.extend(
        column
        for metric in SUMMARY_METRICS
        for column in (f"{metric}_n", f"{metric}_mean", f"{metric}_sd")
    )
    return pd.DataFrame(rows, columns=columns)


def split_manifest(
    n_rows: int,
    target: str,
    splits: list[SplitSpec],
) -> pd.DataFrame:
    if isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer")
    row_count = int(n_rows)
    split_specs = _validate_splits(splits, row_count, require_partition=True)
    rows: list[pd.DataFrame] = []
    row_indices = np.arange(row_count, dtype=int)

    for split in split_specs:
        labels = np.empty(row_count, dtype=object)
        labels[split.train_idx] = "train"
        labels[split.test_idx] = "test"
        rows.append(
            pd.DataFrame(
                {
                    "target": target,
                    "run": split.run,
                    "seed": split.seed,
                    "row_index": row_indices,
                    "split": labels,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def module_feature_conditions(
    all_cols: set[int],
    module_cols: dict[str, set[int]],
    target: str,
    excluded_global: set[int],
) -> dict[str, set[int]]:
    if not all_cols:
        raise ValueError("all_cols must not be empty")
    if len(module_cols) < 2:
        raise ValueError("module_cols must contain at least two modules")
    if target not in module_cols:
        raise ValueError(f"unknown target module: {target}")
    empty_modules = [name for name, cols in module_cols.items() if not cols]
    if empty_modules:
        raise ValueError(
            f"module column sets must not be empty: {', '.join(empty_modules)}"
        )

    all_module_cols = set().union(*module_cols.values())
    referenced_cols = all_module_cols | set(excluded_global)
    if not referenced_cols.issubset(all_cols):
        raise ValueError("module and excluded columns must be present in all_cols")

    target_cols = set(module_cols[target])
    other_module_cols = set().union(
        *(cols for name, cols in module_cols.items() if name != target)
    )
    return {
        "all_non_target_features": all_cols - target_cols - excluded_global,
        "non_cde_background": all_cols - all_module_cols - excluded_global,
        "other_cde_modules_only": (
            other_module_cols - target_cols - excluded_global
        ),
    }


def subcategory_predictor_cols(
    all_cols: set[int],
    target_cols: set[int],
    adfu_cols: set[int],
) -> set[int]:
    if not target_cols:
        raise ValueError("target_cols must not be empty")
    if not (target_cols | adfu_cols).issubset(all_cols):
        raise ValueError("target and ADFU columns must be present in all_cols")
    return all_cols - target_cols - adfu_cols


def cog_target_codes(
    mapping: pd.DataFrame,
    categories: tuple[str, ...] = ("L", "K", "J"),
    excluded_codes: AbstractSet[str] = frozenset({"ADFU"}),
) -> dict[str, set[str]]:
    required_columns = {"COG_category", "Code"}
    if not required_columns.issubset(mapping.columns):
        raise ValueError("mapping must contain COG_category and Code columns")

    if not categories or any(not isinstance(category, str) for category in categories):
        raise ValueError("categories must contain nonempty strings")
    normalized_categories = tuple(category.strip().upper() for category in categories)
    if any(not category for category in normalized_categories):
        raise ValueError("categories must contain nonempty strings")
    if len(set(normalized_categories)) != len(normalized_categories):
        raise ValueError("categories must be unique after normalization")

    data = mapping.loc[:, ["COG_category", "Code"]].copy()
    data["COG_category"] = (
        data["COG_category"].astype("string").str.strip().str.upper()
    )
    data = data[data["COG_category"].isin(normalized_categories)].copy()
    data["Code"] = data["Code"].astype("string").str.strip().str.upper()
    if (data["Code"].isna() | data["Code"].eq("")).any():
        raise ValueError("Code contains missing or blank values in requested categories")

    excluded = {code.strip().upper() for code in excluded_codes}
    data = data[~data["Code"].isin(excluded)]
    return {
        category: set(data.loc[data["COG_category"].eq(category), "Code"])
        for category in normalized_categories
    }
