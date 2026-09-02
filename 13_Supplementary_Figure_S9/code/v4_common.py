"""Shared strict utilities for the V4 machine-learning analyses."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
try:
    import lightgbm
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # Optional when running SGD-only in a minimal environment.
    lightgbm = None
    LGBMClassifier = LGBMRegressor = None
try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:  # Optional when running SGD-only in a minimal environment.
    xgboost = None
    XGBClassifier = None


SEEDS = (42, 43, 44)
MODEL_ORDER = (
    "LightGBM",
    "XGBoost",
    "LogisticRegression",
    "LinearSVC",
    "RandomForest",
    "SGDClassifier",
    "ComplementNB",
)
N_CANDIDATES = 50
EXPECTED_MATRIX_ROWS = 61_961


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_primary_data(matrix_dir: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray]:
    matrix_path = matrix_dir / "X_plasmids_by_codes_no_tRNA.npz"
    labels_path = matrix_dir / "derived_labels_rebuilt.csv"
    names_path = matrix_dir / "feature_names_no_tRNA.npy"
    matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
    labels = pd.read_csv(labels_path, keep_default_na=False)
    feature_names = np.load(names_path, allow_pickle=True).astype(str)
    if matrix.shape[0] != EXPECTED_MATRIX_ROWS:
        raise ValueError(f"Expected {EXPECTED_MATRIX_ROWS} matrix rows, found {matrix.shape[0]}")
    if len(labels) != matrix.shape[0] or len(feature_names) != matrix.shape[1]:
        raise ValueError("Matrix, labels, and feature names are not aligned")
    if labels["Sample_ID"].duplicated().any():
        raise ValueError("Sample_ID must be unique")
    if matrix.data.size and float(matrix.data.min()) < 0:
        raise ValueError("The product-count matrix must be non-negative")
    return matrix, labels, feature_names


def positive_weight(y: np.ndarray) -> float:
    positive_n = int(np.sum(y))
    negative_n = int(len(y) - positive_n)
    if positive_n == 0 or negative_n == 0:
        raise ValueError("Training data must contain both classes")
    return negative_n / positive_n


def balanced_class_weight(y: np.ndarray) -> dict[int, float]:
    ratio = positive_weight(y)
    return {0: 1.0, 1: ratio}


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low <= 0 or high <= low:
        raise ValueError("Invalid log-uniform interval")
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _unique_random_candidates(draw, count: int) -> list[dict]:
    candidates: list[dict] = []
    signatures: set[str] = set()
    while len(candidates) < count:
        candidate = draw()
        signature = json.dumps(candidate, sort_keys=True, default=json_default)
        if signature not in signatures:
            signatures.add(signature)
            candidates.append(candidate)
    return candidates


def random_choice(rng: np.random.Generator, values: tuple):
    """Choose one value without NumPy coercing mixed Python types."""
    return values[int(rng.integers(0, len(values)))]


def classifier_candidates(model_name: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)

    if model_name == "LightGBM":
        return _unique_random_candidates(
            lambda: {
                "n_estimators": int(rng.choice((300, 600, 900, 1200))),
                "learning_rate": log_uniform(rng, 0.01, 0.10),
                "num_leaves": int(rng.choice((31, 63, 95, 127))),
                "max_depth": int(rng.choice((-1, 8, 12, 16))),
                "min_child_samples": int(rng.choice((5, 10, 20, 40, 80))),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "subsample_freq": 1,
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "reg_alpha": log_uniform(rng, 1e-4, 1.0),
                "reg_lambda": log_uniform(rng, 1e-4, 10.0),
            },
            N_CANDIDATES,
        )
    if model_name == "XGBoost":
        return _unique_random_candidates(
            lambda: {
                "n_estimators": int(rng.choice((300, 600, 900, 1200))),
                "learning_rate": log_uniform(rng, 0.01, 0.10),
                "max_depth": int(rng.choice((3, 5, 7, 9, 12))),
                "min_child_weight": int(rng.choice((1, 3, 5, 10))),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "gamma": float(rng.uniform(0.0, 1.0)),
                "reg_alpha": log_uniform(rng, 1e-4, 1.0),
                "reg_lambda": log_uniform(rng, 0.1, 10.0),
            },
            N_CANDIDATES,
        )
    if model_name == "LogisticRegression":
        return _unique_random_candidates(
            lambda: {
                "C": log_uniform(rng, 1e-4, 1e2),
                "penalty": str(rng.choice(("l1", "l2"))),
                "class_weight": random_choice(rng, (None, "balanced")),
            },
            N_CANDIDATES,
        )
    if model_name == "LinearSVC":
        return _unique_random_candidates(
            lambda: {
                "C": log_uniform(rng, 1e-4, 1e2),
                "loss": str(rng.choice(("hinge", "squared_hinge"))),
                "class_weight": random_choice(rng, (None, "balanced")),
            },
            N_CANDIDATES,
        )
    if model_name == "RandomForest":
        return _unique_random_candidates(
            lambda: {
                "n_estimators": int(rng.choice((300, 500, 800, 1200))),
                "max_depth": random_choice(rng, (None, 10, 20, 40, 80)),
                "max_features": random_choice(rng, ("sqrt", "log2", 0.01, 0.03, 0.05)),
                "min_samples_split": int(rng.choice((2, 5, 10, 20))),
                "min_samples_leaf": int(rng.choice((1, 2, 5, 10))),
                "class_weight": random_choice(rng, (None, "balanced")),
            },
            N_CANDIDATES,
        )
    if model_name == "SGDClassifier":
        return _unique_random_candidates(
            lambda: {
                "loss": str(rng.choice(("log_loss", "modified_huber"))),
                "alpha": log_uniform(rng, 1e-6, 1e-2),
                "penalty": str(rng.choice(("l1", "l2", "elasticnet"))),
                "l1_ratio": float(rng.uniform(0.05, 0.95)),
                "class_weight": random_choice(rng, (None, "balanced")),
            },
            N_CANDIDATES,
        )
    if model_name == "ComplementNB":
        return _unique_random_candidates(
            lambda: {
                "alpha": log_uniform(rng, 1e-3, 10.0),
                "norm": bool(rng.integers(0, 2)),
            },
            N_CANDIDATES,
        )
    raise KeyError(model_name)


def regression_candidates(seed: int) -> list[dict]:
    return classifier_candidates("LightGBM", seed)


def build_classifier(
    model_name: str,
    params: dict,
    seed: int,
    train_y: np.ndarray,
    n_jobs: int,
):
    params = dict(params)
    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError("LightGBM is required for LightGBM models")
        return LGBMClassifier(
            objective="binary",
            metric="average_precision",
            force_col_wise=True,
            verbosity=-1,
            random_state=seed,
            n_jobs=n_jobs,
            scale_pos_weight=positive_weight(train_y),
            **params,
        )
    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is required for XGBoost models")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            verbosity=0,
            random_state=seed,
            n_jobs=n_jobs,
            scale_pos_weight=positive_weight(train_y),
            **params,
        )
    if model_name == "LogisticRegression":
        dual = params["penalty"] == "l2"
        return make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(
                solver="liblinear",
                dual=dual,
                max_iter=1000,
                tol=1e-3,
                random_state=seed,
                **params,
            ),
        )
    if model_name == "LinearSVC":
        return make_pipeline(
            MaxAbsScaler(),
            LinearSVC(
                dual=True,
                max_iter=10000,
                tol=1e-3,
                random_state=seed,
                **params,
            ),
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if model_name == "SGDClassifier":
        if params["penalty"] != "elasticnet":
            params.pop("l1_ratio")
        return make_pipeline(
            MaxAbsScaler(),
            SGDClassifier(
                max_iter=5000,
                tol=1e-4,
                random_state=seed,
                **params,
            ),
        )
    if model_name == "ComplementNB":
        return ComplementNB(**params)
    raise KeyError(model_name)


def build_regressor(params: dict, seed: int, n_jobs: int) -> LGBMRegressor:
    if LGBMRegressor is None:
        raise ImportError("LightGBM is required for regression")
    return LGBMRegressor(
        objective="regression",
        metric="rmse",
        force_col_wise=True,
        verbosity=-1,
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )


def continuous_score(model, matrix: sparse.csr_matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return np.asarray(model.decision_function(matrix), dtype=float)


def f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        raise ValueError("Validation data produced no classification threshold")
    denominator = np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    f1_values = 2 * precision[:-1] * recall[:-1] / denominator
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_rank = pd.Series(np.asarray(y_true, dtype=float)).rank(method="average").to_numpy()
    pred_rank = pd.Series(np.asarray(y_pred, dtype=float)).rank(method="average").to_numpy()
    true_centered = true_rank - true_rank.mean()
    pred_centered = pred_rank - pred_rank.mean()
    denominator = float(
        np.sqrt(np.sum(true_centered * true_centered) * np.sum(pred_centered * pred_centered))
    )
    if denominator <= 0:
        raise ValueError("Spearman rank variance is zero")
    rho = float(np.sum(true_centered * pred_centered) / denominator)
    if not np.isfinite(rho):
        raise ValueError("Spearman correlation is not finite")
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "Spearman_rho": float(rho),
    }


def software_metadata() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__ if lightgbm is not None else "unavailable",
        "xgboost": xgboost.__version__ if xgboost is not None else "unavailable",
    }
