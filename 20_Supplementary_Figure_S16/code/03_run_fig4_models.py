#!/usr/bin/env python
"""Unified Fig4 orchestration for real-data LightGBM runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

if __package__:
    from . import fig4_model_core as core
else:
    script_dir = str(Path(__file__).resolve().parent)
    if not sys.path or sys.path[0] != script_dir:
        sys.path.insert(0, script_dir)
    import fig4_model_core as core


MODULE_ORDER = ("Replication", "Transcription", "Translation")
COG_ORDER = ("L", "K", "J")
HYPOTHETICAL_PRODUCT = "hypothetical protein"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PACKAGE_ROOT / "supplementary_tables" / "00_preprocessing"
DEFAULT_MATRIX_DIR = PREPROCESSING_DIR
DEFAULT_CDE_PATH = PACKAGE_ROOT / "supplementary_tables" / "04_fig4_figS8" / "cde_mapping_audit" / "CDE_product_to_code_matched_detail_latest.csv"
DEFAULT_COG_PATH = PREPROCESSING_DIR / "COG_multilabel_noS_long_code_product.csv"
DEFAULT_PARAMS_PATH = PREPROCESSING_DIR / "best_random_lightgbm_params.json"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "generated_fig4_run"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_PREDICTION_BATCH_SIZE = 4096
CONDITION_ORDER = (
    "all_non_target_features",
    "non_cde_background",
    "other_cde_modules_only",
)
COG_TO_MODULE = {"L": "Replication", "K": "Transcription", "J": "Translation"}
PREDICTION_COLUMNS = [
    *core.PREDICTION_COLUMNS,
    "module",
    "subcategory",
]
EACH_RUN_OUTPUT_COLUMNS = [
    *core.EACH_RUN_COLUMNS,
    "module",
    "subcategory",
]
SUMMARY_OUTPUT_COLUMNS = [
    "analysis",
    "target",
    "condition",
    "module",
    "subcategory",
    "run_n",
    "AUPRC_n",
    "AUPRC_mean",
    "AUPRC_sd",
    "AUROC_n",
    "AUROC_mean",
    "AUROC_sd",
    "R2_n",
    "R2_mean",
    "R2_sd",
    "RMSE_n",
    "RMSE_mean",
    "RMSE_sd",
    "MAE_n",
    "MAE_mean",
    "MAE_sd",
    "positive_rate_test_n",
    "positive_rate_test_mean",
    "positive_rate_test_sd",
]
FEATURE_MANIFEST_COLUMNS = [
    "analysis",
    "target",
    "condition",
    "feature_col",
    "Code",
    "included_in_predictors",
    "is_target_code",
    "is_adfu",
]
TARGET_CODE_QC_COLUMNS = [
    "analysis",
    "target",
    "module",
    "subcategory",
    "feature_col",
    "Code",
    "included_in_target",
    "removed_from_predictors",
    "overlap_categories",
    "exclusion_reason",
]
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


class MatrixBundle:
    def __init__(
        self,
        x: sparse.csr_matrix,
        feature_names: np.ndarray,
        mapping: pd.DataFrame,
        samples: pd.DataFrame,
        adfu_cols: set[int],
    ) -> None:
        self.x = x
        self.feature_names = feature_names
        self.mapping = mapping
        self.samples = samples
        self.adfu_cols = adfu_cols


class CdeTargets:
    def __init__(
        self,
        module_sets: dict[str, set[int]],
        subcategory_specs: list[dict[str, object]],
        qc: dict[str, object],
    ) -> None:
        self.module_sets = module_sets
        self.subcategory_specs = subcategory_specs
        self.qc = qc


class CogTargets:
    def __init__(
        self,
        category_to_codes: dict[str, set[str]],
        category_to_cols: dict[str, set[int]],
        source_categories_by_code: dict[str, list[str]],
        qc: dict[str, object],
    ) -> None:
        self.category_to_codes = category_to_codes
        self.category_to_cols = category_to_cols
        self.source_categories_by_code = source_categories_by_code
        self.qc = qc


class Logger:
    def __init__(self, canonical_path: Path, mirror_path: Path | None) -> None:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        self._canonical_handle = canonical_path.open("w", encoding="utf-8")
        self._mirror_handle = None
        if mirror_path is not None:
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            self._mirror_handle = mirror_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._canonical_handle.write(message + "\n")
        self._canonical_handle.flush()
        if self._mirror_handle is not None:
            self._mirror_handle.write(message + "\n")
            self._mirror_handle.flush()

    def close(self) -> None:
        self._canonical_handle.close()
        if self._mirror_handle is not None:
            self._mirror_handle.close()


class StreamingCsvWriter:
    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError(f"Refusing to reuse existing CSV: {self.path}")
        self._header_written = False
        self.row_count = 0

    def append(self, frame: pd.DataFrame) -> None:
        _require_exact_schema(frame, self.columns, str(self.path))
        frame.to_csv(
            self.path,
            mode="a" if self._header_written else "x",
            index=False,
            header=not self._header_written,
            encoding="utf-8",
        )
        self._header_written = True
        self.row_count += int(len(frame))


def _require_exact_schema(
    frame: pd.DataFrame,
    expected_columns: list[str] | tuple[str, ...],
    context: str,
) -> None:
    expected = list(expected_columns)
    actual = frame.columns.tolist()
    if actual != expected:
        raise ValueError(
            f"{context} schema mismatch: expected {expected}, got {actual}"
        )


class FeatureManifestTracker:
    def __init__(self, writer: StreamingCsvWriter, feature_names: np.ndarray, adfu_cols: set[int]) -> None:
        self.writer = writer
        self.feature_names = feature_names.astype(str)
        self.adfu_cols = set(adfu_cols)
        self.feature_n = int(len(self.feature_names))
        self.combo_counts: dict[tuple[str, str, str], int] = {}
        self.combo_target_counts: dict[tuple[str, str, str], int] = {}
        self.combo_adfu_counts: dict[tuple[str, str, str], int] = {}

    def append(
        self,
        *,
        analysis: str,
        target: str,
        condition: str,
        predictor_cols: set[int],
        target_cols: set[int],
    ) -> None:
        rows = []
        combo = (analysis, target, condition)
        target_count = 0
        adfu_count = 0
        for feature_col, code in enumerate(self.feature_names.tolist()):
            is_target = feature_col in target_cols
            is_adfu = feature_col in self.adfu_cols
            included = feature_col in predictor_cols
            if included and is_target:
                raise ValueError(
                    f"Target leakage detected in feature manifest for {analysis} / {target} / {condition}"
                )
            target_count += int(is_target)
            adfu_count += int(is_adfu)
            rows.append(
                {
                    "analysis": analysis,
                    "target": target,
                    "condition": condition,
                    "feature_col": feature_col,
                    "Code": code,
                    "included_in_predictors": included,
                    "is_target_code": is_target,
                    "is_adfu": is_adfu,
                }
            )
        frame = pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)
        self.writer.append(frame)
        self.combo_counts[combo] = self.combo_counts.get(combo, 0) + len(rows)
        self.combo_target_counts[combo] = target_count
        self.combo_adfu_counts[combo] = adfu_count

    def validate(self) -> dict[str, object]:
        expected_total = len(self.combo_counts) * self.feature_n
        if self.writer.row_count != expected_total:
            raise ValueError("feature_exclusion_manifest row count mismatch")
        for combo, count in self.combo_counts.items():
            if count != self.feature_n:
                raise ValueError(f"feature_exclusion_manifest combo row count mismatch for {combo}")
            if self.combo_target_counts[combo] < 1:
                raise ValueError(f"feature_exclusion_manifest missing target rows for {combo}")
            if self.combo_adfu_counts[combo] != len(self.adfu_cols):
                raise ValueError(f"feature_exclusion_manifest ADFU count mismatch for {combo}")
        return {
            "row_count": self.writer.row_count,
            "combo_n": len(self.combo_counts),
            "feature_n": self.feature_n,
        }


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _norm_module(value: object) -> str:
    key = _clean_text(value).casefold()
    mapping = {
        "replication": "Replication",
        "transcription": "Transcription",
        "translation": "Translation",
    }
    return mapping.get(key, _clean_text(value))


def _slugify(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("target label cannot be empty")
    chars = []
    for char in stripped:
        chars.append(char if char.isalnum() else "_")
    base = "".join(chars).strip("_")
    while "__" in base:
        base = base.replace("__", "_")
    if not base:
        raise ValueError("target label cannot be empty")
    if base.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("target label resolves to reserved Windows name")
    digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:12]
    if len(base) <= 80:
        return base
    prefix = base[:67].rstrip("_")
    if not prefix:
        raise ValueError("target label cannot be safely slugged")
    slug = f"{prefix}_{digest}"
    if len(slug) > 80:
        raise ValueError("target label slug exceeds length limit")
    if slug.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError("target label resolves to reserved Windows name")
    return slug


def _read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _has_complete_published_results(out_dir: Path) -> bool:
    metadata_path = out_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata.json must contain a JSON object")
    if metadata.get("STATUS") != "COMPLETE":
        return False
    inventory = metadata.get("output_inventory")
    if not isinstance(inventory, list) or not inventory or any(
        not isinstance(item, str) or not item for item in inventory
    ):
        raise ValueError("COMPLETE run_metadata.json has invalid output_inventory")
    published_paths = [
        out_dir / item
        for item in inventory
        if item not in {"run.log", "run_metadata.json"}
    ]
    return bool(published_paths) and all(path.is_file() for path in published_paths)


def _ensure_unique(values: list[str], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _feature_lookup(feature_names: np.ndarray) -> dict[str, int]:
    values = [str(value) for value in feature_names.tolist()]
    _ensure_unique(values, name="feature_names")
    return {value: idx for idx, value in enumerate(values)}


def _parse_row_index_exact(value: str) -> int:
    text = _clean_text(value)
    if text == "" or text.casefold() in {"true", "false"}:
        raise ValueError("RowIndex must equal 0..n_rows-1 in file order")
    try:
        parsed = int(text, 10)
    except ValueError as error:
        raise ValueError("RowIndex must equal 0..n_rows-1 in file order") from error
    if str(parsed) != text:
        raise ValueError("RowIndex must equal 0..n_rows-1 in file order")
    return parsed


def _validate_samples(samples: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    required = {"Sample_ID", "RowIndex", "Assembly_ID", "Replicon_ID"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"sample_ids.csv missing required columns: {', '.join(sorted(missing))}")
    validated = samples.loc[:, ["Sample_ID", "RowIndex", "Assembly_ID", "Replicon_ID"]].copy()
    validated["Sample_ID"] = validated["Sample_ID"].map(_clean_text)
    if validated["Sample_ID"].eq("").any() or validated["Sample_ID"].duplicated().any():
        raise ValueError("Sample_ID must be nonempty and unique")
    row_index_values = [_parse_row_index_exact(value) for value in validated["RowIndex"].tolist()]
    if row_index_values != list(range(n_rows)):
        raise ValueError("RowIndex must equal 0..n_rows-1 in file order")
    validated["RowIndex"] = [str(value) for value in row_index_values]
    return validated


def load_matrix_bundle(matrix_dir: Path) -> MatrixBundle:
    x = sparse.load_npz(matrix_dir / "X_plasmids_by_codes.npz").tocsr().astype(np.float32)
    feature_names = np.load(matrix_dir / "feature_names.npy", allow_pickle=True).astype(str)
    mapping = _read_csv_str(matrix_dir / "feature_code_mapping.csv")
    samples = _read_csv_str(matrix_dir / "sample_ids.csv")
    required_mapping_cols = {"Code", "Product"}
    missing_mapping = required_mapping_cols.difference(mapping.columns)
    if missing_mapping:
        raise ValueError(
            f"feature_code_mapping.csv missing required columns: {', '.join(sorted(missing_mapping))}"
        )
    if feature_names.ndim != 1:
        raise ValueError("feature_names must be one-dimensional")
    feature_name_values = feature_names.astype(str).tolist()
    _ensure_unique(feature_name_values, name="feature_names")
    if x.shape[1] != len(feature_name_values):
        raise ValueError("feature_names length must match matrix columns")
    if mapping.shape[0] != x.shape[1]:
        raise ValueError("feature_code_mapping.csv rows must match matrix columns")
    if samples.shape[0] != x.shape[0]:
        raise ValueError("sample_ids.csv rows must match matrix rows")
    samples = _validate_samples(samples, x.shape[0])
    mapping = mapping.reset_index(drop=True).copy()
    mapping["Code"] = mapping["Code"].map(_clean_text)
    mapping["Product"] = mapping["Product"].map(_clean_text)
    if mapping["Code"].tolist() != feature_name_values:
        raise ValueError("feature_code_mapping.csv Code column must align with feature_names")
    adfu_cols = {
        int(idx)
        for idx, row in mapping.iterrows()
        if row["Code"].upper() == "ADFU"
        and row["Product"].casefold() == HYPOTHETICAL_PRODUCT.casefold()
    }
    return MatrixBundle(x=x, feature_names=feature_names, mapping=mapping, samples=samples, adfu_cols=adfu_cols)


def load_cde_targets(cde_table: pd.DataFrame, feature_names: np.ndarray) -> CdeTargets:
    required = {"module", "subcategory", "matched_matrix_col", "matched_code"}
    missing = required.difference(cde_table.columns)
    if missing:
        raise ValueError(f"CDE target table missing required columns: {', '.join(sorted(missing))}")
    data = cde_table.loc[:, ["module", "subcategory", "matched_matrix_col", "matched_code"]].copy()
    data["module"] = data["module"].map(_norm_module)
    data["subcategory"] = data["subcategory"].map(_clean_text)
    data["matched_code"] = data["matched_code"].map(_clean_text)
    data["matched_matrix_col"] = pd.to_numeric(data["matched_matrix_col"], errors="coerce")
    if data["matched_matrix_col"].isna().any():
        raise ValueError("matched_matrix_col must be an integer for every CDE target row")
    data["matched_matrix_col"] = data["matched_matrix_col"].astype(int)
    if (data["matched_matrix_col"] < 0).any() or (data["matched_matrix_col"] >= len(feature_names)).any():
        raise ValueError("matched_matrix_col is outside the feature matrix range")
    expected_codes = feature_names[data["matched_matrix_col"].to_numpy(dtype=int)].astype(str)
    if not np.array_equal(expected_codes, data["matched_code"].to_numpy(dtype=str)):
        raise ValueError("matched_code does not match feature_names at matched_matrix_col")
    modules = tuple(sorted(data["module"].unique().tolist(), key=MODULE_ORDER.index))
    if modules != MODULE_ORDER:
        raise ValueError("CDE targets must contain exactly Replication, Transcription, and Translation")
    pair_frame = data.loc[:, ["module", "subcategory"]].drop_duplicates(ignore_index=True)
    if len(pair_frame) != 14:
        raise ValueError("CDE targets must contain exactly 14 unique module/subcategory pairs")
    module_sets: dict[str, set[int]] = {}
    for module_name in MODULE_ORDER:
        cols = set(data.loc[data["module"].eq(module_name), "matched_matrix_col"].astype(int).tolist())
        if not cols:
            raise ValueError(f"CDE module {module_name} has no matched columns")
        module_sets[module_name] = cols
    subcategory_specs: list[dict[str, object]] = []
    for pair in pair_frame.sort_values(["module", "subcategory"], kind="mergesort").to_dict("records"):
        pair_rows = data[
            data["module"].eq(pair["module"]) & data["subcategory"].eq(pair["subcategory"])
        ].copy()
        cols = set(pair_rows["matched_matrix_col"].astype(int).tolist())
        if not cols:
            raise ValueError(f"CDE subcategory {pair['module']} | {pair['subcategory']} has no matched columns")
        subcategory_specs.append(
            {
                "module": pair["module"],
                "subcategory": pair["subcategory"],
                "target": f"{pair['module']} | {pair['subcategory']}",
                "cols": cols,
                "codes": sorted(pair_rows["matched_code"].astype(str).unique().tolist()),
            }
        )
    qc = {
        "module_counts": {module_name: len(module_sets[module_name]) for module_name in MODULE_ORDER},
        "subcategory_pair_n": int(len(subcategory_specs)),
    }
    return CdeTargets(module_sets=module_sets, subcategory_specs=subcategory_specs, qc=qc)


def load_cog_targets(
    cog_table: pd.DataFrame,
    feature_names: np.ndarray,
    adfu_code: str = "ADFU",
) -> CogTargets:
    required = {"COG_category", "Code"}
    missing = required.difference(cog_table.columns)
    if missing:
        raise ValueError(f"COG target table missing required columns: {', '.join(sorted(missing))}")
    lookup = _feature_lookup(feature_names)
    filtered = cog_table.loc[:, ["COG_category", "Code"]].copy()
    filtered["COG_category"] = filtered["COG_category"].map(_clean_text)
    filtered["Code"] = filtered["Code"].map(_clean_text)
    normalized = filtered.copy()
    normalized["COG_category"] = normalized["COG_category"].str.upper()
    normalized["Code"] = normalized["Code"].str.upper()
    normalized = normalized[normalized["COG_category"].isin(COG_ORDER)]
    source_categories_by_code = {
        str(code): sorted(set(group["COG_category"].tolist()))
        for code, group in normalized.groupby("Code", sort=True)
    }
    category_to_codes = core.cog_target_codes(
        filtered,
        categories=COG_ORDER,
        excluded_codes=frozenset({adfu_code}),
    )
    category_to_cols: dict[str, set[int]] = {}
    included_codes: dict[str, list[str]] = {}
    overlap_categories: dict[str, list[str]] = {}
    for category in COG_ORDER:
        codes = sorted(code for code in category_to_codes[category] if code in lookup)
        if not codes:
            raise ValueError(f"COG target category {category} has no codes present in feature_names")
        category_to_cols[category] = {lookup[code] for code in codes}
        included_codes[category] = codes
        for code in codes:
            overlap_categories.setdefault(code, []).append(category)
    overlap_categories = {
        code: sorted(cats) for code, cats in sorted(overlap_categories.items()) if len(cats) > 1
    }
    qc = {
        "overlap_categories": overlap_categories,
        "included_codes": included_codes,
        "excluded_codes": [adfu_code.upper()],
    }
    return CogTargets(
        category_to_codes=category_to_codes,
        category_to_cols=category_to_cols,
        source_categories_by_code=source_categories_by_code,
        qc=qc,
    )


def required_output_inventory(
    cde_modules: tuple[str, ...],
    cde_subcategories: list[dict[str, str]],
    cog_categories: tuple[str, ...],
) -> list[str]:
    inventory = [
        "results/fig4cd_cde_module_each_run.csv",
        "results/fig4cd_cde_module_summary.csv",
        "results/fig4e_subcategory_each_run.csv",
        "results/fig4e_subcategory_summary.csv",
        "results/figs5_cog_lkj_each_run.csv",
        "results/figs5_cog_lkj_summary.csv",
        "predictions/fig4cd_cde_module_predictions.csv",
        "predictions/fig4e_subcategory_predictions.csv",
        "predictions/figs5_cog_lkj_predictions.csv",
        "splits/fig4cd_cde_module_split_manifest.csv",
    ]
    inventory.extend(
        f"splits/fig4cd_cde_module_{_slugify(module_name)}_split_manifest.csv"
        for module_name in cde_modules
    )
    inventory.append("splits/fig4e_subcategory_split_manifest.csv")
    used_split_names: set[str] = set()
    for item in sorted(cde_subcategories, key=lambda row: (row["module"], row["subcategory"])):
        slug = f"{_slugify(item['module'])}__{_slugify(item['subcategory'])}"
        if slug in used_split_names:
            raise ValueError(f"split manifest slug collision: {slug}")
        used_split_names.add(slug)
        inventory.append(f"splits/fig4e_subcategory_{slug}_split_manifest.csv")
    inventory.append("splits/figs5_cog_lkj_split_manifest.csv")
    inventory.extend(
        f"splits/figs5_cog_lkj_{_slugify(category)}_split_manifest.csv"
        for category in sorted(cog_categories)
    )
    inventory.extend(
        [
            "qc/cde_target_codes.csv",
            "qc/cog_lkj_target_codes.csv",
            "qc/feature_exclusion_manifest.csv",
            "qc/target_prevalence.csv",
            "run.log",
            "run_metadata.json",
        ]
    )
    return inventory


def _split_bundle(
    target: str,
    n_rows: int,
    splits: list[core.SplitSpec],
    *,
    analysis: str,
    module_name: str | None = None,
    subcategory: str | None = None,
) -> pd.DataFrame:
    frame = core.split_manifest(n_rows, target, splits)
    frame.insert(0, "analysis", analysis)
    if module_name is not None:
        frame["module"] = module_name
    if subcategory is not None:
        frame["subcategory"] = subcategory
    return frame


def _presence_row(
    *,
    analysis: str,
    target: str,
    module_name: str | None,
    subcategory: str | None,
    counts: np.ndarray,
) -> dict[str, object]:
    presence = (counts > 0).astype(int)
    return {
        "analysis": analysis,
        "target": target,
        "module": module_name or "",
        "subcategory": subcategory or "",
        "target_positive_n": int(presence.sum()),
        "target_positive_rate": float(presence.mean()),
        "target_count_mean": float(counts.mean()),
        "target_count_max": float(np.max(counts) if counts.size else 0.0),
    }


def _write_split_outputs(
    split_dir: Path,
    combined_name: str,
    combined_frames: list[pd.DataFrame],
    per_target_frames: dict[str, pd.DataFrame],
) -> None:
    pd.concat(combined_frames, ignore_index=True).to_csv(split_dir / combined_name, index=False)
    for filename, frame in per_target_frames.items():
        frame.to_csv(split_dir / filename, index=False)


def _prediction_writer_map(predictions_dir: Path) -> dict[str, StreamingCsvWriter]:
    return {
        "cde_module": StreamingCsvWriter(predictions_dir / "fig4cd_cde_module_predictions.csv", PREDICTION_COLUMNS),
        "cde_subcategory": StreamingCsvWriter(predictions_dir / "fig4e_subcategory_predictions.csv", PREDICTION_COLUMNS),
        "cog_lkj": StreamingCsvWriter(predictions_dir / "figs5_cog_lkj_predictions.csv", PREDICTION_COLUMNS),
    }


def _append_predictions(
    writer: StreamingCsvWriter,
    predictions: pd.DataFrame,
    *,
    module_name: str,
    subcategory: str,
) -> None:
    _require_exact_schema(predictions, core.PREDICTION_COLUMNS, "core predictions")
    frame = predictions.copy()
    frame["module"] = module_name
    frame["subcategory"] = subcategory
    _require_exact_schema(frame, PREDICTION_COLUMNS, "runner predictions")
    writer.append(frame)


def _append_each_run(
    store: list[pd.DataFrame],
    each_run: pd.DataFrame,
    *,
    module_name: str,
    subcategory: str,
) -> None:
    _require_exact_schema(each_run, core.EACH_RUN_COLUMNS, "core each_run")
    frame = each_run.copy()
    frame["module"] = module_name
    frame["subcategory"] = subcategory
    _require_exact_schema(frame, EACH_RUN_OUTPUT_COLUMNS, "runner each_run")
    store.append(frame)


def _category_json(values: list[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=True, separators=(",", ":"))


def _build_cde_target_qc(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    memberships = {
        str(code): _category_json(group["target"].astype(str).tolist())
        for code, group in frame.groupby("Code", sort=True)
    }
    frame["included_in_target"] = True
    frame["removed_from_predictors"] = True
    frame["overlap_categories"] = frame["Code"].astype(str).map(memberships)
    frame["exclusion_reason"] = "target code excluded from predictors"
    _require_exact_schema(frame, TARGET_CODE_QC_COLUMNS, "CDE target QC")
    return frame


def _build_cog_target_qc(
    rows: list[dict[str, object]],
    bundle: MatrixBundle,
    cog_targets: CogTargets,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        code = str(row["Code"]).upper()
        records.append(
            {
                "analysis": row["analysis"],
                "target": row["target"],
                "module": row["module"],
                "subcategory": "",
                "feature_col": row["feature_col"],
                "Code": row["Code"],
                "included_in_target": True,
                "removed_from_predictors": True,
                "overlap_categories": _category_json(
                    cog_targets.source_categories_by_code.get(code, [])
                ),
                "exclusion_reason": "target code excluded from predictors",
            }
        )
    adfu_categories = _category_json(
        cog_targets.source_categories_by_code.get("ADFU", [])
    )
    for feature_col in sorted(bundle.adfu_cols):
        for category in COG_ORDER:
            records.append(
                {
                    "analysis": "cog_lkj",
                    "target": category,
                    "module": COG_TO_MODULE[category],
                    "subcategory": "",
                    "feature_col": feature_col,
                    "Code": str(bundle.feature_names[feature_col]),
                    "included_in_target": False,
                    "removed_from_predictors": True,
                    "overlap_categories": adfu_categories,
                    "exclusion_reason": "ADFU excluded from COG target and predictors",
                }
            )
    frame = pd.DataFrame(records)
    _require_exact_schema(frame, TARGET_CODE_QC_COLUMNS, "COG target QC")
    return frame


def _build_metadata(
    *,
    matrix_dir: Path,
    cde_path: Path = DEFAULT_CDE_PATH,
    cog_path: Path,
    params_path: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    seeds: tuple[int, ...],
    test_size: float,
    prediction_batch_size: int,
    canonical_log_path: Path,
    mirror_log_path: Path | None,
    status: str,
    params: dict[str, object] | None = None,
    bundle: MatrixBundle | None = None,
    cde_targets: CdeTargets | None = None,
    cog_targets: CogTargets | None = None,
    inventory: list[str] | None = None,
    feature_manifest_qc: dict[str, object] | None = None,
    error: Exception | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "inputs": {
            "matrix_dir": str(matrix_dir),
            "cde_path": str(cde_path),
            "cog_path": str(cog_path),
            "params_path": str(params_path),
            "out_dir": str(out_dir),
        },
        "seeds": [int(seed) for seed in seeds],
        "test_size": float(test_size),
        "prediction_batch_size": int(prediction_batch_size),
        "logs": {
            "canonical": str(canonical_log_path),
            "mirror": None if mirror_log_path is None else str(mirror_log_path),
        },
        "status": status,
        "STATUS": status,
    }
    if params is not None:
        metadata["params"] = dict(params)
    if bundle is not None:
        metadata["dimensions"] = {
            "n_rows": int(bundle.x.shape[0]),
            "n_features": int(bundle.x.shape[1]),
            "cde_subcategory_n": 0 if cde_targets is None else int(len(cde_targets.subcategory_specs)),
        }
        metadata["adfu"] = {
            "code": "ADFU",
            "product": HYPOTHETICAL_PRODUCT,
            "feature_cols": sorted(bundle.adfu_cols),
        }
    if bundle is not None and all(path.exists() for path in (
        matrix_dir / "X_plasmids_by_codes.npz",
        matrix_dir / "feature_names.npy",
        matrix_dir / "feature_code_mapping.csv",
        matrix_dir / "sample_ids.csv",
        cde_path,
        cog_path,
        params_path,
    )):
        metadata["input_sha256"] = {
            "matrix": _sha256_file(matrix_dir / "X_plasmids_by_codes.npz"),
            "feature_names": _sha256_file(matrix_dir / "feature_names.npy"),
            "mapping": _sha256_file(matrix_dir / "feature_code_mapping.csv"),
            "samples": _sha256_file(matrix_dir / "sample_ids.csv"),
            "cde": _sha256_file(cde_path),
            "cog": _sha256_file(cog_path),
            "params": _sha256_file(params_path),
        }
    if cde_targets is not None or cog_targets is not None:
        metadata["target_rules"] = {
            "cde_modules": {} if cde_targets is None else cde_targets.qc,
            "cog_lkj": {} if cog_targets is None else cog_targets.qc,
            "presence_rule": "presence = sparse_row_sum(target_cols) > 0",
            "target_split_reuse": {
                "cde_module": "one split per target reused across all conditions",
                "cde_subcategory": "one split per target reused for the only condition",
                "cog_lkj": "one split per target reused across all conditions",
            },
        }
    if inventory is not None:
        metadata["output_inventory"] = inventory
    if feature_manifest_qc is not None:
        metadata["feature_manifest_qc"] = feature_manifest_qc
    if error is not None:
        metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return metadata


def _run_cde_module_analysis(
    *,
    bundle: MatrixBundle,
    cde_targets: CdeTargets,
    base_params: dict[str, object],
    seeds: tuple[int, ...],
    test_size: float,
    prediction_batch_size: int,
    prediction_writer: StreamingCsvWriter,
    feature_tracker: FeatureManifestTracker,
    logger: Logger,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    all_cols = set(range(bundle.x.shape[1]))
    each_run_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    split_files: dict[str, pd.DataFrame] = {}
    target_rows: list[dict[str, object]] = []
    prevalence_rows: list[dict[str, object]] = []
    for module_name in MODULE_ORDER:
        target_cols = cde_targets.module_sets[module_name]
        y_count = core.sparse_row_sum(bundle.x, sorted(target_cols))
        y_presence = (y_count > 0).astype(int)
        splits = core.make_target_splits(y_presence, seeds=seeds, test_size=test_size)
        split_frame = _split_bundle(
            module_name,
            bundle.x.shape[0],
            splits,
            analysis="cde_module",
            module_name=module_name,
        )
        split_frames.append(split_frame)
        split_files[f"fig4cd_cde_module_{_slugify(module_name)}_split_manifest.csv"] = split_frame
        prevalence_rows.append(
            _presence_row(
                analysis="cde_module",
                target=module_name,
                module_name=module_name,
                subcategory=None,
                counts=y_count,
            )
        )
        for feature_col in sorted(target_cols):
            target_rows.append(
                {
                    "analysis": "cde_module",
                    "target": module_name,
                    "module": module_name,
                    "subcategory": "",
                    "feature_col": feature_col,
                    "Code": str(bundle.feature_names[feature_col]),
                }
            )
        condition_sets = core.module_feature_conditions(all_cols, cde_targets.module_sets, module_name, bundle.adfu_cols)
        for condition in CONDITION_ORDER:
            predictor_cols = set(condition_sets[condition])
            feature_tracker.append(
                analysis="cde_module",
                target=module_name,
                condition=condition,
                predictor_cols=predictor_cols,
                target_cols=target_cols,
            )
            each_run = core.fit_target_condition(
                bundle.x[:, sorted(predictor_cols)].tocsr(),
                y_presence,
                y_count,
                splits,
                base_params,
                identity={"analysis": "cde_module", "target": module_name, "condition": condition},
                prediction_sink=lambda frame: _append_predictions(
                    prediction_writer,
                    frame,
                    module_name=module_name,
                    subcategory="",
                ),
                prediction_batch_size=prediction_batch_size,
            )
            _append_each_run(each_run_frames, each_run, module_name=module_name, subcategory="")
            logger.log(f"cde_module {module_name} {condition} done")
    return (
        pd.concat(each_run_frames, ignore_index=True),
        split_frames,
        split_files,
        target_rows,
        prevalence_rows,
    )


def _run_cde_subcategory_analysis(
    *,
    bundle: MatrixBundle,
    cde_targets: CdeTargets,
    base_params: dict[str, object],
    seeds: tuple[int, ...],
    test_size: float,
    prediction_batch_size: int,
    prediction_writer: StreamingCsvWriter,
    feature_tracker: FeatureManifestTracker,
    logger: Logger,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    all_cols = set(range(bundle.x.shape[1]))
    each_run_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    split_files: dict[str, pd.DataFrame] = {}
    target_rows: list[dict[str, object]] = []
    prevalence_rows: list[dict[str, object]] = []
    used_slugs: set[str] = set()
    for spec in cde_targets.subcategory_specs:
        module_name = str(spec["module"])
        subcategory = str(spec["subcategory"])
        target_name = str(spec["target"])
        target_cols = set(spec["cols"])
        y_count = core.sparse_row_sum(bundle.x, sorted(target_cols))
        y_presence = (y_count > 0).astype(int)
        splits = core.make_target_splits(y_presence, seeds=seeds, test_size=test_size)
        slug = f"{_slugify(module_name)}__{_slugify(subcategory)}"
        if slug in used_slugs:
            raise ValueError(f"split manifest slug collision: {slug}")
        used_slugs.add(slug)
        split_frame = _split_bundle(
            target_name,
            bundle.x.shape[0],
            splits,
            analysis="cde_subcategory",
            module_name=module_name,
            subcategory=subcategory,
        )
        split_frames.append(split_frame)
        split_files[f"fig4e_subcategory_{slug}_split_manifest.csv"] = split_frame
        prevalence_rows.append(
            _presence_row(
                analysis="cde_subcategory",
                target=target_name,
                module_name=module_name,
                subcategory=subcategory,
                counts=y_count,
            )
        )
        for feature_col in sorted(target_cols):
            target_rows.append(
                {
                    "analysis": "cde_subcategory",
                    "target": target_name,
                    "module": module_name,
                    "subcategory": subcategory,
                    "feature_col": feature_col,
                    "Code": str(bundle.feature_names[feature_col]),
                }
            )
        predictor_cols = core.subcategory_predictor_cols(all_cols, target_cols, bundle.adfu_cols)
        condition = "all_minus_target_subcategory_adfu"
        feature_tracker.append(
            analysis="cde_subcategory",
            target=target_name,
            condition=condition,
            predictor_cols=predictor_cols,
            target_cols=target_cols,
        )
        each_run = core.fit_target_condition(
            bundle.x[:, sorted(predictor_cols)].tocsr(),
            y_presence,
            y_count,
            splits,
            base_params,
            identity={"analysis": "cde_subcategory", "target": target_name, "condition": condition},
            prediction_sink=lambda frame: _append_predictions(
                prediction_writer,
                frame,
                module_name=module_name,
                subcategory=subcategory,
            ),
            prediction_batch_size=prediction_batch_size,
        )
        _append_each_run(each_run_frames, each_run, module_name=module_name, subcategory=subcategory)
        logger.log(f"cde_subcategory {target_name} done")
    return (
        pd.concat(each_run_frames, ignore_index=True),
        split_frames,
        split_files,
        target_rows,
        prevalence_rows,
    )


def _run_cog_analysis(
    *,
    bundle: MatrixBundle,
    cde_targets: CdeTargets,
    cog_targets: CogTargets,
    base_params: dict[str, object],
    seeds: tuple[int, ...],
    test_size: float,
    prediction_batch_size: int,
    prediction_writer: StreamingCsvWriter,
    feature_tracker: FeatureManifestTracker,
    logger: Logger,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    all_cols = set(range(bundle.x.shape[1]))
    each_run_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    split_files: dict[str, pd.DataFrame] = {}
    target_rows: list[dict[str, object]] = []
    prevalence_rows: list[dict[str, object]] = []
    for category in COG_ORDER:
        module_name = COG_TO_MODULE[category]
        target_cols = set(cog_targets.category_to_cols[category])
        y_count = core.sparse_row_sum(bundle.x, sorted(target_cols))
        y_presence = (y_count > 0).astype(int)
        splits = core.make_target_splits(y_presence, seeds=seeds, test_size=test_size)
        split_frame = _split_bundle(
            category,
            bundle.x.shape[0],
            splits,
            analysis="cog_lkj",
            module_name=module_name,
        )
        split_frames.append(split_frame)
        split_files[f"figs5_cog_lkj_{_slugify(category)}_split_manifest.csv"] = split_frame
        prevalence_rows.append(
            _presence_row(
                analysis="cog_lkj",
                target=category,
                module_name=module_name,
                subcategory=None,
                counts=y_count,
            )
        )
        for feature_col in sorted(target_cols):
            target_rows.append(
                {
                    "analysis": "cog_lkj",
                    "target": category,
                    "module": module_name,
                    "feature_col": feature_col,
                    "Code": str(bundle.feature_names[feature_col]),
                }
            )
        condition_sets = core.module_feature_conditions(
            all_cols,
            cde_targets.module_sets,
            module_name,
            bundle.adfu_cols | target_cols,
        )
        for condition in CONDITION_ORDER:
            predictor_cols = set(condition_sets[condition])
            feature_tracker.append(
                analysis="cog_lkj",
                target=category,
                condition=condition,
                predictor_cols=predictor_cols,
                target_cols=target_cols,
            )
            each_run = core.fit_target_condition(
                bundle.x[:, sorted(predictor_cols)].tocsr(),
                y_presence,
                y_count,
                splits,
                base_params,
                identity={"analysis": "cog_lkj", "target": category, "condition": condition},
                prediction_sink=lambda frame: _append_predictions(
                    prediction_writer,
                    frame,
                    module_name=module_name,
                    subcategory="",
                ),
                prediction_batch_size=prediction_batch_size,
            )
            _append_each_run(each_run_frames, each_run, module_name=module_name, subcategory="")
            logger.log(f"cog_lkj {category} {condition} done")
    return (
        pd.concat(each_run_frames, ignore_index=True),
        split_frames,
        split_files,
        target_rows,
        prevalence_rows,
    )


def _publish_staging(
    staging_root: Path,
    out_dir: Path,
    inventory: list[str],
) -> None:
    for relative_path in inventory:
        if relative_path == "run.log":
            continue
        staged_path = staging_root / relative_path
        if not staged_path.is_file():
            raise ValueError(f"Missing staged output: {relative_path}")

    publish_names = ("results", "predictions", "splits", "qc", "plots", "run_metadata.json")
    for name in publish_names[:-1]:
        if not (staging_root / name).is_dir():
            raise ValueError(f"Missing staged output directory: {name}")

    backup_root = Path(tempfile.mkdtemp(prefix=".fig4-backup-", dir=out_dir))
    installed: list[tuple[Path, Path, Path, bool]] = []
    failed_attempt_path = out_dir / "failed_attempt_metadata.json"
    failed_attempt_backup = backup_root / "failed_attempt_metadata.json"
    removed_failed_attempt = False
    try:
        if failed_attempt_path.exists():
            os.replace(failed_attempt_path, failed_attempt_backup)
            removed_failed_attempt = True
        for name in publish_names:
            source = staging_root / name
            destination = out_dir / name
            backup = backup_root / name
            had_destination = destination.exists()
            if had_destination:
                os.replace(destination, backup)
            try:
                os.replace(source, destination)
            except Exception:
                if had_destination:
                    os.replace(backup, destination)
                raise
            installed.append((source, destination, backup, had_destination))
    except Exception:
        for source, destination, backup, had_destination in reversed(installed):
            if destination.exists():
                os.replace(destination, source)
            if had_destination:
                os.replace(backup, destination)
        if removed_failed_attempt and failed_attempt_backup.exists():
            os.replace(failed_attempt_backup, failed_attempt_path)
        raise
    finally:
        shutil.rmtree(backup_root)


def run_workflow(
    *,
    matrix_dir: Path = DEFAULT_MATRIX_DIR,
    cde_path: Path,
    cog_path: Path = DEFAULT_COG_PATH,
    params_path: Path = DEFAULT_PARAMS_PATH,
    out_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    test_size: float = DEFAULT_TEST_SIZE,
    prediction_batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE,
    log_path: Path | None = None,
) -> int:
    canonical_log_path = out_dir / "run.log"
    logger = Logger(canonical_log_path, log_path)
    base_params: dict[str, object] | None = None
    bundle: MatrixBundle | None = None
    cde_targets: CdeTargets | None = None
    cog_targets: CogTargets | None = None
    inventory: list[str] | None = None
    feature_manifest_qc: dict[str, object] | None = None
    staging_root: Path | None = None
    published_results_preserved = False
    try:
        published_results_preserved = _has_complete_published_results(out_dir)
        logger.log("STATUS=RUNNING")
        staging_root = Path(tempfile.mkdtemp(prefix=".fig4-staging-", dir=out_dir))
        results_dir = staging_root / "results"
        predictions_dir = staging_root / "predictions"
        splits_dir = staging_root / "splits"
        qc_dir = staging_root / "qc"
        plots_dir = staging_root / "plots"
        for directory in (results_dir, predictions_dir, splits_dir, qc_dir, plots_dir):
            directory.mkdir(parents=True, exist_ok=True)

        bundle = load_matrix_bundle(matrix_dir)
        base_params = core.load_fig2_params(params_path)
        cde_table = _read_csv_str(cde_path)
        cog_table = _read_csv_str(cog_path)
        cde_targets = load_cde_targets(cde_table, bundle.feature_names)
        cog_targets = load_cog_targets(cog_table, bundle.feature_names)
        inventory = required_output_inventory(
            cde_modules=MODULE_ORDER,
            cde_subcategories=[
                {"module": str(spec["module"]), "subcategory": str(spec["subcategory"])}
                for spec in cde_targets.subcategory_specs
            ],
            cog_categories=COG_ORDER,
        )

        prediction_writers = _prediction_writer_map(predictions_dir)
        feature_tracker = FeatureManifestTracker(
            StreamingCsvWriter(qc_dir / "feature_exclusion_manifest.csv", FEATURE_MANIFEST_COLUMNS),
            bundle.feature_names,
            bundle.adfu_cols,
        )

        module_each_run, module_split_frames, module_split_files, cde_module_target_rows, module_prevalence = _run_cde_module_analysis(
            bundle=bundle,
            cde_targets=cde_targets,
            base_params=base_params,
            seeds=seeds,
            test_size=test_size,
            prediction_batch_size=prediction_batch_size,
            prediction_writer=prediction_writers["cde_module"],
            feature_tracker=feature_tracker,
            logger=logger,
        )
        module_summary = core.summarize_runs(
            module_each_run,
            ["analysis", "target", "condition", "module", "subcategory"],
        )
        _require_exact_schema(module_each_run, EACH_RUN_OUTPUT_COLUMNS, "CDE module each_run")
        _require_exact_schema(module_summary, SUMMARY_OUTPUT_COLUMNS, "CDE module summary")
        module_each_run.to_csv(results_dir / "fig4cd_cde_module_each_run.csv", index=False)
        module_summary.to_csv(results_dir / "fig4cd_cde_module_summary.csv", index=False)
        _write_split_outputs(
            splits_dir,
            "fig4cd_cde_module_split_manifest.csv",
            module_split_frames,
            module_split_files,
        )

        sub_each_run, sub_split_frames, sub_split_files, cde_sub_target_rows, sub_prevalence = _run_cde_subcategory_analysis(
            bundle=bundle,
            cde_targets=cde_targets,
            base_params=base_params,
            seeds=seeds,
            test_size=test_size,
            prediction_batch_size=prediction_batch_size,
            prediction_writer=prediction_writers["cde_subcategory"],
            feature_tracker=feature_tracker,
            logger=logger,
        )
        sub_summary = core.summarize_runs(
            sub_each_run,
            ["analysis", "target", "condition", "module", "subcategory"],
        )
        _require_exact_schema(sub_each_run, EACH_RUN_OUTPUT_COLUMNS, "CDE subcategory each_run")
        _require_exact_schema(sub_summary, SUMMARY_OUTPUT_COLUMNS, "CDE subcategory summary")
        sub_each_run.to_csv(results_dir / "fig4e_subcategory_each_run.csv", index=False)
        sub_summary.to_csv(results_dir / "fig4e_subcategory_summary.csv", index=False)
        _write_split_outputs(
            splits_dir,
            "fig4e_subcategory_split_manifest.csv",
            sub_split_frames,
            sub_split_files,
        )

        cog_each_run, cog_split_frames, cog_split_files, cog_target_rows, cog_prevalence = _run_cog_analysis(
            bundle=bundle,
            cde_targets=cde_targets,
            cog_targets=cog_targets,
            base_params=base_params,
            seeds=seeds,
            test_size=test_size,
            prediction_batch_size=prediction_batch_size,
            prediction_writer=prediction_writers["cog_lkj"],
            feature_tracker=feature_tracker,
            logger=logger,
        )
        cog_summary = core.summarize_runs(
            cog_each_run,
            ["analysis", "target", "condition", "module", "subcategory"],
        )
        _require_exact_schema(cog_each_run, EACH_RUN_OUTPUT_COLUMNS, "COG each_run")
        _require_exact_schema(cog_summary, SUMMARY_OUTPUT_COLUMNS, "COG summary")
        cog_each_run.to_csv(results_dir / "figs5_cog_lkj_each_run.csv", index=False)
        cog_summary.to_csv(results_dir / "figs5_cog_lkj_summary.csv", index=False)
        _write_split_outputs(
            splits_dir,
            "figs5_cog_lkj_split_manifest.csv",
            cog_split_frames,
            cog_split_files,
        )

        _build_cde_target_qc(cde_module_target_rows + cde_sub_target_rows).to_csv(
            qc_dir / "cde_target_codes.csv", index=False
        )
        _build_cog_target_qc(cog_target_rows, bundle, cog_targets).to_csv(
            qc_dir / "cog_lkj_target_codes.csv", index=False
        )
        pd.DataFrame(module_prevalence + sub_prevalence + cog_prevalence).to_csv(qc_dir / "target_prevalence.csv", index=False)

        feature_manifest_qc = feature_tracker.validate()
        metadata = _build_metadata(
            matrix_dir=matrix_dir,
            cde_path=cde_path,
            cog_path=cog_path,
            params_path=params_path,
            out_dir=out_dir,
            seeds=seeds,
            test_size=test_size,
            prediction_batch_size=prediction_batch_size,
            canonical_log_path=canonical_log_path,
            mirror_log_path=log_path,
            status="COMPLETE",
            params=base_params,
            bundle=bundle,
            cde_targets=cde_targets,
            cog_targets=cog_targets,
            inventory=inventory,
            feature_manifest_qc=feature_manifest_qc,
        )
        _write_json(staging_root / "run_metadata.json", metadata)
        _publish_staging(staging_root, out_dir, inventory)
        logger.log("STATUS=COMPLETE")
        return 0
    except Exception as error:
        logger.log(f"ERROR {type(error).__name__}: {error}")
        failed_metadata = _build_metadata(
            matrix_dir=matrix_dir,
            cde_path=cde_path,
            cog_path=cog_path,
            params_path=params_path,
            out_dir=out_dir,
            seeds=seeds,
            test_size=test_size,
            prediction_batch_size=prediction_batch_size,
            canonical_log_path=canonical_log_path,
            mirror_log_path=log_path,
            status="FAILED",
            params=base_params,
            bundle=bundle,
            cde_targets=cde_targets,
            cog_targets=cog_targets,
            inventory=inventory,
            feature_manifest_qc=feature_manifest_qc,
            error=error,
        )
        failed_metadata["attempted_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        failed_metadata["published_results_preserved"] = published_results_preserved
        _write_json(out_dir / "failed_attempt_metadata.json", failed_metadata)
        if not published_results_preserved:
            _write_json(out_dir / "run_metadata.json", failed_metadata)
        logger.log("STATUS=FAILED")
        raise
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root)
        logger.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--cde-match", type=Path, required=True)
    parser.add_argument("--cog-map", type=Path, default=DEFAULT_COG_PATH)
    parser.add_argument("--best-params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--prediction-batch-size", type=int, default=DEFAULT_PREDICTION_BATCH_SIZE)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    seed_values = tuple(int(part.strip()) for part in str(args.seeds).split(",") if part.strip())
    if not seed_values:
        raise ValueError("seeds must not be empty")
    args.seeds = seed_values
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_workflow(
        matrix_dir=args.matrix_dir,
        cde_path=args.cde_match,
        cog_path=args.cog_map,
        params_path=args.best_params,
        out_dir=args.out_dir,
        seeds=args.seeds,
        test_size=args.test_size,
        prediction_batch_size=args.prediction_batch_size,
        log_path=args.log_file,
    )


if __name__ == "__main__":
    sys.exit(main())
