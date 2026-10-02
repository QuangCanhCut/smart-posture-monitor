"""
Canonical training entry point cho Smart Posture Monitor V03.

Run:
    python -m src.train

Vai trò của file này là production training, không phải research notebook.
Notebook `notebooks/02_training_experiments.ipynb` chịu trách nhiệm thử nhiều
model, thử imbalance strategy và tune hyperparameter. `train.py` chỉ tái tạo
winner đã chốt:

    Personal Baseline Delta + XGBoost + imbalance_strategy="none"

Luồng chính:
    RAW features.csv
        -> prepare_dataset()
        -> person-based holdout split
        -> Personal Baseline Delta cho train recordings
        -> fit final XGBoost
        -> save best_model + metadata + split manifest + calibration config

Holdout test persons chỉ được khóa trong manifest. File này không predict hoặc
tính metric trên holdout test; evaluation riêng nằm ở `src.evaluate`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.personal_calibration import (  # noqa: E402
    DeltaDataset,
    build_delta_dataset_for_recordings,
)
from src.preprocessing import (  # noqa: E402
    EXPECTED_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    PreparedDataset,
    prepare_dataset,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SAMPLES = 30
EXPECTED_FEATURE_COUNT = 29

VERSION = "V03"
REPRESENTATION = "personal_baseline_delta"
MODEL_NAME = "XGBoost Tuned"
IMBALANCE_STRATEGY = "none"

EXPERIMENT_CV_MACRO_F1_MEAN = 0.688936
EXPERIMENT_CV_MACRO_F1_STD = 0.140001

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.02,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 5,
}

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
SPLIT_MANIFEST_PATH = MODELS_DIR / "split_manifest.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
CALIBRATION_CONFIG_PATH = MODELS_DIR / "calibration_config.json"
CALIBRATION_AUDIT_PATH = RESULTS_DIR / "calibration_audit.csv"


@dataclass
class SplitData:
    train_indices: np.ndarray
    test_indices: np.ndarray
    manifest: dict[str, Any]
    reused_manifest: bool


@dataclass
class RawSplit:
    X_train_raw: pd.DataFrame
    y_train_raw: pd.Series
    groups_train_raw: pd.Series
    metadata_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_test_raw: pd.Series
    groups_test_raw: pd.Series
    metadata_test_raw: pd.DataFrame


@dataclass
class DeltaPreparedSplit:
    X_delta: pd.DataFrame
    y_delta: pd.Series
    groups_delta: pd.Series
    metadata_delta: pd.DataFrame
    calibration_audit: pd.DataFrame


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON phải chứa object/dict: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def encoded_class_counts(y_values: pd.Series) -> dict[str, int]:
    counts = pd.Series(y_values).value_counts().sort_index()
    return {
        ID_TO_LABEL[int(class_id)]: int(counts.get(class_id, 0))
        for class_id in sorted(ID_TO_LABEL)
    }


def assert_all_classes_present(name: str, y_values: pd.Series) -> None:
    present = set(int(value) for value in pd.Series(y_values).unique())
    missing = sorted(set(ID_TO_LABEL) - present)
    if missing:
        missing_labels = [ID_TO_LABEL[class_id] for class_id in missing]
        raise RuntimeError(f"{name} bị thiếu class: {missing_labels}")


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False))


# ============================================================
# Dataset and split
# ============================================================

def load_training_data(data_path: Path = DATA_PATH) -> tuple[PreparedDataset, str]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {data_path}")

    prepared = prepare_dataset(csv_path=data_path, excluded_persons=())
    dataset_sha256 = sha256_file(data_path)

    feature_columns = list(prepared.feature_columns)
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"nhưng nhận được {len(feature_columns)}."
        )
    if prepared.X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"X_raw phải có {EXPECTED_FEATURE_COUNT} features.")

    return prepared, dataset_sha256


def _indices_for_persons(groups: pd.Series, persons: list[str]) -> np.ndarray:
    mask = groups.astype(str).isin(set(persons)).to_numpy()
    return np.flatnonzero(mask)


def _build_split_manifest(
    prepared: PreparedDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dataset_sha256: str,
    reused_manifest: bool,
) -> dict[str, Any]:
    groups = prepared.groups.astype(str).reset_index(drop=True)
    metadata = prepared.metadata.reset_index(drop=True)
    y = prepared.y.reset_index(drop=True)

    train_persons = sorted(groups.iloc[train_indices].unique())
    test_persons = sorted(groups.iloc[test_indices].unique())
    train_recordings = sorted(metadata.iloc[train_indices]["recording_id"].astype(str).unique())
    test_recordings = sorted(metadata.iloc[test_indices]["recording_id"].astype(str).unique())

    person_overlap = sorted(set(train_persons) & set(test_persons))
    recording_overlap = sorted(set(train_recordings) & set(test_recordings))
    if person_overlap:
        raise RuntimeError(f"Person leakage giữa train/test: {person_overlap}")
    if recording_overlap:
        raise RuntimeError(f"Recording leakage giữa train/test: {recording_overlap}")

    return {
        "version": VERSION,
        "split_strategy": (
            "reused_locked_person_holdout"
            if reused_manifest
            else "GroupShuffleSplit person holdout"
        ),
        "group_column": "person_id",
        "recording_column": "recording_id",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_persons": train_persons,
        "test_persons": test_persons,
        "train_recording_ids": train_recordings,
        "test_recording_ids": test_recordings,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "train_class_distribution_raw": encoded_class_counts(y.iloc[train_indices]),
        "test_class_distribution_raw": encoded_class_counts(y.iloc[test_indices]),
        "person_overlap": person_overlap,
        "recording_overlap": recording_overlap,
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha256,
    }


def _manifest_is_compatible(
    prepared: PreparedDataset,
    dataset_sha256: str,
    manifest: dict[str, Any],
) -> bool:
    if str(manifest.get("dataset_sha256")) != dataset_sha256:
        return False

    train_persons = [str(person) for person in manifest.get("train_persons", [])]
    test_persons = [str(person) for person in manifest.get("test_persons", [])]
    if not train_persons or not test_persons:
        return False

    all_persons = set(prepared.groups.astype(str).unique())
    manifest_persons = set(train_persons) | set(test_persons)
    if manifest_persons != all_persons:
        return False
    if set(train_persons) & set(test_persons):
        return False

    train_indices = _indices_for_persons(prepared.groups, train_persons)
    test_indices = _indices_for_persons(prepared.groups, test_persons)
    if len(train_indices) == 0 or len(test_indices) == 0:
        return False
    if len(train_indices) + len(test_indices) != len(prepared.X):
        return False

    expected_train = manifest.get("train_samples")
    expected_test = manifest.get("test_samples")
    if expected_train is not None and int(expected_train) != len(train_indices):
        return False
    if expected_test is not None and int(expected_test) != len(test_indices):
        return False

    metadata = prepared.metadata.reset_index(drop=True)
    actual_train_recordings = sorted(
        metadata.iloc[train_indices]["recording_id"].astype(str).unique()
    )
    actual_test_recordings = sorted(
        metadata.iloc[test_indices]["recording_id"].astype(str).unique()
    )
    manifest_train_recordings = sorted(
        str(recording) for recording in manifest.get("train_recording_ids", [])
    )
    manifest_test_recordings = sorted(
        str(recording) for recording in manifest.get("test_recording_ids", [])
    )
    if manifest_train_recordings and manifest_train_recordings != actual_train_recordings:
        return False
    if manifest_test_recordings and manifest_test_recordings != actual_test_recordings:
        return False

    return True


def create_or_load_person_split(
    prepared: PreparedDataset,
    dataset_sha256: str,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
) -> SplitData:
    groups = prepared.groups.astype(str).reset_index(drop=True)

    if split_manifest_path.is_file():
        manifest = read_json(split_manifest_path)
        if _manifest_is_compatible(prepared, dataset_sha256, manifest):
            train_persons = [str(person) for person in manifest["train_persons"]]
            test_persons = [str(person) for person in manifest["test_persons"]]
            train_indices = _indices_for_persons(groups, train_persons)
            test_indices = _indices_for_persons(groups, test_persons)
            canonical_manifest = _build_split_manifest(
                prepared=prepared,
                train_indices=train_indices,
                test_indices=test_indices,
                dataset_sha256=dataset_sha256,
                reused_manifest=True,
            )
            return SplitData(
                train_indices=train_indices,
                test_indices=test_indices,
                manifest=canonical_manifest,
                reused_manifest=True,
            )

        print("[WARNING] split_manifest.json không còn tương thích, tạo split mới.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(
        splitter.split(prepared.X, prepared.y, groups=groups)
    )
    manifest = _build_split_manifest(
        prepared=prepared,
        train_indices=train_indices,
        test_indices=test_indices,
        dataset_sha256=dataset_sha256,
        reused_manifest=False,
    )
    return SplitData(
        train_indices=train_indices,
        test_indices=test_indices,
        manifest=manifest,
        reused_manifest=False,
    )


def slice_raw_split(prepared: PreparedDataset, split: SplitData) -> RawSplit:
    X_raw = prepared.X.reset_index(drop=True)
    y = prepared.y.reset_index(drop=True)
    groups = prepared.groups.reset_index(drop=True).astype(str)
    metadata = prepared.metadata.reset_index(drop=True)

    return RawSplit(
        X_train_raw=X_raw.iloc[split.train_indices].copy(),
        y_train_raw=y.iloc[split.train_indices].copy(),
        groups_train_raw=groups.iloc[split.train_indices].copy(),
        metadata_train_raw=metadata.iloc[split.train_indices].copy(),
        X_test_raw=X_raw.iloc[split.test_indices].copy(),
        y_test_raw=y.iloc[split.test_indices].copy(),
        groups_test_raw=groups.iloc[split.test_indices].copy(),
        metadata_test_raw=metadata.iloc[split.test_indices].copy(),
    )


# ============================================================
# Delta construction and model
# ============================================================

def _canonical_calibration_audit(audit: pd.DataFrame) -> pd.DataFrame:
    result = audit.copy()
    result["removed_samples"] = result["calibration_samples"].astype(int)
    result["remaining_samples"] = result["remaining_total"].astype(int)
    result["skip_reason"] = result["skip_reason"].fillna("")

    columns = [
        "recording_id",
        "person_id",
        "total_samples",
        "correct_samples",
        "calibration_samples",
        "removed_samples",
        "remaining_samples",
        "remaining_correct",
        "baseline_created",
        "order_source",
        "skip_reason",
    ]
    return result[columns]


def build_delta_split(
    raw_split: RawSplit,
    feature_columns: list[str],
    calibration_samples: int = CALIBRATION_SAMPLES,
) -> DeltaPreparedSplit:
    delta: DeltaDataset = build_delta_dataset_for_recordings(
        X_raw=raw_split.X_train_raw,
        y=raw_split.y_train_raw,
        groups=raw_split.groups_train_raw,
        metadata=raw_split.metadata_train_raw,
        feature_columns=feature_columns,
        calibration_samples=calibration_samples,
        correct_label="correct",
    )

    audit = _canonical_calibration_audit(delta.audit)
    validate_delta_split(delta, feature_columns)

    return DeltaPreparedSplit(
        X_delta=delta.X_delta,
        y_delta=delta.y_delta,
        groups_delta=delta.groups_delta,
        metadata_delta=delta.metadata_delta,
        calibration_audit=audit,
    )


def validate_delta_split(delta: DeltaDataset, feature_columns: list[str]) -> None:
    if not (
        len(delta.X_delta)
        == len(delta.y_delta)
        == len(delta.groups_delta)
        == len(delta.metadata_delta)
    ):
        raise RuntimeError("X/y/groups/metadata của delta split không align.")

    if delta.X_delta.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"X_delta phải có {EXPECTED_FEATURE_COUNT} features, "
            f"nhưng nhận được {delta.X_delta.shape[1]}."
        )
    if list(delta.X_delta.columns) != list(feature_columns):
        raise RuntimeError("Feature order trong X_delta không khớp feature_columns.")
    if not np.isfinite(delta.X_delta.to_numpy(dtype=float)).all():
        raise RuntimeError("X_delta chứa NaN hoặc Infinity.")

    assert_all_classes_present("Delta train dataset", delta.y_delta)


def build_model() -> XGBClassifier:
    # XGBoost là tree-based model, nên production pipeline không cần StandardScaler.
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(EXPECTED_CLASSES),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        **XGB_PARAMS,
    )


# ============================================================
# Artifacts and smoke test
# ============================================================

def build_training_metadata(
    prepared: PreparedDataset,
    raw_split: RawSplit,
    delta_split: DeltaPreparedSplit,
    split_manifest: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    calibration_samples_removed = int(
        delta_split.calibration_audit["removed_samples"].sum()
    )
    best_params = dict(XGB_PARAMS)

    return {
        "version": VERSION,
        "representation": REPRESENTATION,
        "feature_mode": REPRESENTATION,
        "model_name": MODEL_NAME,
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "resampling_method": None,
        "calibration_samples": CALIBRATION_SAMPLES,
        "baseline_statistic": "median",
        "baseline_scope": "recording_id",
        "feature_count": EXPECTED_FEATURE_COUNT,
        "n_features": EXPECTED_FEATURE_COUNT,
        "feature_columns": list(prepared.feature_columns),
        "classes": list(EXPECTED_CLASSES),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(key): value for key, value in ID_TO_LABEL.items()},
        "best_params": best_params,
        "xgb_params": best_params,
        "experiment_cv_macro_f1_mean": EXPERIMENT_CV_MACRO_F1_MEAN,
        "experiment_cv_macro_f1_std": EXPERIMENT_CV_MACRO_F1_STD,
        "train_persons": split_manifest["train_persons"],
        "test_persons": split_manifest["test_persons"],
        "train_recordings": split_manifest["train_recording_ids"],
        "test_recordings": split_manifest["test_recording_ids"],
        "n_raw_samples": int(len(prepared.X)),
        "raw_train_samples": int(len(raw_split.X_train_raw)),
        "raw_test_samples": int(len(raw_split.X_test_raw)),
        "calibration_samples_removed": calibration_samples_removed,
        "delta_train_samples": int(len(delta_split.X_delta)),
        "raw_class_distribution": encoded_class_counts(raw_split.y_train_raw),
        "delta_class_distribution": encoded_class_counts(delta_split.y_delta),
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha256,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_calibration_config() -> dict[str, Any]:
    return {
        "version": VERSION,
        "feature_mode": REPRESENTATION,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "baseline_statistic": "median",
        "baseline_scope": "recording_id",
        "calibration_samples": CALIBRATION_SAMPLES,
    }


def save_training_artifacts(
    model: XGBClassifier,
    split_manifest: dict[str, Any],
    training_metadata: dict[str, Any],
    calibration_config: dict[str, Any],
    calibration_audit: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    write_json(SPLIT_MANIFEST_PATH, split_manifest)
    write_json(TRAINING_METADATA_PATH, training_metadata)
    write_json(CALIBRATION_CONFIG_PATH, calibration_config)
    calibration_audit.to_csv(CALIBRATION_AUDIT_PATH, index=False)


def reload_smoke_test(model: XGBClassifier, X_train_delta: pd.DataFrame) -> np.ndarray:
    loaded_model = joblib.load(MODEL_PATH)
    if not hasattr(loaded_model, "predict"):
        raise RuntimeError("Loaded model không có method predict().")

    model_n_features = getattr(loaded_model, "n_features_in_", None)
    if model_n_features is not None and int(model_n_features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Loaded model expected {model_n_features} features, "
            f"không phải {EXPECTED_FEATURE_COUNT}."
        )

    smoke_X = X_train_delta.iloc[:10].copy()
    original_predictions = np.asarray(model.predict(smoke_X), dtype=int)
    loaded_predictions = np.asarray(loaded_model.predict(smoke_X), dtype=int)
    if not np.array_equal(original_predictions, loaded_predictions):
        raise RuntimeError("Prediction của loaded model không khớp model vừa fit.")

    return loaded_predictions


# ============================================================
# Main orchestration
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train canonical Smart Posture Monitor V03 model "
            "(Personal Baseline Delta + XGBoost)."
        )
    )
    return parser.parse_args()


def main() -> None:
    parse_args()

    print("=" * 72)
    print("SMART POSTURE MONITOR V03 - CANONICAL TRAINING")
    print("=" * 72)

    print("[1/7] Loading RAW dataset through prepare_dataset()...")
    prepared, dataset_sha256 = load_training_data(DATA_PATH)
    feature_columns = list(prepared.feature_columns)
    print(f"      Raw samples: {len(prepared.X)}")
    print(f"      Features: {len(feature_columns)}")
    print(f"      Dataset SHA256: {dataset_sha256}")

    print("[2/7] Creating or reusing person-based holdout split...")
    split = create_or_load_person_split(prepared, dataset_sha256, SPLIT_MANIFEST_PATH)
    raw_split = slice_raw_split(prepared, split)
    assert_all_classes_present("Raw train split", raw_split.y_train_raw)
    assert_all_classes_present("Raw holdout split", raw_split.y_test_raw)
    print(f"      Reused manifest: {split.reused_manifest}")
    print(f"      Train persons: {split.manifest['train_persons']}")
    print(f"      Test persons: {split.manifest['test_persons']}")

    print("[3/7] Building Personal Baseline Delta train split...")
    delta_split = build_delta_split(
        raw_split=raw_split,
        feature_columns=feature_columns,
        calibration_samples=CALIBRATION_SAMPLES,
    )
    calibration_removed = int(delta_split.calibration_audit["removed_samples"].sum())
    print(f"      Calibration samples removed: {calibration_removed}")
    print(f"      Delta train samples: {len(delta_split.X_delta)}")

    print("[4/7] Fitting final XGBoost winner...")
    model = build_model()
    model.fit(delta_split.X_delta, delta_split.y_delta)

    print("[5/7] Saving model and metadata artifacts...")
    training_metadata = build_training_metadata(
        prepared=prepared,
        raw_split=raw_split,
        delta_split=delta_split,
        split_manifest=split.manifest,
        dataset_sha256=dataset_sha256,
    )
    calibration_config = build_calibration_config()
    save_training_artifacts(
        model=model,
        split_manifest=split.manifest,
        training_metadata=training_metadata,
        calibration_config=calibration_config,
        calibration_audit=delta_split.calibration_audit,
    )

    print("[6/7] Reload smoke test on train delta samples...")
    smoke_predictions = reload_smoke_test(model, delta_split.X_delta)

    print("[7/7] Training summary...")
    print_table("Raw train class distribution", pd.Series(training_metadata["raw_class_distribution"]).rename_axis("label").reset_index(name="samples"))
    print_table("Delta train class distribution", pd.Series(training_metadata["delta_class_distribution"]).rename_axis("label").reset_index(name="samples"))

    print("\nArtifacts")
    print("---------")
    print(f"Model              : {MODEL_PATH}")
    print(f"Split manifest     : {SPLIT_MANIFEST_PATH}")
    print(f"Training metadata  : {TRAINING_METADATA_PATH}")
    print(f"Calibration config : {CALIBRATION_CONFIG_PATH}")
    print(f"Calibration audit  : {CALIBRATION_AUDIT_PATH}")

    print("\nFinal model")
    print("-----------")
    print(f"Representation     : {REPRESENTATION}")
    print(f"Model              : {MODEL_NAME}")
    print(f"Imbalance strategy : {IMBALANCE_STRATEGY}")
    print(f"XGBoost params     : {XGB_PARAMS}")
    print(f"Experiment CV F1   : {EXPERIMENT_CV_MACRO_F1_MEAN:.6f} +/- {EXPERIMENT_CV_MACRO_F1_STD:.6f}")
    print(f"Smoke predictions  : {smoke_predictions.tolist()}")
    print("\nHoldout test persons were NOT used for prediction or scoring in train.py.")
    print("=" * 72)


if __name__ == "__main__":
    main()
