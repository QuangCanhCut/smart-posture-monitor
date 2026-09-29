from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on local environment
    XGBClassifier = None

from src import preprocessing as prep
from src.personal_calibration import PersonalCalibration
from src.posture_predictor_v03 import PosturePredictorV03


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
V02_SPLIT_PATH = PROJECT_ROOT / "models" / "split_manifest.json"
V02_METRICS_PATH = PROJECT_ROOT / "results" / "evaluation" / "test_metrics.json"
V02_CLASS_REPORT_PATH = PROJECT_ROOT / "results" / "evaluation" / "classification_report.csv"
V02_PER_PERSON_PATH = PROJECT_ROOT / "results" / "evaluation" / "per_person_metrics.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "v03"
RESULTS_DIR = PROJECT_ROOT / "results" / "v03"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42
CALIBRATION_SAMPLES = 30
MIN_CALIBRATION_SAMPLES = 10
CLASS_LABELS = prep.EXPECTED_CLASSES
CLASS_IDS = [prep.LABEL_TO_ID[label] for label in CLASS_LABELS]


@dataclass
class CalibrationStats:
    original_samples: int
    original_persons: int
    original_sessions: int
    original_recordings: int
    transformed_samples: int
    calibration_samples_removed: int
    skipped_sessions: list[dict[str, Any]]
    per_recording: list[dict[str, Any]]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            sanitize_for_json(payload),
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v02_split(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"V02 split manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    train_persons = [str(person) for person in manifest.get("train_persons", [])]
    test_persons = [str(person) for person in manifest.get("test_persons", [])]

    if not train_persons or not test_persons:
        raise ValueError("V02 split manifest must contain train_persons and test_persons.")

    overlap = sorted(set(train_persons) & set(test_persons))
    if overlap:
        raise ValueError(f"Person overlap in V02 split manifest: {overlap}")

    return sorted(train_persons), sorted(test_persons), manifest


def make_train_val_test_persons(
    v02_train_persons: list[str],
    v02_test_persons: list[str],
    validation_size: float,
) -> tuple[list[str], list[str], list[str]]:
    train_persons, validation_persons = train_test_split(
        sorted(v02_train_persons),
        test_size=validation_size,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return sorted(train_persons), sorted(validation_persons), sorted(v02_test_persons)


def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in CLASS_LABELS}


def apply_session_delta_calibration(
    df: pd.DataFrame,
    feature_columns: list[str],
    calibration_samples: int,
    min_calibration_samples: int,
) -> tuple[pd.DataFrame, CalibrationStats]:
    transformed_parts: list[pd.DataFrame] = []
    skipped_sessions: list[dict[str, Any]] = []
    per_recording: list[dict[str, Any]] = []
    removed_count = 0

    for recording_id, group in df.groupby("recording_id", sort=True):
        group = group.copy()
        person_id = str(group["person_id"].iloc[0])
        session_id = str(group["session_id"].iloc[0])

        correct = group[group["label"] == "correct"].sort_values("image_path", kind="mergesort")
        correct_count = int(len(correct))

        if correct_count < min_calibration_samples:
            skipped = {
                "recording_id": str(recording_id),
                "person_id": person_id,
                "session_id": session_id,
                "correct_samples": correct_count,
                "reason": "not enough correct samples for calibration",
            }
            skipped_sessions.append(skipped)
            per_recording.append({**skipped, "skipped": True})
            continue

        calibration_count = min(calibration_samples, correct_count)
        calibration_indices = correct.head(calibration_count).index

        calibrator = PersonalCalibration(feature_count=len(feature_columns))
        for _, sample in correct.head(calibration_count).iterrows():
            calibrator.add_sample(sample[feature_columns].to_numpy(dtype=float))
        baseline = calibrator.calculate_baseline()

        usable = group.drop(index=calibration_indices).copy()
        raw_features = usable[feature_columns].to_numpy(dtype=float)
        usable.loc[:, feature_columns] = raw_features - baseline

        transformed_parts.append(usable)
        removed_count += calibration_count
        per_recording.append(
            {
                "recording_id": str(recording_id),
                "person_id": person_id,
                "session_id": session_id,
                "correct_samples": correct_count,
                "calibration_samples": calibration_count,
                "usable_samples": int(len(usable)),
                "skipped": False,
            }
        )

    if not transformed_parts:
        raise RuntimeError("No sessions were usable after calibration.")

    transformed = pd.concat(transformed_parts, ignore_index=True)
    transformed = transformed.sort_values("image_path", kind="mergesort").reset_index(drop=True)

    stats = CalibrationStats(
        original_samples=int(len(df)),
        original_persons=int(df["person_id"].nunique()),
        original_sessions=int(df["session_id"].nunique()),
        original_recordings=int(df["recording_id"].nunique()),
        transformed_samples=int(len(transformed)),
        calibration_samples_removed=int(removed_count),
        skipped_sessions=skipped_sessions,
        per_recording=per_recording,
    )
    return transformed, stats


def split_dataframe(
    df: pd.DataFrame,
    train_persons: list[str],
    validation_persons: list[str],
    test_persons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    person_series = df["person_id"].astype(str)
    train_df = df[person_series.isin(train_persons)].copy()
    validation_df = df[person_series.isin(validation_persons)].copy()
    test_df = df[person_series.isin(test_persons)].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise RuntimeError(
            "Train/validation/test split produced an empty split after calibration."
        )

    overlap = (
        set(train_df["person_id"])
        & set(validation_df["person_id"])
        | set(train_df["person_id"])
        & set(test_df["person_id"])
        | set(validation_df["person_id"])
        & set(test_df["person_id"])
    )
    if overlap:
        raise RuntimeError(f"Person leakage across splits: {sorted(overlap)}")

    return train_df, validation_df, test_df


def make_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns].copy().astype(float)
    y = df["label"].map(prep.LABEL_TO_ID).astype(int)
    return X, y


def fit_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    param_grid = {
        "model__C": [1, 10, 100],
        "model__gamma": [0.001, 0.01, "scale"],
        "model__class_weight": [None, "balanced"],
    }

    n_splits = min(5, int(groups.nunique()))
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)

    info = {
        "classifier": "SVM RBF",
        "hyperparameters": search.best_params_,
        "cv_macro_f1_mean": float(search.best_score_),
        "cv_n_splits": n_splits,
    }
    return search.best_estimator_, info


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, dict[str, Any]] | None:
    if XGBClassifier is None:
        return None

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    info = {
        "classifier": "XGBoost",
        "hyperparameters": model.get_params(),
        "cv_macro_f1_mean": None,
        "cv_n_splits": None,
    }
    return model, info


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = np.asarray(model.predict(X), dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=CLASS_IDS,
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(CLASS_LABELS)
    }

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", labels=CLASS_IDS)),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted", labels=CLASS_IDS)
        ),
        "per_class": per_class,
    }
    return metrics, predictions


def per_person_metrics(
    df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    rows = []
    eval_df = df[["person_id", "label"]].copy()
    eval_df["true_id"] = eval_df["label"].map(prep.LABEL_TO_ID).astype(int)
    eval_df["pred_id"] = predictions

    for person_id, group in eval_df.groupby("person_id", sort=True):
        rows.append(
            {
                "person_id": person_id,
                "samples": int(len(group)),
                "accuracy": float(accuracy_score(group["true_id"], group["pred_id"])),
                "macro_f1": float(
                    f1_score(
                        group["true_id"],
                        group["pred_id"],
                        average="macro",
                        labels=CLASS_IDS,
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        group["true_id"],
                        group["pred_id"],
                        average="weighted",
                        labels=CLASS_IDS,
                        zero_division=0,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_LABELS)), labels=CLASS_LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(CLASS_LABELS)), labels=CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("V03 Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_v02_comparison() -> dict[str, Any] | None:
    if not V02_METRICS_PATH.is_file():
        return None

    with V02_METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    comparison: dict[str, Any] = {
        "metrics": metrics.get("overall_metrics", {}),
        "test_persons": metrics.get("test_persons", []),
    }

    if V02_CLASS_REPORT_PATH.is_file():
        class_report = pd.read_csv(V02_CLASS_REPORT_PATH)
        comparison["per_class_f1"] = {
            str(row["class"]): float(row["f1_score"])
            for _, row in class_report.iterrows()
            if str(row["class"]) in CLASS_LABELS
        }

    if V02_PER_PERSON_PATH.is_file():
        per_person = pd.read_csv(V02_PER_PERSON_PATH)
        comparison["per_person_macro_f1"] = {
            str(row["person_id"]): float(row["macro_f1"])
            for _, row in per_person.iterrows()
        }

    return comparison


def save_outputs(
    best_model: Any,
    candidate_results: list[dict[str, Any]],
    best_info: dict[str, Any],
    feature_columns: list[str],
    prepared: prep.PreparedDataset,
    delta_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calibration_stats: CalibrationStats,
    v02_manifest: dict[str, Any],
    train_persons: list[str],
    validation_persons: list[str],
    test_persons: list[str],
    dataset_sha256: str,
    test_metrics: dict[str, Any],
    test_predictions: np.ndarray,
    validation_metrics: dict[str, Any],
    v02_comparison: dict[str, Any] | None,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version": "V03",
        "feature_mode": "delta",
        "number_of_features": len(feature_columns),
        "feature_columns": feature_columns,
        "calibration_samples": CALIBRATION_SAMPLES,
        "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
        "calibration_statistic": "median",
        "baseline_scope": "session",
        "classifier": best_info["classifier"],
        "hyperparameters": best_info["hyperparameters"],
        "candidate_results": candidate_results,
        "train_persons": train_persons,
        "validation_persons": validation_persons,
        "test_persons": test_persons,
        "random_seed": RANDOM_SEED,
        "label_to_id": prep.LABEL_TO_ID,
        "id_to_label": prep.ID_TO_LABEL,
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha256,
        "dataset_statistics": asdict(calibration_stats),
        "split_samples_after_calibration": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "train_class_distribution": class_distribution(train_df),
        "validation_class_distribution": class_distribution(validation_df),
        "test_class_distribution": class_distribution(test_df),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_timestamp": timestamp,
    }
    write_json(MODEL_DIR / "training_metadata.json", metadata)

    split_manifest = {
        "version": "V03",
        "feature_mode": "delta",
        "split_strategy": "Reuse V02 test persons; deterministic validation holdout from V02 train persons",
        "source_v02_split_manifest": v02_manifest,
        "group_column": "person_id",
        "recording_column": "recording_id",
        "random_seed": RANDOM_SEED,
        "train_persons": train_persons,
        "validation_persons": validation_persons,
        "test_persons": test_persons,
        "train_recording_ids": sorted(train_df["recording_id"].unique().tolist()),
        "validation_recording_ids": sorted(validation_df["recording_id"].unique().tolist()),
        "test_recording_ids": sorted(test_df["recording_id"].unique().tolist()),
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(validation_df)),
        "test_samples": int(len(test_df)),
        "person_overlap": [],
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_sha256,
    }
    write_json(MODEL_DIR / "split_manifest.json", split_manifest)

    write_json(
        MODEL_DIR / "calibration_config.json",
        {
            "feature_mode": "delta",
            "calibration_samples": CALIBRATION_SAMPLES,
            "minimum_calibration_samples": MIN_CALIBRATION_SAMPLES,
            "calibration_statistic": "median",
            "baseline_scope": "session",
            "sort_column": "image_path",
            "calibration_rows_excluded_from_training_and_evaluation": True,
            "stats": asdict(calibration_stats),
        },
    )

    write_json(
        MODEL_DIR / "feature_schema.json",
        {
            "version": "V03",
            "feature_mode": "delta",
            "source": "FeatureExtractor.FEATURE_NAMES",
            "number_of_features": len(feature_columns),
            "feature_columns": feature_columns,
        },
    )

    y_test = test_df["label"].map(prep.LABEL_TO_ID).astype(int)
    save_confusion_matrix(y_test, test_predictions, RESULTS_DIR / "confusion_matrix.png")

    report_text = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_IDS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_IDS,
        target_names=CLASS_LABELS,
        zero_division=0,
        output_dict=True,
    )
    write_json(RESULTS_DIR / "classification_report.json", report_dict)

    person_metrics = per_person_metrics(test_df, test_predictions)
    person_metrics.to_csv(RESULTS_DIR / "per_person_metrics.csv", index=False)

    predictions_df = test_df[
        ["image_path", "session_id", "person_id", "recording_id", "label"]
    ].copy()
    predictions_df["true_id"] = y_test.to_numpy()
    predictions_df["pred_id"] = test_predictions
    predictions_df["pred_label"] = [prep.ID_TO_LABEL[int(value)] for value in test_predictions]
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    metrics_payload = {
        "version": "V03",
        "feature_mode": "delta",
        "best_model": best_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "candidate_results": candidate_results,
        "class_order": CLASS_LABELS,
        "test_persons": test_persons,
        "test_samples": int(len(test_df)),
        "dataset_statistics": asdict(calibration_stats),
        "v02_comparison": v02_comparison,
    }
    write_json(RESULTS_DIR / "metrics.json", metrics_payload)

    if v02_comparison is not None:
        comparison_payload = {
            "v02_raw_29_features": v02_comparison,
            "v03_delta_29_features": {
                "metrics": test_metrics,
                "per_class_f1": {
                    label: test_metrics["per_class"][label]["f1"] for label in CLASS_LABELS
                },
                "per_person_macro_f1": {
                    row["person_id"]: float(row["macro_f1"])
                    for _, row in person_metrics.iterrows()
                },
            },
            "note": "Do not claim V03 is better unless these metrics support it.",
        }
        write_json(RESULTS_DIR / "v02_vs_v03_summary.json", comparison_payload)

    write_report(
        best_info=best_info,
        test_metrics=test_metrics,
        validation_metrics=validation_metrics,
        person_metrics=person_metrics,
        calibration_stats=calibration_stats,
        train_persons=train_persons,
        validation_persons=validation_persons,
        test_persons=test_persons,
        v02_comparison=v02_comparison,
        prepared=prepared,
        delta_df=delta_df,
    )


def write_report(
    best_info: dict[str, Any],
    test_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    person_metrics: pd.DataFrame,
    calibration_stats: CalibrationStats,
    train_persons: list[str],
    validation_persons: list[str],
    test_persons: list[str],
    v02_comparison: dict[str, Any] | None,
    prepared: prep.PreparedDataset,
    delta_df: pd.DataFrame,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DOCS_DIR / f"v03_personal_baseline_calibration_{datetime.now().date()}.md"

    per_class_lines = "\n".join(
        f"- {label}: F1={test_metrics['per_class'][label]['f1']:.4f}, "
        f"precision={test_metrics['per_class'][label]['precision']:.4f}, "
        f"recall={test_metrics['per_class'][label]['recall']:.4f}"
        for label in CLASS_LABELS
    )
    per_person_lines = "\n".join(
        f"- {row['person_id']}: macro F1={row['macro_f1']:.4f}, "
        f"accuracy={row['accuracy']:.4f}, samples={int(row['samples'])}"
        for _, row in person_metrics.iterrows()
    )

    v02_lines = "V02 artifacts were not available for comparison."
    if v02_comparison is not None:
        v02_metrics = v02_comparison.get("metrics", {})
        v02_lines = (
            f"- V02 accuracy: {v02_metrics.get('accuracy')}\n"
            f"- V02 macro F1: {v02_metrics.get('macro_f1')}\n"
            f"- V03 accuracy: {test_metrics['accuracy']:.4f}\n"
            f"- V03 macro F1: {test_metrics['macro_f1']:.4f}"
        )

    content = f"""# V03 Personal Baseline Calibration

## 1. Goal
Implement V03 delta features: `delta_f = raw_f - session_baseline`, where the
session baseline is the median of an initial correct-posture calibration window.

## 2. Design
V03 is implemented beside V02. Existing V02 files and artifacts are not
overwritten. Runtime prediction uses `PosturePredictorV03` and
`PersonalCalibration`.

## 3. Files Added
- `src/personal_calibration.py`
- `src/posture_predictor_v03.py`
- `src/train_v03.py`
- `scripts/test_webcam_v03.py`
- `scripts/dry_run_v03_inference.py`
- `tests/test_personal_calibration.py`

## 4. Files Modified
No V02 source files are modified by the V03 implementation.

## 5. Calibration Algorithm
- Scope: per `recording_id` (`person_id__session_id`)
- Sort calibration candidates by `image_path`
- Use up to {CALIBRATION_SAMPLES} initial `correct` samples
- Require at least {MIN_CALIBRATION_SAMPLES} `correct` samples
- Baseline statistic: median
- Remove calibration rows from train/validation/test samples

## 6. Data Leakage Prevention
- Reused V02 person holdout test persons
- Created validation persons only from V02 train persons
- No person appears in more than one split
- Calibration rows are excluded before model fitting/evaluation
- Baseline subtraction is done in raw feature space before model preprocessing
- Scaler fitting happens inside the SVM sklearn pipeline on train folds only

## 7. Training Protocol
- Train persons: {train_persons}
- Validation persons: {validation_persons}
- Test persons: {test_persons}
- Model selection: validation macro F1
- Final metrics: held-out test persons

## 8. Dataset Statistics
- Raw samples: {calibration_stats.original_samples}
- Raw persons: {calibration_stats.original_persons}
- Raw sessions: {calibration_stats.original_sessions}
- Raw recordings: {calibration_stats.original_recordings}
- Calibration samples removed: {calibration_stats.calibration_samples_removed}
- Delta samples: {len(delta_df)}
- Skipped sessions: {len(calibration_stats.skipped_sessions)}

## 9. Model Results
- Best model: {best_info['classifier']}
- Validation accuracy: {validation_metrics['accuracy']:.4f}
- Validation macro F1: {validation_metrics['macro_f1']:.4f}
- Test accuracy: {test_metrics['accuracy']:.4f}
- Test macro F1: {test_metrics['macro_f1']:.4f}
- Test weighted F1: {test_metrics['weighted_f1']:.4f}

Per-class test metrics:
{per_class_lines}

## 10. V02 vs V03
{v02_lines}

## 11. Realtime Usage
Run:

```bash
python scripts/test_webcam_v03.py
```

Keys:
- `C`: calibrate or recalibrate
- `R`: reset calibration
- `Q` or `Esc`: quit

## 12. Limitations
- Offline calibration assumes the earliest `correct` images in each recording
  are a valid calibration window.
- The first V03 experiment uses delta-only 29-dimensional features.
- Webcam access was not required for training and should be verified on the
  target machine.

## 13. Next Experiments
- Compare repeated live calibration windows for stability.
- Try `raw_delta` features only after delta-only behavior is understood.
- Add calibration quality checks such as median absolute deviation thresholds.
"""
    report_path.write_text(content, encoding="utf-8")


def run_dataset_dry_run() -> dict[str, Any]:
    prepared = prep.prepare_dataset(DATASET_PATH, excluded_persons=())
    feature_columns = prepared.feature_columns
    _, test_persons, _ = load_v02_split(V02_SPLIT_PATH)

    df = prepared.df[prepared.df["person_id"].astype(str).isin(test_persons)].copy()
    if df.empty:
        raise RuntimeError("No V02 test rows available for dry-run.")

    for recording_id, group in df.groupby("recording_id", sort=True):
        correct = group[group["label"] == "correct"].sort_values("image_path", kind="mergesort")
        if len(correct) < CALIBRATION_SAMPLES + 1:
            continue

        calibration_rows = correct.head(CALIBRATION_SAMPLES)
        candidate = group.drop(index=calibration_rows.index).sort_values("image_path", kind="mergesort").head(1)
        if candidate.empty:
            continue

        calibrator = PersonalCalibration(feature_count=len(feature_columns))
        for _, sample in calibration_rows.iterrows():
            calibrator.add_sample(sample[feature_columns].to_numpy(dtype=float))
        calibrator.calculate_baseline()

        predictor = PosturePredictorV03()
        raw_features = candidate.iloc[0][feature_columns].to_numpy(dtype=float)
        delta_features = calibrator.transform(raw_features)
        pred_label, probability, _ = predictor.predict_delta(delta_features)

        result = {
            "recording_id": str(recording_id),
            "true_label": str(candidate.iloc[0]["label"]),
            "predicted_label": pred_label,
            "probability": probability,
            "feature_shape": list(raw_features.shape),
            "delta_feature_shape": list(delta_features.shape),
            "image_path": str(candidate.iloc[0]["image_path"]),
        }
        write_json(RESULTS_DIR / "dry_run_inference.json", result)
        return result

    raise RuntimeError("No suitable test recording found for dry-run.")


def train_v03(args: argparse.Namespace) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared = prep.prepare_dataset(DATASET_PATH, excluded_persons=())
    feature_columns = prepared.feature_columns
    dataset_sha256 = sha256_file(DATASET_PATH)

    v02_train_persons, v02_test_persons, v02_manifest = load_v02_split(V02_SPLIT_PATH)
    train_persons, validation_persons, test_persons = make_train_val_test_persons(
        v02_train_persons=v02_train_persons,
        v02_test_persons=v02_test_persons,
        validation_size=args.validation_size,
    )

    delta_df, calibration_stats = apply_session_delta_calibration(
        df=prepared.df,
        feature_columns=feature_columns,
        calibration_samples=args.calibration_samples,
        min_calibration_samples=args.min_calibration_samples,
    )

    train_df, validation_df, test_df = split_dataframe(
        delta_df,
        train_persons=train_persons,
        validation_persons=validation_persons,
        test_persons=test_persons,
    )

    X_train, y_train = make_xy(train_df, feature_columns)
    X_validation, y_validation = make_xy(validation_df, feature_columns)
    X_test, y_test = make_xy(test_df, feature_columns)
    groups_train = train_df["person_id"].astype(str)

    candidates: list[tuple[str, Any, dict[str, Any]]] = []
    svm_model, svm_info = fit_svm(X_train, y_train, groups_train)
    candidates.append(("svm", svm_model, svm_info))

    xgb_result = fit_xgboost(X_train, y_train)
    if xgb_result is not None:
        xgb_model, xgb_info = xgb_result
        candidates.append(("xgboost", xgb_model, xgb_info))

    candidate_results: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_info: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_validation_score = -np.inf

    for _, model, info in candidates:
        validation_metrics, _ = evaluate_model(model, X_validation, y_validation)
        result = {
            **info,
            "validation_metrics": validation_metrics,
        }
        candidate_results.append(result)

        if validation_metrics["macro_f1"] > best_validation_score:
            best_validation_score = validation_metrics["macro_f1"]
            best_model = model
            best_info = info
            best_validation_metrics = validation_metrics

    if best_model is None or best_info is None or best_validation_metrics is None:
        raise RuntimeError("No model candidates were trained.")

    test_metrics, test_predictions = evaluate_model(best_model, X_test, y_test)
    v02_comparison = load_v02_comparison()

    save_outputs(
        best_model=best_model,
        candidate_results=candidate_results,
        best_info=best_info,
        feature_columns=feature_columns,
        prepared=prepared,
        delta_df=delta_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        calibration_stats=calibration_stats,
        v02_manifest=v02_manifest,
        train_persons=train_persons,
        validation_persons=validation_persons,
        test_persons=test_persons,
        dataset_sha256=dataset_sha256,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        validation_metrics=best_validation_metrics,
        v02_comparison=v02_comparison,
    )

    dry_run_result = run_dataset_dry_run()

    summary = {
        "dataset": {
            "raw_samples": int(len(prepared.df)),
            "raw_persons": int(prepared.df["person_id"].nunique()),
            "raw_recordings": int(prepared.df["recording_id"].nunique()),
            "delta_samples": int(len(delta_df)),
        },
        "calibration": asdict(calibration_stats),
        "train_persons": train_persons,
        "validation_persons": validation_persons,
        "test_persons": test_persons,
        "best_model": best_info,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "dry_run": dry_run_result,
    }
    write_json(RESULTS_DIR / "training_summary.json", summary)

    print(
        json.dumps(
            sanitize_for_json(summary),
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V03 personal baseline model.")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=CALIBRATION_SAMPLES,
        help="Target number of correct samples used for each session baseline.",
    )
    parser.add_argument(
        "--min-calibration-samples",
        type=int,
        default=MIN_CALIBRATION_SAMPLES,
        help="Minimum correct samples required to keep a session.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.25,
        help="Fraction of V02 train persons held out for validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global CALIBRATION_SAMPLES, MIN_CALIBRATION_SAMPLES

    args = parse_args(argv)
    if args.min_calibration_samples <= 0:
        raise ValueError("--min-calibration-samples must be positive.")
    if args.calibration_samples < args.min_calibration_samples:
        raise ValueError("--calibration-samples must be >= --min-calibration-samples.")

    CALIBRATION_SAMPLES = args.calibration_samples
    MIN_CALIBRATION_SAMPLES = args.min_calibration_samples

    train_v03(args)


if __name__ == "__main__":
    main(sys.argv[1:])
