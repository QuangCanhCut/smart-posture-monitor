"""
Official held-out evaluation for Smart Posture Monitor V02.

Run:
    python -m src.evaluate

Inputs:
    data/processed/features.csv
    models/best_model.joblib
    models/split_manifest.json
    models/training_metadata.json

This script never retrains, retunes, reselects a model, or creates a new split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import EXPECTED_CLASSES, ID_TO_LABEL, LABEL_TO_ID, prepare_dataset

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "models" / "split_manifest.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file:\n{path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON phải chứa object/dict: {path}")
    return payload


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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False))


# ============================================================
# Artifact and contract validation
# ============================================================

def load_artifacts(
    model_path: Path,
    split_path: Path,
    metadata_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy trained model:\n{model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError("Model artifact không có method predict().")

    return model, read_json(split_path), read_json(metadata_path)


def validate_dataset_hash(
    data_path: Path,
    split_manifest: dict[str, Any],
    training_metadata: dict[str, Any],
    allow_hash_mismatch: bool,
) -> str:
    actual = sha256_file(data_path)
    expected = []

    if split_manifest.get("dataset_sha256"):
        expected.append(("split_manifest.json", str(split_manifest["dataset_sha256"])))
    if training_metadata.get("dataset_sha256"):
        expected.append(("training_metadata.json", str(training_metadata["dataset_sha256"])))

    mismatches = [(source, value) for source, value in expected if value != actual]
    if mismatches and not allow_hash_mismatch:
        details = "\n".join(f"- {source}: {value}" for source, value in mismatches)
        raise RuntimeError(
            "DATASET SHA256 MISMATCH. features.csv hiện tại không phải đúng dataset "
            "đã dùng lúc training/split.\n"
            f"Actual: {actual}\n{details}"
        )

    if mismatches:
        print("[WARNING] Dataset hash mismatch was explicitly allowed.")

    return actual


def validate_training_contract(
    model: Any,
    feature_columns: list[str],
    training_metadata: dict[str, Any],
) -> None:
    trained_features = training_metadata.get("feature_columns")
    if not isinstance(trained_features, list):
        raise ValueError("training_metadata.json thiếu feature_columns.")

    trained_features = [str(x) for x in trained_features]
    if feature_columns != trained_features:
        raise RuntimeError("FEATURE SCHEMA MISMATCH giữa dataset và training metadata.")

    if training_metadata.get("n_features") is not None:
        if int(training_metadata["n_features"]) != len(feature_columns):
            raise RuntimeError("n_features trong training metadata không khớp.")

    model_n_features = getattr(model, "n_features_in_", None)
    if model_n_features is not None and int(model_n_features) != len(feature_columns):
        raise RuntimeError("Model mong đợi số feature khác dataset hiện tại.")

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        if [str(x) for x in model_feature_names] != feature_columns:
            raise RuntimeError("Feature names/order trong model không khớp dataset.")

    metadata_mapping = training_metadata.get("label_to_id")
    if metadata_mapping is not None:
        normalized = {str(k): int(v) for k, v in metadata_mapping.items()}
        expected = {str(k): int(v) for k, v in LABEL_TO_ID.items()}
        if normalized != expected:
            raise RuntimeError("LABEL MAPPING MISMATCH.")

    metadata_classes = training_metadata.get("classes")
    if metadata_classes is not None:
        if [str(x) for x in metadata_classes] != list(EXPECTED_CLASSES):
            raise RuntimeError("CLASS ORDER MISMATCH.")


# ============================================================
# Locked test reconstruction
# ============================================================

def build_locked_test_set(
    prepared: Any,
    split_manifest: dict[str, Any],
    training_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    test_persons_raw = split_manifest.get("test_persons")
    if not isinstance(test_persons_raw, list) or not test_persons_raw:
        raise ValueError("split_manifest.json không chứa test_persons hợp lệ.")

    test_persons = sorted(str(x) for x in test_persons_raw)
    train_persons = sorted(str(x) for x in split_manifest.get("train_persons", []))

    overlap = sorted(set(train_persons) & set(test_persons))
    if overlap:
        raise RuntimeError(f"Manifest chứa person leakage: {overlap}")

    metadata_test_persons = training_metadata.get("test_persons")
    if metadata_test_persons is not None:
        metadata_test_persons = sorted(str(x) for x in metadata_test_persons)
        if metadata_test_persons != test_persons:
            raise RuntimeError(
                "test_persons trong split_manifest và training_metadata không khớp."
            )

    groups = prepared.groups.astype(str)
    missing = sorted(set(test_persons) - set(groups.unique()))
    if missing:
        raise RuntimeError(f"Test persons không tồn tại trong dataset: {missing}")

    mask = groups.isin(test_persons)
    X_test = prepared.X.loc[mask].copy()
    y_test = prepared.y.loc[mask].copy()
    metadata_test = prepared.metadata.loc[mask].copy()

    expected_samples = split_manifest.get("test_samples")
    if expected_samples is not None and int(expected_samples) != len(X_test):
        raise RuntimeError(
            f"Test sample count mismatch: expected={expected_samples}, actual={len(X_test)}."
        )

    actual_recordings = sorted(metadata_test["recording_id"].astype(str).unique())
    expected_recordings = split_manifest.get("test_recording_ids")
    if expected_recordings is not None:
        if actual_recordings != sorted(str(x) for x in expected_recordings):
            raise RuntimeError("Test recording IDs không khớp split_manifest.")

    present_ids = set(int(x) for x in y_test.unique())
    missing_ids = sorted(set(ID_TO_LABEL) - present_ids)
    if missing_ids:
        raise RuntimeError(
            "Locked test set thiếu class: "
            f"{[ID_TO_LABEL[class_id] for class_id in missing_ids]}"
        )

    return X_test, y_test, metadata_test, test_persons


# ============================================================
# Metrics and tables
# ============================================================

def overall_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(ID_TO_LABEL)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def class_report_frame(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    class_ids = sorted(ID_TO_LABEL)
    names = [ID_TO_LABEL[class_id] for class_id in class_ids]
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(
        [
            {
                "class": name,
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in names
        ]
    )


def per_person_frame(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata_test: pd.DataFrame,
) -> pd.DataFrame:
    labels = sorted(ID_TO_LABEL)
    work = metadata_test[["person_id"]].copy()
    work["y_true"] = np.asarray(y_true, dtype=int)
    work["y_pred"] = np.asarray(y_pred, dtype=int)

    rows = []
    for person_id, frame in work.groupby("person_id", sort=True):
        truth = frame["y_true"].to_numpy()
        pred = frame["y_pred"].to_numpy()
        rows.append(
            {
                "person_id": str(person_id),
                "samples": int(len(frame)),
                "accuracy": float(accuracy_score(truth, pred)),
                "macro_precision": float(
                    precision_score(
                        truth, pred, labels=labels, average="macro", zero_division=0
                    )
                ),
                "macro_recall": float(
                    recall_score(
                        truth, pred, labels=labels, average="macro", zero_division=0
                    )
                ),
                "macro_f1": float(
                    f1_score(truth, pred, labels=labels, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(
                        truth, pred, labels=labels, average="weighted", zero_division=0
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def predictions_frame(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata_test: pd.DataFrame,
    model: Any,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    result = metadata_test[
        ["image_path", "session_id", "person_id", "recording_id", "label"]
    ].copy()

    true_ids = np.asarray(y_true, dtype=int)
    pred_ids = np.asarray(y_pred, dtype=int)

    result["true_label_id"] = true_ids
    result["predicted_label_id"] = pred_ids
    result["true_label"] = [ID_TO_LABEL[int(x)] for x in true_ids]
    result["predicted_label"] = [ID_TO_LABEL[int(x)] for x in pred_ids]
    result["correct"] = result["true_label_id"] == result["predicted_label_id"]

    # Optional: SVM may have probability=False, so this column may not exist.
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X_test), dtype=float)
            classes = np.asarray(getattr(model, "classes_", []))
            values = []

            for row_index, pred_id in enumerate(pred_ids):
                probability = np.nan
                if probs.ndim == 2 and row_index < probs.shape[0]:
                    if classes.size == probs.shape[1]:
                        positions = np.where(classes == pred_id)[0]
                        if positions.size == 1:
                            probability = float(probs[row_index, positions[0]])
                values.append(probability)

            result["predicted_probability"] = values
        except (AttributeError, TypeError, ValueError):
            pass

    return result


# ============================================================
# Plots
# ============================================================

def save_confusion_plot(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    normalized: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_per_person_plot(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("person_id")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["person_id"], ordered["macro_f1"])
    ax.set(
        xlabel="Held-out person",
        ylabel="Macro F1",
        title="Macro F1 by held-out person",
        ylim=(0.0, 1.0),
    )
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Markdown report
# ============================================================

def write_summary(
    path: Path,
    model_name: str,
    test_persons: list[str],
    metrics: dict[str, float],
    class_report: pd.DataFrame,
    per_person: pd.DataFrame,
    confusion: np.ndarray,
    cv_mean: float | None,
    cv_std: float | None,
    dataset_sha256: str,
) -> None:
    lines = [
        "# Smart Posture Monitor V02 - Held-out Evaluation",
        "",
        f"- Model: `{model_name}`",
        f"- Test persons: `{test_persons}`",
        f"- Test samples: `{int(per_person['samples'].sum())}`",
        f"- Dataset SHA256: `{dataset_sha256}`",
        "",
        "## Overall metrics",
        "",
    ]

    for name, value in metrics.items():
        lines.append(f"- {name}: `{value:.6f}`")

    if cv_mean is not None:
        lines += [
            "",
            "## Training-CV reference",
            "",
            f"- Selected-model CV Macro F1 mean: `{cv_mean:.6f}`",
        ]
        if cv_std is not None:
            lines.append(f"- Selected-model CV Macro F1 std: `{cv_std:.6f}`")
        lines.append(
            f"- Held-out minus CV Macro F1: `{metrics['macro_f1'] - cv_mean:+.6f}`"
        )

    lines += [
        "",
        "## Per-class metrics",
        "",
        class_report.to_markdown(index=False),
        "",
        "## Per-person metrics",
        "",
        per_person.to_markdown(index=False),
        "",
        "## Confusion matrix",
        "",
        "Rows are true labels; columns are predicted labels.",
        "",
        "```text",
        np.array2string(confusion),
        "```",
        "",
        "## Protocol",
        "",
        "- No fit/retrain.",
        "- No hyperparameter tuning.",
        "- No new split.",
        "- Test persons come only from `models/split_manifest.json`.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Official evaluation
# ============================================================

def evaluate(
    data_path: Path,
    model_path: Path,
    split_path: Path,
    metadata_path: Path,
    output_dir: Path,
    allow_hash_mismatch: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SMART POSTURE MONITOR V02 - OFFICIAL HELD-OUT EVALUATION")
    print("=" * 72)
    print(f"Dataset : {data_path}")
    print(f"Model   : {model_path}")
    print(f"Split   : {split_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Results : {output_dir}")

    print("\n[1/6] Loading saved training artifacts...")
    model, split_manifest, training_metadata = load_artifacts(
        model_path=model_path,
        split_path=split_path,
        metadata_path=metadata_path,
    )

    print("[2/6] Verifying exact dataset version...")
    dataset_sha256 = validate_dataset_hash(
        data_path=data_path,
        split_manifest=split_manifest,
        training_metadata=training_metadata,
        allow_hash_mismatch=allow_hash_mismatch,
    )

    print("[3/6] Loading dataset through src.preprocessing.prepare_dataset()...")
    prepared = prepare_dataset(csv_path=data_path, excluded_persons=())
    feature_columns = list(prepared.feature_columns)
    validate_training_contract(model, feature_columns, training_metadata)

    print("[4/6] Reconstructing locked unseen-person test set...")
    X_test, y_test, metadata_test, test_persons = build_locked_test_set(
        prepared=prepared,
        split_manifest=split_manifest,
        training_metadata=training_metadata,
    )

    print(f"      Test persons ({len(test_persons)}): {test_persons}")
    print(f"      Test samples: {len(X_test)}")
    print(f"      Recordings: {metadata_test['recording_id'].nunique()}")

    print("[5/6] Running the official prediction pass...")
    y_pred = np.asarray(model.predict(X_test), dtype=int).reshape(-1)

    if len(y_pred) != len(y_test):
        raise RuntimeError("Số prediction không bằng số test samples.")

    unknown_ids = sorted(set(int(x) for x in y_pred) - set(ID_TO_LABEL))
    if unknown_ids:
        raise RuntimeError(f"Model trả về unknown class IDs: {unknown_ids}")

    metrics = overall_metrics(y_test, y_pred)
    class_report = class_report_frame(y_test, y_pred)
    per_person = per_person_frame(y_test, y_pred, metadata_test)
    predictions = predictions_frame(y_test, y_pred, metadata_test, model, X_test)
    mistakes = predictions.loc[~predictions["correct"]].copy()

    class_ids = sorted(ID_TO_LABEL)
    class_names = [ID_TO_LABEL[class_id] for class_id in class_ids]
    cm = confusion_matrix(y_test, y_pred, labels=class_ids)
    cm_norm = confusion_matrix(y_test, y_pred, labels=class_ids, normalize="true")

    cm_frame = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_norm_frame = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    print("[6/6] Saving metrics, predictions, errors and plots...")

    model_name = str(training_metadata.get("model_name", type(model).__name__))
    cv_mean_raw = training_metadata.get("cv_macro_f1_mean")
    cv_std_raw = training_metadata.get("cv_macro_f1_std")
    cv_mean = float(cv_mean_raw) if cv_mean_raw is not None else None
    cv_std = float(cv_std_raw) if cv_std_raw is not None else None

    metrics_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_sha256": dataset_sha256,
        "test_persons": test_persons,
        "test_samples": int(len(X_test)),
        "test_recordings": int(metadata_test["recording_id"].nunique()),
        "class_order": class_names,
        "overall_metrics": metrics,
        "selected_model_cv_macro_f1_mean": cv_mean,
        "selected_model_cv_macro_f1_std": cv_std,
        "heldout_minus_cv_macro_f1": (
            float(metrics["macro_f1"] - cv_mean) if cv_mean is not None else None
        ),
        "correct_predictions": int(predictions["correct"].sum()),
        "incorrect_predictions": int((~predictions["correct"]).sum()),
    }

    write_json(output_dir / "test_metrics.json", metrics_payload)
    class_report.to_csv(output_dir / "classification_report.csv", index=False)
    per_person.to_csv(output_dir / "per_person_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    mistakes.to_csv(output_dir / "misclassifications.csv", index=False)
    cm_frame.to_csv(output_dir / "confusion_matrix.csv", index_label="true_label")
    cm_norm_frame.to_csv(
        output_dir / "confusion_matrix_normalized.csv",
        index_label="true_label",
    )

    save_confusion_plot(
        cm,
        class_names,
        output_dir / "confusion_matrix.png",
        "Held-out Test Confusion Matrix",
        normalized=False,
    )
    save_confusion_plot(
        cm_norm,
        class_names,
        output_dir / "confusion_matrix_normalized.png",
        "Held-out Test Confusion Matrix (Normalized)",
        normalized=True,
    )
    save_per_person_plot(per_person, output_dir / "per_person_macro_f1.png")

    write_summary(
        output_dir / "evaluation_summary.md",
        model_name=model_name,
        test_persons=test_persons,
        metrics=metrics,
        class_report=class_report,
        per_person=per_person,
        confusion=cm,
        cv_mean=cv_mean,
        cv_std=cv_std,
        dataset_sha256=dataset_sha256,
    )

    print("\n" + "=" * 72)
    print("OFFICIAL HELD-OUT TEST RESULTS")
    print("=" * 72)
    print(f"Model          : {model_name}")
    print(f"Test persons   : {test_persons}")
    print(f"Test samples   : {len(X_test)}")
    print(f"Accuracy       : {metrics['accuracy']:.6f}")
    print(f"Balanced Acc.  : {metrics['balanced_accuracy']:.6f}")
    print(f"Macro Precision: {metrics['macro_precision']:.6f}")
    print(f"Macro Recall   : {metrics['macro_recall']:.6f}")
    print(f"Macro F1       : {metrics['macro_f1']:.6f}")
    print(f"Weighted F1    : {metrics['weighted_f1']:.6f}")

    if cv_mean is not None:
        print(f"CV Macro F1    : {cv_mean:.6f}")
        print(f"Test - CV F1   : {metrics['macro_f1'] - cv_mean:+.6f}")

    print_table("Per-class metrics", class_report)
    print_table("Per-person metrics", per_person)

    print("\nConfusion matrix")
    print("----------------")
    print(cm_frame.to_string())

    print(f"\nCorrect        : {int(predictions['correct'].sum())}")
    print(f"Incorrect      : {int((~predictions['correct']).sum())}")
    print(f"Results saved  : {output_dir}")
    print("\nProtocol confirmed: no retrain, no tuning, no new split.")
    print("=" * 72)

    return {
        "metrics": metrics_payload,
        "classification_report": class_report,
        "per_person_metrics": per_person,
        "confusion_matrix": cm_frame,
        "predictions": predictions,
        "misclassifications": mistakes,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official unseen-person evaluation for Smart Posture Monitor V02."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--training-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Debug only: allow evaluation if current features.csv hash differs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        data_path=resolve_project_path(args.data),
        model_path=resolve_project_path(args.model),
        split_path=resolve_project_path(args.split_manifest),
        metadata_path=resolve_project_path(args.training_metadata),
        output_dir=resolve_project_path(args.output_dir),
        allow_hash_mismatch=args.allow_hash_mismatch,
    )


if __name__ == "__main__":
    main()
