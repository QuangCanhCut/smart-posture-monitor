"""
Train and Export Production Hybrid Model with Personal Baseline Delta.

Trains an SVM RBF classifier on the 40D Hybrid Features (32 absolute + 8 personal deltas)
using the locked training subjects from models/split_manifest.json, and exports:
- baseline_calibration/models/best_hybrid_model.joblib
- baseline_calibration/models/hybrid_training_metadata.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    SENSITIVE_DELTA_FEATURES_8,
    get_hybrid_feature_names,
)
from baseline_calibration.offline_dataset_builder import build_offline_baseline_dataset

SPLIT_MANIFEST_PATH = PROJECT_ROOT / "models" / "split_manifest.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"

CLASS_NAMES = ["correct", "forward_slouch", "lean_left", "lean_right"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASS_NAMES)}


def train_hybrid_model(scheme: str = "selective", n_calib_frames: int = 30) -> Dict[str, Any]:
    """
    Trains the hybrid model and saves artifacts into baseline_calibration/models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_persons = manifest["train_persons"]
    test_persons = manifest["test_persons"]

    # 1. Build offline dataset
    print(f"[TrainHybrid] Building {scheme} hybrid dataset (calib N={n_calib_frames})...")
    df = build_offline_baseline_dataset(
        n_calib_frames=n_calib_frames,
        scheme=scheme,
        save_output=False,
    )

    feature_cols = get_hybrid_feature_names(scheme=scheme)
    print(f"[TrainHybrid] Feature vector dimension: {len(feature_cols)}")

    train_mask = df["person_id"].isin(train_persons)
    test_mask = df["person_id"].isin(test_persons)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "label"].map(LABEL_TO_ID).values
    groups_train = df.loc[train_mask, "person_id"].values

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "label"].map(LABEL_TO_ID).values

    print(f"[TrainHybrid] Training on {len(X_train)} samples across {len(train_persons)} persons...")
    print(f"[TrainHybrid] Testing on {len(X_test)} samples across {len(test_persons)} persons...")

    # 2. Hyperparameter Grid Search with GroupKFold
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "svm__C": [1.0, 5.0, 10.0],
        "svm__gamma": [0.005, 0.01, 0.02, "scale"],
    }

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups_train))
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=splits,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_score = float(grid.best_score_)

    print(f"[TrainHybrid] Best CV Macro F1: {best_cv_score:.4f} with params: {best_params}")

    # 3. Evaluate on locked test set
    y_pred = best_pipeline.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    class_f1_dict = {cls: float(per_class_f1[i]) for i, cls in enumerate(CLASS_NAMES)}

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    fs_as_lr = int(cm[1, 3])
    lr_as_fs = int(cm[3, 1])

    print("\n[TrainHybrid] Test Set Evaluation Results:")
    print(f" -> Accuracy: {test_acc*100:.2f}%")
    print(f" -> Macro F1: {test_macro_f1:.4f}")
    print(f" -> Class F1s: {class_f1_dict}")
    print(f" -> FS as LR: {fs_as_lr} | LR as FS: {lr_as_fs} | Total FS<->LR: {fs_as_lr + lr_as_fs}")

    # 4. Save artifacts
    model_save_path = MODELS_DIR / "best_hybrid_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"[TrainHybrid] Saved trained pipeline to: {model_save_path}")

    metadata = {
        "model_name": f"SVM RBF Hybrid ({scheme})",
        "scheme": scheme,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "best_params": best_params,
        "cv_macro_f1": best_cv_score,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_class_f1": class_f1_dict,
        "fs_as_lr_confusion": fs_as_lr,
        "lr_as_fs_confusion": lr_as_fs,
        "classes": CLASS_NAMES,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "train_persons": train_persons,
        "test_persons": test_persons,
    }

    meta_save_path = MODELS_DIR / "hybrid_training_metadata.json"
    with meta_save_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[TrainHybrid] Saved metadata to: {meta_save_path}")

    return metadata


if __name__ == "__main__":
    train_hybrid_model(scheme="selective", n_calib_frames=30)
