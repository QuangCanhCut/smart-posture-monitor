"""
Offline Evaluator for Personal Baseline Delta.

Compares models with and without Personal Baseline Calibration on the locked split:
- Test persons: person01, person12, person13 (from models/split_manifest.json)
- Evaluates:
  1. Baseline-Free 32 Features (V03)
  2. Hybrid 40 Features (32 absolute + 8 sensitive deltas)
  3. Clean Hybrid 28 Features (20 clean absolute + 8 sensitive deltas - Config G-30)
  4. Full Hybrid 64 Features (32 absolute + 32 deltas)
- Measures:
  - Accuracy, Macro F1, Balanced Accuracy
  - Per-class F1 (specifically tracking forward_slouch and lean_right)
  - Cross-confusion count between forward_slouch and lean_right (FS <-> LR)
  - Per-person F1 scores and variance across test subjects
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    SENSITIVE_DELTA_FEATURES_8,
    CLEAN_FEATURES_20,
)
from baseline_calibration.offline_dataset_builder import build_offline_baseline_dataset
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "models" / "split_manifest.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CLASS_NAMES = ["correct", "forward_slouch", "lean_left", "lean_right"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_split_manifest(manifest_path: Optional[Union[str, Path]] = None) -> Tuple[List[str], List[str]]:
    """Loads train_persons and test_persons from manifest."""
    path = Path(manifest_path) if manifest_path else SPLIT_MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data["train_persons"], data["test_persons"]


def train_and_eval_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_persons: List[str],
    test_persons: List[str],
    model_name: str,
    C: float = 10.0,
    gamma: float = 0.01,
) -> Dict[str, Any]:
    """
    Train an SVM model on train_persons and evaluate on locked test_persons.
    """
    train_mask = df["person_id"].isin(train_persons)
    test_mask = df["person_id"].isin(test_persons)

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "label"].map(LABEL_TO_ID).values

    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "label"].map(LABEL_TO_ID).values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    per_class_f1 = f1_score(y_test, y_pred, average=None)
    class_f1_dict = {cls: float(per_class_f1[i]) for i, cls in enumerate(CLASS_NAMES)}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    # Class indices: 0: correct, 1: forward_slouch, 2: lean_left, 3: lean_right
    fs_as_lr = int(cm[1, 3])  # forward_slouch misclassified as lean_right
    lr_as_fs = int(cm[3, 1])  # lean_right misclassified as forward_slouch
    total_fs_lr_confusion = fs_as_lr + lr_as_fs

    # Per-person F1 evaluation
    per_person_f1 = {}
    for p_id in test_persons:
        p_mask = df.loc[test_mask, "person_id"] == p_id
        if p_mask.sum() > 0:
            y_p_true = y_test[p_mask.values]
            y_p_pred = y_pred[p_mask.values]
            p_f1 = float(f1_score(y_p_true, y_p_pred, average="macro", zero_division=0))
            per_person_f1[p_id] = p_f1

    f1_values = list(per_person_f1.values())
    per_person_std = float(np.std(f1_values)) if f1_values else 0.0
    worst_person_f1 = float(np.min(f1_values)) if f1_values else 0.0

    return {
        "model_name": model_name,
        "n_features": len(feature_cols),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "f1_correct": class_f1_dict.get("correct", 0.0),
        "f1_forward_slouch": class_f1_dict.get("forward_slouch", 0.0),
        "f1_lean_left": class_f1_dict.get("lean_left", 0.0),
        "f1_lean_right": class_f1_dict.get("lean_right", 0.0),
        "fs_as_lr": fs_as_lr,
        "lr_as_fs": lr_as_fs,
        "total_fs_lr_confusion": total_fs_lr_confusion,
        "per_person_f1": per_person_f1,
        "cross_person_std": per_person_std,
        "worst_person_f1": worst_person_f1,
        "confusion_matrix": cm.tolist(),
    }


def run_comprehensive_evaluation() -> pd.DataFrame:
    """
    Run evaluation across multiple feature configurations.
    """
    train_persons, test_persons = load_split_manifest()
    print(f"[Evaluator] Train persons ({len(train_persons)}): {train_persons}")
    print(f"[Evaluator] Test persons ({len(test_persons)}): {test_persons}")

    # Build hybrid dataset with N=30
    df = build_offline_baseline_dataset(scheme="full", n_calib_frames=30, save_output=False)

    # 1. Base 32 Features (Baseline-Free V03)
    feats_base_32 = FEATURE_NAMES_32
    
    # 2. Hybrid 40 Features (32 absolute + 8 sensitive deltas)
    delta_sens_8 = [f"delta_{c}" for c in SENSITIVE_DELTA_FEATURES_8]
    feats_hybrid_40 = FEATURE_NAMES_32 + delta_sens_8

    # 3. Clean Hybrid 28 Features (20 clean absolute + 8 sensitive deltas - Config G-30)
    feats_clean_hybrid_28 = CLEAN_FEATURES_20 + delta_sens_8

    # 4. Full Hybrid 64 Features (32 absolute + 32 deltas)
    delta_all_32 = [f"delta_{c}" for c in FEATURE_NAMES_32]
    feats_full_hybrid_64 = FEATURE_NAMES_32 + delta_all_32

    configs = [
        ("Base 32D (Baseline-Free)", feats_base_32, 10.0, 0.01),
        ("Hybrid 40D (32 Abs + 8 Delta)", feats_hybrid_40, 5.0, 0.01),
        ("Clean Hybrid 28D (20 Clean + 8 Delta)", feats_clean_hybrid_28, 5.0, 0.01),
        ("Full Hybrid 64D (32 Abs + 32 Delta)", feats_full_hybrid_64, 2.0, 0.005),
    ]

    results = []
    for name, cols, C, gamma in configs:
        print(f"\n[Evaluator] Running configuration: {name} (dim={len(cols)})...")
        res = train_and_eval_pipeline(
            df=df,
            feature_cols=cols,
            train_persons=train_persons,
            test_persons=test_persons,
            model_name=name,
            C=C,
            gamma=gamma,
        )
        results.append(res)
        print(f" -> Accuracy: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f} | FS<->LR Confusions: {res['total_fs_lr_confusion']} | Worst Person F1: {res['worst_person_f1']:.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Convert to DataFrame
    df_res = pd.DataFrame([{
        "Model Configuration": r["model_name"],
        "Features": r["n_features"],
        "Accuracy": f"{r['accuracy']*100:.2f}%",
        "Macro F1": f"{r['macro_f1']:.4f}",
        "F1 (FS)": f"{r['f1_forward_slouch']:.4f}",
        "F1 (LR)": f"{r['f1_lean_right']:.4f}",
        "FS<->LR Confusions": r["total_fs_lr_confusion"],
        "Person F1 Std": f"{r['cross_person_std']:.4f}",
        "Worst Person F1": f"{r['worst_person_f1']:.4f}",
    } for r in results])

    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Evaluator] Evaluation complete! Saved report to: {summary_path}")
    return df_res


if __name__ == "__main__":
    df_summary = run_comprehensive_evaluation()
    print("\n=== SUMMARY COMPARISON ===")
    print(df_summary.to_string(index=False))
