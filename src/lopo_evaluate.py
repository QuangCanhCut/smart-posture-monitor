"""
Leave-One-Person-Out (LOPO) Evaluation — So sánh 4 cấu hình feature.

Chạy:
    python -m src.lopo_evaluate

Mục đích:
    Đánh giá trung thực hiệu năng model bằng cách luân phiên giữ lại
    mỗi person làm test set (14 folds). Mỗi fold train trên 13 người,
    test trên 1 người bị giữ lại.

    Kết quả LOPO cho biết model sẽ hoạt động thế nào với một người MỚI
    chưa từng thấy — đây là metric trung thực nhất cho bài toán này.

4 configs được so sánh:
    A  (V02)     : 29 features gốc (không có depth proxy)
    B  (Clean20) : 20 features (V02 bỏ 9 features âm)
    V03          : 32 features, C=10 cố định
    D4           : 30 features (V02 + chỉ face_rotation_proxy)

Tất cả configs đều dùng SVM RBF với C=10, gamma=0.001 để so sánh
công bằng (chỉ khác feature set).

Output:
    results/lopo/lopo_summary.csv        — Bảng tổng hợp 4 configs
    results/lopo/lopo_per_person.csv     — Chi tiết từng person × config
    results/lopo/lopo_report.md          — Báo cáo dạng Markdown
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    EXPECTED_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    load_dataset,
    get_feature_columns,
    validate_schema,
    validate_integrity,
    add_recording_id,
)

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "lopo"


# ============================================================
# Feature configs
# ============================================================

# 3 depth proxy features thêm ở V03.
DEPTH_PROXY_FEATURES = [
    "face_shoulder_scale_ratio",
    "ear_nose_depth_proxy",
    "face_rotation_proxy",
]

# 9 features có Permutation Importance âm (gây hại model).
HARMFUL_FEATURES = [
    "left_ear_x_body",
    "eye_vertical_difference",
    "nose_ear_ratio",
    "nose_x_body",
    "eye_center_x_body",
    "ear_eye_ratio",
    "face_pitch_angle",
    "nose_shoulder_asymmetry",
    "head_height_spread",
]


def get_config_features(
    all_features: list[str],
) -> dict[str, list[str]]:
    """
    Trả về dict mapping config_name -> danh sách features.

    Tất cả configs được tạo bằng cách lọc từ danh sách 32 features
    hiện có trong features.csv.
    """

    all_set = set(all_features)

    # Config A (V02): 29 features = tất cả trừ 3 depth proxy
    config_a = [
        f for f in all_features
        if f not in DEPTH_PROXY_FEATURES
    ]

    # Config B (Clean 20): V02 trừ 9 features âm
    excluded_b = set(DEPTH_PROXY_FEATURES) | set(HARMFUL_FEATURES)
    config_b = [
        f for f in all_features
        if f not in excluded_b
    ]

    # Config V03: Giữ nguyên tất cả 32 features
    config_v03 = list(all_features)

    # Config D4: V02 + chỉ face_rotation_proxy
    excluded_d4 = set(DEPTH_PROXY_FEATURES) - {"face_rotation_proxy"}
    config_d4 = [
        f for f in all_features
        if f not in excluded_d4
    ]

    configs = {
        "A_V02_29feat": config_a,
        "B_Clean20": config_b,
        "V03_32feat_C10": config_v03,
        "D4_V02+yaw_30feat": config_d4,
    }

    # Sanity check.
    expected_counts = {
        "A_V02_29feat": 29,
        "B_Clean20": 20,
        "V03_32feat_C10": 32,
        "D4_V02+yaw_30feat": 30,
    }

    for name, feats in configs.items():
        actual = len(feats)
        expected = expected_counts[name]

        if actual != expected:
            raise ValueError(
                f"Config {name}: kỳ vọng {expected} features, "
                f"nhưng có {actual}."
            )

        # Đảm bảo tất cả feature đều tồn tại trong dataset.
        missing = [f for f in feats if f not in all_set]
        if missing:
            raise ValueError(
                f"Config {name}: features không tồn tại: {missing}"
            )

    return configs


# ============================================================
# LOPO Core
# ============================================================

def run_lopo_single_config(
    df: pd.DataFrame,
    feature_cols: list[str],
    config_name: str,
    C: float = 10.0,
    gamma: float = 0.001,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Chạy LOPO cho một config duy nhất.

    Returns:
        List of dicts, mỗi dict chứa metrics của 1 fold (1 person).
    """

    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].map(LABEL_TO_ID).values.astype(int)
    groups = df["person_id"].values.astype(str)

    logo = LeaveOneGroupOut()
    results = []

    class_ids = sorted(LABEL_TO_ID.values())

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups)
    ):
        test_person = groups[test_idx[0]]
        n_test = len(test_idx)
        n_train = len(train_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Pipeline: Scale -> SVM RBF.
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                random_state=random_state,
                class_weight=None,
            )),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics.
        acc = accuracy_score(y_test, y_pred)

        macro_f1 = f1_score(
            y_test, y_pred,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )

        per_class_f1 = f1_score(
            y_test, y_pred,
            labels=class_ids,
            average=None,
            zero_division=0,
        )

        # Confusion matrix cho FS ↔ LR.
        cm = confusion_matrix(
            y_test, y_pred,
            labels=class_ids,
        )

        # FS = 1, LR = 3
        fs_idx = LABEL_TO_ID["forward_slouch"]
        lr_idx = LABEL_TO_ID["lean_right"]
        fs_as_lr = int(cm[fs_idx, lr_idx])
        lr_as_fs = int(cm[lr_idx, fs_idx])
        confuse_fs_lr = fs_as_lr + lr_as_fs

        fold_result = {
            "config": config_name,
            "fold": fold_idx,
            "test_person": test_person,
            "n_train": n_train,
            "n_test": n_test,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "f1_correct": float(per_class_f1[0]),
            "f1_forward_slouch": float(per_class_f1[1]),
            "f1_lean_left": float(per_class_f1[2]),
            "f1_lean_right": float(per_class_f1[3]),
            "confuse_fs_lr": confuse_fs_lr,
            "fs_as_lr": fs_as_lr,
            "lr_as_fs": lr_as_fs,
        }

        results.append(fold_result)

    return results


# ============================================================
# Summary helpers
# ============================================================

def summarize_lopo(
    per_person_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Tổng hợp LOPO results thành bảng summary theo config.
    """

    df_results = pd.DataFrame(per_person_results)

    summary_rows = []

    for config_name, group in df_results.groupby("config", sort=False):
        row = {
            "config": config_name,
            "n_folds": len(group),
            "n_features": None,  # sẽ được điền sau
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(),
            "macro_f1_mean": group["macro_f1"].mean(),
            "macro_f1_std": group["macro_f1"].std(),
            "f1_correct_mean": group["f1_correct"].mean(),
            "f1_fs_mean": group["f1_forward_slouch"].mean(),
            "f1_ll_mean": group["f1_lean_left"].mean(),
            "f1_lr_mean": group["f1_lean_right"].mean(),
            "confuse_fs_lr_total": int(group["confuse_fs_lr"].sum()),
            "accuracy_min": group["accuracy"].min(),
            "accuracy_max": group["accuracy"].max(),
            "macro_f1_min": group["macro_f1"].min(),
            "macro_f1_max": group["macro_f1"].max(),
            "worst_person": group.loc[
                group["macro_f1"].idxmin(), "test_person"
            ],
            "best_person": group.loc[
                group["macro_f1"].idxmax(), "test_person"
            ],
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def generate_report(
    summary_df: pd.DataFrame,
    per_person_df: pd.DataFrame,
    n_features_map: dict[str, int],
    elapsed_seconds: float,
) -> str:
    """
    Tạo báo cáo Markdown từ kết quả LOPO.
    """

    # Cập nhật n_features.
    for idx, row in summary_df.iterrows():
        config = row["config"]
        if config in n_features_map:
            summary_df.at[idx, "n_features"] = n_features_map[config]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append("# LOPO Evaluation Report")
    lines.append("")
    lines.append(f"> Generated: {now}")
    lines.append(f"> Total time: {elapsed_seconds:.1f}s")
    lines.append(f"> SVM RBF: C=10, gamma=0.001 (cố định cho tất cả configs)")
    lines.append(f"> Số persons: {per_person_df['test_person'].nunique()}")
    lines.append("")

    # --------------------------------------------------------
    # Bảng tổng hợp
    # --------------------------------------------------------
    lines.append("## 1. Tổng hợp 4 configs")
    lines.append("")
    lines.append(
        "| Config | Features | "
        "Acc (mean±std) | Macro F1 (mean±std) | "
        "F1 correct | F1 FS | F1 LL | F1 LR | "
        "FS<>LR total | Worst Person |"
    )
    lines.append(
        "|--------|----------|"
        "----------------|---------------------|"
        "------------|-------|-------|-------|"
        "-------------|--------------|"
    )

    for _, row in summary_df.iterrows():
        nf = int(row["n_features"]) if row["n_features"] else "?"
        lines.append(
            f"| {row['config']} | {nf} | "
            f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
            f"{row['f1_correct_mean']:.4f} | "
            f"{row['f1_fs_mean']:.4f} | "
            f"{row['f1_ll_mean']:.4f} | "
            f"{row['f1_lr_mean']:.4f} | "
            f"{int(row['confuse_fs_lr_total'])} | "
            f"{row['worst_person']} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Bảng chi tiết per-person
    # --------------------------------------------------------
    lines.append("## 2. Chi tiết từng person")
    lines.append("")

    for config_name in summary_df["config"].values:
        config_data = per_person_df[
            per_person_df["config"] == config_name
        ].sort_values("test_person")

        lines.append(f"### {config_name}")
        lines.append("")
        lines.append(
            "| Person | Samples | Accuracy | Macro F1 | "
            "F1 correct | F1 FS | F1 LL | F1 LR | FS<>LR |"
        )
        lines.append(
            "|--------|---------|----------|----------|"
            "------------|-------|-------|-------|-------|"
        )

        for _, row in config_data.iterrows():
            lines.append(
                f"| {row['test_person']} | {row['n_test']} | "
                f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                f"{row['f1_correct']:.4f} | "
                f"{row['f1_forward_slouch']:.4f} | "
                f"{row['f1_lean_left']:.4f} | "
                f"{row['f1_lean_right']:.4f} | "
                f"{int(row['confuse_fs_lr'])} |"
            )

        lines.append("")

    # --------------------------------------------------------
    # Phân tích
    # --------------------------------------------------------
    lines.append("## 3. Phân tích")
    lines.append("")

    # Tìm config tốt nhất theo Macro F1.
    best_config = summary_df.loc[summary_df["macro_f1_mean"].idxmax()]
    worst_config = summary_df.loc[summary_df["macro_f1_mean"].idxmin()]

    lines.append(
        f"- **Config tốt nhất** (Macro F1): "
        f"**{best_config['config']}** "
        f"({best_config['macro_f1_mean']:.4f}±{best_config['macro_f1_std']:.4f})"
    )
    lines.append(
        f"- **Config kém nhất** (Macro F1): "
        f"{worst_config['config']} "
        f"({worst_config['macro_f1_mean']:.4f}±{worst_config['macro_f1_std']:.4f})"
    )
    lines.append("")

    # Tìm person có F1 thấp nhất qua tất cả configs.
    all_persons = per_person_df.groupby("test_person")["macro_f1"].mean()
    worst_person = all_persons.idxmin()
    best_person = all_persons.idxmax()
    lines.append(
        f"- **Person khó nhất** (trung bình qua 4 configs): "
        f"**{worst_person}** (mean F1 = {all_persons[worst_person]:.4f})"
    )
    lines.append(
        f"- **Person dễ nhất**: "
        f"**{best_person}** (mean F1 = {all_persons[best_person]:.4f})"
    )
    lines.append("")
    lines.append(
        f"- **Khoảng cách F1 giữa người dễ nhất và khó nhất**: "
        f"{all_persons[best_person] - all_persons[worst_person]:.4f} "
        f"({(all_persons[best_person] - all_persons[worst_person]) * 100:.1f}pp)"
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main(
    csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Entry point: chạy LOPO cho 4 configs và xuất kết quả.
    """

    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOPO EVALUATION — 4 FEATURE CONFIGS")
    print("=" * 70)
    print(f"Dataset: {csv_path}")
    print(f"Output : {output_dir}")
    print()

    # --------------------------------------------------------
    # 1. Load và validate dataset
    # --------------------------------------------------------
    print("[1/4] Loading dataset...")
    df = load_dataset(csv_path)
    all_features = get_feature_columns(df)
    validate_schema(df, all_features)
    validate_integrity(df, all_features)
    df = add_recording_id(df)

    persons = sorted(df["person_id"].unique())
    n_persons = len(persons)
    print(f"      Samples: {len(df)}")
    print(f"      Persons: {n_persons} — {persons}")
    print(f"      Features in CSV: {len(all_features)}")
    print()

    # --------------------------------------------------------
    # 2. Tạo 4 configs
    # --------------------------------------------------------
    print("[2/4] Building feature configs...")
    configs = get_config_features(all_features)
    n_features_map = {}

    for name, feats in configs.items():
        n_features_map[name] = len(feats)
        print(f"      {name}: {len(feats)} features")

    print()

    # --------------------------------------------------------
    # 3. Chạy LOPO cho từng config
    # --------------------------------------------------------
    print("[3/4] Running LOPO...")
    all_results = []
    start_time = time.time()

    for config_idx, (config_name, feature_cols) in enumerate(
        configs.items(), start=1
    ):
        config_start = time.time()
        print(
            f"      [{config_idx}/4] {config_name} "
            f"({len(feature_cols)} features)...",
            end=" ",
            flush=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_results = run_lopo_single_config(
                df=df,
                feature_cols=feature_cols,
                config_name=config_name,
                C=10.0,
                gamma=0.001,
            )

        all_results.extend(fold_results)

        mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
        elapsed = time.time() - config_start
        print(f"done ({elapsed:.1f}s) — mean F1 = {mean_f1:.4f}")

    total_time = time.time() - start_time
    print(f"\n      Total LOPO time: {total_time:.1f}s")
    print()

    # --------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------
    print("[4/4] Saving results...")

    per_person_df = pd.DataFrame(all_results)
    summary_df = summarize_lopo(all_results)

    # CSV files.
    per_person_path = output_dir / "lopo_per_person.csv"
    summary_path = output_dir / "lopo_summary.csv"

    per_person_df.to_csv(per_person_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"      Per-person: {per_person_path}")
    print(f"      Summary   : {summary_path}")

    # Markdown report.
    report = generate_report(
        summary_df=summary_df,
        per_person_df=per_person_df,
        n_features_map=n_features_map,
        elapsed_seconds=total_time,
    )
    report_path = output_dir / "lopo_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"      Report    : {report_path}")

    # --------------------------------------------------------
    # Print summary to console
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("LOPO SUMMARY")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        nf = int(row["n_features"]) if row["n_features"] else "?"
        print(
            f"  {row['config']:30s} | "
            f"{nf:2d} feat | "
            f"Acc {row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"F1 {row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
            f"FS<>LR {int(row['confuse_fs_lr_total']):3d}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
