"""
Realtime webcam smoke test for Smart Posture Monitor V02.

Pipeline:
Webcam -> PoseDetector -> FeatureExtractor (32 features)
       -> best_model.joblib -> predicted posture

Run from project root:
    python scripts/test_webcam_model.py
    python scripts/test_webcam_model.py --camera 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths + imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.pose_detector import PoseDetector  # noqa: E402
from src.temporal_monitor import TemporalMonitor, HEAD_TURNED_LABEL  # noqa: E402

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"


# ============================================================
# 2. Load + validate training artifacts
# ============================================================

def load_artifacts(
    model_path: Path,
    metadata_path: Path,
) -> tuple[Any, dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy model:\n{model_path}\n"
            "Hãy chạy training pipeline trước."
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy training metadata:\n{metadata_path}"
        )

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


def get_id_to_label(metadata: dict[str, Any]) -> dict[int, str]:
    """Hỗ trợ cả id_to_label và label_to_id trong metadata."""

    raw = metadata.get("id_to_label")
    if isinstance(raw, dict) and raw:
        return {int(class_id): str(label) for class_id, label in raw.items()}

    raw = metadata.get("label_to_id")
    if isinstance(raw, dict) and raw:
        return {int(class_id): str(label) for label, class_id in raw.items()}

    raise ValueError(
        "training_metadata.json cần có 'id_to_label' "
        "hoặc 'label_to_id'."
    )


def validate_feature_schema(
    model: Any,
    metadata: dict[str, Any],
) -> list[str]:
    """
    Đảm bảo feature schema realtime giống hệt lúc training.
    Không cho model chạy nếu tên/thứ tự feature đã thay đổi.
    """

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "training_metadata.json không có 'feature_columns' hợp lệ."
        )

    feature_columns = [str(name) for name in feature_columns]
    current_features = list(FeatureExtractor.FEATURE_NAMES)

    if feature_columns != current_features:
        raise RuntimeError(
            "FeatureExtractor hiện tại KHÔNG khớp model đã train.\n"
            f"Current : {current_features}\n"
            f"Trained : {feature_columns}"
        )

    expected_count = len(FeatureExtractor.FEATURE_NAMES)
    if len(feature_columns) != expected_count:
        raise RuntimeError(
            f"Kỳ vọng {expected_count} features nhưng metadata có "
            f"{len(feature_columns)}."
        )

    # Pipeline/sklearn estimator thường lưu feature_names_in_ nếu fit bằng DataFrame.
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = [str(name) for name in model_feature_names]
        if model_feature_names != feature_columns:
            raise RuntimeError(
                "Feature names trong model không khớp training metadata."
            )

    return feature_columns


# ============================================================
# 3. Pose -> features -> model prediction
# ============================================================

def predict_posture(
    model: Any,
    extractor: FeatureExtractor,
    pose: dict[str, Any],
    feature_columns: list[str],
    id_to_label: dict[int, str],
) -> tuple[str | None, float | None, float | None, str]:
    """
    Returns:
        label: predicted posture name, or None.
        probability: predicted probability if supported, otherwise None.
        d4_value: value of face_rotation_proxy if present, otherwise None.
        status: debug status for realtime display.
    """

    features = extractor.extract(pose)
    if features is None:
        return None, None, None, "FEATURE EXTRACTION FAILED"

    features_array = np.asarray(features, dtype=np.float64).reshape(-1)

    if len(features_array) != len(feature_columns):
        return (
            None,
            None,
            None,
            f"INVALID FEATURE LENGTH {len(features_array)}/{len(feature_columns)}",
        )

    if not np.isfinite(features_array).all():
        return None, None, None, "NON-FINITE FEATURES"

    d4_value = None
    if "face_rotation_proxy" in feature_columns:
        d4_idx = feature_columns.index("face_rotation_proxy")
        d4_value = float(features_array[d4_idx])

    # Dùng DataFrame để giữ đúng tên + thứ tự feature như khi train.
    X_live = pd.DataFrame([features_array], columns=feature_columns)

    pred_id = int(np.asarray(model.predict(X_live)).reshape(-1)[0])
    if pred_id not in id_to_label:
        return None, None, d4_value, f"UNKNOWN CLASS ID: {pred_id}"

    label = id_to_label[pred_id]
    probability = None

    # SVM có thể probability=False nên probability chỉ là optional.
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(X_live), dtype=float)
            classes = np.asarray(getattr(model, "classes_", []))

            if probs.ndim == 2 and probs.shape[0] == 1:
                if classes.size == probs.shape[1]:
                    positions = np.where(classes == pred_id)[0]
                    if positions.size == 1:
                        probability = float(probs[0, positions[0]])
                else:
                    probability = float(probs[0].max())
        except (AttributeError, TypeError, ValueError):
            probability = None

    return label, probability, d4_value, "OK"


# ============================================================
# 4. Visualization helpers
# ============================================================

def parse_bbox(bbox: Any) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        for keys in (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
        ):
            if all(key in bbox for key in keys):
                return tuple(int(float(bbox[key])) for key in keys)
        return None

    try:
        values = np.asarray(bbox, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if values.size < 4 or not np.isfinite(values[:4]).all():
        return None

    return tuple(int(value) for value in values[:4])


def draw_pose(frame: np.ndarray, pose: dict[str, Any] | None) -> None:
    if not isinstance(pose, dict):
        return

    bbox = parse_bbox(pose.get("bbox"))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    keypoints = pose.get("keypoints")
    if not isinstance(keypoints, dict):
        return

    for name, keypoint in keypoints.items():
        try:
            values = np.asarray(keypoint, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue

        if values.size < 2 or not np.isfinite(values[:2]).all():
            continue

        x, y = int(values[0]), int(values[1])
        confidence = float(values[2]) if values.size >= 3 else None

        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

        text = str(name)
        if confidence is not None and np.isfinite(confidence):
            text = f"{name}:{confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


COLOR_MAP = {
    "correct": (60, 220, 100),         # Xanh lá dịu
    "forward_slouch": (50, 50, 240),    # Đỏ tươi
    "lean_left": (240, 160, 40),        # Cam sáng
    "lean_right": (230, 80, 210),       # Hồng/Tím
    "head_turned": (0, 215, 255),       # Vàng cam
    "unknown": (180, 180, 180),         # Xám
}


def draw_status(
    frame: np.ndarray,
    smoothed_label: str | None,
    raw_label: str | None,
    model_probability: float | None,
    person_confidence: float | None,
    d4_value: float | None,
    is_head_turned: bool,
    smoothing_enabled: bool,
    status: str,
    fps: float,
) -> None:
    display_label = smoothed_label if smoothing_enabled else raw_label
    posture = "--" if display_label is None else display_label.upper().replace("_", " ")

    color_key = display_label.lower() if display_label else "unknown"
    posture_color = COLOR_MAP.get(color_key, (255, 255, 255))

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"Posture: {posture}", posture_color),
    ]

    if smoothing_enabled and raw_label:
        raw_str = raw_label.upper().replace("_", " ")
        lines.append((f"Raw Model: {raw_str}", (200, 200, 200)))

    lines.append((f"Status: {status}", (255, 255, 255)))
    lines.append((f"FPS: {fps:.1f}", (220, 220, 220)))

    if person_confidence is not None:
        lines.append((f"Person conf: {person_confidence:.2f}", (200, 200, 200)))

    if model_probability is not None:
        lines.append((f"Model prob: {model_probability:.2f}", (200, 200, 200)))

    if d4_value is not None:
        yaw_note = " [HEAD TURNED GATED]" if is_head_turned else ""
        yaw_color = (0, 215, 255) if is_head_turned else (190, 190, 190)
        lines.append((f"Yaw D4: {d4_value:.2f}{yaw_note}", yaw_color))

    filter_text = "Smoothing: ON (Hysteresis)" if smoothing_enabled else "Smoothing: OFF (Raw Frame)"
    filter_color = (100, 240, 100) if smoothing_enabled else (120, 120, 240)
    lines.append((filter_text, filter_color))

    panel_width = 460
    panel_height = 20 + 26 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for index, (text, color) in enumerate(lines):
        scale = 0.70 if index == 0 else 0.52
        thick = 2 if index == 0 else 1
        cv2.putText(
            frame,
            text,
            (25, 38 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )

    # Footer hướng dẫn phím tắt
    footer_text = "Q/ESC: Quit | S: Toggle Smooth | P: Toggle Pose"
    cv2.putText(
        frame,
        footer_text,
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 5. Realtime webcam loop
# ============================================================

def run_webcam(
    camera_index: int,
    model_path: Path,
    metadata_path: Path,
    show_pose: bool,
    smoothing_enabled: bool = True,
    window_seconds: float = 3.0,
    hysteresis_frames: int = 3,
    yaw_mode: str = "conservative",
    device: str | None = None,
) -> None:
    print("=" * 68)
    print("SMART POSTURE MONITOR - V03 REALTIME WEBCAM TEST")
    print("=" * 68)
    print(f"Model        : {model_path}")
    print(f"Metadata     : {metadata_path}")
    print(f"Device       : {device if device else 'auto'}")
    print(f"Smoothing    : {'ENABLED' if smoothing_enabled else 'DISABLED'}")
    print(f"Window / Hyst: {window_seconds}s / {hysteresis_frames} frames")
    print(f"Yaw Mode     : {yaw_mode}")

    model, metadata = load_artifacts(model_path, metadata_path)
    feature_columns = validate_feature_schema(model, metadata)
    id_to_label = get_id_to_label(metadata)

    detector = PoseDetector(device=device)
    extractor = FeatureExtractor()

    # Khởi tạo bộ lọc thời gian & Yaw Gating
    monitor = TemporalMonitor(
        fps=15,
        window_seconds=window_seconds,
        hysteresis_frames=hysteresis_frames,
        yaw_mode=yaw_mode,  # type: ignore[arg-type]
    )

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Không mở được webcam index={camera_index}.")

    print(f"Features     : {len(feature_columns)}")
    print(f"Classes      : {id_to_label}")
    print(
        "\nĐặt camera cùng protocol với dataset "
        "(xấp xỉ 45° bên trái người dùng)."
    )
    print("Frame KHÔNG được mirror vì sẽ đảo ý nghĩa lean_left/lean_right.")
    print("Phím tắt:")
    print("  - Q hoặc ESC : Thoát")
    print("  - S          : Bật / Tắt bộ lọc làm mượt (Smoothing)")
    print("  - P          : Bật / Tắt vẽ khung xương (Pose)\n")

    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[WARNING] Không đọc được frame từ webcam.")
                break

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            pose = None
            raw_label = None
            smoothed_label = None
            probability = None
            d4_value = None
            person_confidence = None
            is_head_turned = False
            status = "NO VALID POSE"

            try:
                pose = detector.detect(frame)
            except Exception as error:
                status = f"POSE ERROR: {type(error).__name__}"

            if isinstance(pose, dict):
                raw_conf = pose.get("person_confidence")
                if raw_conf is not None:
                    try:
                        value = float(raw_conf)
                        if np.isfinite(value):
                            person_confidence = value
                    except (TypeError, ValueError):
                        pass

                try:
                    raw_label, probability, d4_value, status = predict_posture(
                        model=model,
                        extractor=extractor,
                        pose=pose,
                        feature_columns=feature_columns,
                        id_to_label=id_to_label,
                    )
                except Exception as error:
                    raw_label = None
                    probability = None
                    d4_value = None
                    status = f"PREDICT ERROR: {type(error).__name__}"

                # Cập nhật TemporalMonitor nếu dự đoán thành công
                if raw_label is not None:
                    if smoothing_enabled:
                        smoothed_label = monitor.update(
                            raw_prediction=raw_label,
                            face_rotation_proxy=d4_value,
                        )
                        is_head_turned = monitor.state.is_head_turned
                    else:
                        smoothed_label = raw_label

            if show_pose:
                draw_pose(frame, pose)

            draw_status(
                frame=frame,
                smoothed_label=smoothed_label,
                raw_label=raw_label,
                model_probability=probability,
                person_confidence=person_confidence,
                d4_value=d4_value,
                is_head_turned=is_head_turned,
                smoothing_enabled=smoothing_enabled,
                status=status,
                fps=fps,
            )

            cv2.imshow("Smart Posture Monitor - Webcam Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                smoothing_enabled = not smoothing_enabled
                print(f"[HOTKEY] Temporal Smoothing toggled -> {'ON' if smoothing_enabled else 'OFF'}")
            elif key in (ord("p"), ord("P")):
                show_pose = not show_pose
                print(f"[HOTKEY] Pose Skeleton toggled -> {'ON' if show_pose else 'OFF'}")

    finally:
        capture.release()
        cv2.destroyAllWindows()


# ============================================================
# 6. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime webcam test for Smart Posture Monitor V03."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Default: models/best_model.joblib",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Default: models/training_metadata.json",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Không vẽ bbox/keypoints để giảm overhead.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Tắt bộ lọc làm mượt thời gian (chỉ lấy raw model prediction).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="Độ dài cửa sổ làm mượt theo giây (default: 3.0s).",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=3,
        help="Số frame liên tục cần đồng thuận trước khi đổi nhãn (default: 3).",
    )
    parser.add_argument(
        "--yaw-mode",
        type=str,
        default="conservative",
        choices=["conservative", "informative", "off"],
        help="Chế độ Yaw Gating khi quay đầu (default: 'conservative').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị chạy model: 'cpu', 'cuda', hoặc None (mặc định tự động, tự fallback CPU nếu CUDA lỗi).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model
    metadata_path = args.metadata

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    run_webcam(
        camera_index=args.camera,
        model_path=model_path.resolve(),
        metadata_path=metadata_path.resolve(),
        show_pose=not args.no_pose,
        smoothing_enabled=not args.no_smooth,
        window_seconds=args.window,
        hysteresis_frames=args.hysteresis,
        yaw_mode=args.yaw_mode,
        device=args.device,
    )


if __name__ == "__main__":
    main()
