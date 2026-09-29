"""
Realtime webcam demo for V03 personal baseline calibration.

Run from project root:
    python scripts/test_webcam_v03.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.personal_calibration import PersonalCalibration  # noqa: E402
from src.pose_detector import PoseDetector  # noqa: E402
from src.posture_predictor_v03 import PosturePredictorV03  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "v03" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "v03" / "training_metadata.json"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "models" / "v03" / "feature_schema.json"
DEFAULT_CALIBRATION_SAMPLES = 30


def parse_bbox(bbox: Any) -> tuple[int, int, int, int] | None:
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

        text = name
        if confidence is not None and np.isfinite(confidence):
            text = f"{name}:{confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def extract_valid_features(
    extractor: FeatureExtractor,
    pose: dict[str, Any] | None,
) -> tuple[np.ndarray | None, str]:
    if not isinstance(pose, dict):
        return None, "NO VALID POSE"

    features = extractor.extract(pose)
    if features is None:
        return None, "FEATURE EXTRACTION FAILED"

    features = np.asarray(features, dtype=np.float64).reshape(-1)
    if features.shape != (len(FeatureExtractor.FEATURE_NAMES),):
        return None, f"INVALID FEATURE LENGTH {features.shape[0]}"

    if not np.isfinite(features).all():
        return None, "NON-FINITE FEATURES"

    return features, "OK"


def draw_status(
    frame: np.ndarray,
    mode: str,
    calibration: PersonalCalibration,
    calibration_target: int,
    label: str | None,
    probability: float | None,
    status: str,
    fps: float,
) -> None:
    posture = "--" if label is None else label.upper().replace("_", " ")

    lines = [
        "PERSONAL CALIBRATION V03",
        f"Mode: {mode}",
        f"Calibration: {calibration.sample_count}/{calibration_target}",
        f"Posture: {posture}",
        f"Status: {status}",
        f"FPS: {fps:.1f}",
    ]

    if probability is not None:
        lines.insert(5, f"Confidence: {probability:.2f}")

    panel_height = 18 + 27 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (25, 38 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "C: calibrate/recalibrate   R: reset   Q/ESC: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_webcam(
    camera_index: int,
    model_path: Path,
    metadata_path: Path,
    schema_path: Path,
    calibration_target: int,
    show_pose: bool,
) -> None:
    print("=" * 72)
    print("PERSONAL CALIBRATION V03")
    print("=" * 72)
    print(f"Model    : {model_path}")
    print(f"Metadata : {metadata_path}")
    print(f"Schema   : {schema_path}")
    print("Press C to start calibration.")

    predictor = PosturePredictorV03(
        model_path=model_path,
        metadata_path=metadata_path,
        schema_path=schema_path,
    )
    detector = PoseDetector()
    extractor = FeatureExtractor()
    calibration = PersonalCalibration(feature_count=len(predictor.feature_columns))

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam index={camera_index}.")

    mode = "CALIBRATION REQUIRED"
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[WARNING] Could not read frame from webcam.")
                break

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            pose = None
            raw_features = None
            label = None
            probability = None
            status = "Calibration required" if not calibration.is_ready else "OK"

            try:
                pose = detector.detect(frame)
                raw_features, feature_status = extract_valid_features(extractor, pose)
            except Exception as error:
                feature_status = f"POSE/FEATURE ERROR: {type(error).__name__}"

            if mode == "CALIBRATING":
                status = feature_status
                if raw_features is not None:
                    try:
                        calibration.add_sample(raw_features)
                        status = f"Calibration: {calibration.sample_count}/{calibration_target}"
                    except ValueError as error:
                        status = f"CALIBRATION REJECTED: {error}"

                    if calibration.sample_count >= calibration_target:
                        calibration.calculate_baseline()
                        mode = "MONITORING"
                        status = "Calibration completed"

            elif calibration.is_ready:
                mode = "MONITORING"
                if raw_features is None:
                    status = feature_status
                else:
                    try:
                        label, probability, _ = predictor.predict_raw(raw_features, calibration)
                        status = "OK"
                    except Exception as error:
                        status = f"PREDICT ERROR: {type(error).__name__}"
            else:
                mode = "CALIBRATION REQUIRED"
                status = "Press C to start calibration"

            if show_pose:
                draw_pose(frame, pose)

            draw_status(
                frame=frame,
                mode=mode,
                calibration=calibration,
                calibration_target=calibration_target,
                label=label,
                probability=probability,
                status=status,
                fps=fps,
            )

            cv2.imshow("Smart Posture Monitor V03", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("c"), ord("C")):
                calibration.reset()
                mode = "CALIBRATING"
                print("Calibration started.")

            if key in (ord("r"), ord("R")):
                calibration.reset()
                mode = "CALIBRATION REQUIRED"
                print("Calibration reset.")

    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_self_test(
    model_path: Path,
    metadata_path: Path,
    schema_path: Path,
) -> None:
    predictor = PosturePredictorV03(
        model_path=model_path,
        metadata_path=metadata_path,
        schema_path=schema_path,
    )
    calibration = PersonalCalibration(feature_count=len(predictor.feature_columns))
    calibration.add_sample(np.zeros(len(predictor.feature_columns), dtype=np.float64))
    calibration.add_sample(np.ones(len(predictor.feature_columns), dtype=np.float64))
    calibration.calculate_baseline()

    raw_features = np.ones(len(predictor.feature_columns), dtype=np.float64)
    label, probability, _ = predictor.predict_raw(raw_features, calibration)

    print("V03 webcam self-test OK")
    print(f"Features : {len(predictor.feature_columns)}")
    print(f"Label    : {label}")
    print(f"Prob     : {probability}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime V03 webcam demo.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of valid frames to collect for live calibration.",
    )
    parser.add_argument("--no-pose", action="store_true", help="Do not draw bbox/keypoints.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load V03 artifacts and run a no-camera prediction pipeline check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    metadata_path = (
        args.metadata if args.metadata.is_absolute() else PROJECT_ROOT / args.metadata
    )
    schema_path = args.schema if args.schema.is_absolute() else PROJECT_ROOT / args.schema

    if args.self_test:
        run_self_test(
            model_path=model_path.resolve(),
            metadata_path=metadata_path.resolve(),
            schema_path=schema_path.resolve(),
        )
        return

    run_webcam(
        camera_index=args.camera,
        model_path=model_path.resolve(),
        metadata_path=metadata_path.resolve(),
        schema_path=schema_path.resolve(),
        calibration_target=args.calibration_samples,
        show_pose=not args.no_pose,
    )


if __name__ == "__main__":
    main()
