"""
Realtime integration test cho Smart Posture Monitor V03.

Pipeline được test:

    Webcam
      -> PostureInferenceEngine
      -> PoseDetector
      -> FeatureExtractor
      -> RAW12
      -> PersonalCalibration
      -> RepresentationBuilder
      -> REP13
      -> PosturePredictor
      -> SVM RBF prediction

Lưu ý:
- Script này CHƯA dùng TemporalMonitor.
- Mục tiêu là quan sát prediction thô của model V03.
- Không mirror frame vì có thể đảo ý nghĩa lean_left / lean_right.

Cách chạy từ project root:

    python scripts/test_webcam_model.py

Hoặc:

    python scripts/test_webcam_model.py --camera 0

Phím điều khiển:
    C      : bắt đầu calibration mới
    R      : reset và calibration lại
    Q/ESC  : thoát
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# 1. Project paths + imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import PostureInferenceEngine  # noqa: E402
from src.posture_predictor import PosturePredictor  # noqa: E402


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
DEFAULT_YOLO_PATH = PROJECT_ROOT / "models" / "yolo26n-pose.pt"


# ============================================================
# 2. Helpers
# ============================================================

def resolve_path(path: Path | None) -> Path | None:
    """Chuyển đường dẫn tương đối thành đường dẫn tuyệt đối trong project."""
    if path is None:
        return None

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def draw_bbox(
    frame: np.ndarray,
    bbox: list[int] | tuple[int, ...] | None,
) -> None:
    """Vẽ bounding box người được chọn."""
    if bbox is None or len(bbox) < 4:
        return

    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_keypoints(
    frame: np.ndarray,
    keypoints: dict[str, list[float]] | None,
) -> None:
    """Vẽ các keypoint đã được PostureInferenceEngine trả về."""
    if not isinstance(keypoints, dict):
        return

    for name, values in keypoints.items():
        if len(values) < 2:
            continue

        x = int(values[0])
        y = int(values[1])
        confidence = float(values[2]) if len(values) >= 3 else None

        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

        text = name
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


def draw_status_panel(
    frame: np.ndarray,
    result: dict[str, Any],
    engine: PostureInferenceEngine,
    fps: float,
) -> None:
    """Hiển thị trạng thái calibration và prediction của V03."""
    status = str(result.get("status", "UNKNOWN"))
    label = result.get("raw_label")
    probability = result.get("probability")
    message = str(result.get("message", ""))

    label_text = "--" if label is None else str(label).upper().replace("_", " ")

    calibration_text = (
        "READY"
        if engine.is_calibrated
        else (
            f"{engine.calibration_progress * 100:.0f}%"
            if engine.is_calibrating
            else "REQUIRED"
        )
    )

    lines = [
        "Smart Posture Monitor V03",
        f"Status: {status}",
        f"Calibration: {calibration_text}",
        f"Posture: {label_text}",
        f"FPS: {fps:.1f}",
    ]

    if probability is not None:
        lines.insert(4, f"Probability: {float(probability):.3f}")

    if message:
        lines.append(f"Message: {message}")

    panel_height = 25 + 28 * len(lines)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (620, panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame,
    )

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (25, 40 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    controls = "C: calibrate | R: recalibrate | Q/ESC: quit"

    cv2.putText(
        frame,
        controls,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def print_startup_guide(calibration_samples: int) -> None:
    """In hướng dẫn ngắn trước khi bắt đầu webcam."""
    print("=" * 72)
    print("SMART POSTURE MONITOR V03 - REALTIME INFERENCE TEST")
    print("=" * 72)
    print()
    print("Pipeline:")
    print("Webcam -> RAW12 -> Personal Baseline -> REP13 -> SVM RBF")
    print()
    print("Lưu ý:")
    print("- Đây là raw frame-level prediction, CHƯA temporal smoothing.")
    print("- Camera nên giữ cùng protocol dataset (~45° bên trái người dùng).")
    print("- KHÔNG mirror frame vì sẽ đảo lean_left / lean_right.")
    print()
    print("Điều khiển:")
    print("C     : bắt đầu calibration")
    print("R     : reset và calibration lại")
    print("Q/ESC : thoát")
    print()
    print(
        f"Khi calibration, hãy ngồi ở tư thế correct tự nhiên "
        f"cho đến khi đủ {calibration_samples} mẫu hợp lệ."
    )
    print("=" * 72)


# ============================================================
# 3. Realtime webcam integration test
# ============================================================

def run_webcam(
    camera_index: int,
    model_path: Path,
    metadata_path: Path,
    yolo_path: Path | None,
    calibration_samples: int,
    show_pose: bool,
    device: str | None = None,
) -> None:
    """Chạy integration test realtime cho toàn bộ inference pipeline V03."""
    print_startup_guide(calibration_samples)

    engine = PostureInferenceEngine(
        model_path=model_path,
        metadata_path=metadata_path,
        yolo_path=yolo_path,
        calibration_samples=calibration_samples,
    )

    # ``device`` là tùy chọn của PoseDetector, không phải tham số
    # khởi tạo của PostureInferenceEngine. Engine đã tự chọn thiết bị
    # khi device là None/auto; chỉ ghi đè khi CLI yêu cầu rõ ràng.
    if device not in (None, "auto"):
        engine.detector.device = device

    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Không mở được webcam index={camera_index}."
        )

    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                print("[CẢNH BÁO] Không đọc được frame từ webcam.")
                break

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            try:
                result = engine.process_frame(frame)

            except Exception as error:
                # Giữ webcam sống để người dùng nhìn thấy lỗi realtime,
                # đồng thời in traceback ngắn gọn qua console.
                print(
                    f"[LỖI INFERENCE] "
                    f"{type(error).__name__}: {error}"
                )

                result = {
                    "status": "INFERENCE_ERROR",
                    "message": f"{type(error).__name__}: {error}",
                    "raw_label": None,
                    "probability": None,
                    "bbox": None,
                    "keypoints": {},
                }

            if show_pose:
                draw_bbox(frame, result.get("bbox"))
                draw_keypoints(frame, result.get("keypoints"))

            draw_status_panel(
                frame=frame,
                result=result,
                engine=engine,
                fps=fps,
            )

            cv2.imshow(
                "Smart Posture Monitor V03 - Webcam Test",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("c"), ord("C")):
                engine.start_calibration()
                print(
                    "\n[CALIBRATION] Bắt đầu calibration mới. "
                    "Hãy ngồi ở tư thế correct tự nhiên."
                )

            if key in (ord("r"), ord("R")):
                engine.reset_calibration()
                engine.start_calibration()
                print(
                    "\n[RECALIBRATION] Đã xóa baseline cũ. "
                    "Đang thu baseline mới."
                )

    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_artifact_check(model_path: Path, metadata_path: Path) -> None:
    """Load canonical artifacts và predict REP13 mà không mở webcam."""
    predictor = PosturePredictor(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    smoke_rep13 = np.zeros(predictor.feature_count, dtype=np.float32)
    prediction = predictor.predict(smoke_rep13)
    print("[OK] Model và metadata tương thích với REP13.")
    print(f"[OK] Feature count: {predictor.feature_count}")
    print(f"[OK] Smoke prediction: {prediction}")


# ============================================================
# 4. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime integration test cho Smart Posture Monitor V03 "
            "với Personal Baseline Delta."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Mặc định: 0",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Mặc định: models/best_model.joblib",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Mặc định: models/training_metadata.json",
    )

    parser.add_argument(
        "--yolo",
        type=Path,
        default=DEFAULT_YOLO_PATH,
        help="Mặc định: models/yolo26n-pose.pt",
    )

    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=30,
        help="Số mẫu correct dùng để tạo Personal Baseline. Mặc định: 30",
    )

    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Không vẽ bbox/keypoints để giảm overhead.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device để chạy YOLO Pose: 'cpu', 'cuda', hoặc None/auto (tự động fallback về CPU nếu CUDA không tương thích).",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra model/metadata REP13, không mở webcam.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.calibration_samples <= 0:
        raise ValueError(
            "--calibration-samples phải lớn hơn 0."
        )

    model_path = resolve_path(args.model)
    metadata_path = resolve_path(args.metadata)
    yolo_path = resolve_path(args.yolo)

    if model_path is None or not model_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy model:\n{model_path}\n"
            "Hãy chạy `python -m src.train` trước."
        )

    if metadata_path is None or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy training metadata:\n{metadata_path}"
        )

    if not args.check_only and yolo_path is not None and not yolo_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy YOLO model:\n{yolo_path}"
        )

    if args.check_only:
        run_artifact_check(
            model_path=model_path,
            metadata_path=metadata_path,
        )
        return

    run_webcam(
        camera_index=args.camera,
        model_path=model_path,
        metadata_path=metadata_path,
        yolo_path=yolo_path,
        calibration_samples=args.calibration_samples,
        show_pose=not args.no_pose,
        device=args.device,
    )


if __name__ == "__main__":
    main()
