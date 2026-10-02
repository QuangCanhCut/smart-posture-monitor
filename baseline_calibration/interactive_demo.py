"""
Interactive Webcam Demo for Personal Baseline Delta.

Features:
- Real-time webcam capture with PoseDetector (YOLOv8-pose).
- Visual on-screen HUD with calibration countdown (3-2-1).
- Live telemetry displaying the 8 Personal Baseline Delta values (Δf).
- Color-coded posture overlay (Green: correct, Red: forward_slouch, Orange: lean_left/right, Cyan: head_turned).
- Keyboard shortcuts:
  * 'C': Start 3-second baseline calibration
  * 'R': Reset baseline profile
  * 'S': Toggle temporal smoothing
  * 'P': Toggle skeleton overlay
  * 'Q' or ESC: Quit demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_detector import PoseDetector
from baseline_calibration.posture_predictor import PosturePredictor
from baseline_calibration.calibrator import CalibrationConfig

# Color Palette (BGR)
COLOR_CORRECT = (76, 175, 80)       # Green
COLOR_SLOUCH = (48, 59, 255)       # Red
COLOR_LEAN = (0, 165, 255)         # Orange
COLOR_HEAD_TURN = (255, 200, 0)    # Cyan / Yellow
COLOR_CALIB = (255, 140, 0)        # Deep Orange
COLOR_BG = (25, 25, 25)            # Dark grey
COLOR_TEXT = (255, 255, 255)       # White

COLOR_MAP = {
    "correct": COLOR_CORRECT,
    "forward_slouch": COLOR_SLOUCH,
    "lean_left": COLOR_LEAN,
    "lean_right": COLOR_LEAN,
    "head_turned": COLOR_HEAD_TURN,
    "calibrating": COLOR_CALIB,
}

LABEL_NAMES_VI = {
    "correct": "Ngoi Thang (Correct)",
    "forward_slouch": "Gu Lung / Cui Dau (Slouch)",
    "lean_left": "Nghieng Trai (Lean Left)",
    "lean_right": "Nghieng Phai (Lean Right)",
    "head_turned": "Quay Dau (Head Turned)",
    "calibrating": "Dang Hieu Chuan (Calibrating)",
}


def draw_hud_panel(
    frame: np.ndarray,
    pred: Optional[Any],
    fps: float,
    show_pose: bool,
    show_smoothing: bool,
) -> np.ndarray:
    """Renders informative HUD overlay onto webcam frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top status bar
    cv2.rectangle(overlay, (0, 0), (w, 55), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    fps_text = f"FPS: {fps:4.1f} | Key: [C] Calib | [R] Reset | [S] Smooth ({'ON' if show_smoothing else 'OFF'}) | [P] Pose | [Q] Quit"
    cv2.putText(frame, fps_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    if pred is None:
        cv2.putText(frame, "Khong tim thay nguoi trong khung hinh", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    # 2. Main Posture Banner
    banner_y = 110
    active_color = COLOR_MAP.get(pred.smoothed_label, (200, 200, 200))
    vi_label = LABEL_NAMES_VI.get(pred.smoothed_label, pred.smoothed_label)

    # Banner background
    cv2.rectangle(frame, (15, 65), (420, banner_y), COLOR_BG, -1)
    cv2.rectangle(frame, (15, 65), (25, banner_y), active_color, -1)  # Color accent bar

    cv2.putText(frame, f"TU THE: {vi_label}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, active_color, 2, cv2.LINE_AA)

    # 3. Calibration Status Section
    calib_y = 130
    cv2.rectangle(frame, (15, calib_y), (420, calib_y + 80), COLOR_BG, -1)

    if pred.is_calibrating:
        # Progress bar
        bar_w = 380
        fill_w = int(bar_w * pred.calibration_progress)
        cv2.rectangle(frame, (25, calib_y + 15), (25 + bar_w, calib_y + 35), (60, 60, 60), -1)
        cv2.rectangle(frame, (25, calib_y + 15), (25 + fill_w, calib_y + 35), COLOR_CALIB, -1)
        cv2.rectangle(frame, (25, calib_y + 15), (25 + bar_w, calib_y + 35), (200, 200, 200), 1)

        msg = f"HIEU CHUAN: {int(pred.calibration_progress * 100)}% (Giu thang nguoi 3s)"
        cv2.putText(frame, msg, (25, calib_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1, cv2.LINE_AA)

    elif pred.is_calibrated:
        badge = "BASELINE: ACTIVE [OK]"
        cv2.putText(frame, badge, (25, calib_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CORRECT, 2, cv2.LINE_AA)
        
        # Display deltas
        deltas = pred.active_deltas
        if deltas:
            d_nose_y = deltas.get("nose_y_body", 0.0)
            d_head_g = deltas.get("head_gravity_angle", 0.0)
            d_yaw = deltas.get("face_rotation_proxy", 0.0)
            d_scale = deltas.get("face_shoulder_scale_ratio", 0.0)

            delta_text = f"d(NoseY)={d_nose_y:+.2f} | d(HeadAng)={d_head_g:+.1f}deg | d(Yaw)={d_yaw:+.2f}"
            cv2.putText(frame, delta_text, (25, calib_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "BASELINE: CHUA HIEU CHUAN", (25, calib_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        cv2.putText(frame, "Nhan phim [C] de bat dau hieu chuan 3s", (25, calib_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

    # 4. Status message
    if pred.status_message:
        cv2.putText(frame, pred.status_message, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame: np.ndarray, pose_result: dict, color=(0, 255, 0)) -> None:
    """Draws keypoints and skeleton connections."""
    if not pose_result or "keypoints" not in pose_result:
        return

    kps = pose_result["keypoints"]
    pts = {}
    for name, kp in kps.items():
        x, y, conf = kp
        if conf >= 0.3:
            pts[name] = (int(x), int(y))
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    # Connections
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_eye", "right_eye"),
        ("left_eye", "nose"),
        ("right_eye", "nose"),
        ("left_eye", "left_ear"),
        ("left_shoulder", "left_ear"),
    ]

    for p1, p2 in connections:
        if p1 in pts and p2 in pts:
            cv2.line(frame, pts[p1], pts[p2], color, 2)


def run_demo(camera_id: int = 0) -> None:
    """Main execution loop."""
    print("=" * 60)
    print("Personal Baseline Delta — Interactive Webcam Demo")
    print("=" * 60)
    print(f"Connecting to Camera {camera_id}...")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}. Retrying with default backend...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Failed to access camera {camera_id}. Exiting.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize components
    print("Loading YOLOv8-pose detector and PosturePredictor...")
    detector = PoseDetector()
    predictor = PosturePredictor()

    show_pose = True
    show_smoothing = True
    prev_time = time.time()
    fps = 0.0

    cv2.namedWindow("Smart Posture Monitor - Baseline Delta Demo", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Frame rate estimation
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # 1. Pose detection
            pose_result = detector.detect(frame)

            # 2. Prediction with Baseline Delta
            pred = predictor.predict_frame(pose_result=pose_result)

            # 3. Draw skeleton overlay
            if show_pose and pose_result:
                color = COLOR_MAP.get(pred.smoothed_label if pred else "correct", (0, 255, 0))
                draw_skeleton(frame, pose_result, color=color)

            # 4. Draw HUD
            frame = draw_hud_panel(frame, pred, fps, show_pose, show_smoothing)

            cv2.imshow("Smart Posture Monitor - Baseline Delta Demo", frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("Exiting demo...")
                break
            elif key in (ord("c"), ord("C")):
                print("Starting 3-second Baseline Calibration...")
                predictor.start_calibration(n_frames=30)
            elif key in (ord("r"), ord("R")):
                print("Resetting baseline...")
                predictor.reset_baseline()
            elif key in (ord("p"), ord("P")):
                show_pose = not show_pose
            elif key in (ord("s"), ord("S")):
                show_smoothing = not show_smoothing
                predictor.temporal_monitor.hysteresis_frames = 3 if show_smoothing else 0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Calibration Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run_demo(camera_id=args.camera)
