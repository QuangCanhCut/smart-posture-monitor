"""
Smart Posture Monitor - Streamlit Dashboard (Premium UI)
Ứng dụng giám sát tư thế thời gian thực - 100% Python (Frontend + Backend).

Tương thích kiến trúc V03 Folder 1:
    YOLO Pose -> RAW12 -> Personal Calibration -> REP13 -> SVM RBF -> Temporal Smoothing

Chạy bằng lệnh:
    .venv\\Scripts\\streamlit run app/streamlit_app.py

Tính năng:
    - Giám sát webcam real-time với skeleton overlay
    - Personal Baseline Calibration (30 frames / ~2-3s)
    - 4 KPI cards: Tư thế, Điểm công thái học, Thời gian, Cảnh báo
    - Góc sinh trắc học real-time (Shoulder, Head Gravity, Face Yaw Proxy, Body Diff)
    - Biểu đồ phân phối tư thế (Bar chart)
    - Nhật ký cảnh báo trực tiếp
    - Chụp ảnh đơn lẻ & Upload ảnh kiểm tra
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 1. Đường dẫn dự án & Nạp module từ src/ (Folder 1)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor  # noqa: E402
from src.inference import PostureInferenceEngine  # noqa: E402
from src.personal_calibration import PersonalCalibration  # noqa: E402
from src.pose_detector import PoseDetector  # noqa: E402
from src.posture_predictor import PosturePredictor  # noqa: E402
from src.representation_builder import RepresentationBuilder  # noqa: E402
from src.temporal_monitor import HEAD_TURNED_LABEL, VALID_LABELS  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n-pose.pt"

# Baseline tham chiếu mẫu từ tập huấn luyện V03 (median của 1,467 mẫu correct)
# Dùng làm fallback cho Single Snapshot / Upload khi người dùng chưa hiệu chuẩn webcam.
DEFAULT_BASELINE = np.array(
    [
        -175.39,  # shoulder_roll_deg
        171.67,  # eye_roll_deg
        -9.43,  # head_shoulder_roll_diff_deg
        167.39,  # neck_pitch_deg
        853.53,  # eye_center_x_px
        419.76,  # eye_center_y_px
        972.32,  # shoulder_center_x_px
        343.57,  # eye_shoulder_vertical_gap_px
        -130.74,  # eye_shoulder_horizontal_offset_px
        85.26,  # inter_eye_distance_px
        212.99,  # ear_nose_horizontal_span_px
        437.66,  # shoulder_width_px
    ],
    dtype=np.float32,
)

# ──────────────────────────────────────────────────────────────
# 2. Cấu hình Trang Streamlit & Premium Custom CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Posture Monitor",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Google Fonts: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"],
    .main, [data-testid="stApp"] {
        background-color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: -0.01em;
    }

    .shadcn-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
        transition: all 0.2s ease-in-out;
        margin-bottom: 6px;
    }
    .shadcn-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 8px -2px rgba(0,0,0,0.06);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 6px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .badge-correct {
        background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;
    }
    .badge-slouch {
        background: #fff1f2; color: #be123c; border: 1px solid #fecdd3;
    }
    .badge-lean-left {
        background: #fffbeb; color: #b45309; border: 1px solid #fde68a;
    }
    .badge-lean-right {
        background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe;
    }
    .badge-head-turned {
        background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd;
    }
    .badge-no-person {
        background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0;
    }
    .badge-evaluating {
        background: #fefce8; color: #a16207; border: 1px solid #fef08a;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
    }
    .pulse-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        animation: pulse-dot 2s cubic-bezier(0.4,0,0.6,1) infinite;
    }

    .gauge-container {
        margin: 10px 0;
    }
    .gauge-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .gauge-label {
        font-size: 12px;
        font-weight: 500;
        color: #475569;
    }
    .gauge-value {
        font-size: 12px;
        font-weight: 700;
        font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
        color: #0f172a;
    }
    .gauge-track {
        width: 100%;
        height: 8px;
        background: #f1f5f9;
        border-radius: 999px;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease, background 0.3s ease;
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    .section-badge {
        font-size: 11px;
        font-weight: 500;
        color: #64748b;
        background: #f1f5f9;
        padding: 3px 8px;
        border-radius: 6px;
    }

    .log-entry {
        display: flex;
        align-items: baseline;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 12px;
        margin-bottom: 4px;
        line-height: 1.4;
    }
    .log-time {
        font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
        font-size: 11px;
        color: #94a3b8;
        flex-shrink: 0;
    }
    .log-info { background: #f8fafc; color: #475569; }
    .log-warn { background: #fffbeb; color: #b45309; }
    .log-alert { background: #fff1f2; color: #be123c; font-weight: 600; }
    .log-good { background: #ecfdf5; color: #047857; }

    .calib-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
    }
    .calib-badge-active {
        background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;
    }
    .calib-badge-pending {
        background: #fffbeb; color: #b45309; border: 1px solid #fde68a;
    }
    .calib-badge-collecting {
        background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd;
    }
    .calib-info-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .calib-info-card .calib-title {
        font-size: 12px;
        font-weight: 600;
        color: #166534;
        margin-bottom: 4px;
    }
    .calib-info-card .calib-detail {
        font-size: 11px;
        color: #15803d;
        line-height: 1.5;
    }

    .guide-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
    }
    .guide-card-title {
        font-size: 13px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .footer-bar {
        border-top: 1px solid #e2e8f0;
        padding-top: 16px;
        margin-top: 24px;
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #94a3b8;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# 3. Cache Resource: Tải model & Inference Engine một lần duy nhất
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_posture_engine():
    """Khởi tạo và cache PostureInferenceEngine V03."""
    engine = PostureInferenceEngine(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        yolo_path=YOLO_MODEL_PATH,
        calibration_samples=30,
        fps=15,
    )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return engine, metadata


engine, metadata = load_posture_engine()

# ──────────────────────────────────────────────────────────────
# 4. Session State Management
# ──────────────────────────────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_label" not in st.session_state:
    st.session_state.last_label = None

if "last_prob" not in st.session_state:
    st.session_state.last_prob = None


def reset_session():
    """Reset phiên làm việc và nhật ký."""
    engine.reset_session()
    st.session_state.logs = []
    st.session_state.last_label = None
    st.session_state.last_prob = None


def append_log(text: str, log_type: str = "info"):
    """Ghi nhận log vào danh sách hiển thị."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, {"time": timestamp, "text": text, "type": log_type})
    if len(st.session_state.logs) > 30:
        st.session_state.logs = st.session_state.logs[:30]


# ──────────────────────────────────────────────────────────────
# 5. Hàm vẽ trực quan lên ảnh OpenCV (Skeleton + HUD)
# ──────────────────────────────────────────────────────────────
COLOR_MAP_BGR = {
    "correct": (116, 199, 72),  # Xanh lá
    "forward_slouch": (60, 60, 235),  # Đỏ
    "lean_left": (30, 160, 245),  # Cam
    "lean_right": (220, 80, 210),  # Tím
    "head_turned": (230, 175, 40),  # Vàng/Xanh dương
    "evaluating": (60, 180, 240),  # Vàng nhạt
    "unknown": (180, 180, 180),  # Xám
    "no_person": (150, 150, 150),  # Xám nhạt
}

LABEL_VN = {
    "correct": "TƯ THẾ CHUẨN",
    "forward_slouch": "GÙ LƯNG / CÚI ĐẦU",
    "lean_left": "LỆCH TRÁI",
    "lean_right": "LỆCH PHẢI",
    "head_turned": "ĐANG QUAY ĐẦU",
    "evaluating": "ĐANG ĐÁNH GIÁ",
    "no_person": "KHÔNG CÓ NGƯỜI",
    "unknown": "CHƯA XÁC ĐỊNH",
}

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


def draw_hud(
    frame: np.ndarray,
    bbox: list[int] | None,
    keypoints: dict[str, list[float]] | None,
    smoothed_label: str | None,
    raw_label: str | None,
    prob: float | None,
    angles: dict[str, float],
    fps: float,
    show_skeleton: bool = True,
    d4_value: float | None = None,
    is_head_turned: bool = False,
    is_calibrating: bool = False,
    calibration_progress: float = 0.0,
    calibration_samples_collected: int = 0,
    calibration_target_samples: int = 30,
    is_calibrated: bool = False,
) -> np.ndarray:
    """Vẽ Skeleton, HUD Box và màn hình Calibration Countdown lên ảnh."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    label_key = (smoothed_label or "unknown").lower()
    bgr_color = COLOR_MAP_BGR.get(label_key, (200, 200, 200))

    # 1. Vẽ Bounding box & Skeleton
    if show_skeleton:
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

        if keypoints:
            for p1, p2 in SKELETON_CONNECTIONS:
                if p1 in keypoints and p2 in keypoints:
                    pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                    pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                    cv2.line(annotated, pt1, pt2, bgr_color, 3, cv2.LINE_AA)

            for name, pt in keypoints.items():
                if len(pt) >= 2:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(annotated, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, (px, py), 6, bgr_color, 2, cv2.LINE_AA)

    # 2. Màn hình Countdown khi đang Calibrate
    if is_calibrating:
        calib_overlay = annotated.copy()
        cv2.rectangle(calib_overlay, (0, 0), (w, h), (180, 100, 20), -1)
        cv2.addWeighted(calib_overlay, 0.25, annotated, 0.75, 0, annotated)

        countdown_text = f"{calibration_samples_collected}/{calibration_target_samples}"
        font_scale = 3.0
        thickness = 6
        text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(
            annotated,
            countdown_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        inst_text = "Giu tu the CHUAN - Dang thu thap baseline..."
        inst_size = cv2.getTextSize(inst_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        ix = (w - inst_size[0]) // 2
        cv2.putText(
            annotated,
            inst_text,
            (ix, ty + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 240, 255),
            1,
            cv2.LINE_AA,
        )

        bar_w = int(w * max(0.0, min(1.0, calibration_progress)))
        cv2.rectangle(annotated, (0, 0), (bar_w, 8), (100, 220, 100), -1)
        return annotated

    # 3. Bảng HUD hiển thị kết quả phân tích
    lines: list[tuple[str, tuple[int, int, int]]] = []
    label_vn = LABEL_VN.get(smoothed_label or "", "CHƯA CÓ NGƯỜI")
    lines.append((f"Tu the: {label_vn}", bgr_color))

    if raw_label and raw_label != smoothed_label and raw_label != "no_person":
        raw_vn = LABEL_VN.get(raw_label, raw_label.upper())
        lines.append((f"Raw Model: {raw_vn}", (200, 200, 200)))

    conf_text = f"Tin cay: {prob * 100:.1f}%" if prob else "Trang thai: Da loc nhieu"
    lines.append((conf_text, (220, 220, 220)))
    lines.append((f"FPS: {fps:.1f}", (220, 220, 220)))

    if d4_value is not None:
        yaw_note = " [HEAD TURNED]" if is_head_turned else ""
        yaw_color = (0, 215, 255) if is_head_turned else (190, 190, 190)
        lines.append((f"Yaw D4: {d4_value:.2f}{yaw_note}", yaw_color))

    sh_text = f"Goc vai: {angles.get('shoulder_angle', 0.0):+.1f} deg"
    lines.append((sh_text, (180, 190, 200)))

    neck_text = f"Goc co: {angles.get('head_gravity_angle', 0.0):.1f} deg"
    lines.append((neck_text, (180, 190, 200)))

    if is_calibrated:
        lines.append(("[Baseline: DA HIEU CHUAN]", (100, 240, 100)))
    else:
        lines.append(("[Baseline: CHUA CO]", (120, 120, 180)))

    panel_height = 20 + 24 * len(lines)
    panel_width = 410
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

    for idx, (text, color) in enumerate(lines):
        scale = 0.62 if idx == 0 else 0.45
        thick = 2 if idx == 0 else 1
        cv2.putText(
            annotated,
            text,
            (22, 34 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
            cv2.LINE_AA,
        )

    return annotated


# ──────────────────────────────────────────────────────────────
# 6. Hàm phân tích ảnh đơn lẻ (Snapshot & Upload)
# ──────────────────────────────────────────────────────────────
def analyze_static_frame(img_bgr: np.ndarray) -> dict[str, Any]:
    """
    Phân tích ảnh tĩnh (Snapshot / Upload).
    Nếu đã hiệu chuẩn webcam, dùng baseline của người dùng.
    Nếu chưa, dùng DEFAULT_BASELINE tham chiếu để suy luận.
    """
    pose = engine.detector.detect(img_bgr)
    if pose is None:
        return {
            "has_person": False,
            "status": "NO_PERSON",
            "message": "Không tìm thấy người trong ảnh.",
            "bbox": None,
            "keypoints": {},
            "angles": {},
            "raw_label": None,
            "probability": None,
            "d4_value": None,
            "is_default_baseline": False,
        }

    keypoints = PostureInferenceEngine._extract_keypoints(pose)
    bbox = PostureInferenceEngine._extract_bbox(pose)
    raw_features = engine.extractor.extract(pose)

    if raw_features is None:
        return {
            "has_person": True,
            "status": "LOW_CONFIDENCE",
            "message": "Keypoint bị che khuất hoặc không đủ độ tin cậy.",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": {},
            "raw_label": None,
            "probability": None,
            "d4_value": None,
            "is_default_baseline": False,
        }

    raw_arr = np.asarray(raw_features, dtype=np.float64)
    angles = PostureInferenceEngine._extract_angles(raw_arr)
    d4_val = PostureInferenceEngine._calculate_face_rotation_proxy(pose)

    # Lấy baseline
    if engine.is_calibrated and engine.calibrator.baseline is not None:
        baseline = engine.calibrator.baseline
        is_default = False
    else:
        baseline = DEFAULT_BASELINE
        is_default = True

    # RAW12 -> REP13 -> Predict
    rep = engine.representation_builder.transform(raw_arr, baseline)
    pred = engine.predictor.predict(rep)

    return {
        "has_person": True,
        "status": "OK",
        "message": "Phát hiện tư thế thành công",
        "bbox": bbox,
        "keypoints": keypoints,
        "angles": angles,
        "raw_label": pred["raw_label"],
        "probability": pred.get("probability"),
        "d4_value": d4_val,
        "is_default_baseline": is_default,
        "rep_features": rep,
    }


# ──────────────────────────────────────────────────────────────
# 7. UI Helper Functions
# ──────────────────────────────────────────────────────────────
def render_posture_badge(label: str) -> str:
    """Trả về mã HTML cho posture status badge."""
    badge_map = {
        "correct": ("badge-correct", "🟢", "Tư thế Chuẩn"),
        "forward_slouch": ("badge-slouch", "🔴", "Gù Lưng / Cúi Đầu"),
        "lean_left": ("badge-lean-left", "🟡", "Nghiêng Trái"),
        "lean_right": ("badge-lean-right", "🟣", "Nghiêng Phải"),
        "head_turned": ("badge-head-turned", "🔵", "Đang Quay Đầu"),
        "evaluating": ("badge-evaluating", "🟡", "Đang Đánh Giá"),
    }
    cls, emoji, text = badge_map.get(label, ("badge-no-person", "⚪", "Không có người"))
    return f'<span class="badge {cls}"><span class="pulse-dot" style="background:currentColor"></span> {text}</span>'


def render_gauge(
    label: str,
    value: float,
    val_text: str,
    pct: float,
    color: str,
    min_label: str = "",
    mid_label: str = "",
    max_label: str = "",
) -> str:
    """Trả về HTML cho thanh đo sinh trắc học."""
    clamped_pct = max(3, min(97, pct))
    return f"""
    <div class="gauge-container">
        <div class="gauge-header">
            <span class="gauge-label">{label}</span>
            <span class="gauge-value">{val_text}</span>
        </div>
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{clamped_pct:.1f}%; background:{color};"></div>
        </div>
        <div class="gauge-labels">
            <span>{min_label}</span>
            <span style="color:{color}; font-weight:500;">{mid_label}</span>
            <span>{max_label}</span>
        </div>
    </div>
    """


def render_log_entries(logs: list[dict]) -> str:
    """Hiển thị nhật ký sự kiện trực tiếp."""
    if not logs:
        return '<div class="log-entry log-info" style="justify-content:center;">Chưa có dữ liệu. Bật webcam để bắt đầu giám sát.</div>'

    html_parts = []
    for log in logs[:20]:
        type_class = f"log-{log['type']}"
        html_parts.append(
            f'<div class="log-entry {type_class}">'
            f'<span class="log-time">{log["time"]}</span>'
            f'<span>{log["text"]}</span>'
            f"</div>"
        )
    return "".join(html_parts)


def format_elapsed(seconds: int) -> str:
    """Định dạng thời gian theo HH:MM:SS."""
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# ──────────────────────────────────────────────────────────────
# 8. Sidebar: Cấu hình & Nút điều khiển
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt Giám sát")

    camera_idx = st.selectbox("📷 Chọn cổng Camera:", [0, 1, 2], index=0)

    use_smoothing = st.toggle(
        "🔄 Lọc mịn Temporal Smoothing",
        value=True,
        help="Sử dụng bộ lọc majority vote & yaw gating V03 để giảm nhiễu dự đoán.",
    )

    show_skeleton = st.toggle(
        "🦴 Hiển thị khung xương (Skeleton)",
        value=True,
        help="Bật/tắt hiển thị keypoints và skeleton trên khung hình camera.",
    )

    alert_threshold_sec = st.slider(
        "⏱️ Ngưỡng cảnh báo sai tư thế (giây):",
        min_value=1,
        max_value=10,
        value=int(engine.stats.alert_threshold_seconds),
        step=1,
        help="Số giây ngồi sai tư thế liên tục trước khi kích hoạt cảnh báo.",
    )
    engine.stats.alert_threshold_seconds = float(alert_threshold_sec)

    st.markdown("---")

    # ── Baseline Calibration Section ──
    st.markdown("### 🎯 Personal Baseline Calibration")

    if engine.is_calibrating:
        st.markdown(
            f'<span class="calib-badge calib-badge-collecting">🔵 Đang thu thập ({engine.calibrator.sample_count}/{engine.calibration_samples})...</span>',
            unsafe_allow_html=True,
        )
        st.progress(engine.calibration_progress)
        st.caption("Hãy giữ tư thế ngồi chuẩn! Đang thu thập các frame mốc...")
    elif engine.is_calibrated:
        n_frames = engine.calibrator.sample_count
        st.markdown(
            f'<span class="calib-badge calib-badge-active">✅ Đã hiệu chuẩn ({n_frames} mẫu)</span>',
            unsafe_allow_html=True,
        )
        sh_w = engine.calibrator.baseline_shoulder_width
        baseline_vec = engine.calibrator.baseline
        neck_p = baseline_vec[3] if baseline_vec is not None else 0.0
        sh_r = baseline_vec[0] if baseline_vec is not None else 0.0

        st.markdown(
            f"""
            <div class="calib-info-card">
                <div class="calib-title">📐 Personal Baseline V03 (REP13)</div>
                <div class="calib-detail">
                    • Vai chuẩn: {sh_w:.1f} px<br>
                    • Góc vai gốc: {sh_r:.1f}°<br>
                    • Góc gập cổ: {neck_p:.1f}°
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Reset Baseline", use_container_width=True):
            engine.reset_calibration()
            append_log("Đã đặt lại mốc hiệu chuẩn baseline cá nhân.", "info")
            st.rerun()
    else:
        st.markdown(
            '<span class="calib-badge calib-badge-pending">⚠ Chưa hiệu chuẩn</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Ngồi đúng tư thế chuẩn, sau đó nhấn nút dưới đây để hệ thống thu thập 30 mẫu mốc chuẩn cá nhân."
        )
        if st.button("🎯 Bắt đầu Hiệu chuẩn Baseline", use_container_width=True, type="primary"):
            engine.start_calibration()
            append_log("Bắt đầu thu thập mốc chuẩn cá nhân (30 mẫu)...", "info")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Thông tin Model (Folder 1)")
    lopo_f1 = metadata.get("lopo_metrics", {}).get("LOPO Macro F1 Mean", 0.92) * 100
    st.info(
        f"**Model:** {metadata.get('model_name', 'SVM RBF')}\n\n"
        f"**Kiến trúc:** RAW12 → REP13\n\n"
        f"**LOPO Macro F1:** {lopo_f1:.1f}%\n\n"
        f"**Pose Detector:** YOLO Pose (yolo26n-pose.pt)"
    )

    st.markdown("---")
    st.markdown("### 🎛️ Điều khiển")

    if st.button("🔄 Đặt lại thống kê phiên", use_container_width=True):
        reset_session()
        st.success("✅ Đã reset toàn bộ số liệu phiên!")
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:11px; color:#94a3b8; line-height:1.6;">
        <strong>Smart Posture Monitor v3.0</strong><br>
        100% Python (Streamlit + OpenCV)<br>
        YOLO Pose • SVM RBF • Temporal Yaw Gating<br>
        Personal Baseline Calibration (REP13)
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# 9. Dashboard Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <div style="width:40px; height:40px; border-radius:10px; background:rgba(16,185,129,0.1);
             border:1px solid rgba(16,185,129,0.2); display:flex; align-items:center;
             justify-content:center; font-size:20px;">
            🧘
        </div>
        <div>
            <div style="display:flex; align-items:center; gap:8px;">
                <h2 style="margin:0; font-size:20px; font-weight:700; color:#0f172a; letter-spacing:-0.02em;">
                    Smart Posture Monitor
                </h2>
                <span style="font-size:11px; padding:2px 8px; border-radius:999px; background:#f1f5f9;
                      color:#475569; border:1px solid #e2e8f0; font-weight:500;">
                    v3.0 • RAW12 + REP13
                </span>
            </div>
            <p style="margin:2px 0 0; font-size:12px; color:#64748b;">
                Giám sát tư thế thời gian thực qua webcam & công nghệ AI — 100% Python Stack
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# 10. KPI Metric Cards (Row 1)
# ──────────────────────────────────────────────────────────────
session_stats = engine.stats.to_dict()
ergo_score = session_stats["ergonomics_score"]
elapsed_str = format_elapsed(session_stats["elapsed_seconds"])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    last_lbl = st.session_state.last_label or "no_person"
    badge_html = render_posture_badge(last_lbl)
    prob_val = st.session_state.last_prob
    prob_text = f"{prob_val * 100:.1f}%" if prob_val is not None else "--"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">TƯ THẾ HIỆN TẠI</div>
            {badge_html}
            <div class="kpi-sub">Độ tin cậy: <strong style="color:#334155">{prob_text}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    score_color = "#047857" if ergo_score >= 80 else ("#b45309" if ergo_score >= 60 else "#be123c")
    bar_color = "#10b981" if ergo_score >= 80 else ("#f59e0b" if ergo_score >= 60 else "#f43f5e")
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">ĐIỂM CÔNG THÁI HỌC</div>
            <div class="kpi-value" style="color:{score_color}">{ergo_score}%</div>
            <div style="width:100%; height:6px; background:#f1f5f9; border-radius:999px;
                 overflow:hidden; margin-top:10px;">
                <div style="width:{ergo_score}%; height:100%; background:{bar_color};
                     border-radius:999px; transition:width 0.5s;"></div>
            </div>
            <div class="kpi-sub">{"Đạt chuẩn ✓" if ergo_score >= 80 else "Cần cải thiện ↓"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    correct_sec = int(session_stats.get("correct_seconds", 0))
    correct_min = correct_sec // 60
    correct_rem = correct_sec % 60
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">THỜI GIAN PHIÊN</div>
            <div class="kpi-value">{elapsed_str}</div>
            <div class="kpi-sub">Đúng chuẩn: <strong style="color:#047857">{correct_min}m {correct_rem}s ({ergo_score}%)</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">CẢNH BÁO SAI TƯ THẾ</div>
            <div class="kpi-value">{session_stats['alert_count']}</div>
            <div class="kpi-sub">Ngưỡng báo động: > {int(engine.stats.alert_threshold_seconds)} giây</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# 11. Main Workspace: Tabs
# ──────────────────────────────────────────────────────────────
tab_live, tab_snapshot, tab_upload = st.tabs([
    "📹 Giám sát Webcam Trực tiếp",
    "📸 Chụp ảnh đơn lẻ",
    "📁 Tải ảnh lên kiểm tra",
])

# ─── Tab 1: Real-time Webcam Monitoring ──────────────────────
with tab_live:
    left_col, right_col = st.columns([7, 5])

    with left_col:
        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">📹 Khung hình Webcam Giám sát</div>
                <span class="section-badge">Pipeline: YOLO + REP13 + SVM</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_camera = st.checkbox(
            "▶ Bật Webcam Giám sát Liên tục",
            value=False,
            help="Tick để bật webcam. Bỏ tick để dừng.",
        )

        frame_placeholder = st.empty()

        if not run_camera:
            st.markdown(
                """
                <div class="shadcn-card" style="text-align:center; padding:40px 20px; background:#f8fafc;">
                    <div style="font-size:36px; margin-bottom:10px;">📷</div>
                    <p style="font-size:14px; font-weight:600; color:#334155;">Webcam chưa được kích hoạt</p>
                    <p style="font-size:12px; color:#64748b; max-width:360px; margin:6px auto 0;">
                        Tick vào checkbox <strong>"▶ Bật Webcam Giám sát Liên tục"</strong> ở trên
                        để hệ thống bắt đầu nhận diện khung xương và giám sát tư thế thời gian thực.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">🧭 Góc Sinh trắc học Real-time</div>
                <span class="section-badge">RAW12 & Vi phân REP13</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        angle_placeholder = st.empty()

        angle_placeholder.markdown(
            render_gauge(
                "Góc nghiêng vai (Shoulder)",
                0.0,
                "0.0°",
                50,
                "#10b981",
                "Lệch trái (-15°)",
                "Chuẩn (0°)",
                "Lệch phải (+15°)",
            )
            + render_gauge(
                "Góc gập cổ (Neck Pitch)",
                0.0,
                "0.0°",
                30,
                "#3b82f6",
                "0°",
                "Bình thường (< 25°)",
                "Cúi nhiều (> 45°)",
            )
            + render_gauge(
                "Tỷ lệ quay mặt Yaw (D4 Proxy)",
                1.0,
                "1.00",
                50,
                "#6366f1",
                "0.45 (Trái)",
                "Nhìn thẳng (1.0)",
                "1.80 (Phải)",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-header" style="margin-top:16px;">
                <div class="section-title">📋 Nhật ký Giám sát Trực tiếp</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        log_placeholder = st.empty()
        log_placeholder.markdown(
            render_log_entries(st.session_state.logs), unsafe_allow_html=True
        )

    # ── Webcam Loop ──
    if run_camera:
        cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        time.sleep(1.0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        frame_counter = 0

        try:
            while run_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("⚠️ Không thể đọc webcam. Kiểm tra xem webcam có đang bị ứng dụng khác khóa không!")
                    time.sleep(1)
                    break

                now = time.time()
                fps = 1.0 / max(0.001, (now - prev_time))
                prev_time = now

                # Xử lý frame qua PostureInferenceEngine V03
                result = engine.process_frame(frame)
                status = result.get("status", "OK")

                smoothed_label = result.get("smoothed_label")
                raw_label = result.get("raw_label")
                prob = result.get("probability")
                angles = result.get("angles", {})
                bbox = result.get("bbox")
                keypoints = result.get("keypoints", {})
                trigger_alert = result.get("trigger_alert", False)

                d4_val = engine.temporal_monitor.state.yaw_value
                is_head_turned = engine.temporal_monitor.state.is_head_turned

                # Quản lý sự kiện và log
                display_label = smoothed_label if use_smoothing else raw_label
                if display_label != st.session_state.last_label:
                    if display_label == "correct":
                        append_log("Tư thế đã trở lại trạng thái chuẩn.", "good")
                    elif display_label == "forward_slouch":
                        append_log("Phát hiện dấu hiệu cúi gù lưng về phía trước.", "warn")
                    elif display_label == "lean_left":
                        append_log("Phát hiện lệch vai/cơ thể sang bên trái.", "warn")
                    elif display_label == "lean_right":
                        append_log("Phát hiện lệch vai/cơ thể sang bên phải.", "warn")
                    elif display_label == HEAD_TURNED_LABEL:
                        append_log("Phát hiện quay đầu sang hướng khác (Yaw Gated).", "info")
                    st.session_state.last_label = display_label

                st.session_state.last_prob = prob

                if trigger_alert:
                    append_log("⚠️ Cảnh báo: Bạn đang ngồi sai tư thế liên tục! Hãy điều chỉnh lại dáng ngồi.", "alert")

                # Vẽ khung hình HUD
                annotated_frame = draw_hud(
                    frame=frame,
                    bbox=bbox,
                    keypoints=keypoints,
                    smoothed_label=display_label,
                    raw_label=raw_label,
                    prob=prob,
                    angles=angles,
                    fps=fps,
                    show_skeleton=show_skeleton,
                    d4_value=d4_val,
                    is_head_turned=is_head_turned,
                    is_calibrating=engine.is_calibrating,
                    calibration_progress=engine.calibration_progress,
                    calibration_samples_collected=engine.calibrator.sample_count,
                    calibration_target_samples=engine.calibration_samples,
                    is_calibrated=engine.is_calibrated,
                )

                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                if status == "CALIBRATED":
                    append_log(f"✅ Baseline cá nhân hiệu chuẩn thành công ({engine.calibrator.sample_count} mẫu)!", "good")
                    st.rerun()

                # Cập nhật gauges và nhật ký định kỳ
                frame_counter += 1
                if frame_counter % 5 == 0:
                    sh_angle = angles.get("shoulder_angle", 0.0)
                    neck_angle = angles.get("head_gravity_angle", 0.0)
                    head_body = angles.get("head_body_angle", 0.0)
                    d4 = d4_val if d4_val is not None else 1.0

                    sh_pct = ((sh_angle + 15) / 30) * 100
                    sh_color = "#10b981" if abs(sh_angle) <= 5.0 else "#f43f5e"

                    neck_pct = (neck_angle / 50) * 100
                    neck_color = "#3b82f6" if neck_angle < 25.0 else "#f59e0b"

                    d4_pct = ((d4 - 0.45) / (1.80 - 0.45)) * 100

                    diff_pct = ((head_body + 20) / 40) * 100
                    diff_color = "#10b981" if abs(head_body) <= 8.0 else "#f59e0b"

                    angle_placeholder.markdown(
                        render_gauge(
                            "Góc nghiêng vai (Shoulder)",
                            sh_angle,
                            f"{sh_angle:+.1f}°",
                            sh_pct,
                            sh_color,
                            "Lệch trái (-15°)",
                            "Chuẩn (0°)",
                            "Lệch phải (+15°)",
                        )
                        + render_gauge(
                            "Góc gập cổ (Neck Pitch)",
                            neck_angle,
                            f"{neck_angle:.1f}°",
                            neck_pct,
                            neck_color,
                            "0°",
                            "Bình thường (< 25°)",
                            "Cúi nhiều (> 45°)",
                        )
                        + render_gauge(
                            "Tỷ lệ quay mặt Yaw (D4 Proxy)",
                            d4,
                            f"{d4:.2f}",
                            d4_pct,
                            "#6366f1",
                            "0.45 (Trái)",
                            "Nhìn thẳng (1.0)",
                            "1.80 (Phải)",
                        )
                        + render_gauge(
                            "Độ lệch đầu - thân (Head-Body Diff)",
                            head_body,
                            f"{head_body:+.1f}°",
                            diff_pct,
                            diff_color,
                            "-20°",
                            "0° (Cân đối)",
                            "+20°",
                        ),
                        unsafe_allow_html=True,
                    )

                    log_placeholder.markdown(
                        render_log_entries(st.session_state.logs), unsafe_allow_html=True
                    )

                time.sleep(0.02)
        finally:
            if "cap" in locals() and cap.isOpened():
                cap.release()

# ─── Tab 2: Snapshot Test ─────────────────────────────────────
with tab_snapshot:
    st.markdown(
        """
        <div class="shadcn-card" style="text-align:center; padding:20px;">
            <p style="font-size:14px; color:#334155; font-weight:500;">
                📸 Chụp một bức ảnh từ webcam để phân tích tĩnh góc sinh trắc học và tư thế
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_photo = st.camera_input("Chụp ảnh kiểm tra tư thế")

    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Đang phân tích tư thế..."):
            res = analyze_static_frame(img)

        if res["has_person"] and res["raw_label"] is not None:
            annotated = draw_hud(
                frame=img,
                bbox=res["bbox"],
                keypoints=res["keypoints"],
                smoothed_label=res["raw_label"],
                raw_label=res["raw_label"],
                prob=res["probability"],
                angles=res["angles"],
                fps=0.0,
                show_skeleton=True,
                d4_value=res["d4_value"],
                is_calibrated=engine.is_calibrated,
            )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            label = res["raw_label"]
            prob = res["probability"]
            badge_html = render_posture_badge(label)
            prob_text = f"{prob * 100:.1f}%" if prob else "N/A"
            note_extra = " (Baseline tham chiếu mẫu)" if res["is_default_baseline"] else " (Personal Baseline)"

            st.markdown(
                f"""
                <div class="shadcn-card" style="margin-top:12px;">
                    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                        <span style="font-size:13px; font-weight:600; color:#334155;">Kết quả:</span>
                        {badge_html}
                        <span style="font-size:12px; color:#64748b;">Độ tin cậy: <strong>{prob_text}</strong>{note_extra}</span>
                    </div>
                    <div style="margin-top:12px; padding-top:10px; border-top:1px solid #f1f5f9;">
                        <table style="width:100%; font-size:12px; color:#475569;">
                            <tr>
                                <td>Góc nghiêng vai:</td>
                                <td style="font-weight:600;">{res['angles'].get('shoulder_angle', 0.0):+.1f}°</td>
                                <td>Góc gập cổ (Neck Pitch):</td>
                                <td style="font-weight:600;">{res['angles'].get('head_gravity_angle', 0.0):.1f}°</td>
                            </tr>
                            <tr>
                                <td>Tỷ lệ quay mặt D4:</td>
                                <td style="font-weight:600;">{res['d4_value'] if res['d4_value'] is not None else 1.0:.2f}</td>
                                <td>Góc lệch đầu - thân:</td>
                                <td style="font-weight:600;">{res['angles'].get('head_body_angle', 0.0):+.1f}°</td>
                            </tr>
                        </table>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif res["has_person"]:
            st.warning("⚠️ Keypoint chưa đủ tin cậy để dự đoán tư thế.")
        else:
            st.error("❌ Không tìm thấy người trong ảnh. Hãy đảm bảo đủ ánh sáng và ngồi rõ trong khung hình.")

# ─── Tab 3: Upload Image Test ─────────────────────────────────
with tab_upload:
    st.markdown(
        """
        <div class="shadcn-card" style="text-align:center; padding:20px;">
            <p style="font-size:14px; color:#334155; font-weight:500;">
                📁 Tải ảnh từ máy tính (JPG/PNG) để phân tích tư thế
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh cần kiểm tra",
        type=["jpg", "jpeg", "png"],
        help="Hỗ trợ JPG, JPEG, PNG. Ảnh nên chứa người ngồi ở góc 45° bên trái.",
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Đang phân tích tư thế..."):
            res = analyze_static_frame(img)

        col_img, col_result = st.columns([6, 4])

        with col_img:
            if res["has_person"] and res["raw_label"] is not None:
                annotated = draw_hud(
                    frame=img,
                    bbox=res["bbox"],
                    keypoints=res["keypoints"],
                    smoothed_label=res["raw_label"],
                    raw_label=res["raw_label"],
                    prob=res["probability"],
                    angles=res["angles"],
                    fps=0.0,
                    show_skeleton=True,
                    d4_value=res["d4_value"],
                    is_calibrated=engine.is_calibrated,
                )
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Ảnh đã phân tích với skeleton overlay",
                )
            else:
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption="Ảnh gốc (không phát hiện được người)",
                )

        with col_result:
            if res["has_person"] and res["raw_label"] is not None:
                label = res["raw_label"]
                prob = res["probability"]
                angles = res["angles"]
                badge_html = render_posture_badge(label)
                note_extra = " (Baseline tham chiếu mẫu)" if res["is_default_baseline"] else " (Personal Baseline)"

                st.markdown(
                    f"""
                    <div class="shadcn-card">
                        <div class="section-header">
                            <div class="section-title">📊 Kết quả Phân tích</div>
                        </div>
                        <div style="margin:12px 0;">
                            {badge_html}
                        </div>
                        <div style="font-size:12px; color:#475569; margin-top:8px;">
                            Độ tin cậy: <strong style="color:#0f172a">{prob * 100:.1f}%</strong>{note_extra}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                sh = angles.get("shoulder_angle", 0.0)
                neck = angles.get("head_gravity_angle", 0.0)
                d4 = res["d4_value"] if res["d4_value"] is not None else 1.0

                st.markdown(
                    render_gauge(
                        "Góc nghiêng vai",
                        sh,
                        f"{sh:+.1f}°",
                        ((sh + 15) / 30) * 100,
                        "#10b981" if abs(sh) <= 5 else "#f43f5e",
                        "-15°",
                        "0°",
                        "+15°",
                    )
                    + render_gauge(
                        "Góc gập cổ",
                        neck,
                        f"{neck:.1f}°",
                        (neck / 50) * 100,
                        "#3b82f6" if neck < 25 else "#f59e0b",
                        "0°",
                        "25°",
                        "50°",
                    )
                    + render_gauge(
                        "Tỷ lệ quay mặt D4",
                        d4,
                        f"{d4:.2f}",
                        ((d4 - 0.45) / (1.80 - 0.45)) * 100,
                        "#6366f1",
                        "0.45",
                        "1.0",
                        "1.80",
                    ),
                    unsafe_allow_html=True,
                )
            elif res["has_person"]:
                st.warning("⚠️ Keypoint chưa đủ tin cậy để phân tích.")
            else:
                st.error("❌ Không tìm thấy người trong ảnh.")

# ──────────────────────────────────────────────────────────────
# 12. Row 3: Analytics & Posture Distribution Charts
# ──────────────────────────────────────────────────────────────
st.markdown("---")

chart_col, guide_col = st.columns([4, 8])

with chart_col:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">🥧 Tỷ lệ Phân phối Tư thế</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_data = pd.DataFrame(
        {
            "Tư thế": ["Chuẩn", "Gù Lưng", "Lệch Trái", "Lệch Phải"],
            "Số Frame": [
                max(session_stats.get("correct_frames", 0), 1),
                session_stats.get("slouch_frames", 0),
                session_stats.get("lean_left_frames", 0),
                session_stats.get("lean_right_frames", 0),
            ],
        }
    ).set_index("Tư thế")

    st.bar_chart(chart_data, height=220, color="#10b981")

with guide_col:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">📖 Cơ chế Temporal Smoothing & Hướng dẫn sử dụng</div>
            <span style="font-size:11px; padding:3px 10px; border-radius:6px; background:#ecfdf5;
                  color:#047857; border:1px solid #a7f3d0; font-weight:500;">
                Khử nhiễu Hysteresis 2 frames + Yaw Gating
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-card-title">
                    <span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;"></span>
                    Tư thế Chuẩn (Correct)
                </div>
                <p style="color:#64748b; margin:0; font-size:11px; line-height:1.5;">
                    Lưng thẳng, mắt nhìn ngang tầm màn hình, hai vai cân đối trong biên độ ±5°.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g2:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-card-title">
                    <span style="width:8px;height:8px;border-radius:50%;background:#f43f5e;display:inline-block;"></span>
                    Gù lưng (Forward Slouch)
                </div>
                <p style="color:#64748b; margin:0; font-size:11px; line-height:1.5;">
                    Cổ vươn về phía trước, khoảng cách mắt-vai co lại. Cảnh báo sau > 2s liên tục.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g3:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-card-title">
                    <span style="width:8px;height:8px;border-radius:50%;background:#f59e0b;display:inline-block;"></span>
                    Nghiêng người (Lean)
                </div>
                <p style="color:#64748b; margin:0; font-size:11px; line-height:1.5;">
                    Nghiêng sang trái/phải do tì cằm hoặc ngồi lệch trọng tâm cơ thể.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────
# 13. Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-bar">
        <span>Backend: Streamlit • YOLO Pose • Sklearn SVM RBF • Temporal Yaw Gating</span>
        <span style="font-weight:600; color:#475569;">100% Python — Smart Posture Monitor V03</span>
    </div>
    """,
    unsafe_allow_html=True,
)
