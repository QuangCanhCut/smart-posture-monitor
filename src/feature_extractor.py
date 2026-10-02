from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class FeatureExtractor:
    """
    Trích xuất 12 RAW geometric features từ 6 keypoint.

    Chỉ đo hình học của frame hiện tại:
    - Không trừ personal baseline.
    - Không normalize bằng shoulder_width hiện tại.
    - RepresentationBuilder sẽ xử lý delta / normalize / log-ratio.
    """

    KEYPOINT_NAMES = (
        "nose", "left_eye", "right_eye",
        "left_ear", "left_shoulder", "right_shoulder",
    )

    # Thứ tự cố định của vector RAW12.
    FEATURE_NAMES = (
        "shoulder_roll_deg",
        "eye_roll_deg",
        "head_shoulder_roll_diff_deg",
        "neck_pitch_deg",
        "eye_center_x_px",
        "eye_center_y_px",
        "shoulder_center_x_px",
        "eye_shoulder_vertical_gap_px",
        "eye_shoulder_horizontal_offset_px",
        "inter_eye_distance_px",
        "ear_nose_horizontal_span_px",
        "shoulder_width_px",
    )

    # Dùng ở RepresentationBuilder để áp đúng kiểu biến đổi.
    ANGLE_FEATURES = FEATURE_NAMES[:4]
    SPATIAL_FEATURES = FEATURE_NAMES[4:9]
    SCALE_FEATURES = FEATURE_NAMES[9:]

    def __init__(self, min_keypoint_confidence: float = 0.35) -> None:
        if not 0.0 <= min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be in [0, 1].")
        self.min_keypoint_confidence = float(min_keypoint_confidence)

    @property
    def feature_count(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract(
        self,
        pose_result: Mapping[str, Any] | Sequence[Any] | np.ndarray | None,
    ) -> np.ndarray | None:
        """Pose -> validate keypoint -> 4 nhóm công thức -> RAW12."""
        p = self._parse_points(pose_result)
        if p is None:
            return None

        features = np.asarray(
            (
                *self._world_orientation(p),
                *self._sagittal(p),
                *self._spatial(p),
                *self._perspective(p),
            ),
            dtype=np.float32,
        )

        if features.size != self.feature_count or not np.all(np.isfinite(features)):
            return None
        return features

    def as_dict(self, features: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Đổi RAW12 sang dict để debug/log dễ đọc."""
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        if vector.size != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} features, got {vector.size}.")
        return dict(zip(self.FEATURE_NAMES, vector.tolist()))

    # ==========================================================
    # NHÓM 1 — WORLD-FRAME ORIENTATION
    # ==========================================================

    def _world_orientation(
        self, p: dict[str, np.ndarray]
    ) -> tuple[float, float, float]:
        # RAW01: góc nghiêng vai so với trục ngang ảnh.
        shoulder_roll = self._angle_deg(p["left_shoulder"], p["right_shoulder"])

        # RAW02: góc nghiêng đầu, đo bằng đường nối hai mắt.
        eye_roll = self._angle_deg(p["left_eye"], p["right_eye"])

        # RAW03: đầu nghiêng khác thân bao nhiêu.
        roll_diff = self._wrap_angle_deg(eye_roll - shoulder_roll)

        return shoulder_roll, eye_roll, roll_diff

    # ==========================================================
    # NHÓM 2 — SAGITTAL / NECK FLEXION
    # ==========================================================

    def _sagittal(self, p: dict[str, np.ndarray]) -> tuple[float]:
        # RAW04: tai trái -> mũi làm proxy cho góc cúi/ngẩng đầu
        # với camera khoảng 45° từ bên trái.
        neck_pitch = self._angle_deg(p["left_ear"], p["nose"])
        return (neck_pitch,)

    # ==========================================================
    # NHÓM 3 — SPATIAL DISPLACEMENT
    # ==========================================================

    def _spatial(
        self, p: dict[str, np.ndarray]
    ) -> tuple[float, float, float, float, float]:
        # Tâm mắt đại diện vị trí đầu; tâm vai đại diện thân trên.
        eye_center = self._midpoint(p["left_eye"], p["right_eye"])
        shoulder_center = self._midpoint(p["left_shoulder"], p["right_shoulder"])

        # Y ảnh tăng từ trên xuống dưới.
        vertical_gap = shoulder_center[1] - eye_center[1]

        # Có dấu để giữ thông tin lệch trái/phải của đầu so với thân.
        horizontal_offset = eye_center[0] - shoulder_center[0]

        return (
            float(eye_center[0]),          # RAW05
            float(eye_center[1]),          # RAW06
            float(shoulder_center[0]),     # RAW07
            float(vertical_gap),           # RAW08
            float(horizontal_offset),      # RAW09
        )

    # ==========================================================
    # NHÓM 4 — PERSPECTIVE / DEPTH PROXY
    # ==========================================================

    def _perspective(
        self, p: dict[str, np.ndarray]
    ) -> tuple[float, float, float]:
        # RAW10: scale biểu kiến của vùng mặt.
        inter_eye_distance = self._distance(p["left_eye"], p["right_eye"])

        # RAW11: chỉ lấy độ dài theo trục X tai trái - mũi.
        ear_nose_span = abs(float(p["nose"][0] - p["left_ear"][0]))

        # RAW12: chỉ là RAW feature, không dùng làm mẫu số ở đây.
        shoulder_width = self._distance(p["left_shoulder"], p["right_shoulder"])

        return inter_eye_distance, ear_nose_span, shoulder_width

    # ==========================================================
    # INPUT VALIDATION
    # ==========================================================

    def _parse_points(
        self,
        pose_result: Mapping[str, Any] | Sequence[Any] | np.ndarray | None,
    ) -> dict[str, np.ndarray] | None:
        if pose_result is None:
            return None

        # Hỗ trợ {"keypoints": ...}, dict trực tiếp hoặc array/list 6 điểm.
        raw = (
            pose_result.get("keypoints", pose_result)
            if isinstance(pose_result, Mapping)
            else pose_result
        )

        if isinstance(raw, Mapping):
            items = [raw.get(name) for name in self.KEYPOINT_NAMES]
        else:
            try:
                if len(raw) < len(self.KEYPOINT_NAMES):
                    return None
                items = list(raw[: len(self.KEYPOINT_NAMES)])
            except (TypeError, IndexError):
                return None

        points: dict[str, np.ndarray] = {}

        for name, item in zip(self.KEYPOINT_NAMES, items):
            parsed = self._parse_keypoint(item)
            if parsed is None:
                return None

            x, y, conf = parsed
            if conf < self.min_keypoint_confidence:
                return None

            points[name] = np.array([x, y], dtype=np.float64)

        # Hai mắt hoặc hai vai trùng nhau -> hình học không hợp lệ.
        if self._distance(points["left_eye"], points["right_eye"]) <= 1e-6:
            return None
        if self._distance(points["left_shoulder"], points["right_shoulder"]) <= 1e-6:
            return None

        return points

    @staticmethod
    def _parse_keypoint(item: Any) -> tuple[float, float, float] | None:
        """Chuẩn hóa một keypoint về (x, y, confidence)."""
        if item is None:
            return None

        try:
            if isinstance(item, Mapping):
                x, y = float(item["x"]), float(item["y"])
                conf = float(
                    item.get("confidence", item.get("conf", item.get("score", 1.0)))
                )
            else:
                arr = np.asarray(item, dtype=np.float64).reshape(-1)
                if arr.size < 2:
                    return None
                x, y = float(arr[0]), float(arr[1])
                conf = float(arr[2]) if arr.size >= 3 else 1.0
        except (TypeError, ValueError, KeyError):
            return None

        return (x, y, conf) if np.all(np.isfinite([x, y, conf])) else None

    # ==========================================================
    # GEOMETRY PRIMITIVES
    # ==========================================================

    @staticmethod
    def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Trung điểm hai điểm 2D."""
        return (a + b) * 0.5

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Khoảng cách Euclidean."""
        return float(np.linalg.norm(b - a))

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        """Góc vector a -> b so với trục X ảnh, đơn vị độ."""
        dx, dy = b - a
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        """Đưa góc về [-180, 180)."""
        return float((angle + 180.0) % 360.0 - 180.0)
