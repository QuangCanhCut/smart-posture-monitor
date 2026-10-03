from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.feature_extractor import FeatureExtractor
from src.personal_calibration import PersonalCalibration
from src.pose_detector import PoseDetector
from src.posture_predictor import PosturePredictor
from src.representation_builder import RepresentationBuilder
from src.session_statistics import SessionStatistics
from src.temporal_monitor import TemporalMonitor


class PostureInferenceEngine:
    """
    Điều phối toàn diện pipeline inference cho Smart Posture Monitor V03.
    Tích hợp trọn gói:
        YOLO Pose -> RAW12 -> PersonalCalibration -> REP13 -> Predictor
        -> Temporal Smoothing & Yaw Gating -> Session Statistics & Cảnh báo.
    """

    RAW_FEATURE_COUNT = PersonalCalibration.FEATURE_DIM
    REP_FEATURE_COUNT = len(RepresentationBuilder.OUTPUT_FEATURE_NAMES)

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        yolo_path: Optional[Path] = None,
        calibration_samples: int = 30,
        fps: int = 15,
    ) -> None:
        if calibration_samples <= 0:
            raise ValueError("calibration_samples phải lớn hơn 0.")

        if len(FeatureExtractor.FEATURE_NAMES) != self.RAW_FEATURE_COUNT:
            raise RuntimeError(
                f"FeatureExtractor ({len(FeatureExtractor.FEATURE_NAMES)}) "
                f"không khớp PersonalCalibration ({self.RAW_FEATURE_COUNT})"
            )

        # 1. Pipeline thị giác & Mô hình AI
        self.detector = PoseDetector(model_path=yolo_path, person_conf_threshold=0.45)
        self.extractor = FeatureExtractor(min_keypoint_confidence=0.35)
        self.calibrator = PersonalCalibration(target_samples=calibration_samples)
        self.representation_builder = RepresentationBuilder()
        self.predictor = PosturePredictor(model_path=model_path, metadata_path=metadata_path)

        self.calibration_samples = calibration_samples
        self._is_calibrating = False

        # 2. Xử lý hậu dự đoán & Thống kê phiên thời gian thực (15 FPS)
        self.fps = fps
        self.temporal_monitor = TemporalMonitor(
            fps=fps,
            window_seconds=3.0,       # Cửa sổ trượt 3 giây để triệt tiêu flickering
            hysteresis_frames=4,       # Cần 4 frame ổn định liên tục để đổi trạng thái
            yaw_mode="informative",   # Trả về nhãn 'head_turned' khi quay đầu
        )
        self.stats = SessionStatistics(alert_threshold_frames=int(fps * 2))  # 2 giây liên tục = 30 frame

    # ============================================================
    # Calibration state
    # ============================================================

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.is_ready

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    @property
    def calibration_progress(self) -> float:
        return self.calibrator.progress

    def start_calibration(self) -> None:
        """Bắt đầu thu 30 mẫu Personal Baseline mới."""
        self.calibrator.reset()
        self._is_calibrating = True

    def reset_calibration(self) -> None:
        """Xóa baseline và dừng calibration."""
        self.calibrator.reset()
        self._is_calibrating = False

    def reset_session(self) -> None:
        """Đặt lại số liệu thống kê phiên làm việc."""
        self.temporal_monitor.reset()
        self.stats.reset()

    # ============================================================
    # Main inference flow
    # ============================================================

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        pose = self.detector.detect(frame_bgr)

        # 1. Không tìm thấy người
        if pose is None:
            self.temporal_monitor.update("no_person")
            self.stats.update("no_person", status="NO_PERSON")
            return self._build_no_person_result()

        keypoints = self._extract_keypoints(pose)
        bbox = self._extract_bbox(pose)
        raw_features = self.extractor.extract(pose)

        # 2. Keypoints không đủ độ tin cậy
        if raw_features is None:
            self.temporal_monitor.update("evaluating")
            self.stats.update("evaluating", status="LOW_CONFIDENCE")
            return self._build_low_confidence_result(bbox=bbox, keypoints=keypoints)

        raw_features = self._validate_raw_features(raw_features)
        angles = self._extract_angles(raw_features)

        # Tính proxy quay đầu (face_rotation_proxy): dist(left_eye, nose) / dist(right_eye, nose)
        face_rotation_proxy = self._calculate_face_rotation_proxy(pose)

        # 3. Đang thu 30 mẫu mốc chuẩn Baseline
        if self.is_calibrating:
            return self._process_calibration_frame(
                raw_features=raw_features,
                bbox=bbox,
                keypoints=keypoints,
                angles=angles,
            )

        # 4. Chưa hiệu chuẩn Baseline
        if not self.is_calibrated:
            self.stats.update(None, status="CALIBRATION_REQUIRED")
            return self._build_calibration_required_result(
                bbox=bbox,
                keypoints=keypoints,
                angles=angles,
            )

        # 5. Đã có baseline: RAW12 -> REP13 -> Predict
        baseline = self.calibrator.baseline
        if baseline is None:
            raise RuntimeError("Baseline RAW12 không tồn tại dù đã sẵn sàng.")

        rep_features = self.representation_builder.transform(raw_features, baseline)
        prediction = self.predictor.predict(rep_features)
        raw_label = prediction["raw_label"]

        # 6. Làm mượt bằng Temporal Smoothing + Yaw Gating
        smoothed_label = self.temporal_monitor.update(
            raw_prediction=raw_label,
            face_rotation_proxy=face_rotation_proxy,
        )

        # 7. Thống kê phiên & Kích hoạt chuông còi cảnh báo
        trigger_alert = self.stats.update(
            label=smoothed_label,
            status="OK",
            alert_threshold_frames=self.stats.alert_threshold_frames,
            debounce_seconds=4.0,
        )

        return {
            "has_person": True,
            "status": "OK",
            "message": "Phát hiện tư thế thành công",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": raw_label,
            "smoothed_label": smoothed_label,
            "label_id": prediction["label_id"],
            "probability": prediction["probability"],
            "trigger_alert": trigger_alert,
            "calibrated": True,
            "calibration_progress": 1.0,
            "stats": self.stats.to_dict(),
        }

    # ============================================================
    # Helpers
    # ============================================================

    def _process_calibration_frame(
        self,
        raw_features: np.ndarray,
        bbox: list[int] | None,
        keypoints: dict[str, list[float]],
        angles: dict[str, float],
    ) -> dict[str, Any]:
        accepted = self.calibrator.add_sample(raw_features)

        if not accepted:
            return {
                "has_person": True,
                "status": "CALIBRATION_SAMPLE_REJECTED",
                "message": "Mẫu calibration không hợp lệ",
                "bbox": bbox,
                "keypoints": keypoints,
                "angles": angles,
                "raw_label": None,
                "smoothed_label": None,
                "probability": None,
                "trigger_alert": False,
                "calibrated": self.is_calibrated,
                "calibration_progress": self.calibration_progress,
                "stats": self.stats.to_dict(),
            }

        if self.calibrator.is_ready:
            self._is_calibrating = False
            return {
                "has_person": True,
                "status": "CALIBRATED",
                "message": "Calibration hoàn thành",
                "bbox": bbox,
                "keypoints": keypoints,
                "angles": angles,
                "raw_label": None,
                "smoothed_label": None,
                "probability": None,
                "trigger_alert": False,
                "calibrated": True,
                "calibration_progress": 1.0,
                "stats": self.stats.to_dict(),
            }

        return {
            "has_person": True,
            "status": "CALIBRATING",
            "message": f"Đang hiệu chuẩn: {self.calibrator.sample_count}/{self.calibration_samples}",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": None,
            "smoothed_label": None,
            "probability": None,
            "trigger_alert": False,
            "calibrated": False,
            "calibration_progress": self.calibration_progress,
            "stats": self.stats.to_dict(),
        }

    @staticmethod
    def _calculate_face_rotation_proxy(pose: dict[str, Any]) -> Optional[float]:
        try:
            kps = pose.get("keypoints", {})
            nose = np.asarray(kps["nose"][:2], dtype=float)
            left_eye = np.asarray(kps["left_eye"][:2], dtype=float)
            right_eye = np.asarray(kps["right_eye"][:2], dtype=float)

            dist_left = float(np.linalg.norm(nose - left_eye))
            dist_right = float(np.linalg.norm(nose - right_eye))

            if dist_right <= 1e-5:
                return None
            return dist_left / dist_right
        except Exception:
            return None

    def _validate_raw_features(self, features: np.ndarray | list[float]) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 1 or array.shape != (self.RAW_FEATURE_COUNT,) or not np.isfinite(array).all():
            raise ValueError(f"RAW features không hợp lệ: shape {array.shape}")
        return array

    @staticmethod
    def _extract_keypoints(pose: dict[str, Any]) -> dict[str, list[float]]:
        output: dict[str, list[float]] = {}
        for name, value in pose.get("keypoints", {}).items():
            array = np.asarray(value, dtype=float).reshape(-1)
            if len(array) >= 2 and np.isfinite(array[:2]).all():
                conf = float(array[2]) if len(array) >= 3 else 1.0
                output[name] = [round(float(array[0]), 1), round(float(array[1]), 1), round(conf, 2)]
        return output

    @staticmethod
    def _extract_bbox(pose: dict[str, Any]) -> list[int] | None:
        bbox = pose.get("bbox")
        if bbox is None:
            return None
        array = np.asarray(bbox, dtype=float).reshape(-1)
        if len(array) >= 4 and np.isfinite(array[:4]).all():
            return [int(array[0]), int(array[1]), int(array[2]), int(array[3])]
        return None

    @staticmethod
    def _extract_angles(raw_features: np.ndarray) -> dict[str, float]:
        """
        Quy đổi các góc RAW12 sang chuẩn hiển thị công thái học trên giao diện:
        - Vai: Đưa về quanh mốc 0° (thay vì +-180° do ngược chiều trục X).
        - Đầu - Trọng lực: Quy đổi sang độ lệch góc so với phương thẳng đứng (0° - 45°).
        - Đầu - Thân: Giữ nguyên độ lệch quanh mốc 0°.
        """
        feature_map = dict(zip(FeatureExtractor.FEATURE_NAMES, raw_features))

        raw_sh = float(feature_map.get("shoulder_roll_deg", 0.0))
        raw_neck = float(feature_map.get("neck_pitch_deg", 0.0))
        raw_diff = float(feature_map.get("head_shoulder_roll_diff_deg", 0.0))

        # 1. Chuẩn hóa góc vai về quanh mốc 0°:
        # Trong FeatureExtractor: arctan2(left -> right) dao động quanh +-180°
        sh = raw_sh - 180.0 if raw_sh > 0 else raw_sh + 180.0

        # 2. Chuẩn hóa góc gập cổ (neck pitch) sang góc so với trọng lực (0° - 45°):
        # raw_neck tính so với phương ngang ảnh (~150° - 165° lúc ngồi thẳng)
        neck = abs(180.0 - raw_neck)

        # 3. Chuẩn hóa góc lệch đầu - thân về [-90°, 90°]:
        diff = (raw_diff + 180.0) % 360.0 - 180.0

        sh_clean = round(sh, 1)
        neck_clean = round(neck, 1)
        diff_clean = round(diff, 1)

        return {
            # Key theo giao diện Frontend (index.html)
            "shoulder_angle": sh_clean,
            "head_gravity_angle": neck_clean,
            "head_body_angle": diff_clean,

            # Key theo tên biến V03
            "shoulder_roll_deg": sh_clean,
            "neck_pitch_deg": neck_clean,
            "head_shoulder_roll_diff_deg": diff_clean,
        }

    def _build_no_person_result(self) -> dict[str, Any]:
        return {
            "has_person": False,
            "status": "NO_PERSON",
            "message": "Không tìm thấy người trong camera",
            "bbox": None,
            "keypoints": {},
            "angles": {},
            "raw_label": "no_person",
            "smoothed_label": "no_person",
            "label_id": None,
            "probability": None,
            "trigger_alert": False,
            "calibrated": self.is_calibrated,
            "calibration_progress": self.calibration_progress,
            "stats": self.stats.to_dict(),
        }

    def _build_low_confidence_result(self, bbox: list[int] | None, keypoints: dict[str, list[float]]) -> dict[str, Any]:
        return {
            "has_person": True,
            "status": "LOW_CONFIDENCE",
            "message": "Keypoint bị che khuất hoặc không rõ",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": {},
            "raw_label": None,
            "smoothed_label": None,
            "probability": None,
            "trigger_alert": False,
            "calibrated": self.is_calibrated,
            "calibration_progress": self.calibration_progress,
            "stats": self.stats.to_dict(),
        }

    def _build_calibration_required_result(
        self,
        bbox: list[int] | None,
        keypoints: dict[str, list[float]],
        angles: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "has_person": True,
            "status": "CALIBRATION_REQUIRED",
            "message": "Cần calibration trước khi dự đoán tư thế",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": None,
            "smoothed_label": None,
            "probability": None,
            "trigger_alert": False,
            "calibrated": False,
            "calibration_progress": self.calibration_progress,
            "stats": self.stats.to_dict(),
        }