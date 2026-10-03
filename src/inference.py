from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.feature_extractor import FeatureExtractor
from src.personal_calibration import PersonalCalibration
from src.pose_detector import PoseDetector
from src.posture_predictor import PosturePredictor
from src.representation_builder import RepresentationBuilder


class PostureInferenceEngine:
    """
    Điều phối pipeline inference cho Smart Posture Monitor V03.

    Luồng:
        Frame
        -> PoseDetector
        -> FeatureExtractor
        -> RAW12
        -> PersonalCalibration
        -> RepresentationBuilder
        -> REP13
        -> PosturePredictor

    File này chỉ xử lý frame-level inference.
    Temporal smoothing và session statistics sẽ được xử lý ở bước sau.
    """

    RAW_FEATURE_COUNT = PersonalCalibration.FEATURE_DIM
    REP_FEATURE_COUNT = len(RepresentationBuilder.OUTPUT_FEATURE_NAMES)

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        yolo_path: Optional[Path] = None,
        calibration_samples: int = 30,
        device: Optional[str] = None,
    ) -> None:
        if calibration_samples <= 0:
            raise ValueError(
                "calibration_samples phải lớn hơn 0."
            )

        # FeatureExtractor và PersonalCalibration phải cùng contract RAW12.
        if len(FeatureExtractor.FEATURE_NAMES) != self.RAW_FEATURE_COUNT:
            raise RuntimeError(
                "FeatureExtractor không khớp PersonalCalibration: "
                f"{len(FeatureExtractor.FEATURE_NAMES)} != "
                f"{self.RAW_FEATURE_COUNT}"
            )

        self.detector = PoseDetector(
            model_path=yolo_path,
            person_conf_threshold=0.45,
            device=device,
        )

        self.extractor = FeatureExtractor(
            min_keypoint_confidence=0.35,
        )

        self.calibrator = PersonalCalibration(
            target_samples=calibration_samples,
        )
        self.representation_builder = RepresentationBuilder()

        self.predictor = PosturePredictor(
            model_path=model_path,
            metadata_path=metadata_path,
        )

        self.calibration_samples = calibration_samples

        # Chỉ thu baseline khi user chủ động yêu cầu.
        self._is_calibrating = False

    # ============================================================
    # Calibration state
    # ============================================================

    @property
    def is_calibrated(self) -> bool:
        """Cho biết Personal Baseline đã được tạo hay chưa."""
        return self.calibrator.is_ready

    @property
    def is_calibrating(self) -> bool:
        """Cho biết engine có đang thu calibration samples hay không."""
        return self._is_calibrating

    @property
    def calibration_progress(self) -> float:
        """Tiến trình calibration trong khoảng [0.0, 1.0]."""
        return self.calibrator.progress

    def start_calibration(self) -> None:
        """
        Bắt đầu một phiên calibration mới.

        Baseline cũ sẽ bị xóa.
        Người dùng cần ngồi ở tư thế correct tự nhiên.
        """
        self.calibrator.reset()
        self._is_calibrating = True

    def reset_calibration(self) -> None:
        """Xóa baseline hiện tại và dừng quá trình calibration."""
        self.calibrator.reset()
        self._is_calibrating = False

    # ============================================================
    # Main inference flow
    # ============================================================

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """
        Xử lý một frame webcam.

        Khi calibration:
            Frame -> RAW12 -> lưu calibration sample.

        Khi baseline đã sẵn sàng:
            Frame -> RAW12 + baseline -> REP13 -> prediction.
        """
        pose = self.detector.detect(frame_bgr)

        if pose is None:
            return self._build_no_person_result()

        keypoints = self._extract_keypoints(pose)
        bbox = self._extract_bbox(pose)

        raw_features = self.extractor.extract(pose)

        if raw_features is None:
            return self._build_low_confidence_result(
                bbox=bbox,
                keypoints=keypoints,
            )

        raw_features = self._validate_raw_features(
            raw_features
        )

        angles = self._extract_angles(
            raw_features
        )

        # Đang thu Personal Baseline.
        if self.is_calibrating:
            return self._process_calibration_frame(
                raw_features=raw_features,
                bbox=bbox,
                keypoints=keypoints,
                angles=angles,
            )

        # Chưa có baseline thì không được đưa RAW trực tiếp vào model.
        if not self.is_calibrated:
            return self._build_calibration_required_result(
                bbox=bbox,
                keypoints=keypoints,
                angles=angles,
            )

        baseline = self.calibrator.baseline
        if baseline is None:
            raise RuntimeError(
                "Calibration đã sẵn sàng nhưng không có baseline RAW12."
            )

        rep_features = self.representation_builder.transform(
            raw_features,
            baseline,
        )
        if rep_features.shape != (self.REP_FEATURE_COUNT,):
            raise RuntimeError(f"REP13 không đúng shape: {rep_features.shape}.")

        prediction = self.predictor.predict(
            rep_features
        )

        return {
            "has_person": True,
            "status": "OK",
            "message": "Phát hiện tư thế thành công",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": prediction["raw_label"],
            "label_id": prediction["label_id"],
            "probability": prediction["probability"],
            "calibrated": True,
            "calibration_progress": 1.0,
        }

    # ============================================================
    # Calibration
    # ============================================================

    def _process_calibration_frame(
        self,
        raw_features: np.ndarray,
        bbox: list[int] | None,
        keypoints: dict[str, list[float]],
        angles: dict[str, float],
    ) -> dict[str, Any]:
        """
        Thêm một RAW feature vector vào calibration buffer.

        PersonalCalibration.add_sample() sẽ tự tính baseline
        ngay khi đủ target_samples.
        """
        accepted = self.calibrator.add_sample(
            raw_features
        )

        if not accepted:
            return {
                "has_person": True,
                "status": "CALIBRATION_SAMPLE_REJECTED",
                "message": "Mẫu calibration không hợp lệ",
                "bbox": bbox,
                "keypoints": keypoints,
                "angles": angles,
                "raw_label": None,
                "label_id": None,
                "probability": None,
                "calibrated": self.is_calibrated,
                "calibration_progress": self.calibration_progress,
            }

        # add_sample() tự động compute baseline ở sample cuối.
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
                "label_id": None,
                "probability": None,
                "calibrated": True,
                "calibration_progress": 1.0,
            }

        return {
            "has_person": True,
            "status": "CALIBRATING",
            "message": (
                f"Đang calibration: "
                f"{self.calibrator.sample_count}/"
                f"{self.calibration_samples}"
            ),
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": None,
            "label_id": None,
            "probability": None,
            "calibrated": False,
            "calibration_progress": self.calibration_progress,
        }

    # ============================================================
    # Feature validation
    # ============================================================

    def _validate_raw_features(
        self,
        features: np.ndarray | list[float],
    ) -> np.ndarray:
        """Kiểm tra RAW feature vector trước khi calibration."""
        try:
            array = np.asarray(
                features,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "RAW features phải chứa dữ liệu dạng số."
            ) from error

        if array.ndim != 1:
            raise ValueError(
                "RAW features phải là vector một chiều."
            )

        if array.shape != (self.RAW_FEATURE_COUNT,):
            raise ValueError(
                f"RAW features phải có shape "
                f"({self.RAW_FEATURE_COUNT},), "
                f"nhưng nhận được {array.shape}."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "RAW features chứa NaN hoặc Infinity."
            )

        return array

    # ============================================================
    # Pose output helpers
    # ============================================================

    @staticmethod
    def _extract_keypoints(
        pose: dict[str, Any],
    ) -> dict[str, list[float]]:
        """Chuẩn hóa keypoints để trả cho frontend."""
        output: dict[str, list[float]] = {}

        for name, value in pose.get(
            "keypoints",
            {},
        ).items():
            try:
                array = np.asarray(
                    value,
                    dtype=float,
                ).reshape(-1)
            except (TypeError, ValueError):
                continue

            if (
                len(array) < 2
                or not np.isfinite(array[:2]).all()
            ):
                continue

            confidence = (
                float(array[2])
                if len(array) >= 3
                else 1.0
            )

            output[name] = [
                round(float(array[0]), 1),
                round(float(array[1]), 1),
                round(confidence, 2),
            ]

        return output

    @staticmethod
    def _extract_bbox(
        pose: dict[str, Any],
    ) -> list[int] | None:
        """Chuẩn hóa bounding box để trả cho frontend."""
        bbox = pose.get("bbox")

        if bbox is None:
            return None

        try:
            array = np.asarray(
                bbox,
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError):
            return None

        if (
            len(array) < 4
            or not np.isfinite(array[:4]).all()
        ):
            return None

        return [
            int(array[0]),
            int(array[1]),
            int(array[2]),
            int(array[3]),
        ]

    @staticmethod
    def _extract_angles(
        raw_features: np.ndarray,
    ) -> dict[str, float]:
        """
        Lấy một số góc RAW phục vụ hiển thị/debug.

        REP13 dùng cho model.
        RAW angle dùng để thể hiện tư thế vật lý hiện tại.
        """
        feature_map = dict(
            zip(
                FeatureExtractor.FEATURE_NAMES,
                raw_features,
            )
        )

        angle_names = tuple(FeatureExtractor.ANGLE_FEATURES)

        return {
            name: round(
                float(feature_map.get(name, 0.0)),
                1,
            )
            for name in angle_names
        }

    # ============================================================
    # Response builders
    # ============================================================

    def _build_no_person_result(
        self,
    ) -> dict[str, Any]:
        """
        Response khi YOLO không tìm thấy người.

        Không reset calibration vì mất người tạm thời
        không đồng nghĩa với việc baseline cũ bị mất.
        """
        return {
            "has_person": False,
            "status": "NO_PERSON",
            "message": "Không tìm thấy người trong camera",
            "bbox": None,
            "keypoints": {},
            "angles": {},
            "raw_label": "no_person",
            "label_id": None,
            "probability": None,
            "calibrated": self.is_calibrated,
            "calibration_progress": self.calibration_progress,
        }

    def _build_low_confidence_result(
        self,
        bbox: list[int] | None,
        keypoints: dict[str, list[float]],
    ) -> dict[str, Any]:
        """
        Response khi có pose nhưng keypoint không đủ chất lượng.

        Frame này không được tính vào calibration.
        """
        return {
            "has_person": True,
            "status": "LOW_CONFIDENCE",
            "message": "Keypoint bị che khuất hoặc không rõ",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": {},
            "raw_label": None,
            "label_id": None,
            "probability": None,
            "calibrated": self.is_calibrated,
            "calibration_progress": self.calibration_progress,
        }

    def _build_calibration_required_result(
        self,
        bbox: list[int] | None,
        keypoints: dict[str, list[float]],
        angles: dict[str, float],
    ) -> dict[str, Any]:
        """Response khi chưa có Personal Baseline."""
        return {
            "has_person": True,
            "status": "CALIBRATION_REQUIRED",
            "message": "Cần calibration trước khi dự đoán tư thế",
            "bbox": bbox,
            "keypoints": keypoints,
            "angles": angles,
            "raw_label": None,
            "label_id": None,
            "probability": None,
            "calibrated": False,
            "calibration_progress": self.calibration_progress,
        }
