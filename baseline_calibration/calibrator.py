"""
Baseline Calibrator Module for Personal Baseline Delta.

This module implements the core logic for:
1. Online collection of N calibration frames (e.g., 30 frames ≈ 2-3s).
2. Physiological sanity checks to avoid calibrating while the user is slouching or crooked.
3. Stability check to verify the user sat reasonably still during calibration.
4. Computation of baseline feature vector: f_baseline = mean(calib_frames).
5. Dynamic delta feature computation: Δf = f_current - f_baseline.
6. Hybrid feature vector generation (Selective 40D, Clean 28D, or Full 64D).
7. Runtime drift detection (tracking changes in shoulder width / camera distance).
8. Serialization for user profile persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    FEATURE_INDEX_32,
    SENSITIVE_DELTA_FEATURES_8,
    SENSITIVE_DELTA_INDICES_8,
    CLEAN_INDICES_20,
    get_hybrid_feature_names,
)


class CalibrationState(str, Enum):
    """Lifecycle states of the BaselineCalibrator."""
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    FAILED = "failed"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class CalibrationConfig:
    """Configuration parameters for Baseline Calibration."""
    n_frames: int = 30                     # Target frames for baseline collection
    min_confidence: float = 0.35           # Keypoint min confidence
    feature_scheme: str = "selective"      # 'selective' (40D), 'clean_selective' (28D), 'full' (64D)
    
    # Sanity checks during calibration (Physiological safety bounds)
    max_abs_shoulder_angle: float = 10.0   # Shoulder tilt must be <= 10 deg
    max_abs_head_gravity: float = 15.0     # Head gravity angle must be <= 15 deg
    max_abs_eye_shoulder_angle: float = 12.0 # Eye-shoulder tilt <= 12 deg
    max_feature_std: float = 0.08          # Max allowed std deviation across calib window (movement check)

    # Drift detection during monitoring
    drift_scale_threshold: float = 0.25    # 25% change in shoulder_width triggers drift
    drift_patience_frames: int = 15        # Number of consecutive drift frames required


class BaselineCalibrator:
    """
    Manages personal baseline acquisition, delta extraction, and drift tracking.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.state: CalibrationState = CalibrationState.NOT_CALIBRATED
        
        # Buffer for collecting frames during calibration
        self._frame_buffer: List[np.ndarray] = []
        self._shoulder_widths_buffer: List[float] = []
        
        # Computed baseline results
        self.baseline_features: Optional[np.ndarray] = None  # shape (32,)
        self.baseline_shoulder_width: Optional[float] = None
        self.baseline_std: Optional[np.ndarray] = None
        
        # Drift tracking
        self._drift_counter: int = 0
        self.last_status_message: str = "Chưa hiệu chuẩn baseline."

    # =========================================================
    # Calibration Flow
    # =========================================================

    def start_calibration(self, n_frames: Optional[int] = None) -> None:
        """Initiate calibration sequence."""
        if n_frames is not None and n_frames > 0:
            self.config.n_frames = n_frames
            
        self._frame_buffer.clear()
        self._shoulder_widths_buffer.clear()
        self.baseline_features = None
        self.baseline_shoulder_width = None
        self.baseline_std = None
        self._drift_counter = 0
        self.state = CalibrationState.CALIBRATING
        self.last_status_message = f"Đang hiệu chuẩn... Vui lòng ngồi thẳng tự nhiên (0/{self.config.n_frames})."

    def cancel_calibration(self) -> None:
        """Cancel ongoing calibration."""
        self._frame_buffer.clear()
        self._shoulder_widths_buffer.clear()
        self.state = CalibrationState.NOT_CALIBRATED
        self.last_status_message = "Đã hủy hiệu chuẩn."

    def reset(self) -> None:
        """Reset calibrator to initial state."""
        self._frame_buffer.clear()
        self._shoulder_widths_buffer.clear()
        self.baseline_features = None
        self.baseline_shoulder_width = None
        self.baseline_std = None
        self._drift_counter = 0
        self.state = CalibrationState.NOT_CALIBRATED
        self.last_status_message = "Chưa hiệu chuẩn baseline."

    def is_calibrated(self) -> bool:
        """Return True if baseline is successfully calibrated and active."""
        return self.state in (CalibrationState.CALIBRATED, CalibrationState.DRIFT_DETECTED) and self.baseline_features is not None

    def is_calibrating(self) -> bool:
        """Return True if currently in progress of collecting calibration frames."""
        return self.state == CalibrationState.CALIBRATING

    def get_progress(self) -> float:
        """Returns calibration progress between 0.0 and 1.0."""
        if not self.is_calibrating():
            return 1.0 if self.is_calibrated() else 0.0
        return min(1.0, float(len(self._frame_buffer)) / float(self.config.n_frames))

    # =========================================================
    # Frame Validation and Addition
    # =========================================================

    def _validate_single_frame_sanity(self, features: np.ndarray) -> Tuple[bool, str]:
        """
        Verify that a single candidate frame represents a plausible 'correct' posture.
        Prevents calibrating when user is already slouched, leaning, or looking sideways.
        """
        shoulder_angle = abs(features[FEATURE_INDEX_32["shoulder_angle"]])
        head_gravity = abs(features[FEATURE_INDEX_32["head_gravity_angle"]])
        eye_shoulder = abs(features[FEATURE_INDEX_32["eye_shoulder_angle"]])

        if shoulder_angle > self.config.max_abs_shoulder_angle:
            return False, f"Vai bị lệch ({shoulder_angle:.1f}° > {self.config.max_abs_shoulder_angle}°). Hãy ngồi thăng bằng."

        if head_gravity > self.config.max_abs_head_gravity:
            return False, f"Đầu bị nghiêng/cúi ({head_gravity:.1f}° > {self.config.max_abs_head_gravity}°). Hãy giữ đầu thẳng."

        if eye_shoulder > self.config.max_abs_eye_shoulder_angle:
            return False, f"Trục mắt lệch vai ({eye_shoulder:.1f}° > {self.config.max_abs_eye_shoulder_angle}°)."

        return True, "Hợp lệ"

    def add_frame(
        self,
        features: np.ndarray,
        pose_result: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Add a frame during the calibration phase.
        
        Args:
            features: 32D numpy array from FeatureExtractor.
            pose_result: Optional dict containing keypoints and bbox for geometry validation.

        Returns:
            Tuple of (accepted: bool, status_message: str).
        """
        if self.state != CalibrationState.CALIBRATING:
            return False, "Calibrator không ở trạng thái CALIBRATING."

        if features is None or len(features) != 32 or not np.all(np.isfinite(features)):
            return False, "Frame không hợp lệ (thiếu keypoint hoặc NaN)."

        # 1. Sanity check for the candidate frame
        is_sane, reason = self._validate_single_frame_sanity(features)
        if not is_sane:
            return False, f"Frame không đạt tiêu chuẩn: {reason}"

        # 2. Extract shoulder width if available in pose_result
        if pose_result and "keypoints" in pose_result:
            kps = pose_result["keypoints"]
            if "left_shoulder" in kps and "right_shoulder" in kps:
                ls = np.array(kps["left_shoulder"][:2])
                rs = np.array(kps["right_shoulder"][:2])
                sw = float(np.linalg.norm(rs - ls))
                self._shoulder_widths_buffer.append(sw)

        # 3. Append to buffer
        self._frame_buffer.append(features.copy())
        count = len(self._frame_buffer)
        self.last_status_message = f"Đang hiệu chuẩn... ({count}/{self.config.n_frames})"

        # 4. Check if collection target reached
        if count >= self.config.n_frames:
            return self._finalize_calibration()

        return True, self.last_status_message

    def _finalize_calibration(self) -> Tuple[bool, str]:
        """
        Calculate baseline vector and perform stability verification.
        """
        buffer_arr = np.array(self._frame_buffer, dtype=np.float32)  # shape (N, 32)
        mean_vector = np.mean(buffer_arr, axis=0)
        std_vector = np.std(buffer_arr, axis=0)

        # Check stability on the sensitive features
        sensitive_stds = std_vector[SENSITIVE_DELTA_INDICES_8]
        max_std = float(np.max(sensitive_stds))

        if max_std > self.config.max_feature_std:
            self.state = CalibrationState.FAILED
            self.last_status_message = (
                f"Hiệu chuẩn thất bại: bạn cử động quá nhiều (std={max_std:.3f} > {self.config.max_feature_std}). "
                "Vui lòng ngồi yên và thử lại."
            )
            return False, self.last_status_message

        # Success: commit baseline
        self.baseline_features = mean_vector
        self.baseline_std = std_vector
        if self._shoulder_widths_buffer:
            self.baseline_shoulder_width = float(np.mean(self._shoulder_widths_buffer))
        else:
            self.baseline_shoulder_width = None

        self.state = CalibrationState.CALIBRATED
        self.last_status_message = "Hiệu chuẩn baseline thành công! Sẵn sàng theo dõi."
        return True, self.last_status_message

    # =========================================================
    # Delta and Hybrid Feature Computation
    # =========================================================

    def compute_deltas(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute delta vector: Δf = features - baseline_features.
        Returns:
            np.ndarray of shape (32,) or None if not calibrated.
        """
        if not self.is_calibrated() or self.baseline_features is None:
            return None
        return features - self.baseline_features

    def get_delta_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Returns a dictionary of delta values for the 8 sensitive features,
        useful for visualization, UI gauges, and logging.
        """
        deltas = self.compute_deltas(features)
        if deltas is None:
            return {}

        result = {}
        for name in SENSITIVE_DELTA_FEATURES_8:
            idx = FEATURE_INDEX_32[name]
            result[name] = float(deltas[idx])
        return result

    def compute_hybrid_features(
        self,
        features: np.ndarray,
        scheme: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate hybrid vector combining absolute features with personal baseline delta features.

        Schemes:
        - 'selective' (default): 32 absolute + 8 sensitive deltas = 40 features
        - 'clean_selective': 20 clean absolute + 8 sensitive deltas = 28 features (Config G-30)
        - 'full': 32 absolute + 32 deltas = 64 features
        - 'delta_only_8': 8 sensitive deltas only

        Returns:
            np.ndarray or None if uncalibrated / invalid input.
        """
        if features is None or len(features) != 32:
            return None

        if not self.is_calibrated() or self.baseline_features is None:
            # Fallback: cannot compute deltas without baseline
            return None

        active_scheme = scheme or self.config.feature_scheme
        deltas = features - self.baseline_features

        if active_scheme == "selective":
            sensitive_deltas = deltas[SENSITIVE_DELTA_INDICES_8]
            return np.concatenate([features, sensitive_deltas]).astype(np.float32)

        elif active_scheme == "clean_selective":
            clean_abs = features[CLEAN_INDICES_20]
            sensitive_deltas = deltas[SENSITIVE_DELTA_INDICES_8]
            return np.concatenate([clean_abs, sensitive_deltas]).astype(np.float32)

        elif active_scheme == "full":
            return np.concatenate([features, deltas]).astype(np.float32)

        elif active_scheme == "delta_only_8":
            return deltas[SENSITIVE_DELTA_INDICES_8].astype(np.float32)

        else:
            raise ValueError(f"Unknown feature scheme: '{active_scheme}'")

    # =========================================================
    # Runtime Drift Detection
    # =========================================================

    def check_drift(self, pose_result: Optional[dict]) -> bool:
        """
        Check if the user's distance/scale has significantly drifted from the baseline position
        (e.g., moved the chair, leaned far forward/back, or stood up).
        """
        if not self.is_calibrated() or self.baseline_shoulder_width is None:
            return False

        if not pose_result or "keypoints" not in pose_result:
            return False

        kps = pose_result["keypoints"]
        if "left_shoulder" not in kps or "right_shoulder" not in kps:
            return False

        ls = np.array(kps["left_shoulder"][:2])
        rs = np.array(kps["right_shoulder"][:2])
        curr_width = float(np.linalg.norm(rs - ls))

        rel_diff = abs(curr_width - self.baseline_shoulder_width) / max(self.baseline_shoulder_width, 1e-6)

        if rel_diff > self.config.drift_scale_threshold:
            self._drift_counter += 1
            if self._drift_counter >= self.config.drift_patience_frames:
                self.state = CalibrationState.DRIFT_DETECTED
                self.last_status_message = "Vị trí ngồi đã thay đổi lớn so với lúc hiệu chuẩn. Nên hiệu chuẩn lại."
                return True
        else:
            self._drift_counter = max(0, self._drift_counter - 1)
            if self.state == CalibrationState.DRIFT_DETECTED and self._drift_counter == 0:
                self.state = CalibrationState.CALIBRATED
                self.last_status_message = "Baseline đang hoạt động bình thường."

        return self.state == CalibrationState.DRIFT_DETECTED

    # =========================================================
    # HUD & Telemetry Info
    # =========================================================

    def get_hud_info(self, features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Produce a dictionary formatted for drawing onto a video HUD or Streamlit dashboard.
        """
        info: Dict[str, Any] = {
            "state": self.state.value,
            "is_calibrated": self.is_calibrated(),
            "is_calibrating": self.is_calibrating(),
            "progress": self.get_progress(),
            "frames_collected": len(self._frame_buffer),
            "target_frames": self.config.n_frames,
            "status_message": self.last_status_message,
            "active_deltas": {},
            "drift_detected": (self.state == CalibrationState.DRIFT_DETECTED),
        }

        if features is not None and self.is_calibrated():
            info["active_deltas"] = self.get_delta_dict(features)

        return info

    # =========================================================
    # Profile Serialization (Save / Load)
    # =========================================================

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save baseline profile to a JSON file."""
        if not self.is_calibrated() or self.baseline_features is None:
            raise ValueError("Không thể lưu khi baseline chưa được hiệu chuẩn.")

        data = {
            "config": asdict(self.config),
            "baseline_features": self.baseline_features.tolist(),
            "baseline_std": self.baseline_std.tolist() if self.baseline_std is not None else None,
            "baseline_shoulder_width": self.baseline_shoulder_width,
            "feature_names": FEATURE_NAMES_32,
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """Load baseline profile from a JSON file."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file baseline: {filepath}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.baseline_features = np.array(data["baseline_features"], dtype=np.float32)
        if data.get("baseline_std"):
            self.baseline_std = np.array(data["baseline_std"], dtype=np.float32)
        self.baseline_shoulder_width = data.get("baseline_shoulder_width")
        
        self.state = CalibrationState.CALIBRATED
        self.last_status_message = "Đã tải baseline cá nhân từ file thành công."
