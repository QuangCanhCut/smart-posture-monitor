"""
Posture Predictor Module with Integrated Baseline Calibration.

Fulfills the designed architecture from `docs/V03_feature_design_proposal.md`:
- Integrates PoseDetector output -> FeatureExtractor -> BaselineCalibrator -> Classifier -> TemporalMonitor.
- Seamlessly handles:
  1. Calibration lifecycle (collection, verification, readiness).
  2. Model inference with either standard 32D features or 40D/28D hybrid delta features.
  3. Delta-assisted boundary correction for ambiguous cases (FS <-> LR confusion).
  4. Temporal smoothing (hysteresis + majority vote) and Yaw gating.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

# Ensure project root is importable if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor
from src.temporal_monitor import TemporalMonitor, HEAD_TURNED_LABEL
from baseline_calibration.calibrator import BaselineCalibrator, CalibrationConfig, CalibrationState
from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    FEATURE_INDEX_32,
    SENSITIVE_DELTA_FEATURES_8,
)


@dataclass
class PosturePrediction:
    """Detailed output of a single frame inference."""
    raw_label: str
    smoothed_label: str
    probabilities: Dict[str, float]
    is_head_turned: bool
    is_calibrated: bool
    calibration_state: str
    calibration_progress: float
    active_deltas: Dict[str, float]
    status_message: str
    confidence: float = 0.0


class PosturePredictor:
    """
    Unified end-to-end predictor with Personal Baseline Calibration and Temporal Smoothing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        calibrator_config: Optional[CalibrationConfig] = None,
        temporal_monitor_kwargs: Optional[Dict[str, Any]] = None,
        enable_delta_correction: bool = True,
    ):
        """
        Initialize the PosturePredictor.

        Args:
            model_path: Path to best_model.joblib. If None, defaults to PROJECT_ROOT/models/best_model.joblib.
            metadata_path: Path to training_metadata.json.
            calibrator_config: Optional CalibrationConfig.
            temporal_monitor_kwargs: Optional kwargs for TemporalMonitor.
            enable_delta_correction: If True and model was trained on 32D features, uses active
                                    baseline delta rules to disambiguate FS vs LR.
        """
        self.project_root = PROJECT_ROOT
        
        # Paths
        if model_path == "hybrid":
            self.model_path = Path(__file__).resolve().parent / "models" / "best_hybrid_model.joblib"
            self.metadata_path = Path(__file__).resolve().parent / "models" / "hybrid_training_metadata.json"
        else:
            self.model_path = Path(model_path) if model_path else self.project_root / "models" / "best_model.joblib"
            self.metadata_path = Path(metadata_path) if metadata_path else self.project_root / "models" / "training_metadata.json"

        # Load artifacts
        self.model, self.metadata = self._load_artifacts()
        self.id_to_label = self._extract_id_to_label()
        self.feature_columns = self.metadata.get("feature_columns", FEATURE_NAMES_32)
        self.expected_n_features = len(self.feature_columns)

        # Components
        self.feature_extractor = FeatureExtractor(min_keypoint_confidence=0.35)
        self.calibrator = BaselineCalibrator(config=calibrator_config)
        
        # Temporal Monitor
        tm_kwargs = {
            "fps": 5,
            "window_seconds": 5.0,
            "hysteresis_frames": 3,
            "yaw_mode": "conservative",
        }
        if temporal_monitor_kwargs:
            tm_kwargs.update(temporal_monitor_kwargs)
        self.temporal_monitor = TemporalMonitor(**tm_kwargs)

        self.enable_delta_correction = enable_delta_correction

    # =========================================================
    # Artifact Loading
    # =========================================================

    def _load_artifacts(self) -> Tuple[Any, dict]:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        model = joblib.load(self.model_path)
        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, metadata

    def _extract_id_to_label(self) -> Dict[int, str]:
        raw = self.metadata.get("id_to_label")
        if isinstance(raw, dict) and raw:
            return {int(k): str(v) for k, v in raw.items()}
        raw = self.metadata.get("label_to_id")
        if isinstance(raw, dict) and raw:
            return {int(v): str(k) for k, v in raw.items()}
        return {0: "correct", 1: "forward_slouch", 2: "lean_left", 3: "lean_right"}

    # =========================================================
    # Calibration Controls
    # =========================================================

    def start_calibration(self, n_frames: int = 30) -> None:
        """Start baseline calibration sequence."""
        self.calibrator.start_calibration(n_frames=n_frames)

    def reset_baseline(self) -> None:
        """Reset calibration and clear temporal monitor history."""
        self.calibrator.reset()
        self.temporal_monitor.reset()

    def is_calibrated(self) -> bool:
        return self.calibrator.is_calibrated()

    def is_calibrating(self) -> bool:
        return self.calibrator.is_calibrating()

    # =========================================================
    # Delta-Assisted Boundary Correction
    # =========================================================

    def _apply_delta_correction(
        self,
        raw_pred: str,
        features: np.ndarray,
        probabilities: Dict[str, float]
    ) -> str:
        """
        When calibrated, use delta cues to prevent cross-subject confusion between
        forward_slouch and lean_right (the primary pain point in 45° camera perspective).
        """
        if not self.calibrator.is_calibrated():
            return raw_pred

        deltas = self.calibrator.get_delta_dict(features)
        if not deltas:
            return raw_pred

        delta_nose_y = deltas.get("nose_y_body", 0.0)
        delta_head_gravity = abs(deltas.get("head_gravity_angle", 0.0))
        delta_scale = deltas.get("face_shoulder_scale_ratio", 0.0)

        # Absolute features
        shoulder_angle = features[FEATURE_INDEX_32["shoulder_angle"]]
        abs_shoulder_tilt = abs(shoulder_angle)

        # Rule 1: Model predicted lean_right, but user actually has severe head drop without shoulder tilt -> Forward Slouch
        if raw_pred == "lean_right":
            # Significant nose drop, face size increased towards camera, and shoulder remains relatively flat
            if delta_nose_y < -0.12 and abs_shoulder_tilt < 6.0 and delta_scale > 0.001:
                return "forward_slouch"

        # Rule 2: Model predicted forward_slouch, but nose did not drop, shoulders are tilted -> Lean Right
        elif raw_pred == "forward_slouch":
            # Nose did not drop significantly from personal baseline, but shoulders are visibly tilted
            if delta_nose_y > -0.04 and abs_shoulder_tilt > 5.5:
                return "lean_right"

        # Rule 3: Model predicted error posture, but all deltas from personal baseline are virtually zero (< tolerance)
        elif raw_pred in ("forward_slouch", "lean_right"):
            if abs(delta_nose_y) < 0.03 and delta_head_gravity < 3.0 and abs_shoulder_tilt < 4.0:
                return "correct"

        return raw_pred

    # =========================================================
    # Prediction Pipeline
    # =========================================================

    def predict_frame(
        self,
        pose_result: Optional[dict],
        features: Optional[np.ndarray] = None
    ) -> Optional[PosturePrediction]:
        """
        Execute prediction pipeline for a single frame.

        Args:
            pose_result: Output from PoseDetector containing keypoints.
            features: Pre-computed 32D features (optional; if None, computed from pose_result).

        Returns:
            PosturePrediction dataclass, or None if frame is invalid / unprocessable.
        """
        # 1. Feature extraction if not provided
        if features is None:
            if pose_result is None:
                return None
            features = self.feature_extractor.extract(pose_result)

        if features is None or len(features) != 32 or not np.all(np.isfinite(features)):
            return None

        # 2. Check drift if calibrated
        self.calibrator.check_drift(pose_result)

        # 3. Handle calibration collection state
        if self.calibrator.is_calibrating():
            accepted, msg = self.calibrator.add_frame(features, pose_result)
            return PosturePrediction(
                raw_label="calibrating",
                smoothed_label="calibrating",
                probabilities={"calibrating": 1.0},
                is_head_turned=False,
                is_calibrated=False,
                calibration_state=self.calibrator.state.value,
                calibration_progress=self.calibrator.get_progress(),
                active_deltas={},
                status_message=msg,
                confidence=1.0,
            )

        # 4. Prepare feature vector for model
        import pandas as pd
        if self.expected_n_features == 32:
            model_input = pd.DataFrame([features], columns=self.feature_columns)
        elif self.expected_n_features == 40 and self.calibrator.is_calibrated():
            hybrid_features = self.calibrator.compute_hybrid_features(features, scheme="selective")
            curr_feats = hybrid_features if hybrid_features is not None else features
            model_input = pd.DataFrame([curr_feats], columns=self.feature_columns)
        elif self.expected_n_features == 28 and self.calibrator.is_calibrated():
            hybrid_features = self.calibrator.compute_hybrid_features(features, scheme="clean_selective")
            curr_feats = hybrid_features if hybrid_features is not None else features
            model_input = pd.DataFrame([curr_feats], columns=self.feature_columns)
        else:
            model_input = pd.DataFrame([features], columns=self.feature_columns[:len(features)])

        # 5. Model Inference
        probs_raw = None
        if hasattr(self.model, "predict_proba"):
            probs_raw = self.model.predict_proba(model_input)[0]
            pred_id = int(np.argmax(probs_raw))
            max_prob = float(probs_raw[pred_id])
        else:
            pred_id = int(self.model.predict(model_input)[0])
            max_prob = 1.0

        raw_label = self.id_to_label.get(pred_id, "correct")

        # Map probabilities dict
        prob_dict: Dict[str, float] = {}
        if probs_raw is not None:
            for cls_idx, p_val in enumerate(probs_raw):
                label_name = self.id_to_label.get(cls_idx, str(cls_idx))
                prob_dict[label_name] = float(p_val)
        else:
            prob_dict[raw_label] = 1.0

        # 6. Apply delta correction if active
        if self.enable_delta_correction and self.calibrator.is_calibrated():
            corrected_label = self._apply_delta_correction(raw_label, features, prob_dict)
        else:
            corrected_label = raw_label

        # 7. Temporal Smoothing & Yaw Gating
        face_rotation_proxy = float(features[FEATURE_INDEX_32["face_rotation_proxy"]])
        smoothed_label = self.temporal_monitor.update(
            raw_prediction=corrected_label,
            face_rotation_proxy=face_rotation_proxy,
        )

        monitor_stats = self.temporal_monitor.get_stats()
        is_head_turned = monitor_stats["is_head_turned"]

        # 8. Active deltas for HUD
        active_deltas = self.calibrator.get_delta_dict(features) if self.calibrator.is_calibrated() else {}

        return PosturePrediction(
            raw_label=raw_label,
            smoothed_label=smoothed_label,
            probabilities=prob_dict,
            is_head_turned=is_head_turned,
            is_calibrated=self.calibrator.is_calibrated(),
            calibration_state=self.calibrator.state.value,
            calibration_progress=self.calibrator.get_progress(),
            active_deltas=active_deltas,
            status_message=self.calibrator.last_status_message,
            confidence=max_prob,
        )
