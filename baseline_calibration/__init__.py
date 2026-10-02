"""
Personal Baseline Delta Module for Smart Posture Monitor.

This package provides an end-to-end implementation of Personal Baseline Calibration:
- BaselineCalibrator: Handles calibration frame collection, physiological sanity checks,
  stability verification, drift detection, and hybrid delta feature computation.
- PosturePredictor: Production-grade predictor combining FeatureExtractor, BaselineCalibrator,
  SVM model inference, delta-assisted boundary correction, and TemporalMonitor.
- Offline Dataset Builder: Simulates production calibration across dataset sessions to build
  hybrid training datasets.
- Offline Evaluator: Quantitative benchmark comparing baseline-free vs hybrid delta configurations.
"""

from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    FEATURE_INDEX_32,
    SENSITIVE_DELTA_FEATURES_8,
    SENSITIVE_DELTA_INDICES_8,
    CLEAN_FEATURES_20,
    CLEAN_INDICES_20,
    HYBRID_FEATURE_NAMES_40,
    CLEAN_HYBRID_FEATURES_28,
    FULL_HYBRID_FEATURE_NAMES_64,
    get_hybrid_feature_names,
)
from baseline_calibration.calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
)
from baseline_calibration.posture_predictor import (
    PosturePredictor,
    PosturePrediction,
)
from baseline_calibration.offline_dataset_builder import (
    compute_baseline_for_session,
    build_offline_baseline_dataset,
)

__all__ = [
    "FEATURE_NAMES_32",
    "FEATURE_INDEX_32",
    "SENSITIVE_DELTA_FEATURES_8",
    "SENSITIVE_DELTA_INDICES_8",
    "CLEAN_FEATURES_20",
    "CLEAN_INDICES_20",
    "HYBRID_FEATURE_NAMES_40",
    "CLEAN_HYBRID_FEATURES_28",
    "FULL_HYBRID_FEATURE_NAMES_64",
    "get_hybrid_feature_names",
    "BaselineCalibrator",
    "CalibrationConfig",
    "CalibrationState",
    "PosturePredictor",
    "PosturePrediction",
    "compute_baseline_for_session",
    "build_offline_baseline_dataset",
]
