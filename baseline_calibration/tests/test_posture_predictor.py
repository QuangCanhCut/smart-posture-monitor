"""
Unit tests for PosturePredictor.
"""

import numpy as np
import pytest

from baseline_calibration.posture_predictor import PosturePredictor, PosturePrediction
from baseline_calibration.calibrator import CalibrationConfig
from baseline_calibration.feature_schema import FEATURE_INDEX_32


def create_mock_pose_result():
    """Returns valid mock pose dict with 6 required keypoints."""
    return {
        "person_confidence": 0.95,
        "bbox": [100, 100, 200, 300],
        "keypoints": {
            "nose": [150.0, 120.0, 0.9],
            "left_eye": [140.0, 110.0, 0.9],
            "right_eye": [160.0, 110.0, 0.9],
            "left_ear": [125.0, 115.0, 0.9],
            "left_shoulder": [110.0, 200.0, 0.95],
            "right_shoulder": [190.0, 200.0, 0.95],
        },
    }


def test_predictor_initialization():
    predictor = PosturePredictor()
    assert predictor.model is not None
    assert predictor.calibrator is not None
    assert predictor.temporal_monitor is not None
    assert not predictor.is_calibrated()


def test_predict_frame_uncalibrated():
    predictor = PosturePredictor()
    mock_pose = create_mock_pose_result()

    pred = predictor.predict_frame(pose_result=mock_pose)
    assert pred is not None
    assert isinstance(pred, PosturePrediction)
    assert pred.raw_label in ["correct", "forward_slouch", "lean_left", "lean_right"]
    assert pred.smoothed_label in ["correct", "forward_slouch", "lean_left", "lean_right"]
    assert not pred.is_calibrated
    assert pred.calibration_state == "not_calibrated"


def test_calibration_via_predictor():
    cfg = CalibrationConfig(n_frames=5)
    predictor = PosturePredictor(calibrator_config=cfg)
    predictor.start_calibration(n_frames=5)

    assert predictor.is_calibrating()

    mock_pose = create_mock_pose_result()

    # Feed 4 frames
    for i in range(4):
        pred = predictor.predict_frame(pose_result=mock_pose)
        assert pred is not None
        assert pred.raw_label == "calibrating"
        assert predictor.is_calibrating()

    # 5th frame finishes calibration
    pred = predictor.predict_frame(pose_result=mock_pose)
    assert pred is not None
    assert predictor.is_calibrated()

    # Next frame performs regular inference with active deltas
    pred_calibrated = predictor.predict_frame(pose_result=mock_pose)
    assert pred_calibrated is not None
    assert pred_calibrated.is_calibrated
    assert "nose_y_body" in pred_calibrated.active_deltas


def test_delta_assisted_correction():
    predictor = PosturePredictor(enable_delta_correction=True)
    # Calibrate with standard sitting
    predictor.start_calibration(n_frames=3)
    mock_pose = create_mock_pose_result()
    for _ in range(3):
        predictor.predict_frame(pose_result=mock_pose)

    assert predictor.is_calibrated()

    # Create synthetic features where model might say lean_right,
    # but nose dropped significantly (-0.16) and shoulders are flat
    features = np.zeros(32, dtype=np.float32)
    # Fill baseline features
    features[:] = predictor.calibrator.baseline_features[:]
    features[FEATURE_INDEX_32["nose_y_body"]] -= 0.16
    features[FEATURE_INDEX_32["shoulder_angle"]] = 1.0  # nearly flat
    features[FEATURE_INDEX_32["face_shoulder_scale_ratio"]] += 0.003  # closer

    # Raw prediction = lean_right
    corrected = predictor._apply_delta_correction(
        raw_pred="lean_right",
        features=features,
        probabilities={"lean_right": 0.6, "forward_slouch": 0.4},
    )
    # Corrected should be forward_slouch!
    assert corrected == "forward_slouch"


def test_hybrid_model_loading_and_prediction():
    hybrid_model_file = PosturePredictor().project_root / "baseline_calibration" / "models" / "best_hybrid_model.joblib"
    if not hybrid_model_file.is_file():
        pytest.skip("Hybrid model artifact not yet generated.")

    predictor = PosturePredictor(model_path="hybrid")
    assert predictor.expected_n_features == 40

    mock_pose = create_mock_pose_result()
    # Calibrate
    predictor.start_calibration(n_frames=3)
    for _ in range(3):
        predictor.predict_frame(pose_result=mock_pose)

    assert predictor.is_calibrated()

    # Predict with hybrid 40D features
    pred = predictor.predict_frame(pose_result=mock_pose)
    assert pred is not None
    assert pred.is_calibrated
    assert pred.raw_label in ["correct", "forward_slouch", "lean_left", "lean_right"]
