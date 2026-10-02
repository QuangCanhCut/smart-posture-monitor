"""
Unit tests for BaselineCalibrator.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from baseline_calibration.calibrator import (
    BaselineCalibrator,
    CalibrationConfig,
    CalibrationState,
)
from baseline_calibration.feature_schema import (
    FEATURE_NAMES_32,
    FEATURE_INDEX_32,
    SENSITIVE_DELTA_FEATURES_8,
)


def create_mock_correct_features(offset: float = 0.0) -> np.ndarray:
    """Creates a synthetic 32D feature vector matching a proper sitting posture."""
    feats = np.zeros(32, dtype=np.float32)
    feats[FEATURE_INDEX_32["shoulder_angle"]] = 1.0 + offset
    feats[FEATURE_INDEX_32["head_gravity_angle"]] = -2.0 + offset
    feats[FEATURE_INDEX_32["eye_shoulder_angle"]] = 0.5 + offset
    feats[FEATURE_INDEX_32["nose_y_body"]] = 0.75 + offset
    feats[FEATURE_INDEX_32["head_mean_height"]] = 0.78 + offset
    feats[FEATURE_INDEX_32["face_shoulder_scale_ratio"]] = 0.012 + offset
    feats[FEATURE_INDEX_32["eye_width_ratio"]] = 0.23 + offset
    feats[FEATURE_INDEX_32["face_rotation_proxy"]] = 1.05 + offset
    feats[FEATURE_INDEX_32["ear_nose_depth_proxy"]] = 1.10 + offset
    return feats


def test_calibrator_initial_state():
    calibrator = BaselineCalibrator()
    assert calibrator.state == CalibrationState.NOT_CALIBRATED
    assert not calibrator.is_calibrated()
    assert not calibrator.is_calibrating()
    assert calibrator.get_progress() == 0.0


def test_calibrator_normal_lifecycle():
    config = CalibrationConfig(n_frames=10)
    calibrator = BaselineCalibrator(config=config)

    calibrator.start_calibration()
    assert calibrator.state == CalibrationState.CALIBRATING
    assert calibrator.is_calibrating()

    # Add 9 frames
    for i in range(9):
        feats = create_mock_correct_features(offset=0.001 * i)
        accepted, msg = calibrator.add_frame(feats)
        assert accepted
        assert calibrator.is_calibrating()
        assert not calibrator.is_calibrated()

    assert calibrator.get_progress() == 0.9

    # Add 10th frame -> completion
    feats_10 = create_mock_correct_features(offset=0.009)
    accepted, msg = calibrator.add_frame(feats_10)
    assert accepted
    assert calibrator.is_calibrated()
    assert calibrator.state == CalibrationState.CALIBRATED
    assert calibrator.get_progress() == 1.0
    assert calibrator.baseline_features is not None
    assert len(calibrator.baseline_features) == 32


def test_delta_computation():
    config = CalibrationConfig(n_frames=5)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()

    base_val = 0.75
    for _ in range(5):
        feats = create_mock_correct_features()
        feats[FEATURE_INDEX_32["nose_y_body"]] = base_val
        calibrator.add_frame(feats)

    assert calibrator.is_calibrated()

    # Current frame with slouch (nose dropped by 0.15)
    curr_feats = create_mock_correct_features()
    curr_feats[FEATURE_INDEX_32["nose_y_body"]] = base_val - 0.15

    deltas = calibrator.compute_deltas(curr_feats)
    assert deltas is not None
    assert np.isclose(deltas[FEATURE_INDEX_32["nose_y_body"]], -0.15, atol=1e-4)

    delta_dict = calibrator.get_delta_dict(curr_feats)
    assert "nose_y_body" in delta_dict
    assert np.isclose(delta_dict["nose_y_body"], -0.15, atol=1e-4)


def test_hybrid_feature_shapes():
    config = CalibrationConfig(n_frames=5)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()
    for _ in range(5):
        calibrator.add_frame(create_mock_correct_features())

    curr_feats = create_mock_correct_features()

    # Selective: 32 abs + 8 deltas = 40
    vec_40 = calibrator.compute_hybrid_features(curr_feats, scheme="selective")
    assert vec_40 is not None
    assert len(vec_40) == 40

    # Clean selective: 20 clean abs + 8 deltas = 28
    vec_28 = calibrator.compute_hybrid_features(curr_feats, scheme="clean_selective")
    assert vec_28 is not None
    assert len(vec_28) == 28

    # Full: 32 abs + 32 deltas = 64
    vec_64 = calibrator.compute_hybrid_features(curr_feats, scheme="full")
    assert vec_64 is not None
    assert len(vec_64) == 64

    # Delta only: 8
    vec_8 = calibrator.compute_hybrid_features(curr_feats, scheme="delta_only_8")
    assert vec_8 is not None
    assert len(vec_8) == 8


def test_sanity_check_rejection():
    config = CalibrationConfig(n_frames=5, max_abs_head_gravity=15.0)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()

    # Slouched frame (head_gravity = 25.0 deg)
    slouch_feats = create_mock_correct_features()
    slouch_feats[FEATURE_INDEX_32["head_gravity_angle"]] = 25.0

    accepted, msg = calibrator.add_frame(slouch_feats)
    assert not accepted
    assert "Đầu bị nghiêng/cúi" in msg
    assert calibrator.get_progress() == 0.0


def test_stability_check_failure():
    # If values oscillate violently during calibration, calibration should fail
    config = CalibrationConfig(n_frames=6, max_feature_std=0.05)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()

    for i in range(6):
        feats = create_mock_correct_features()
        # Alternate nose_y_body dramatically
        feats[FEATURE_INDEX_32["nose_y_body"]] = 0.50 if (i % 2 == 0) else 0.90
        calibrator.add_frame(feats)

    assert calibrator.state == CalibrationState.FAILED
    assert not calibrator.is_calibrated()
    assert "cử động quá nhiều" in calibrator.last_status_message


def test_drift_detection():
    config = CalibrationConfig(n_frames=5, drift_scale_threshold=0.20, drift_patience_frames=3)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()

    mock_pose = {
        "keypoints": {
            "left_shoulder": [100.0, 200.0, 0.9],
            "right_shoulder": [200.0, 200.0, 0.9],  # width = 100
        }
    }
    for _ in range(5):
        calibrator.add_frame(create_mock_correct_features(), mock_pose)

    assert calibrator.is_calibrated()
    assert calibrator.baseline_shoulder_width == 100.0

    # User steps back: shoulder width drops to 60px (40% change > 20%)
    drift_pose = {
        "keypoints": {
            "left_shoulder": [100.0, 200.0, 0.9],
            "right_shoulder": [160.0, 200.0, 0.9],  # width = 60
        }
    }

    # Frame 1 and 2 of drift
    calibrator.check_drift(drift_pose)
    calibrator.check_drift(drift_pose)
    assert calibrator.state == CalibrationState.CALIBRATED

    # Frame 3 triggers drift state
    calibrator.check_drift(drift_pose)
    assert calibrator.state == CalibrationState.DRIFT_DETECTED


def test_save_and_load(tmp_path: Path):
    config = CalibrationConfig(n_frames=5)
    calibrator = BaselineCalibrator(config=config)
    calibrator.start_calibration()
    for _ in range(5):
        calibrator.add_frame(create_mock_correct_features())

    save_file = tmp_path / "test_baseline.json"
    calibrator.save_to_file(save_file)
    assert save_file.is_file()

    # Load into fresh calibrator
    calibrator2 = BaselineCalibrator()
    calibrator2.load_from_file(save_file)
    assert calibrator2.is_calibrated()
    assert np.allclose(calibrator.baseline_features, calibrator2.baseline_features)
