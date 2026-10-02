"""
Feature Schema and Configuration for Personal Baseline Delta.

This module defines:
- The base 32 feature columns (V02 29 features + V03 3 Depth Proxy features).
- The 8 most sensitive features identified for Personal Baseline Delta.
- The 20 "clean" features (excluding 9 negative-importance noisy features).
- Feature schemes for hybrid representations:
  * 'selective' (40 features: 32 absolute + 8 delta)
  * 'clean_selective' (28 features: 20 clean absolute + 8 delta)
  * 'full' (64 features: 32 absolute + 32 delta)
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np


# ============================================================
# 1. Base 32 Features (matching src/feature_extractor.py)
# ============================================================

FEATURE_NAMES_32: List[str] = [
    # --- V02 features (1-29) ---
    "shoulder_angle",
    "eye_shoulder_angle",
    "eye_vertical_difference",
    "nose_x_body",
    "nose_y_body",
    "eye_center_x_body",
    "eye_center_y_body",
    "left_ear_x_body",
    "left_ear_y_body",
    "nose_eye_dx",
    "nose_eye_dy",
    "eye_width_ratio",
    "ear_eye_ratio",
    "nose_shoulder_center_distance",
    "eye_shoulder_center_distance",
    "nose_shoulder_asymmetry",
    "eye_shoulder_asymmetry",
    "nose_ear_ratio",
    "head_body_angle",
    "head_gravity_angle",
    "nose_gravity_angle",
    "face_pitch_angle",
    "eye_vertical_axis_offset",
    "nose_vertical_axis_offset",
    "head_mean_height",
    "head_height_spread",
    "nose_body_angle",
    "ear_body_angle",
    "head_axis_angle_spread",
    # --- V03 Depth Proxy features (30-32) ---
    "face_shoulder_scale_ratio",
    "ear_nose_depth_proxy",
    "face_rotation_proxy",
]

# Quick lookup dictionary for feature index in the 32D vector
FEATURE_INDEX_32: Dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES_32)}


# ============================================================
# 2. 8 Sensitive Delta Features (V03 Design Proposal Section II)
# ============================================================
# Selected because they are the most ambiguous between forward_slouch
# and lean_right, and benefit most from personal baseline subtraction.

SENSITIVE_DELTA_FEATURES_8: List[str] = [
    "nose_y_body",               # Head drop in body frame (primary forward slouch signal)
    "head_mean_height",          # Head center vertical displacement
    "nose_gravity_angle",        # Sagittal head inclination w.r.t gravity
    "head_gravity_angle",        # Head axis inclination w.r.t gravity
    "face_shoulder_scale_ratio", # Depth proxy D1 (face area / shoulder_width^2)
    "eye_width_ratio",           # Depth proxy D2 (apparent eye width vs shoulder width)
    "face_rotation_proxy",       # Yaw proxy D4 (left_eye-nose / right_eye-nose)
    "ear_nose_depth_proxy",      # Depth proxy D3 (ear-nose / ear-eye distance ratio)
]

# Indices of the 8 sensitive features inside the 32D vector
SENSITIVE_DELTA_INDICES_8: List[int] = [
    FEATURE_INDEX_32[feat] for feat in SENSITIVE_DELTA_FEATURES_8
]


# ============================================================
# 3. 20 Clean Features (excluding 9 negative importance features)
# ============================================================
# From permutation importance study (Docs 2026-09-28 & Config B/G-30):
# 9 Harmful/Noisy features removed:
# - left_ear_x_body (-0.0624)
# - eye_vertical_difference (-0.0533)
# - nose_ear_ratio (-0.0097)
# - nose_x_body (-0.0091)
# - eye_center_x_body (-0.0039)
# - ear_eye_ratio (-0.0032)
# - face_pitch_angle (-0.0018)
# - nose_shoulder_asymmetry (-0.0012)
# - head_height_spread (-0.0004)

HARMFUL_FEATURES_9: List[str] = [
    "left_ear_x_body",
    "eye_vertical_difference",
    "nose_ear_ratio",
    "nose_x_body",
    "eye_center_x_body",
    "ear_eye_ratio",
    "face_pitch_angle",
    "nose_shoulder_asymmetry",
    "head_height_spread",
]

CLEAN_FEATURES_20: List[str] = [
    f for f in FEATURE_NAMES_32[:29] if f not in HARMFUL_FEATURES_9
]  # 20 features from V02

CLEAN_INDICES_20: List[int] = [
    FEATURE_INDEX_32[feat] for feat in CLEAN_FEATURES_20
]


# ============================================================
# 4. Hybrid Feature Names Generation
# ============================================================

def get_hybrid_feature_names(scheme: str = "selective") -> List[str]:
    """
    Returns list of feature names for a chosen hybrid scheme.

    Supported schemes:
    - 'selective': 32 absolute + 8 delta = 40 features
    - 'clean_selective': 20 clean absolute + 8 delta = 28 features (Config G-30)
    - 'full': 32 absolute + 32 delta = 64 features
    - 'delta_only_8': 8 delta features only
    """
    if scheme == "selective":
        delta_names = [f"delta_{name}" for name in SENSITIVE_DELTA_FEATURES_8]
        return FEATURE_NAMES_32 + delta_names

    elif scheme == "clean_selective":
        delta_names = [f"delta_{name}" for name in SENSITIVE_DELTA_FEATURES_8]
        return CLEAN_FEATURES_20 + delta_names

    elif scheme == "full":
        delta_names = [f"delta_{name}" for name in FEATURE_NAMES_32]
        return FEATURE_NAMES_32 + delta_names

    elif scheme == "delta_only_8":
        return [f"delta_{name}" for name in SENSITIVE_DELTA_FEATURES_8]

    else:
        raise ValueError(
            f"Unknown scheme: '{scheme}'. Choose from: 'selective', 'clean_selective', 'full', 'delta_only_8'."
        )


HYBRID_FEATURE_NAMES_40 = get_hybrid_feature_names("selective")
CLEAN_HYBRID_FEATURES_28 = get_hybrid_feature_names("clean_selective")
FULL_HYBRID_FEATURE_NAMES_64 = get_hybrid_feature_names("full")


# ============================================================
# 5. Validation Utilities
# ============================================================

def validate_features_shape(features: np.ndarray, expected_dim: int = 32) -> bool:
    """Check if feature vector is 1D with expected dimension."""
    if not isinstance(features, np.ndarray):
        return False
    if features.ndim != 1 or len(features) != expected_dim:
        return False
    return bool(np.all(np.isfinite(features)))
