from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.feature_extractor import FeatureExtractor


class RepresentationBuilder:
    """
    RAW12 current + baseline RAW12 -> REP13.

    Stateless: không lưu baseline/person/session, không đọc dataset,
    không train model. Training và realtime dùng chung công thức này.
    """

    OUTPUT_FEATURE_NAMES = (
        "delta_shoulder_roll_deg",
        "delta_eye_roll_deg",
        "delta_head_shoulder_roll_diff_deg",
        "delta_neck_pitch_deg",
        "delta_eye_center_x_body",
        "delta_eye_center_y_body",
        "delta_shoulder_center_x_body",
        "delta_eye_shoulder_vertical_gap_body",
        "delta_eye_shoulder_horizontal_offset_body",
        "log_ratio_inter_eye_distance",
        "log_ratio_ear_nose_horizontal_span",
        "log_ratio_shoulder_width",
        "head_drift_magnitude",
    )

    INPUT_DIM = len(FeatureExtractor.FEATURE_NAMES)
    SHOULDER_WIDTH_INDEX = FeatureExtractor.FEATURE_NAMES.index("shoulder_width_px")

    # Lấy index từ FeatureExtractor để tránh hard-code sai schema RAW12.
    ANGLE_INDICES = tuple(
        FeatureExtractor.FEATURE_NAMES.index(name)
        for name in FeatureExtractor.ANGLE_FEATURES
    )
    SPATIAL_INDICES = tuple(
        FeatureExtractor.FEATURE_NAMES.index(name)
        for name in FeatureExtractor.SPATIAL_FEATURES
    )
    SCALE_INDICES = tuple(
        FeatureExtractor.FEATURE_NAMES.index(name)
        for name in FeatureExtractor.SCALE_FEATURES
    )

    def __init__(self, eps: float = 1e-6) -> None:
        if eps <= 0:
            raise ValueError("eps phải lớn hơn 0.")
        self.eps = float(eps)

    @property
    def output_feature_count(self) -> int:
        return len(self.OUTPUT_FEATURE_NAMES)

    def transform(
        self,
        current_raw: Sequence[float] | np.ndarray,
        baseline_raw: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Biến đổi 1 frame.
        Input : current_raw (12,), baseline_raw (12,)
        Output: REP13 float32 (13,)
        """
        current = self._validate_vector(current_raw, "current_raw")
        baseline = self._validate_vector(baseline_raw, "baseline_raw")

        # Dùng chung core với batch để train và realtime không lệch công thức.
        rep = self._transform_array(current[None, :], baseline)[0]
        return rep.astype(np.float32, copy=False)

    def transform_batch(
        self,
        raw_samples: Sequence[Sequence[float]] | np.ndarray,
        baseline_raw: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Biến đổi nhiều RAW12 cùng baseline của một person/session.
        Input : raw_samples (N, 12), baseline_raw (12,)
        Output: REP13 matrix (N, 13)
        """
        samples = self._validate_batch(raw_samples)
        baseline = self._validate_vector(baseline_raw, "baseline_raw")

        return self._transform_array(samples, baseline).astype(
            np.float32, copy=False
        )

    def as_dict(
        self,
        representation: Sequence[float] | np.ndarray,
    ) -> dict[str, float]:
        """Đổi REP13 sang dict để debug/log dễ đọc."""
        vector = np.asarray(representation, dtype=np.float64).reshape(-1)

        if vector.shape != (self.output_feature_count,):
            raise ValueError(
                f"REP phải có shape ({self.output_feature_count},), "
                f"nhận được {vector.shape}."
            )

        return dict(zip(self.OUTPUT_FEATURE_NAMES, vector.tolist()))

    # ==========================================================
    # CORE: RAW12 + BASELINE -> REP13
    # ==========================================================

    def _transform_array(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        baseline_width = baseline[self.SHOULDER_WIDTH_INDEX]

        if baseline_width <= self.eps:
            raise ValueError("baseline shoulder_width phải lớn hơn 0.")

        angle_idx = np.asarray(self.ANGLE_INDICES)
        spatial_idx = np.asarray(self.SPATIAL_INDICES)
        scale_idx = np.asarray(self.SCALE_INDICES)

        # REP01-04: góc đã scale-invariant -> delta có wrap.
        angle_delta = self._wrap_angle_deg(
            current[:, angle_idx] - baseline[angle_idx]
        )

        # REP05-09: chuẩn hóa bằng shoulder width CỐ ĐỊNH của baseline.
        spatial_delta = (
            current[:, spatial_idx] - baseline[spatial_idx]
        ) / baseline_width

        # REP10-12: scale/depth dùng log(current / baseline).
        current_scale = current[:, scale_idx]
        baseline_scale = baseline[scale_idx]

        if np.any(current_scale <= self.eps):
            raise ValueError("Scale feature hiện tại phải lớn hơn 0.")
        if np.any(baseline_scale <= self.eps):
            raise ValueError("Scale feature baseline phải lớn hơn 0.")

        scale_delta = np.log(current_scale / baseline_scale)

        # REP13: độ dịch chuyển tổng của tâm đầu.
        # spatial_delta[:, 0:2] = delta eye_center_x, delta eye_center_y.
        head_drift = np.hypot(
            spatial_delta[:, 0], spatial_delta[:, 1]
        )[:, None]

        rep = np.concatenate(
            (angle_delta, spatial_delta, scale_delta, head_drift),
            axis=1,
        )

        if rep.shape[1] != self.output_feature_count:
            raise RuntimeError(
                f"REP phải có {self.output_feature_count} features, "
                f"nhận được {rep.shape[1]}."
            )
        if not np.all(np.isfinite(rep)):
            raise ValueError("REP chứa NaN hoặc Infinity.")

        return rep

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate_vector(
        cls,
        vector: Sequence[float] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        try:
            array = np.asarray(vector, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} phải chứa dữ liệu số.") from exc

        if array.shape != (cls.INPUT_DIM,):
            raise ValueError(
                f"{name} phải có shape ({cls.INPUT_DIM},), "
                f"nhận được {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} chứa NaN hoặc Infinity.")

        return array

    @classmethod
    def _validate_batch(
        cls,
        samples: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        try:
            batch = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw_samples phải chứa dữ liệu số.") from exc

        if batch.ndim != 2 or batch.shape[1] != cls.INPUT_DIM:
            raise ValueError(
                f"raw_samples phải có shape (N, {cls.INPUT_DIM}), "
                f"nhận được {batch.shape}."
            )
        if batch.shape[0] == 0:
            raise ValueError("raw_samples đang rỗng.")
        if not np.all(np.isfinite(batch)):
            raise ValueError("raw_samples chứa NaN hoặc Infinity.")

        return batch

    @staticmethod
    def _wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
        """Đưa angle delta về [-180, 180)."""
        return (angle + 180.0) % 360.0 - 180.0
