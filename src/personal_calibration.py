from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.feature_extractor import FeatureExtractor


class PersonalCalibration:
    """
    Tạo và lưu personal baseline từ các RAW12 feature ở tư thế correct.

    Vai trò:
        RAW12 correct samples -> baseline RAW12

    Quy tắc baseline:
    - Angle features: xử lý theo circular geometry rồi lấy median.
    - Các feature còn lại: median thông thường.

    File này KHÔNG làm:
    - Delta feature
    - Normalize theo body scale
    - Log-ratio
    - Xây dataset train

    RAW12 + baseline -> REP13 thuộc RepresentationBuilder.
    """

    FEATURE_DIM = len(FeatureExtractor.FEATURE_NAMES)

    ANGLE_INDICES = tuple(
        FeatureExtractor.FEATURE_NAMES.index(name)
        for name in FeatureExtractor.ANGLE_FEATURES
    )

    SHOULDER_WIDTH_INDEX = FeatureExtractor.FEATURE_NAMES.index(
        "shoulder_width_px"
    )

    def __init__(self, target_samples: int = 30) -> None:
        if target_samples <= 0:
            raise ValueError("target_samples phải lớn hơn 0.")

        self.target_samples = int(target_samples)

        # Chứa RAW12 của các frame correct trong giai đoạn calibration.
        self._samples: list[np.ndarray] = []

        # Baseline RAW12 sau khi calibration hoàn thành.
        self._baseline: np.ndarray | None = None

    # ==========================================================
    # TRẠNG THÁI CALIBRATION
    # ==========================================================

    @property
    def sample_count(self) -> int:
        """Số RAW12 hợp lệ đã thu được."""
        return len(self._samples)

    @property
    def progress(self) -> float:
        """Tiến trình calibration trong khoảng [0, 1]."""
        return min(self.sample_count / self.target_samples, 1.0)

    @property
    def is_ready(self) -> bool:
        """True khi baseline đã được tạo."""
        return self._baseline is not None

    @property
    def baseline(self) -> np.ndarray | None:
        """Trả bản copy baseline để tránh code ngoài sửa state nội bộ."""
        return None if self._baseline is None else self._baseline.copy()

    @property
    def baseline_shoulder_width(self) -> float | None:
        """
        Shoulder width chuẩn của session.

        RepresentationBuilder dùng giá trị cố định này để normalize
        các spatial delta.
        """
        if self._baseline is None:
            return None

        return float(self._baseline[self.SHOULDER_WIDTH_INDEX])

    # ==========================================================
    # REALTIME CALIBRATION
    # ==========================================================

    def add_sample(
        self,
        raw_features: Sequence[float] | np.ndarray | None,
    ) -> bool:
        """
        Thêm một RAW12 correct sample.

        Return:
            True  -> sample hợp lệ và được nhận.
            False -> sample lỗi hoặc calibration đã hoàn thành.

        Đủ target_samples thì baseline được tính tự động.
        """
        if self.is_ready:
            return False

        sample = self._validate_vector(raw_features)
        if sample is None:
            return False

        self._samples.append(sample)

        if self.sample_count >= self.target_samples:
            self.calculate_baseline()

        return True

    def calculate_baseline(self) -> np.ndarray:
        """
        Tính baseline từ các sample realtime đã thu.

        Input nội bộ:
            (N, 12)

        Output:
            baseline RAW12 shape (12,)
        """
        if self.sample_count < self.target_samples:
            raise RuntimeError(
                f"Chưa đủ mẫu calibration: "
                f"{self.sample_count}/{self.target_samples}."
            )

        samples = np.stack(self._samples, axis=0)
        self._baseline = self._build_baseline(samples)

        return self._baseline.copy()

    # ==========================================================
    # OFFLINE CALIBRATION CHO TRAINING
    # ==========================================================

    def fit(
        self,
        correct_samples: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """
        Tạo baseline từ batch RAW12 correct của một person/session.

        Input:
            correct_samples: shape (N, 12)

        Output:
            baseline RAW12: shape (12,)
        """
        samples = self._validate_batch(correct_samples)

        if len(samples) < self.target_samples:
            raise ValueError(
                f"Cần ít nhất {self.target_samples} calibration samples, "
                f"nhưng chỉ nhận được {len(samples)}."
            )

        # Chỉ dùng đúng số mẫu calibration đã quy định.
        samples = samples[: self.target_samples]

        self._samples = [row.copy() for row in samples]
        self._baseline = self._build_baseline(samples)

        return self._baseline.copy()

    def reset(self) -> None:
        """Xóa samples và baseline để bắt đầu calibration mới."""
        self._samples.clear()
        self._baseline = None

    # ==========================================================
    # BASELINE LOGIC
    # ==========================================================

    @classmethod
    def _build_baseline(cls, samples: np.ndarray) -> np.ndarray:
        """
        Tạo baseline RAW12.

        Feature thường:
            median theo từng cột.

        Feature góc:
            dùng circular median-like để tránh lỗi tại biên ±180°.
        """
        baseline = np.median(samples, axis=0)

        for index in cls.ANGLE_INDICES:
            baseline[index] = cls._circular_median_deg(samples[:, index])

        return baseline

    @classmethod
    def _circular_median_deg(cls, angles: np.ndarray) -> float:
        """
        Median cho dữ liệu góc.

        Ví dụ:
            [-179, 178, 179, -178]

        phải có baseline quanh ±180°, không phải gần 0°.

        Cách làm:
        1. Chọn một góc quan sát làm reference sao cho tổng khoảng cách
           góc tới các sample còn lại là nhỏ nhất.
        2. Unwrap toàn bộ góc quanh reference.
        3. Lấy median tuyến tính.
        4. Wrap kết quả về [-180, 180).
        """
        values = np.asarray(angles, dtype=np.float64).reshape(-1)

        # Ma trận khoảng cách góc giữa mọi cặp sample.
        pairwise_delta = cls._wrap_angle_deg(
            values[:, None] - values[None, :]
        )
        total_distance = np.abs(pairwise_delta).sum(axis=0)

        reference = values[np.argmin(total_distance)]

        # Đưa tất cả sample về cùng lân cận với reference trước khi median.
        unwrapped = reference + cls._wrap_angle_deg(values - reference)
        median_angle = float(np.median(unwrapped))

        return float(cls._wrap_angle_deg(median_angle))

    @staticmethod
    def _wrap_angle_deg(
        angle: float | np.ndarray,
    ) -> float | np.ndarray:
        """Đưa góc về khoảng [-180, 180)."""
        return (angle + 180.0) % 360.0 - 180.0

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate_vector(
        cls,
        sample: Sequence[float] | np.ndarray | None,
    ) -> np.ndarray | None:
        """RAW feature phải có đúng shape (12,) và không có NaN/Inf."""
        if sample is None:
            return None

        try:
            vector = np.asarray(sample, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None

        if vector.shape != (cls.FEATURE_DIM,):
            return None

        if not np.all(np.isfinite(vector)):
            return None

        return vector.copy()

    @classmethod
    def _validate_batch(
        cls,
        samples: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Batch calibration phải có shape (N, 12), N > 0 và hữu hạn."""
        try:
            batch = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Calibration batch phải chứa dữ liệu số.") from exc

        if batch.ndim != 2 or batch.shape[1] != cls.FEATURE_DIM:
            raise ValueError(
                f"Calibration batch phải có shape (N, {cls.FEATURE_DIM}), "
                f"nhận được {batch.shape}."
            )

        if batch.shape[0] == 0:
            raise ValueError("Calibration batch đang rỗng.")

        if not np.all(np.isfinite(batch)):
            raise ValueError("Calibration batch chứa NaN hoặc Infinity.")

        return batch.copy()
