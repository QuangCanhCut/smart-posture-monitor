from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersonalCalibration:
    """Collect valid feature vectors and build a median personal baseline."""

    feature_count: int = 29
    _samples: list[np.ndarray] = field(default_factory=list, init=False)
    _baseline: np.ndarray | None = field(default=None, init=False)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        return self._baseline is not None

    def reset(self) -> None:
        self._samples.clear()
        self._baseline = None

    def add_sample(self, feature_vector: np.ndarray | list[float]) -> None:
        vector = self._validate_vector(feature_vector)
        self._samples.append(vector)

    def calculate_baseline(self) -> np.ndarray:
        if not self._samples:
            raise RuntimeError("Cannot calculate baseline without samples.")

        sample_matrix = np.vstack(self._samples)
        baseline = np.median(sample_matrix, axis=0)

        if baseline.shape != (self.feature_count,):
            raise RuntimeError(
                f"Baseline shape mismatch: expected {(self.feature_count,)}, "
                f"got {baseline.shape}."
            )

        if not np.isfinite(baseline).all():
            raise RuntimeError("Calculated baseline contains NaN or infinity.")

        self._baseline = baseline.astype(np.float64, copy=True)
        return self.get_baseline()

    def transform(self, feature_vector: np.ndarray | list[float]) -> np.ndarray:
        if self._baseline is None:
            raise RuntimeError("Calibration baseline is not ready.")

        vector = self._validate_vector(feature_vector)
        delta = vector - self._baseline

        if not np.isfinite(delta).all():
            raise RuntimeError("Delta feature vector contains NaN or infinity.")

        return delta.astype(np.float64, copy=False)

    def get_baseline(self) -> np.ndarray:
        if self._baseline is None:
            raise RuntimeError("Calibration baseline is not ready.")
        return self._baseline.copy()

    def _validate_vector(
        self,
        feature_vector: np.ndarray | list[float],
    ) -> np.ndarray:
        try:
            vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("Feature vector must be numeric.") from error

        if vector.shape != (self.feature_count,):
            raise ValueError(
                f"Expected feature vector with {self.feature_count} values, "
                f"got shape {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Feature vector contains NaN or infinity.")

        return vector
