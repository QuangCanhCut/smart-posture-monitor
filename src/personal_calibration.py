"""
Personal Baseline Calibration cho Smart Posture Monitor V03.

Vai trò:
- Kiểm tra vector 29 features đầu vào.
- Thu thập các mẫu calibration realtime.
- Tính baseline cá nhân bằng median.
- Chuyển raw features thành delta features.
- Hỗ trợ transform batch cho quá trình training offline.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DeltaDataset:
    """Ket qua bien doi RAW -> Personal Baseline Delta theo recording."""

    X_delta: pd.DataFrame
    y_delta: pd.Series
    groups_delta: pd.Series
    metadata_delta: pd.DataFrame
    audit: pd.DataFrame
    baselines: dict[str, np.ndarray]
    calibration_indices: dict[str, list[int]]


class PersonalCalibration:
    """Cân chỉnh baseline cá nhân dựa trên 29 đặc trưng tư thế."""

    FEATURE_DIM = 29

    def __init__(self, target_samples: int = 30) -> None:
        """
        Khởi tạo bộ calibration.

        Parameters
        ----------
        target_samples:
            Số mẫu hợp lệ cần thu thập trước khi tính baseline.
        """
        if target_samples <= 0:
            raise ValueError("target_samples phải lớn hơn 0.")

        self.target_samples = target_samples

        # Bộ nhớ tạm chứa các vector feature trong quá trình calibration.
        self._samples: list[np.ndarray] = []

        # Baseline cá nhân sau khi calibration thành công.
        self._baseline: Optional[np.ndarray] = None

        # Cho biết calibration đã hoàn thành hay chưa.
        self.is_calibrated = False

    @property
    def collected_samples(self) -> int:
        """Số mẫu calibration hợp lệ đã thu thập."""
        return len(self._samples)

    @property
    def progress(self) -> float:
        """
        Tiến trình calibration trong khoảng [0.0, 1.0].

        Ví dụ:
            15 / 30 mẫu -> progress = 0.5
        """
        return min(
            self.collected_samples / self.target_samples,
            1.0,
        )

    @property
    def baseline(self) -> Optional[np.ndarray]:
        """
        Trả về bản sao của baseline hiện tại.

        Trả None nếu chưa calibration.
        """
        if self._baseline is None:
            return None

        return self._baseline.copy()

    @classmethod
    def _validate_vector(
        cls,
        sample: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Kiểm tra một vector feature đầu vào.

        Vector hợp lệ phải:
        - Không phải None.
        - Có shape (29,).
        - Chứa dữ liệu số.
        - Không có NaN hoặc Infinity.
        """
        if sample is None:
            return None

        try:
            array = np.asarray(sample, dtype=float)
        except (TypeError, ValueError):
            return None

        # Mỗi frame phải tạo đúng một vector 29 chiều.
        if (
            array.ndim != 1
            or array.shape[0] != cls.FEATURE_DIM
        ):
            return None

        # Loại các vector có NaN, +Inf hoặc -Inf.
        if not np.all(np.isfinite(array)):
            return None

        # Trả bản copy để tránh code bên ngoài sửa dữ liệu nội bộ.
        return array.copy()

    @classmethod
    def _validate_batch(
        cls,
        samples: np.ndarray,
    ) -> np.ndarray:
        """
        Kiểm tra ma trận feature dạng batch.

        Shape hợp lệ:
            (N, 29)
        """
        try:
            array = np.asarray(samples, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Batch feature phải chứa dữ liệu dạng số."
            ) from exc

        if (
            array.ndim != 2
            or array.shape[1] != cls.FEATURE_DIM
        ):
            raise ValueError(
                f"Batch phải có shape (N, {cls.FEATURE_DIM}), "
                f"nhưng nhận được {array.shape}."
            )

        if array.shape[0] == 0:
            raise ValueError(
                "Batch feature đang rỗng."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Batch feature chứa NaN hoặc Infinity."
            )

        return array

    def add_sample(
        self,
        raw_features: np.ndarray,
    ) -> bool:
        """
        Thêm một mẫu raw feature vào quá trình calibration realtime.

        Returns
        -------
        True:
            Frame hợp lệ và đã được chấp nhận.

        False:
            Frame không hợp lệ hoặc calibration đã hoàn thành.
        """
        sample = self._validate_vector(raw_features)

        if sample is None:
            return False

        # Không tiếp tục thu thập khi baseline đã được tạo.
        if self.is_calibrated:
            return False

        # Giới hạn buffer tối đa bằng target_samples.
        if self.collected_samples < self.target_samples:
            self._samples.append(sample)

        # Khi đủ số mẫu thì tự động tính baseline.
        if self.collected_samples == self.target_samples:
            self.compute_baseline()

        return True

    def compute_baseline(
        self,
    ) -> Optional[np.ndarray]:
        """
        Tính baseline bằng median theo từng feature.

        Chỉ thực hiện khi đã thu đủ target_samples.
        """
        if self.collected_samples < self.target_samples:
            return None

        # Shape sau stack:
        # (target_samples, 29)
        samples = np.stack(
            self._samples,
            axis=0,
        )

        # Median theo từng cột feature.
        self._baseline = np.median(
            samples,
            axis=0,
        )

        self.is_calibrated = True

        return self._baseline.copy()

    def transform(
        self,
        raw_features: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Chuyển raw features thành delta features.

        Công thức:
            delta = raw_features - baseline
        """
        if (
            not self.is_calibrated
            or self._baseline is None
        ):
            raise RuntimeError(
                "Calibration chưa hoàn thành. "
                "Cần thu đủ mẫu trước khi gọi transform()."
            )

        sample = self._validate_vector(
            raw_features
        )

        if sample is None:
            return None

        return sample - self._baseline

    def reset(self) -> None:
        """
        Xóa baseline và toàn bộ mẫu cũ.

        Dùng khi người dùng muốn recalibrate.
        """
        self._samples.clear()
        self._baseline = None
        self.is_calibrated = False

    @classmethod
    def fit_transform_offline(
        cls,
        correct_samples: np.ndarray,
        all_samples: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo baseline và transform dữ liệu theo batch khi training.

        Parameters
        ----------
        correct_samples:
            Các mẫu tư thế correct dùng để tạo baseline.
            Shape: (K, 29)

        all_samples:
            Toàn bộ các mẫu cần transform.
            Shape: (N, 29)

        Returns
        -------
        baseline:
            Vector baseline shape (29,)

        delta_matrix:
            Ma trận delta shape (N, 29)
        """
        correct = cls._validate_batch(
            correct_samples
        )

        samples = cls._validate_batch(
            all_samples
        )

        # Tính baseline cá nhân bằng median.
        baseline = np.median(
            correct,
            axis=0,
        )

        # NumPy broadcasting:
        # (N, 29) - (29,) -> (N, 29)
        delta_matrix = samples - baseline

        return baseline, delta_matrix


def extract_frame_index(image_path: str) -> int | None:
    """
    Lay frame index tu ten file neu co pattern frame_0001.

    Neu path khong co pattern ro rang thi tra ve None de caller giu thu tu
    DataFrame ban dau.
    """
    name = Path(str(image_path)).name
    match = re.search(r"frame[_-]?(\d+)", name, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _validate_offline_inputs(
    X_raw: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    if not (len(X_raw) == len(y) == len(groups) == len(metadata)):
        raise ValueError("X, y, groups va metadata phai co cung so sample.")

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in X_raw.columns
    ]
    if missing_features:
        raise ValueError(f"Thieu feature columns: {missing_features}")

    required_metadata = {"image_path", "person_id", "recording_id", "label"}
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Thieu metadata columns: {missing_metadata}")

    feature_array = X_raw[feature_columns].to_numpy(dtype=float)
    if feature_array.shape[1] != PersonalCalibration.FEATURE_DIM:
        raise ValueError(
            f"Expected {PersonalCalibration.FEATURE_DIM} features, "
            f"nhung nhan duoc {feature_array.shape[1]}."
        )
    if not np.isfinite(feature_array).all():
        raise ValueError("X_raw chua NaN hoac Infinity.")


def build_delta_dataset_for_recordings(
    X_raw: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
    feature_columns: list[str],
    calibration_samples: int = 30,
    correct_label: str = "correct",
) -> DeltaDataset:
    """
    Tao Personal Baseline Delta dataset cho nhieu recording.

    Protocol:
    - baseline rieng cho moi recording_id;
    - baseline = median(K correct samples dau tien theo thu tu frame);
    - K calibration samples bi loai khoi ML dataset;
    - cac sample con lai duoc bien doi: delta = raw - baseline.
    """
    if calibration_samples <= 0:
        raise ValueError("calibration_samples phai lon hon 0.")

    feature_columns = list(feature_columns)
    _validate_offline_inputs(
        X_raw=X_raw,
        y=y,
        groups=groups,
        metadata=metadata,
        feature_columns=feature_columns,
    )

    X_work = X_raw.reset_index(drop=False).rename(columns={"index": "original_index"})
    y_work = y.reset_index(drop=True)
    groups_work = groups.reset_index(drop=True).astype(str)
    metadata_work = metadata.reset_index(drop=True).copy()
    metadata_work["original_index"] = X_work["original_index"].to_numpy()
    metadata_work["_row_position"] = np.arange(len(metadata_work), dtype=int)
    metadata_work["_frame_index"] = metadata_work["image_path"].map(extract_frame_index)

    delta_frames: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    group_parts: list[pd.Series] = []
    metadata_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    baselines: dict[str, np.ndarray] = {}
    calibration_indices: dict[str, list[int]] = {}

    for recording_id, recording_meta in metadata_work.groupby("recording_id", sort=True):
        recording_id = str(recording_id)
        positions = recording_meta.index.to_numpy()

        if recording_meta["_frame_index"].notna().all():
            ordered_meta = recording_meta.sort_values(
                ["_frame_index", "_row_position"],
                kind="mergesort",
            )
            order_source = "image_path_frame_index"
        else:
            ordered_meta = recording_meta.sort_values("_row_position", kind="mergesort")
            order_source = "dataframe_order"

        ordered_positions = ordered_meta.index.to_numpy()
        correct_positions = ordered_meta.loc[
            ordered_meta["label"].astype(str).eq(correct_label)
        ].index.to_numpy()

        total_samples = int(len(positions))
        correct_samples = int(len(correct_positions))

        if correct_samples < calibration_samples:
            audit_rows.append(
                {
                    "recording_id": recording_id,
                    "person_id": str(recording_meta["person_id"].iloc[0]),
                    "total_samples": total_samples,
                    "correct_samples": correct_samples,
                    "calibration_samples": 0,
                    "remaining_correct": correct_samples,
                    "remaining_total": total_samples,
                    "baseline_created": False,
                    "order_source": order_source,
                    "skip_reason": (
                        f"not_enough_correct_samples: "
                        f"{correct_samples} < {calibration_samples}"
                    ),
                }
            )
            raise ValueError(
                "Recording khong du correct calibration samples: "
                f"{recording_id} ({correct_samples} < {calibration_samples})."
            )

        calibration_pos = correct_positions[:calibration_samples]
        remaining_pos = np.array(
            [pos for pos in ordered_positions if pos not in set(calibration_pos)],
            dtype=int,
        )

        X_calibration = X_work.loc[calibration_pos, feature_columns].to_numpy(dtype=float)
        X_remaining_raw = X_work.loc[remaining_pos, feature_columns].to_numpy(dtype=float)
        baseline, delta_matrix = PersonalCalibration.fit_transform_offline(
            correct_samples=X_calibration,
            all_samples=X_remaining_raw,
        )

        if baseline.shape != (len(feature_columns),):
            raise RuntimeError(
                f"Baseline shape sai cho {recording_id}: {baseline.shape}."
            )
        if delta_matrix.shape != (len(remaining_pos), len(feature_columns)):
            raise RuntimeError(
                f"Delta shape sai cho {recording_id}: {delta_matrix.shape}."
            )

        delta_frame = pd.DataFrame(delta_matrix, columns=feature_columns)
        delta_frames.append(delta_frame)
        y_parts.append(y_work.loc[remaining_pos].reset_index(drop=True))
        group_parts.append(groups_work.loc[remaining_pos].reset_index(drop=True))

        clean_meta = metadata_work.loc[remaining_pos].drop(
            columns=["_row_position", "_frame_index"]
        ).reset_index(drop=True)
        clean_meta["is_calibration_sample"] = False
        metadata_parts.append(clean_meta)

        remaining_labels = metadata_work.loc[remaining_pos, "label"].astype(str)
        baselines[recording_id] = baseline.copy()
        calibration_indices[recording_id] = (
            metadata_work.loc[calibration_pos, "original_index"].astype(int).tolist()
        )

        audit_rows.append(
            {
                "recording_id": recording_id,
                "person_id": str(recording_meta["person_id"].iloc[0]),
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "calibration_samples": int(calibration_samples),
                "remaining_correct": int(remaining_labels.eq(correct_label).sum()),
                "remaining_total": int(len(remaining_pos)),
                "baseline_created": True,
                "order_source": order_source,
                "skip_reason": "",
            }
        )

    if not delta_frames:
        raise ValueError("Khong tao duoc delta dataset nao.")

    X_delta = pd.concat(delta_frames, ignore_index=True)
    y_delta = pd.concat(y_parts, ignore_index=True).astype(int)
    y_delta.name = getattr(y, "name", "label_id")
    groups_delta = pd.concat(group_parts, ignore_index=True).astype(str)
    groups_delta.name = getattr(groups, "name", "person_id")
    metadata_delta = pd.concat(metadata_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)

    if not (len(X_delta) == len(y_delta) == len(groups_delta) == len(metadata_delta)):
        raise RuntimeError("Delta dataset bi mismatch X/y/groups/metadata.")

    if X_delta.shape[1] != PersonalCalibration.FEATURE_DIM:
        raise RuntimeError(
            f"X_delta phai co {PersonalCalibration.FEATURE_DIM} features."
        )

    if not np.isfinite(X_delta.to_numpy(dtype=float)).all():
        raise RuntimeError("X_delta chua NaN hoac Infinity.")

    return DeltaDataset(
        X_delta=X_delta,
        y_delta=y_delta,
        groups_delta=groups_delta,
        metadata_delta=metadata_delta,
        audit=audit,
        baselines=baselines,
        calibration_indices=calibration_indices,
    )
