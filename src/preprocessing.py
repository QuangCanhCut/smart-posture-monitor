from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.feature_extractor import FeatureExtractor


# ==========================================================
# 1. CẤU HÌNH DATASET V03
# ==========================================================

# Metadata do DatasetBuilder ghi ra. Không cột nào được đưa vào model.
METADATA_COLUMNS = (
    "image_path",
    "session_id",
    "person_id",
    "label",
    "frame_index",
)

EXPECTED_CLASSES = (
    "correct",
    "forward_slouch",
    "lean_left",
    "lean_right",
)

LABEL_TO_ID = {
    "correct": 0,
    "forward_slouch": 1,
    "lean_left": 2,
    "lean_right": 3,
}

# FeatureExtractor là single source of truth cho schema RAW12.
RAW_FEATURE_COLUMNS = tuple(FeatureExtractor.FEATURE_NAMES)
RAW_FEATURE_COUNT = len(RAW_FEATURE_COLUMNS)


@dataclass
class PreparedRawDataset:
    """
    Dataset RAW đã sẵn sàng để bước training xử lý tiếp.

    X_raw:
        RAW12, chưa calibration và chưa tạo REP13.

    y:
        Label đã encode thành số.

    groups:
        person_id dùng cho group-based split, tránh subject leakage.

    metadata:
        Thông tin để trace, chia recording và chọn calibration frames.
    """

    df: pd.DataFrame
    X_raw: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    raw_feature_columns: list[str]


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Đọc features.csv và kiểm tra file tồn tại, không rỗng."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset đang rỗng: {path}")

    return df


# ==========================================================
# 3. VALIDATE SCHEMA
# ==========================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Kiểm tra schema của features.csv V03.

    Yêu cầu:
    - Đủ metadata columns.
    - Đúng RAW feature names từ FeatureExtractor.
    - Không có cột lạ.
    - Đủ đúng 4 class.
    - frame_index và RAW features phải numeric.
    """
    expected_columns = set(METADATA_COLUMNS) | set(RAW_FEATURE_COLUMNS)
    actual_columns = set(df.columns)

    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)

    if missing or unexpected:
        raise ValueError(
            "Schema features.csv không hợp lệ.\n"
            f"Thiếu cột: {missing}\n"
            f"Cột không mong đợi: {unexpected}"
        )

    actual_classes = set(df["label"].dropna().astype(str).unique())
    expected_classes = set(EXPECTED_CLASSES)

    if actual_classes != expected_classes:
        raise ValueError(
            "Schema label không hợp lệ.\n"
            f"Thiếu class: {sorted(expected_classes - actual_classes)}\n"
            f"Class lạ: {sorted(actual_classes - expected_classes)}"
        )

    numeric_columns = ["frame_index", *RAW_FEATURE_COLUMNS]
    non_numeric = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            f"Các cột sau phải là numeric: {non_numeric}"
        )


# ==========================================================
# 4. VALIDATE INTEGRITY
# ==========================================================

def validate_integrity(df: pd.DataFrame) -> None:
    """
    Kiểm tra chất lượng dữ liệu nhưng KHÔNG tự sửa dữ liệu lỗi.

    Kiểm:
    - Missing / chuỗi metadata rỗng.
    - RAW12 có NaN/Inf hay không.
    - frame_index có phải số nguyên không âm.
    - duplicated rows / duplicated image_path.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Dataset chứa missing values: {missing.to_dict()}"
        )

    # Các metadata dạng text không được rỗng.
    for column in ("image_path", "session_id", "person_id", "label"):
        empty_count = int(df[column].astype(str).str.strip().eq("").sum())
        if empty_count:
            raise ValueError(
                f"Metadata '{column}' có {empty_count} giá trị rỗng."
            )

    raw = df[list(RAW_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    if not np.isfinite(raw).all():
        row, col = np.argwhere(~np.isfinite(raw))[0]
        raise ValueError(
            "RAW feature chứa NaN/Infinity tại "
            f"row={df.index[row]}, feature='{RAW_FEATURE_COLUMNS[col]}'."
        )

    frame_index = df["frame_index"].to_numpy(dtype=np.float64)

    if not np.isfinite(frame_index).all():
        raise ValueError("frame_index chứa NaN hoặc Infinity.")

    if np.any(frame_index < 0) or np.any(frame_index != np.floor(frame_index)):
        raise ValueError("frame_index phải là số nguyên không âm.")

    duplicated_rows = int(df.duplicated().sum())
    if duplicated_rows:
        raise ValueError(
            f"Dataset chứa {duplicated_rows} duplicated rows."
        )

    duplicated_paths = df["image_path"].duplicated(keep=False)
    if duplicated_paths.any():
        examples = (
            df.loc[duplicated_paths, "image_path"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Dataset chứa duplicated image_path. "
            f"Ví dụ: {examples}"
        )


# ==========================================================
# 5. RECORDING ID
# ==========================================================

def add_recording_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo ID duy nhất cho từng lần quay:

        person07 + session02
        -> person07__session02

    recording_id dùng để chọn baseline riêng theo person/session.
    """
    result = df.copy()

    result["recording_id"] = (
        result["person_id"].astype(str)
        + "__"
        + result["session_id"].astype(str)
    )

    return result


# ==========================================================
# 6. OPTIONAL PROTOCOL FILTER
# ==========================================================

def filter_protocol_invalid(
    df: pd.DataFrame,
    excluded_persons: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Loại các person đã biết chắc chắn vi phạm protocol thu thập.

    Đây KHÔNG phải outlier filtering.
    Mặc định không loại person nào.
    """
    if not excluded_persons:
        return df.reset_index(drop=True).copy()

    excluded = {str(person) for person in excluded_persons}
    result = df.loc[
        ~df["person_id"].astype(str).isin(excluded)
    ].reset_index(drop=True)

    if result.empty:
        raise ValueError("Protocol filtering đã loại toàn bộ dataset.")

    return result


# ==========================================================
# 7. ENTRY POINT: FEATURES.CSV -> PREPARED RAW DATASET
# ==========================================================

def prepare_raw_dataset(
    csv_path: str | Path,
    excluded_persons: Sequence[str] | None = None,
) -> PreparedRawDataset:
    """
    Pipeline preprocessing V03:

        features.csv
        -> load
        -> validate schema
        -> validate integrity
        -> recording_id
        -> optional protocol filter
        -> X_raw / y / groups / metadata

    Lưu ý:
        X_raw vẫn là RAW12.
        Calibration và RAW12 -> REP13 được làm SAU group split ở training.
    """
    df = load_dataset(csv_path)

    validate_schema(df)
    validate_integrity(df)

    df = add_recording_id(df)
    df = filter_protocol_invalid(df, excluded_persons)

    # X_raw chỉ chứa đúng RAW12, tuyệt đối không có metadata.
    X_raw = df[list(RAW_FEATURE_COLUMNS)].astype(np.float32)

    y = df["label"].map(LABEL_TO_ID)
    if y.isna().any():
        unknown = df.loc[y.isna(), "label"].unique().tolist()
        raise ValueError(f"Không encode được label: {unknown}")

    y = y.astype(np.int64)
    y.name = "label_id"

    # Group split theo person để một người không xuất hiện ở cả train/test.
    groups = df["person_id"].astype(str).copy()
    groups.name = "person_id"

    metadata = df[
        [
            "image_path",
            "session_id",
            "person_id",
            "recording_id",
            "label",
            "frame_index",
        ]
    ].copy()

    if not (
        len(X_raw)
        == len(y)
        == len(groups)
        == len(metadata)
    ):
        raise RuntimeError(
            "X_raw, y, groups và metadata không cùng số sample."
        )

    return PreparedRawDataset(
        df=df.copy(),
        X_raw=X_raw,
        y=y,
        groups=groups,
        metadata=metadata,
        raw_feature_columns=list(RAW_FEATURE_COLUMNS),
    )


# ==========================================================
# CHẠY KIỂM TRA NHANH
# ==========================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "processed" / "features.csv"

    prepared = prepare_raw_dataset(dataset_path)

    print("=== PREPROCESSING V03 COMPLETE ===")
    print(f"Samples     : {len(prepared.df)}")
    print(f"RAW features: {prepared.X_raw.shape[1]}")
    print(f"Persons     : {prepared.groups.nunique()}")
    print(f"Recordings  : {prepared.metadata['recording_id'].nunique()}")
    print(f"Classes     : {prepared.df['label'].nunique()}")
    print(f"X_raw shape : {prepared.X_raw.shape}")
