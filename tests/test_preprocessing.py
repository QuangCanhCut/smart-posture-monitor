"""
Unit tests cho src/preprocessing.py.

Mục tiêu:
- Kiểm tra từng function chính trong preprocessing.py.
- Đảm bảo dữ liệu lỗi bị phát hiện sớm.
- Đảm bảo output X / y / groups / metadata đúng thiết kế.
- Đảm bảo utility lọc protocol-invalid person hoạt động generic.

Cách chạy từ thư mục gốc project:

    pytest tests/test_preprocessing.py -v

Hoặc chạy toàn bộ test:

    pytest -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as prep


# ============================================================
# 1. FIXTURE: TẠO DATASET GIẢ HỢP LỆ
# ============================================================

@pytest.fixture
def valid_df() -> pd.DataFrame:
    """
    Tạo một DataFrame nhỏ nhưng đúng schema của project.

    Dataset giả gồm:
    - đủ 4 labels;
    - có nhiều person;
    - có person05 để xác nhận person này không bị loại mặc định;
    - có person07 với nhiều session để test group/recording_id;
    - đúng toàn bộ feature columns từ FeatureExtractor V2;
    - không có NaN / Inf / duplicate.

    Mỗi class được tạo cho nhiều person/session.
    Nhờ vậy sau khi lọc một person bất kỳ, dataset vẫn còn đủ 4 class.
    """

    rows = []

    person_sessions = [
        ("person01", "session01"),
        ("person02", "session01"),
        ("person05", "session01"),
        ("person07", "session01"),
        ("person07", "session02"),
    ]

    labels = prep.EXPECTED_CLASSES

    sample_id = 0

    for person_index, (person_id, session_id) in enumerate(person_sessions):

        for label_index, label in enumerate(labels):

            # ------------------------------------------------
            # Tạo giá trị numeric khác nhau cho từng feature.
            #
            # Không cần mô phỏng geometry thật vì đây là unit test
            # của preprocessing, không phải test FeatureExtractor.
            # ------------------------------------------------
            feature_values = {
                feature_name: float(
                    sample_id
                    + feature_index / 100.0
                    + 1.0
                )
                for feature_index, feature_name
                in enumerate(prep.EXPECTED_FEATURE_COLUMNS)
            }

            row = {
                "image_path": (
                    f"data/raw/{label}/"
                    f"{person_id}_{session_id}/"
                    f"frame_{sample_id:04d}.jpg"
                ),
                "session_id": session_id,
                "person_id": person_id,
                "label": label,
                **feature_values,
            }

            rows.append(row)
            sample_id += 1

    return pd.DataFrame(rows)


# ============================================================
# 2. TEST load_dataset()
# ============================================================

def test_load_dataset_thanh_cong(
    tmp_path: Path,
    valid_df: pd.DataFrame,
) -> None:
    """
    File CSV hợp lệ phải được đọc thành công.
    """

    csv_path = tmp_path / "features.csv"
    valid_df.to_csv(
        csv_path,
        index=False,
    )

    loaded_df = prep.load_dataset(
        csv_path
    )

    assert isinstance(
        loaded_df,
        pd.DataFrame,
    )

    assert len(loaded_df) == len(
        valid_df
    )

    assert list(loaded_df.columns) == list(
        valid_df.columns
    )


def test_load_dataset_bao_loi_khi_file_khong_ton_tai(
    tmp_path: Path,
) -> None:
    """
    Nếu features.csv không tồn tại thì phải raise FileNotFoundError.
    """

    missing_path = (
        tmp_path
        / "khong_ton_tai.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        prep.load_dataset(
            missing_path
        )


def test_load_dataset_bao_loi_khi_csv_rong(
    tmp_path: Path,
) -> None:
    """
    CSV chỉ có header nhưng không có sample phải bị từ chối.
    """

    empty_df = pd.DataFrame(
        columns=(
            prep.METADATA_COLUMNS
            + prep.EXPECTED_FEATURE_COLUMNS
        )
    )

    csv_path = (
        tmp_path
        / "empty_features.csv"
    )

    empty_df.to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(
        ValueError
    ):
        prep.load_dataset(
            csv_path
        )


# ============================================================
# 3. TEST get_feature_columns()
# ============================================================

def test_get_feature_columns_dung_29_features(
    valid_df: pd.DataFrame,
) -> None:
    """
    Dataset hợp lệ phải trả về đúng danh sách feature chuẩn.
    """

    feature_columns = (
        prep.get_feature_columns(
            valid_df
        )
    )

    assert (
        feature_columns
        == prep.EXPECTED_FEATURE_COLUMNS
    )

    assert len(
        feature_columns
    ) == prep.EXPECTED_NUM_FEATURES


def test_get_feature_columns_bao_loi_khi_thieu_feature(
    valid_df: pd.DataFrame,
) -> None:
    """
    Nếu thiếu một feature thì preprocessing phải dừng.
    """

    broken_df = valid_df.drop(
        columns=[
            prep.EXPECTED_FEATURE_COLUMNS[0]
        ]
    )

    with pytest.raises(
        ValueError
    ):
        prep.get_feature_columns(
            broken_df
        )


def test_get_feature_columns_bao_loi_khi_feature_sai_ten(
    valid_df: pd.DataFrame,
) -> None:
    """
    Giữ nguyên số lượng 29 cột nhưng đổi tên một feature thành tên lạ.
    Function vẫn phải phát hiện schema không khớp.
    """

    broken_df = valid_df.rename(
        columns={
            prep.EXPECTED_FEATURE_COLUMNS[0]:
            "feature_khong_hop_le"
        }
    )

    with pytest.raises(
        ValueError
    ):
        prep.get_feature_columns(
            broken_df
        )


# ============================================================
# 4. TEST validate_schema()
# ============================================================

def test_validate_schema_thanh_cong(
    valid_df: pd.DataFrame,
) -> None:
    """
    Dataset đúng schema không được raise exception.
    """

    feature_columns = (
        prep.get_feature_columns(
            valid_df
        )
    )

    prep.validate_schema(
        valid_df,
        feature_columns,
    )


def test_validate_schema_bao_loi_khi_thieu_metadata(
    valid_df: pd.DataFrame,
) -> None:
    """
    Thiếu person_id phải bị phát hiện.
    """

    broken_df = valid_df.drop(
        columns=["person_id"]
    )

    with pytest.raises(
        ValueError
    ):
        prep.validate_schema(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


def test_validate_schema_bao_loi_khi_label_khong_hop_le(
    valid_df: pd.DataFrame,
) -> None:
    """
    Label ngoài 4 class chính thức phải bị từ chối.
    """

    broken_df = valid_df.copy()

    broken_df.loc[
        broken_df.index[0],
        "label",
    ] = "wrong_posture"

    with pytest.raises(
        ValueError
    ):
        prep.validate_schema(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


def test_validate_schema_bao_loi_khi_feature_khong_numeric(
    valid_df: pd.DataFrame,
) -> None:
    """
    Feature bị biến thành string/object phải bị phát hiện.
    """

    broken_df = valid_df.copy()

    feature_name = (
        prep.EXPECTED_FEATURE_COLUMNS[0]
    )

    # Gán cả column thành string để dtype trở thành object.
    broken_df[feature_name] = (
        broken_df[feature_name]
        .astype(str)
    )

    with pytest.raises(
        TypeError
    ):
        prep.validate_schema(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


# ============================================================
# 5. TEST validate_integrity()
# ============================================================

def test_validate_integrity_thanh_cong(
    valid_df: pd.DataFrame,
) -> None:
    """
    Dataset sạch không được raise exception.
    """

    prep.validate_integrity(
        valid_df,
        prep.EXPECTED_FEATURE_COLUMNS,
    )


def test_validate_integrity_bao_loi_khi_co_nan(
    valid_df: pd.DataFrame,
) -> None:
    """
    NaN trong feature phải bị phát hiện.
    """

    broken_df = valid_df.copy()

    broken_df.loc[
        broken_df.index[0],
        prep.EXPECTED_FEATURE_COLUMNS[0],
    ] = np.nan

    with pytest.raises(
        ValueError
    ):
        prep.validate_integrity(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.inf,
        -np.inf,
    ],
)
def test_validate_integrity_bao_loi_khi_co_infinity(
    valid_df: pd.DataFrame,
    invalid_value: float,
) -> None:
    """
    +Inf hoặc -Inf trong feature phải bị phát hiện.
    """

    broken_df = valid_df.copy()

    broken_df.loc[
        broken_df.index[0],
        prep.EXPECTED_FEATURE_COLUMNS[0],
    ] = invalid_value

    with pytest.raises(
        ValueError
    ):
        prep.validate_integrity(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


def test_validate_integrity_bao_loi_khi_duplicate_row(
    valid_df: pd.DataFrame,
) -> None:
    """
    Nếu toàn bộ row bị lặp thì phải báo lỗi.
    """

    duplicated_row = (
        valid_df.iloc[[0]].copy()
    )

    broken_df = pd.concat(
        [
            valid_df,
            duplicated_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError
    ):
        prep.validate_integrity(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


def test_validate_integrity_bao_loi_khi_duplicate_image_path(
    valid_df: pd.DataFrame,
) -> None:
    """
    Cùng image_path xuất hiện ở hai record khác nhau phải bị phát hiện.
    """

    broken_df = valid_df.copy()

    # Cho row 1 dùng lại image_path của row 0,
    # nhưng vẫn giữ các giá trị khác để tránh duplicate toàn row.
    broken_df.loc[
        broken_df.index[1],
        "image_path",
    ] = broken_df.loc[
        broken_df.index[0],
        "image_path",
    ]

    with pytest.raises(
        ValueError
    ):
        prep.validate_integrity(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


def test_validate_integrity_bao_loi_khi_conflicting_label(
    valid_df: pd.DataFrame,
) -> None:
    """
    Một image_path có hai label khác nhau phải bị phát hiện.
    """

    broken_df = valid_df.copy()

    # Cho row 1 dùng image_path của row 0.
    broken_df.loc[
        broken_df.index[1],
        "image_path",
    ] = broken_df.loc[
        broken_df.index[0],
        "image_path",
    ]

    # Bảo đảm label khác nhau.
    broken_df.loc[
        broken_df.index[1],
        "label",
    ] = (
        "lean_right"
        if broken_df.loc[
            broken_df.index[0],
            "label",
        ] != "lean_right"
        else "correct"
    )

    with pytest.raises(
        ValueError
    ):
        prep.validate_integrity(
            broken_df,
            prep.EXPECTED_FEATURE_COLUMNS,
        )


# ============================================================
# 6. TEST add_recording_id()
# ============================================================

def test_add_recording_id_tao_dung_dinh_dang(
    valid_df: pd.DataFrame,
) -> None:
    """
    recording_id phải có dạng:
        person_id__session_id
    """

    result = prep.add_recording_id(
        valid_df
    )

    assert (
        "recording_id"
        in result.columns
    )

    expected = (
        result.loc[0, "person_id"]
        + "__"
        + result.loc[0, "session_id"]
    )

    assert (
        result.loc[
            0,
            "recording_id",
        ]
        == expected
    )


def test_add_recording_id_khong_sua_dataframe_goc(
    valid_df: pd.DataFrame,
) -> None:
    """
    Function phải tạo copy,
    không được thêm recording_id trực tiếp vào DataFrame đầu vào.
    """

    original_df = valid_df.copy(
        deep=True
    )

    _ = prep.add_recording_id(
        valid_df
    )

    pd.testing.assert_frame_equal(
        valid_df,
        original_df,
    )

    assert (
        "recording_id"
        not in valid_df.columns
    )


def test_add_recording_id_phan_biet_person_va_session(
    valid_df: pd.DataFrame,
) -> None:
    """
    recording_id phải ghép person_id và session_id,
    nên session01 của hai person khác nhau vẫn là hai recording khác nhau.
    """

    result = prep.add_recording_id(
        valid_df
    )

    pairs = (
        result[
            [
                "person_id",
                "session_id",
                "recording_id",
            ]
        ]
        .drop_duplicates()
        .set_index(
            [
                "person_id",
                "session_id",
            ]
        )["recording_id"]
        .to_dict()
    )

    assert pairs[
        (
            "person07",
            "session01",
        )
    ] == "person07__session01"

    assert pairs[
        (
            "person07",
            "session02",
        )
    ] == "person07__session02"

    assert pairs[
        (
            "person01",
            "session01",
        )
    ] == "person01__session01"

    assert pairs[
        (
            "person02",
            "session01",
        )
    ] == "person02__session01"


# ============================================================
# 7. TEST filter_protocol_invalid()
# ============================================================

def test_filter_protocol_invalid_none_khong_loai_du_lieu(
    valid_df: pd.DataFrame,
) -> None:
    """excluded_persons=None must keep all rows."""

    result = prep.filter_protocol_invalid(
        valid_df,
        excluded_persons=None,
    )

    assert len(result) == len(valid_df)
    pd.testing.assert_frame_equal(
        result,
        valid_df.reset_index(drop=True),
    )


def test_filter_protocol_invalid_empty_tuple_khong_loai_du_lieu(
    valid_df: pd.DataFrame,
) -> None:
    """excluded_persons=() must keep all rows."""

    result = prep.filter_protocol_invalid(
        valid_df,
        excluded_persons=(),
    )

    assert len(result) == len(valid_df)
    pd.testing.assert_frame_equal(
        result,
        valid_df.reset_index(drop=True),
    )


def test_filter_protocol_invalid_loai_dung_person_duoc_cau_hinh(
    valid_df: pd.DataFrame,
) -> None:
    """Filtering must be generic and must not hard-code person05."""

    result = prep.filter_protocol_invalid(
        valid_df,
        excluded_persons=("person02",),
    )

    assert "person02" not in set(result["person_id"])
    assert "person05" in set(result["person_id"])
    assert len(result) == (
        len(valid_df)
        - int((valid_df["person_id"] == "person02").sum())
    )
    assert list(result.index) == list(range(len(result)))


def test_filter_protocol_invalid_bao_loi_khi_loai_het_dataset(
    valid_df: pd.DataFrame,
) -> None:
    """Filtering all persons must raise ValueError."""

    with pytest.raises(ValueError):
        prep.filter_protocol_invalid(
            valid_df,
            excluded_persons=(
                "person01",
                "person02",
                "person05",
                "person07",
            ),
        )


# ============================================================
# 8. TEST prepare_model_data()
# ============================================================

def test_prepare_model_data_tao_dung_output(
    valid_df: pd.DataFrame,
) -> None:
    """
    Kiểm tra:
    - X có đúng 29 features.
    - y đúng mapping.
    - groups đúng person_id.
    - metadata có đúng các cột cần thiết.
    - metadata không lọt vào X.
    """

    df_with_recording = (
        prep.add_recording_id(
            valid_df
        )
    )

    prepared = (
        prep.prepare_model_data(
            df_with_recording,
            prep.EXPECTED_FEATURE_COLUMNS,
        )
    )

    # X phải đúng N x 29.
    assert prepared.X.shape == (
        len(valid_df),
        prep.EXPECTED_NUM_FEATURES,
    )

    # Thứ tự columns phải đúng chuẩn.
    assert list(
        prepared.X.columns
    ) == prep.EXPECTED_FEATURE_COLUMNS

    # Kiểm tra y theo mapping.
    expected_y = (
        valid_df["label"]
        .map(prep.LABEL_TO_ID)
        .astype(int)
        .tolist()
    )

    assert (
        prepared.y.tolist()
        == expected_y
    )

    # groups phải chính là person_id.
    assert (
        prepared.groups.tolist()
        == valid_df["person_id"]
        .astype(str)
        .tolist()
    )

    person07_rows = (
        df_with_recording["person_id"]
        == "person07"
    )

    assert set(
        df_with_recording.loc[
            person07_rows,
            "session_id",
        ]
    ) == {
        "session01",
        "session02",
    }

    assert set(
        prepared.groups.loc[
            person07_rows
        ]
    ) == {"person07"}

    # Metadata phải có đúng các cột trace/debug.
    assert list(
        prepared.metadata.columns
    ) == [
        "image_path",
        "session_id",
        "person_id",
        "recording_id",
        "label",
    ]

    # Metadata tuyệt đối không được lọt vào X.
    forbidden_columns = {
        "image_path",
        "session_id",
        "person_id",
        "recording_id",
        "label",
    }

    assert (
        forbidden_columns
        .isdisjoint(
            set(prepared.X.columns)
        )
    )


# ============================================================
# 9. TEST prepare_dataset() END-TO-END
# ============================================================

def test_prepare_dataset_end_to_end_khong_exclude_mac_dinh(
    tmp_path: Path,
    valid_df: pd.DataFrame,
) -> None:
    """End-to-end pipeline must keep person05 when excluded_persons is empty."""

    csv_path = tmp_path / "features.csv"
    valid_df.to_csv(csv_path, index=False)

    prepared = prep.prepare_dataset(
        csv_path=csv_path,
        excluded_persons=(),
    )

    assert len(prepared.df) == len(valid_df)
    assert "person05" in set(prepared.df["person_id"])
    assert set(prepared.df["label"]) == set(prep.EXPECTED_CLASSES)
    assert prepared.X.shape[1] == prep.EXPECTED_NUM_FEATURES
    assert prepared.groups.tolist() == prepared.df["person_id"].astype(str).tolist()
    assert list(prepared.metadata.columns) == [
        "image_path",
        "session_id",
        "person_id",
        "recording_id",
        "label",
    ]
    assert (
        len(prepared.X)
        == len(prepared.y)
        == len(prepared.groups)
        == len(prepared.metadata)
        == len(prepared.df)
    )


def test_prepare_dataset_filter_protocol_invalid_generic(
    tmp_path: Path,
    valid_df: pd.DataFrame,
) -> None:
    """prepare_dataset can still exclude a configured protocol-invalid person."""

    csv_path = tmp_path / "features.csv"
    valid_df.to_csv(csv_path, index=False)

    prepared = prep.prepare_dataset(
        csv_path=csv_path,
        excluded_persons=("person02",),
    )

    assert "person02" not in set(prepared.df["person_id"])
    assert "person05" in set(prepared.df["person_id"])
    assert set(prepared.df["label"]) == set(prep.EXPECTED_CLASSES)
    assert prepared.X.shape[1] == prep.EXPECTED_NUM_FEATURES


def test_prepare_dataset_khong_filter_person_khi_excluded_rong(
    tmp_path: Path,
    valid_df: pd.DataFrame,
) -> None:
    """Empty excluded_persons keeps the complete valid dataset."""

    csv_path = tmp_path / "features.csv"
    valid_df.to_csv(csv_path, index=False)

    prepared = prep.prepare_dataset(
        csv_path=csv_path,
        excluded_persons=(),
    )

    assert len(prepared.df) == len(valid_df)
    assert set(prepared.groups.unique()) == {
        "person01",
        "person02",
        "person05",
        "person07",
    }
