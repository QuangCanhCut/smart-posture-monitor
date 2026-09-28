"""
Tiền xử lý dữ liệu cho dự án Smart Posture Monitor.

============================================================
MỤC ĐÍCH CỦA FILE
============================================================

File này chịu trách nhiệm chuẩn bị dữ liệu đầu vào cho quá trình train model.

Luồng xử lý chính:

    data/processed/features.csv
        ↓
    load_dataset()
        ↓
    get_feature_columns()
        ↓
    validate_schema()
        ↓
    validate_integrity()
        ↓
    add_recording_id()
        ↓
    filter_protocol_invalid()
        ↓
    prepare_model_data()
        ↓
    X, y, groups, metadata
        ↓
    train.py

============================================================
LƯU Ý KIẾN TRÚC
============================================================

File này KHÔNG thực hiện:
- StandardScaler trên toàn bộ dataset.
- PCA trên toàn bộ dataset.
- Xóa outlier bằng IQR.
- Xóa feature chỉ vì correlation cao.
- SMOTE / oversampling.
- Random train/validation/test split theo frame.
- Train model SVM/XGBoost/MLP.

Lý do:
Những bước trên nếu thực hiện sai vị trí có thể gây data leakage.
Các thao tác liên quan đến scaling, PCA, GroupKFold, train model
sẽ được thực hiện ở train.py thông qua Pipeline của sklearn.

Vai trò của preprocessing.py là:

    1. Đọc dữ liệu.
    2. Kiểm tra dataset có đúng cấu trúc hay không.
    3. Kiểm tra dữ liệu có lỗi hay không.
    4. Hỗ trợ loại các subject vi phạm protocol thu thập dữ liệu nếu có.
    5. Chuẩn bị X, y, groups và metadata.
    6. Trả dữ liệu sạch cho train.py.
"""

# ============================================================
# HƯỚNG DẪN CẬP NHẬT KHI DATASET THAY ĐỔI
# ============================================================
#
# File preprocessing.py được thiết kế để KHÔNG phải sửa mỗi khi
# dataset tăng thêm số lượng ảnh, person hoặc session.
#
# Các trường hợp KHÔNG cần sửa preprocessing.py:
# - Thêm ảnh mới vào dataset.
# - Thêm person mới, ví dụ person11, person12,...
# - Thêm session mới, ví dụ session02, session03,...
# - Rebuild lại data/processed/features.csv với cùng schema hiện tại.
#
# Sau khi cập nhật dataset, chỉ cần chạy lại:
#
#     python -m src.preprocessing
#
# và test lại:
#
#     python -m pytest tests/test_preprocessing.py -v
#
#
# ============================================================
# CÁC TRƯỜNG HỢP CẦN LƯU Ý / CÓ THỂ PHẢI CHỈNH CODE
# ============================================================
#
# 1. Nếu FeatureExtractor thay đổi số lượng hoặc tên feature:
#
#    Ví dụ:
#        V2: 29 features
#        V3: 35 features
#
#    preprocessing.py đang lấy trực tiếp:
#
#        FeatureExtractor.FEATURE_NAMES
#
#    nên thông thường không cần sửa danh sách feature ở đây.
#
#    Tuy nhiên PHẢI rebuild lại features.csv bằng FeatureExtractor mới.
#    Nếu CSV cũ không khớp với FeatureExtractor hiện tại,
#    preprocessing sẽ báo lỗi schema.
#
#
# 2. Nếu thay đổi hoặc thêm class:
#
#    Ví dụ thêm:
#
#        "lean_forward"
#
#    thì cần cập nhật:
#
#        EXPECTED_CLASSES
#        LABEL_TO_ID
#        ID_TO_LABEL
#
#
# 3. Nếu thay đổi metadata của dataset:
#
#    Ví dụ thêm:
#
#        camera_id
#        camera_angle
#        distance
#
#    thì cần xem xét cập nhật:
#
#        METADATA_COLUMNS
#        phần metadata trong prepare_model_data()
#
#
# 4. Nếu sau này phát hiện một person thật sự vi phạm acquisition protocol:
#
#    Ví dụ:
#
#        excluded_persons = ("personXX",)
#
#    Hiện tại Dataset V02 không có subject nào bị loại mặc định.
#    person05 mới đã hợp lệ và được xử lý như các person khác.
#
#
# 5. Không được tự động sửa preprocessing để:
#
#    - random split theo frame
#    - scale toàn bộ dataset trước khi split
#    - PCA toàn bộ dataset trước khi split
#    - xóa outlier chỉ vì IQR
#    - xóa feature chỉ vì correlation cao
#    - SMOTE trước khi split
#
#    Những bước này thuộc train.py / evaluation pipeline và phải
#    được thực hiện đúng theo group/person để tránh data leakage.
#
#
# ============================================================
# NGUYÊN TẮC CHUNG
# ============================================================
#
# Nếu dataset chỉ "nhiều hơn" nhưng cấu trúc không đổi:
#     -> KHÔNG sửa preprocessing.py
#
# Nếu dataset "đổi schema":
#     -> kiểm tra lại constants + validation logic
#
# Sau mọi lần thay đổi dataset:
#
#     1. Rebuild features.csv
#     2. Chạy python -m src.preprocessing
#     3. Chạy pytest
#     4. Chỉ train model khi tất cả đều PASS
#

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ============================================================
# IMPORT FEATURE EXTRACTOR
# ============================================================
#
# Mục đích:
# Lấy danh sách FEATURE_NAMES trực tiếp từ FeatureExtractor.
#
# Đây là "single source of truth":
# preprocessing.py không tự viết lại 29 tên feature.
#
# Nếu FeatureExtractor thay đổi số lượng/tên feature nhưng features.csv
# chưa được build lại thì phần kiểm tra schema sẽ phát hiện mismatch.
#
try:
    # Cách chạy khuyến nghị:
    #
    #     python -m src.preprocessing
    #
    # Khi chạy theo cách này, import từ package src hoạt động bình thường.
    from src.feature_extractor import FeatureExtractor

except ModuleNotFoundError:
    # Hỗ trợ trường hợp chạy trực tiếp:
    #
    #     python src/preprocessing.py
    #
    from feature_extractor import FeatureExtractor


# ============================================================
# 1. CÁC HẰNG SỐ CẤU HÌNH DATASET
# ============================================================

# 4 cột metadata.
# Các cột này dùng để quản lý/trace dữ liệu,
# KHÔNG được đưa trực tiếp vào X để train model.
METADATA_COLUMNS = [
    "image_path",
    "session_id",
    "person_id",
    "label",
]

# 4 class chính thức của bài toán.
EXPECTED_CLASSES = [
    "correct",
    "forward_slouch",
    "lean_left",
    "lean_right",
]

# Lấy danh sách feature trực tiếp từ FeatureExtractor V2.
# Hiện tại phải có đúng 29 features.
EXPECTED_FEATURE_COLUMNS = list(
    FeatureExtractor.FEATURE_NAMES
)

EXPECTED_NUM_FEATURES = len(
    EXPECTED_FEATURE_COLUMNS
)

# Mapping label text -> integer để dùng thống nhất trong training.
LABEL_TO_ID = {
    "correct": 0,
    "forward_slouch": 1,
    "lean_left": 2,
    "lean_right": 3,
}

# Mapping ngược integer -> label text.
# Hữu ích khi model dự đoán ra số và app cần hiển thị tên class.
ID_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_ID.items()
}


# ============================================================
# 2. CONTAINER CHỨA DỮ LIỆU SAU PREPROCESSING
# ============================================================

@dataclass
class PreparedDataset:
    """
    Gom toàn bộ output của preprocessing vào một object.

    df:
        DataFrame đầy đủ sau khi validate + filter.
        Bao gồm metadata, 29 features và recording_id.

    X:
        Ma trận đầu vào của model.
        Chỉ chứa đúng 29 engineered features.

    y:
        Nhãn đã encode:
            correct         -> 0
            forward_slouch  -> 1
            lean_left       -> 2
            lean_right      -> 3

    groups:
        person_id của từng sample.
        Dùng cho GroupKFold / LeaveOneGroupOut để chống subject leakage.

    metadata:
        Thông tin phục vụ trace/debug:
            image_path
            session_id
            person_id
            recording_id
            label

        Metadata KHÔNG được đưa vào model.

    feature_columns:
        Danh sách 29 feature theo đúng thứ tự chuẩn.
    """

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    feature_columns: list[str]


# ============================================================
# 3. ĐỌC DATASET
# ============================================================

def load_dataset(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Đọc features.csv vào pandas DataFrame.

    Kiểm tra:
    - file có tồn tại không;
    - path có đúng là file không;
    - dataset có rỗng không.

    Function này chỉ đọc dữ liệu.
    Chưa kiểm tra schema hay chất lượng bên trong.
    """

    # Chuẩn hóa đường dẫn thành Path.
    csv_path = Path(csv_path)

    # Không tìm thấy file -> dừng ngay.
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {csv_path}"
        )

    # Tránh truyền nhầm một thư mục thay vì file CSV.
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Đường dẫn dataset không phải file: {csv_path}"
        )

    # Đọc CSV.
    df = pd.read_csv(csv_path)

    # Dataset rỗng thì không thể train.
    if df.empty:
        raise ValueError(
            f"Dataset đang rỗng: {csv_path}"
        )

    return df


# ============================================================
# 4. XÁC ĐỊNH 29 FEATURE COLUMNS
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Xác định các cột feature và kiểm tra chúng có khớp
    với FeatureExtractor V2 hay không.

    Ý tưởng:
    - bỏ metadata;
    - bỏ recording_id nếu đã được thêm trước đó;
    - các cột còn lại phải đúng 29 feature chuẩn.

    Nếu CSV được build từ version cũ hoặc thiếu feature,
    function sẽ báo lỗi thay vì tiếp tục train sai dữ liệu.
    """

    # Các cột này không phải input feature của model.
    non_feature_columns = set(
        METADATA_COLUMNS
    ) | {"recording_id"}

    # Lấy tất cả cột còn lại.
    actual_feature_columns = [
        column
        for column in df.columns
        if column not in non_feature_columns
    ]

    # Kiểm tra số lượng feature.
    if (
        len(actual_feature_columns)
        != EXPECTED_NUM_FEATURES
    ):
        raise ValueError(
            "Số lượng feature không đúng. "
            f"Kỳ vọng {EXPECTED_NUM_FEATURES}, "
            f"nhưng tìm thấy {len(actual_feature_columns)}."
        )

    # Tìm các feature bị thiếu.
    missing_features = [
        feature
        for feature in EXPECTED_FEATURE_COLUMNS
        if feature not in actual_feature_columns
    ]

    # Tìm các feature lạ.
    unexpected_features = [
        feature
        for feature in actual_feature_columns
        if feature not in EXPECTED_FEATURE_COLUMNS
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            "Schema feature không khớp với FeatureExtractor V2.\n"
            f"Feature bị thiếu: {missing_features}\n"
            f"Feature không mong đợi: {unexpected_features}"
        )

    # Trả về thứ tự chuẩn từ FeatureExtractor.
    return EXPECTED_FEATURE_COLUMNS.copy()


# ============================================================
# 5. KIỂM TRA SCHEMA DATASET
# ============================================================

def validate_schema(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """
    Kiểm tra cấu trúc tổng thể của dataset.

    Các bước:
    1. Có đủ 4 metadata columns không?
    2. Có đúng 29 features không?
    3. Có đúng 4 labels không?
    4. Toàn bộ feature có phải numeric không?

    Đây là kiểm tra "cấu trúc".
    NaN/Inf/duplicate sẽ được kiểm tra ở validate_integrity().
    """

    # --------------------------------------------------------
    # BƯỚC 1: Kiểm tra metadata columns
    # --------------------------------------------------------
    missing_metadata = [
        column
        for column in METADATA_COLUMNS
        if column not in df.columns
    ]

    if missing_metadata:
        raise ValueError(
            f"Thiếu metadata columns: {missing_metadata}"
        )

    # --------------------------------------------------------
    # BƯỚC 2: Kiểm tra số lượng feature
    # --------------------------------------------------------
    if len(feature_columns) != EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Kỳ vọng {EXPECTED_NUM_FEATURES} features, "
            f"nhưng tìm thấy {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # BƯỚC 3: Kiểm tra 4 labels
    # --------------------------------------------------------
    actual_classes = set(
        df["label"]
        .dropna()
        .astype(str)
        .unique()
    )

    expected_classes = set(
        EXPECTED_CLASSES
    )

    missing_classes = sorted(
        expected_classes - actual_classes
    )

    unexpected_classes = sorted(
        actual_classes - expected_classes
    )

    if missing_classes or unexpected_classes:
        raise ValueError(
            "Schema label không hợp lệ.\n"
            f"Class bị thiếu: {missing_classes}\n"
            f"Class không mong đợi: {unexpected_classes}"
        )

    # --------------------------------------------------------
    # BƯỚC 4: Kiểm tra các feature phải là numeric
    # --------------------------------------------------------
    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if non_numeric_features:
        raise TypeError(
            "Tất cả engineered features phải là numeric. "
            f"Feature không phải numeric: {non_numeric_features}"
        )


# ============================================================
# 6. KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU
# ============================================================

def validate_integrity(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """
    Kiểm tra chất lượng dữ liệu bên trong dataset.

    Các bước:
    1. Missing value / NaN.
    2. Metadata string rỗng.
    3. NaN / +Inf / -Inf trong feature matrix.
    4. Duplicate toàn bộ row.
    5. Một image_path có nhiều label khác nhau không.
    6. Duplicated image_path.

    Quan điểm:
    Không tự sửa lỗi bằng fillna/drop_duplicates.
    Nếu phát hiện lỗi thì raise error để quay lại kiểm tra pipeline upstream.
    """

    # --------------------------------------------------------
    # BƯỚC 1: Missing values trên toàn dataset
    # --------------------------------------------------------
    missing_count = int(
        df.isna().sum().sum()
    )

    if missing_count > 0:
        columns_with_missing = (
            df.isna()
            .sum()
            .loc[
                lambda series:
                series > 0
            ]
            .to_dict()
        )

        raise ValueError(
            f"Dataset chứa {missing_count} missing values. "
            f"Các cột bị ảnh hưởng: {columns_with_missing}"
        )

    # --------------------------------------------------------
    # BƯỚC 2: Metadata không được là chuỗi rỗng
    # --------------------------------------------------------
    for column in METADATA_COLUMNS:

        empty_mask = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_mask.any():
            raise ValueError(
                f"Metadata column '{column}' có "
                f"{int(empty_mask.sum())} giá trị rỗng."
            )

    # --------------------------------------------------------
    # BƯỚC 3: Feature matrix phải toàn giá trị hữu hạn
    #
    # np.isfinite(False) nếu gặp:
    # - NaN
    # - +Inf
    # - -Inf
    # --------------------------------------------------------
    feature_array = (
        df[list(feature_columns)]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(feature_array).all():

        invalid_locations = np.argwhere(
            ~np.isfinite(feature_array)
        )

        first_row, first_col = (
            invalid_locations[0]
        )

        feature_name = (
            feature_columns[first_col]
        )

        raise ValueError(
            "Feature matrix chứa NaN hoặc Infinity. "
            f"Lỗi đầu tiên ở dataframe index "
            f"{df.index[first_row]}, feature '{feature_name}'."
        )

    # --------------------------------------------------------
    # BƯỚC 4: Kiểm tra duplicated rows
    # --------------------------------------------------------
    duplicated_rows = df.duplicated(
        keep=False
    )

    if duplicated_rows.any():
        raise ValueError(
            "Dataset chứa duplicated rows. "
            f"Số row bị ảnh hưởng: {int(duplicated_rows.sum())}."
        )

    # --------------------------------------------------------
    # BƯỚC 5: Kiểm tra conflicting label
    #
    # Một ảnh không được có nhiều label khác nhau.
    # --------------------------------------------------------
    label_count_per_image = (
        df.groupby("image_path")["label"]
        .nunique()
    )

    conflicting_images = (
        label_count_per_image[
            label_count_per_image > 1
        ]
        .index
        .tolist()
    )

    if conflicting_images:
        preview = conflicting_images[:5]

        raise ValueError(
            "Phát hiện image_path có nhiều label khác nhau. "
            f"Ví dụ: {preview}"
        )

    # --------------------------------------------------------
    # BƯỚC 6: Kiểm tra duplicated image_path
    #
    # Nguyên tắc:
    # 1 ảnh -> 1 record trong features.csv
    # --------------------------------------------------------
    duplicated_image_paths = (
        df["image_path"]
        .duplicated(keep=False)
    )

    if duplicated_image_paths.any():

        duplicated_examples = (
            df.loc[
                duplicated_image_paths,
                "image_path",
            ]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Dataset chứa duplicated image_path. "
            f"Ví dụ: {duplicated_examples}"
        )


# ============================================================
# 7. TẠO recording_id
# ============================================================

def add_recording_id(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo định danh duy nhất cho recording.

    session_id bị reuse giữa nhiều person:
        person01/session01
        person02/session01
        person03/session01

    Vì vậy tạo:
        recording_id = person_id + "__" + session_id

    Ví dụ:
        person01__session01
        person01__session02
        person02__session01

    recording_id dùng để:
    - trace;
    - audit;
    - kiểm tra recording leakage;
    - debug prediction.

    recording_id KHÔNG phải feature của model.
    """

    # Copy để không sửa trực tiếp DataFrame đầu vào.
    result = df.copy()

    result["recording_id"] = (
        result["person_id"].astype(str)
        + "__"
        + result["session_id"].astype(str)
    )

    return result


# ============================================================
# 8. LỌC SUBJECT SAI ACQUISITION PROTOCOL NẾU CÓ
# ============================================================

def filter_protocol_invalid(
    df: pd.DataFrame,
    excluded_persons: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Loại các person đã biết chắc chắn sai protocol thu thập.

    Đây KHÔNG phải outlier filtering.

    Hiện tại Dataset V02 không có subject nào bị loại mặc định.
    person05 mới đã hợp lệ và được xử lý như các person khác.

    Function này được giữ như một utility generic để tương lai có thể
    loại subject nếu phát hiện vi phạm protocol thật sự.
    """

    # Không có person nào cần loại.
    if not excluded_persons:
        return (
            df.copy()
            .reset_index(drop=True)
        )

    excluded_persons = tuple(
        excluded_persons
    )

    # Giữ lại các sample không thuộc danh sách bị loại.
    result = df[
        ~df["person_id"].isin(
            excluded_persons
        )
    ].copy()

    # Reset index sau filtering.
    result.reset_index(
        drop=True,
        inplace=True,
    )

    # Tránh cấu hình sai làm mất toàn bộ dataset.
    if result.empty:
        raise ValueError(
            "Protocol filtering đã loại toàn bộ samples."
        )

    return result


# ============================================================
# 9. CHUẨN BỊ X / y / groups / metadata
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> PreparedDataset:
    """
    Chuyển DataFrame đã validate thành dữ liệu sẵn sàng cho train.py.

    X:
        29 engineered features.

    y:
        label đã encode.

    groups:
        person_id dùng cho GroupKFold / LeaveOneGroupOut.

    metadata:
        thông tin dùng để trace/debug, không đưa vào model.
    """

    # --------------------------------------------------------
    # BƯỚC 1: Tạo X
    #
    # X chỉ chứa đúng 29 engineered features.
    # --------------------------------------------------------
    X = (
        df[list(feature_columns)]
        .copy()
        .astype(float)
    )

    # --------------------------------------------------------
    # BƯỚC 2: Encode y
    # --------------------------------------------------------
    y = df["label"].map(
        LABEL_TO_ID
    )

    # Nếu encode không được nghĩa là xuất hiện label không hợp lệ.
    if y.isna().any():

        unknown_labels = (
            df.loc[
                y.isna(),
                "label",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Không thể encode các label sau: {unknown_labels}"
        )

    y = y.astype(int)
    y.name = "label_id"

    # --------------------------------------------------------
    # BƯỚC 3: groups = person_id
    #
    # groups dùng để ngăn cùng một person xuất hiện ở cả
    # train và validation/test.
    # --------------------------------------------------------
    groups = (
        df["person_id"]
        .copy()
        .astype(str)
    )

    groups.name = "person_id"

    # --------------------------------------------------------
    # BƯỚC 4: Giữ metadata để trace/debug
    # --------------------------------------------------------
    metadata = df[
        [
            "image_path",
            "session_id",
            "person_id",
            "recording_id",
            "label",
        ]
    ].copy()

    # --------------------------------------------------------
    # BƯỚC 5: Safety check chống metadata leakage vào X
    # --------------------------------------------------------
    leaked_columns = (
        set(X.columns)
        & {
            "image_path",
            "session_id",
            "person_id",
            "recording_id",
            "label",
        }
    )

    if leaked_columns:
        raise RuntimeError(
            "Phát hiện metadata leakage trong X: "
            f"{sorted(leaked_columns)}"
        )

    # --------------------------------------------------------
    # BƯỚC 6: X phải đúng 29 features
    # --------------------------------------------------------
    if X.shape[1] != EXPECTED_NUM_FEATURES:
        raise RuntimeError(
            "X sau preprocessing phải có "
            f"{EXPECTED_NUM_FEATURES} features, "
            f"nhưng hiện có {X.shape[1]}."
        )

    # --------------------------------------------------------
    # BƯỚC 7: Tất cả output phải có cùng số sample
    # --------------------------------------------------------
    if not (
        len(X)
        == len(y)
        == len(groups)
        == len(metadata)
    ):
        raise RuntimeError(
            "X, y, groups và metadata có số lượng sample không khớp."
        )

    return PreparedDataset(
        df=df.copy(),
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        feature_columns=list(
            feature_columns
        ),
    )


# ============================================================
# 10. HÀM ENTRY-POINT CHÍNH
# ============================================================

def prepare_dataset(
    csv_path: str | Path,
    excluded_persons: Sequence[str] | None = None,
) -> PreparedDataset:
    """
    Chạy toàn bộ preprocessing pipeline.

    Luồng:
        features.csv
            ↓
        load_dataset()
            ↓
        get_feature_columns()
            ↓
        validate_schema()
            ↓
        validate_integrity()
            ↓
        add_recording_id()
            ↓
        filter_protocol_invalid()
            ↓
        prepare_model_data()
            ↓
        PreparedDataset
    """

    # BƯỚC 1: Đọc features.csv.
    df = load_dataset(
        csv_path
    )

    # BƯỚC 2: Xác định đúng 29 feature columns.
    feature_columns = get_feature_columns(
        df
    )

    # BƯỚC 3: Kiểm tra schema.
    validate_schema(
        df=df,
        feature_columns=feature_columns,
    )

    # BƯỚC 4: Kiểm tra integrity.
    validate_integrity(
        df=df,
        feature_columns=feature_columns,
    )

    # BƯỚC 5: Tạo recording_id.
    df = add_recording_id(
        df
    )

    # BƯỚC 6: Lọc các subject sai protocol nếu có.
    df = filter_protocol_invalid(
        df=df,
        excluded_persons=excluded_persons,
    )

    # BƯỚC 7: Tạo X, y, groups, metadata.
    prepared = prepare_model_data(
        df=df,
        feature_columns=feature_columns,
    )

    return prepared


# ============================================================
# 11. IN SUMMARY SAU PREPROCESSING
# ============================================================

def print_preprocessing_summary(
    prepared: PreparedDataset,
    excluded_persons: Sequence[str] | None = None,
) -> None:
    """
    In báo cáo nhanh để kiểm tra output preprocessing.

    Function này chủ yếu dùng khi chạy thử / review.
    """

    print("=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(
        f"Samples        : {len(prepared.df)}"
    )

    print(
        f"Features       : {prepared.X.shape[1]}"
    )

    print(
        f"Persons        : {prepared.groups.nunique()}"
    )

    print(
        f"Recordings     : "
        f"{prepared.df['recording_id'].nunique()}"
    )

    print(
        f"Classes        : "
        f"{prepared.df['label'].nunique()}"
    )

    print(
        "Excluded       : "
        f"{list(excluded_persons or [])}"
    )

    # Kiểm tra phân bố 4 class sau preprocessing.
    print("\nClass distribution:")
    print(
        prepared.df["label"]
        .value_counts()
        .reindex(
            EXPECTED_CLASSES,
            fill_value=0,
        )
    )

    # Kiểm tra person nào còn trong dataset.
    print("\nPersons:")
    print(
        sorted(
            prepared.df[
                "person_id"
            ].unique()
        )
    )

    # Kiểm tra shape đầu ra.
    print("\nOutput shapes:")
    print(
        f"X        : {prepared.X.shape}"
    )
    print(
        f"y        : {prepared.y.shape}"
    )
    print(
        f"groups   : {prepared.groups.shape}"
    )
    print(
        f"metadata : {prepared.metadata.shape}"
    )

    print("=" * 60)
    print(
        "PASS: Dataset is ready for train.py"
    )
    print("=" * 60)


# ============================================================
# 12. CHẠY THỬ preprocessing.py
# ============================================================
#
# Đoạn này chỉ chạy khi:
#
#     python -m src.preprocessing
#
# Nếu preprocessing.py được import từ train.py,
# phần dưới KHÔNG tự chạy.
#
if __name__ == "__main__":

    # Xác định project root:
    #
    # smart_posture_monitor/src/preprocessing.py
    #                     ↑
    #             parents[1]
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # Đường dẫn mặc định tới features.csv.
    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "features.csv"
    )

    # --------------------------------------------------------
    # CẤU HÌNH LỌC PROTOCOL
    # --------------------------------------------------------
    #
    # Dataset V02 hiện tại không có subject nào cần loại mặc định.
    # person05 mới đã hợp lệ. Giữ tuple rỗng để summary hiển thị:
    #
    #     Excluded       : []
    #
    excluded_persons = ()

    # Chạy preprocessing.
    prepared_dataset = prepare_dataset(
        csv_path=dataset_path,
        excluded_persons=excluded_persons,
    )

    # In summary để kiểm tra.
    print_preprocessing_summary(
        prepared=prepared_dataset,
        excluded_persons=excluded_persons,
    )
