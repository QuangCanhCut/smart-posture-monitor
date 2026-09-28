# Cập nhật Dataset V02, EDA và Preprocessing - 2026-09-27

## 1. Mục tiêu công việc

Tạo baseline Dataset V02 mới cho project Smart Posture Monitor và chuẩn bị lại pipeline trước khi triển khai `train.py`. Trạng thái này dùng để snapshot code hiện tại trên nhánh `CanhCutDev_V02`, tách biệt với nhánh `CanhCutDev` cũ đang được teammate dùng cho training phiên bản trước.

## 2. Dataset V02 hiện tại

Kết quả DatasetBuilder hiện tại:

- Tổng ảnh raw: 4341
- Valid samples: 4014
- Tỷ lệ hợp lệ: 92.47%
- Rejected: 327
- Persons: 14
- Sessions / person-session recordings: 16
- Features: 29
- Classes: 4

Class distribution:

- correct: 1051
- forward_slouch: 889
- lean_left: 1054
- lean_right: 1020

Rejected:

- feature_extraction_failed: 327

## 3. Dataset structure

Cấu trúc raw dataset hiện tại:

```text
data/raw/<label>/<person_id>_<session_id>/<image>
```

Output được build ra:

```text
data/processed/features.csv
data/rejected/rejected_images.csv
```

Các file dữ liệu trong `data/` là artifact runtime/dataset local và không được push lên GitHub.

## 4. Feature pipeline

Pipeline feature hiện tại:

```text
Image
-> PoseDetector
-> 6 keypoints
-> FeatureExtractor
-> 29 engineered features
-> DatasetBuilder
-> features.csv
```

`DatasetBuilder` vẫn giữ pipeline V02, validate đủ 29 features và ghi rejected images kèm reason/detail để audit.

## 5. EDA đã thực hiện

Notebook hiện tại:

```text
notebooks/01_eda_dataset.ipynb
```

Các phân tích chính đang phản ánh Dataset V02:

- 4014 samples
- 29 features
- 14 persons
- 16 person-session recordings
- 4 classes
- không missing values / NaN / Inf
- không duplicate rows
- class distribution tương đối cân bằng
- sample distribution giữa person có chênh lệch
- `person07` có nhiều samples hơn do có 3 sessions
- có nhiều feature correlation cao
- PCA cho thấy redundancy giữa 29 engineered features
- subject-to-subject variation đáng chú ý
- group-based split theo `person_id` cần được dùng khi train

Notebook chỉ phục vụ exploration, không fit preprocessing artifact cuối cùng và không kết luận model tốt/xấu vì chưa train.

## 6. Preprocessing

File preprocessing hiện tại:

```text
src/preprocessing.py
```

Pipeline thực tế:

```text
features.csv
-> load_dataset
-> get_feature_columns
-> validate_schema
-> validate_integrity
-> add_recording_id
-> filter_protocol_invalid
-> prepare_model_data
-> X / y / groups / metadata
```

Output preprocessing:

- `X` chứa đúng 29 engineered features.
- `y` là encoded label theo mapping:
  - correct -> 0
  - forward_slouch -> 1
  - lean_left -> 2
  - lean_right -> 3
- `groups = person_id` để chống subject leakage.
- `metadata` gồm `image_path`, `session_id`, `person_id`, `recording_id`, `label`.
- `recording_id = person_id + "__" + session_id`, chỉ phục vụ trace/debug/audit.
- Metadata không được đưa vào `X`.

## 7. Cập nhật person05

`person05` cũ từng sai acquisition protocol. Dataset hiện tại đã thay bằng dữ liệu `person05` hợp lệ.

Vì vậy:

- `person05` không còn bị loại khỏi modeling.
- `excluded_persons` mặc định hiện là empty tuple `()`.
- `filter_protocol_invalid()` vẫn được giữ như utility generic để loại subject thật sự protocol-invalid trong tương lai nếu cần.

## 8. Data leakage prevention

`src/preprocessing.py` hiện tại không thực hiện:

- StandardScaler trên toàn dataset
- PCA trên toàn dataset
- random frame split
- SMOTE trước split
- xóa outlier bằng IQR
- xóa feature chỉ vì correlation cao
- oversampling / undersampling

Các bước model-dependent sẽ được xử lý trong `train.py` bằng group-based split theo `person_id` và sklearn Pipeline để tránh data leakage.

## 9. Trạng thái hiện tại

- DatasetBuilder: hoàn thành
- Dataset V02: hoàn thành
- EDA V02: hoàn thành
- Preprocessing: đã cập nhật
- Tests preprocessing: đã có và pass với `28 passed`

Bước tiếp theo:

```text
train.py
-> group-based split theo person_id
-> baseline SVM/XGBoost
-> evaluation
```

## 10. Git strategy

`CanhCutDev`:

- giữ nguyên baseline/version trước
- đang được teammate sử dụng để train
- không ghi đè, không rebase, không force-push

`CanhCutDev_V02`:

- snapshot pipeline V02 hiện tại
- dùng để phát triển `train.py` tiếp theo

`master`:

- nhánh ổn định
- chỉ merge khi V02 đã được kiểm tra đầy đủ

