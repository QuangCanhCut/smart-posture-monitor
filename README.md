# Smart Posture Monitor

Smart Posture Monitor là project Python dùng để nhận diện tư thế ngồi từ ảnh hoặc webcam.

Pipeline V02 hiện tại:

```text
YOLO Pose -> keypoints phần thân trên -> 29 engineered features
-> mô hình Machine Learning -> dự đoán tư thế
```

Project hiện không còn ở giai đoạn prototype ban đầu chỉ có PoseDetector + FeatureExtractor. V02 đã có pipeline build dataset, preprocessing, EDA, training experiments, chọn model, official held-out evaluation và prototype realtime webcam inference.

## 1. Các tư thế hỗ trợ

| Label | Ý nghĩa |
|---|---|
| `correct` | Tư thế ngồi tương đối đúng/upright. |
| `forward_slouch` | Cúi hoặc gù người về phía trước. |
| `lean_left` | Nghiêng người/đầu sang trái. |
| `lean_right` | Nghiêng người/đầu sang phải. |

## 2. Pipeline hệ thống

Pipeline offline để build dataset và train model:

```text
Raw images
-> PoseDetector
-> 6 keypoints
-> FeatureExtractor
-> 29 engineered features
-> DatasetBuilder
-> data/processed/features.csv
-> preprocessing / person-based split
-> group-based CV training
-> models/best_model.joblib
```

Pipeline realtime inference:

```text
Webcam frame
-> PoseDetector
-> 6 keypoints
-> FeatureExtractor
-> 29 engineered features
-> best_model.joblib
-> posture prediction
-> realtime display / tích hợp app sau này
```

## 3. Cấu trúc project

```text
smart_posture_monitor/
|-- app/
|   `-- app.py                         # Placeholder cho application
|-- data/                              # Dataset/runtime data local, đang bị ignore bởi Git
|   |-- raw/
|   |-- processed/
|   `-- rejected/
|-- docs/                              # Báo cáo tiến độ và ghi chú thiết kế
|-- models/
|   |-- best_model.joblib              # Classifier V02 đã train
|   |-- split_manifest.json            # Locked train/test person split
|   |-- training_metadata.json         # Metadata model và feature schema
|   `-- yolo26n-pose.pt                # YOLO pose model local, đang bị ignore bởi Git
|-- notebooks/
|   |-- 01_eda_dataset.ipynb
|   `-- 02_training_experiments.ipynb
|-- results/
|   |-- baseline_cv_results.csv
|   |-- model_selection_results.csv
|   |-- svm_search_results.csv
|   |-- xgboost_search_results.csv
|   `-- evaluation/
|-- scripts/
|   |-- copy_rejected_images.py
|   `-- test_webcam_model.py
|-- src/
|   |-- pose_detector.py
|   |-- feature_extractor.py
|   |-- dataset_builder.py
|   |-- preprocessing.py
|   |-- evaluate.py
|   |-- inference.py                   # Placeholder
|   |-- train.py                       # Placeholder
|   |-- posture_predictor.py           # Placeholder
|   |-- temporal_monitor.py            # Placeholder
|   `-- session_statistics.py          # Placeholder
|-- tests/
|   |-- test_preprocessing.py
|   |-- test_pose_detector.py
|   `-- test_feature_extractor.py
|-- requirements.txt
`-- README.md
```

## 4. Pose Detection

`src/pose_detector.py` dùng Ultralytics YOLO Pose với model local:

```text
models/yolo26n-pose.pt
```

Cấu hình mặc định của detector:

- `person_conf_threshold = 0.5`
- Nếu phát hiện nhiều người, hệ thống chọn person theo score:
  - `0.7 * centrality_score`
  - `0.3 * confidence_score`
- Chỉ trả về 6 keypoints phần thân trên:
  - `nose`
  - `left_eye`
  - `right_eye`
  - `left_ear`
  - `left_shoulder`
  - `right_shoulder`

Keypoint `right_ear` hiện không dùng trong feature pipeline V02.

## 5. Feature Engineering

`src/feature_extractor.py` chuyển 6 keypoints thành vector cố định gồm 29 engineered features.

Danh sách feature hiện tại được định nghĩa tại:

```python
FeatureExtractor.FEATURE_NAMES
```

Các nhóm feature chính:

- góc vai và góc đường mắt;
- tọa độ head/eye/nose/ear đã chuẩn hóa theo body frame;
- các tỉ lệ khoảng cách và độ bất đối xứng;
- các proxy về góc head/body/gravity;
- offset so với trục dọc;
- mean/spread của chiều cao head landmarks và độ phân tán góc.

FeatureExtractor kiểm tra confidence keypoint với ngưỡng:

```text
min_keypoint_confidence = 0.35
```

Nếu thiếu keypoint, keypoint confidence thấp, giá trị không finite, hoặc geometry không hợp lệ, frame sẽ bị reject bằng cách trả về `None`.

Tài liệu chi tiết hơn nằm ở [docs/feature_v2_29_features_2026-09-25.md](docs/feature_v2_29_features_2026-09-25.md).

## 6. Dataset V02

Các số liệu Dataset V02 dưới đây được xác nhận từ dataset local, notebook EDA, preprocessing và model metadata hiện tại.

| Hạng mục | Giá trị |
|---|---:|
| Raw images | 4341 |
| Valid samples | 4014 |
| Rejected samples | 327 |
| Persons | 14 |
| Person-session recordings | 16 |
| Classes | 4 |
| Engineered features | 29 |

Phân bố class:

| Class | Samples |
|---|---:|
| `correct` | 1051 |
| `forward_slouch` | 889 |
| `lean_left` | 1054 |
| `lean_right` | 1020 |

Cấu trúc raw dataset mà `DatasetBuilder` yêu cầu:

```text
data/raw/<label>/<person_id>_<session_id>/<image>
```

Các output local được sinh ra:

```text
data/processed/features.csv
data/rejected/rejected_images.csv
data/rejected/rejected_img/
```

Thư mục `data/` là dataset/runtime data local và đang được ignore bởi Git.

## 7. Chuẩn bị dữ liệu

### DatasetBuilder

`src/dataset_builder.py` build file feature CSV theo luồng:

```text
raw images
-> PoseDetector
-> FeatureExtractor
-> 29 features
-> data/processed/features.csv
```

Các ảnh bị reject được ghi lại kèm reason/detail tại:

```text
data/rejected/rejected_images.csv
```

### Preprocessing

`src/preprocessing.py` chuẩn bị dữ liệu cho model:

```text
features.csv
-> load_dataset()
-> get_feature_columns()
-> validate_schema()
-> validate_integrity()
-> add_recording_id()
-> filter_protocol_invalid()
-> prepare_model_data()
-> X, y, groups, metadata
```

Các điểm quan trọng:

- `X` chỉ chứa đúng 29 engineered features.
- `y` dùng mapping label:
  - `correct -> 0`
  - `forward_slouch -> 1`
  - `lean_left -> 2`
  - `lean_right -> 3`
- `groups = person_id` để phục vụ group-based validation và tránh leakage.
- `recording_id = person_id + "__" + session_id`.
- Metadata không được đưa vào feature matrix để train model.

Preprocessing không thực hiện global scaling, PCA, SMOTE, random frame split, IQR outlier removal hoặc drop feature chỉ vì correlation cao.

## 8. Exploratory Data Analysis

Notebook:

[notebooks/01_eda_dataset.ipynb](notebooks/01_eda_dataset.ipynb)

Tóm tắt EDA hiện tại:

- Dataset có `4014` valid samples, `14` persons, `16` person-session recordings và `4` classes.
- Class distribution tương đối cân bằng; tỉ lệ max/min khoảng `1.19x`.
- `person07` có nhiều samples hơn do có 3 sessions.
- Không phát hiện duplicate rows, missing values, NaN hoặc Inf trong feature matrix.
- Có outlier theo IQR ở một số feature, nhưng notebook chỉ thống kê, không xóa/cắt sample.
- Correlation analysis tìm thấy `22` feature pairs có `abs_corr >= 0.90`, gợi ý có redundancy cần theo dõi.
- PCA trên 29 scaled features cần 4 components để đạt ít nhất 90% variance và 6 components để đạt ít nhất 95% variance.
- Có subject-to-subject variation, vì vậy khi đánh giá model cần split theo group `person_id`.

EDA không train model và không kết luận model tốt/xấu.

## 9. Model Training

Notebook:

[notebooks/02_training_experiments.ipynb](notebooks/02_training_experiments.ipynb)

Training protocol:

- Load dataset qua `src.preprocessing.prepare_dataset()`.
- Split holdout theo `person_id`.
- Lock unseen test persons trước khi chọn model.
- Chạy `StratifiedGroupKFold` chỉ trên training persons.
- Dùng Macro F1 làm metric chính để chọn model.
- Không dùng held-out test set để chọn model.

Baseline models đã thử:

- `DummyClassifier`
- `LogisticRegression`
- `SVM RBF`
- `RandomForest`
- `ExtraTrees`
- `XGBoost`

Models đã tune:

- SVM RBF bằng `GridSearchCV`
- XGBoost bằng `RandomizedSearchCV`

Model cuối được chọn:

```text
SVM RBF Tuned
```

Training artifacts:

- [models/best_model.joblib](models/best_model.joblib)
- [models/split_manifest.json](models/split_manifest.json)
- [models/training_metadata.json](models/training_metadata.json)

## 10. Kết quả model

Training-CV reference từ `models/training_metadata.json`:

| Metric | Value |
|---|---:|
| Cross-validation Macro F1 mean | 0.607130 |
| Cross-validation Macro F1 std | 0.170123 |

Đây là mean validation Macro F1 từ group-based cross-validation trên training subjects. Đây không phải độ chính xác trên tập train.

Official held-out evaluation từ `results/evaluation/test_metrics.json`:

| Metric | Value |
|---|---:|
| Accuracy | 0.793494 |
| Balanced accuracy | 0.788075 |
| Macro precision | 0.784227 |
| Macro recall | 0.788075 |
| Macro F1 | 0.779126 |
| Weighted F1 | 0.796576 |

Held-out test set:

| Hạng mục | Giá trị |
|---|---:|
| Unseen test persons | 3 |
| Test samples | 707 |
| Test recordings | 3 |

Locked unseen test persons:

```text
person01, person10, person12
```

Held-out unseen-person Macro F1 là kết quả evaluation riêng trên locked test persons. Metric này không được dùng để tune model.

## 11. Per-class Performance

Từ `results/evaluation/classification_report.csv`:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `correct` | 0.875 | 0.936 | 0.904 | 172 |
| `forward_slouch` | 0.564 | 0.773 | 0.652 | 132 |
| `lean_left` | 0.995 | 0.867 | 0.927 | 226 |
| `lean_right` | 0.703 | 0.576 | 0.634 | 177 |

Confusion matrix từ `results/evaluation/confusion_matrix.csv`:

```text
                correct  forward_slouch  lean_left  lean_right
correct             161               2          0           9
forward_slouch        0             102          0          30
lean_left             9              17        196           4
lean_right           14              60          1         102
```

Nhận xét hiện tại:

- `correct` và `lean_left` là hai class mạnh hơn trong held-out test hiện tại.
- Confusion đáng chú ý nhất nằm giữa `forward_slouch` và `lean_right`.
- `lean_right` thường bị predict thành `forward_slouch`.

## 12. Realtime Webcam Inference

Script:

[scripts/test_webcam_model.py](scripts/test_webcam_model.py)

Chạy từ project root:

```bash
python scripts/test_webcam_model.py
```

Các argument hỗ trợ:

```bash
python scripts/test_webcam_model.py --camera 0
python scripts/test_webcam_model.py --no-pose
```

Script load:

```text
models/best_model.joblib
models/training_metadata.json
```

và predict ở mức frame-level:

```text
Webcam
-> PoseDetector
-> FeatureExtractor
-> trained model
-> realtime posture label
```

Trạng thái hiện tại:

- Đây là prototype/integration test cho realtime inference.
- Có vẽ bbox/keypoints, trừ khi truyền `--no-pose`.
- Chưa có temporal smoothing.
- Chưa refactor qua `PosturePredictor`.

## 13. Official Evaluation

Script:

[src/evaluate.py](src/evaluate.py)

Chạy:

```bash
python -m src.evaluate
```

Evaluation script thực hiện:

- load `features.csv`, `best_model.joblib`, `split_manifest.json` và `training_metadata.json`;
- verify dataset SHA256;
- dùng `prepare_dataset()`;
- reconstruct locked unseen-person test set;
- chạy `model.predict()`;
- ghi metrics, predictions, confusion matrices, per-person metrics và plots.

Script này không:

- retrain;
- tune;
- tạo split mới;
- chọn model lại.

Output được lưu trong:

[results/evaluation/](results/evaluation/)

## 14. Cài đặt môi trường

Tạo và activate virtual environment trên Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

Dependencies hiện tại trong [requirements.txt](requirements.txt):

- `ultralytics`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `joblib`
- `ipykernel`
- `pytest`
- `tabulate`

YOLO pose model cần có local tại:

```text
models/yolo26n-pose.pt
```

File `.pt` đang được ignore bởi Git.

## 15. Các lệnh chính

| Nhiệm vụ | Command |
|---|---|
| Build dataset | `python -m src.dataset_builder` |
| Validate preprocessing | `python -m src.preprocessing` |
| Chạy preprocessing tests | `python -m pytest tests/test_preprocessing.py -v` |
| Chạy realtime webcam prototype | `python scripts/test_webcam_model.py` |
| Chạy official held-out evaluation | `python -m src.evaluate` |

Các script webcam/debug thủ công:

```bash
python -m tests.test_pose_detector
python -m tests.test_feature_extractor
python scripts/copy_rejected_images.py
```

Hai file test pose/feature mở cửa sổ webcam OpenCV, nên chúng không phải unit test tự động thông thường.

## 16. Trạng thái project hiện tại

| Component | Status |
|---|---|
| Pose detection | Done |
| Feature extraction với 29 features | Done |
| Dataset builder | Done |
| Rejected-image audit script | Done |
| Dataset V02 | Done |
| EDA notebook | Done |
| Preprocessing | Done |
| Training experiment notebook | Done |
| Model selection | Done |
| Official held-out evaluation | Done |
| Realtime webcam prototype | Done |
| `src/inference.py` | Placeholder |
| `src/train.py` | Placeholder |
| `src/posture_predictor.py` | Placeholder |
| `src/temporal_monitor.py` | Placeholder |
| `src/session_statistics.py` | Placeholder |
| `app/app.py` | Placeholder |

## 17. Hướng phát triển tiếp theo

Hướng V02 product gần nhất:

```text
PosturePredictor
-> TemporalMonitor
-> SessionStatistics
-> Web/Application integration
-> stable V02 product
```

Hướng nghiên cứu V03 sau này:

- giảm subject-to-subject variation;
- phân tích confusion giữa `forward_slouch` và `lean_right`;
- cân nhắc scale/camera normalization;
- giữ nguyên locked evaluation protocol khi so sánh V02 và V03;
- đánh giá feature/model mới nhưng không tune trực tiếp trên held-out test set.

## 18. Tài liệu liên quan

- [docs/Cap_nhat_Dataset_V02_EDA_Preprocessing_2026-09-27.md](docs/Cap_nhat_Dataset_V02_EDA_Preprocessing_2026-09-27.md)
- [docs/Training_Realtime_Inference_Evaluation_V02_2026-09-27.md](docs/Training_Realtime_Inference_Evaluation_V02_2026-09-27.md)
- [docs/feature_v2_29_features_2026-09-25.md](docs/feature_v2_29_features_2026-09-25.md)
- [notebooks/01_eda_dataset.ipynb](notebooks/01_eda_dataset.ipynb)
- [notebooks/02_training_experiments.ipynb](notebooks/02_training_experiments.ipynb)
- [results/evaluation/evaluation_summary.md](results/evaluation/evaluation_summary.md)
