# Smart Posture Monitor

Smart Posture Monitor là hệ thống nhận diện tư thế ngồi từ ảnh hoặc webcam bằng YOLO Pose, đặc trưng hình học thủ công và mô hình Machine Learning cổ điển.

Phiên bản hiện tại là **V03 — RAW12 + Personal Calibration + REP13**. Research FULL MODE đã hoàn tất bằng nested Leave-One-Person-Out (LOPO); model thắng là **SVM RBF**.

## Trạng thái hiện tại

- Geometry đầu vào: **RAW12**.
- Baseline cá nhân: 30 frame `correct` cho từng recording.
- Representation cho model: **REP13**.
- Evaluation chính: nested LOPO theo `person_id`.
- Model tốt nhất: SVM RBF.
- Temporal smoothing chưa được tích hợp vào webcam demo.
- `src/train.py` và `src/evaluate.py` vẫn là pipeline cũ, **chưa phải entry point production của V03 hiện tại**.

## Pipeline

### Realtime

```text
Webcam frame
  -> PoseDetector
  -> 6 upper-body keypoints
  -> FeatureExtractor
  -> RAW12
  -> PersonalCalibration (baseline RAW12)
  -> RepresentationBuilder
  -> REP13
  -> PosturePredictor
  -> SVM RBF
  -> frame-level posture prediction
```

### Research/training

```text
data/processed/features.csv
  -> prepare_raw_dataset()
  -> exclude person08, person11 theo protocol audit
  -> baseline riêng cho từng recording_id
  -> loại 30 calibration frames khỏi tập ML
  -> RAW12 + baseline -> REP13
  -> outer LeaveOneGroupOut(person_id)
  -> inner StratifiedGroupKFold(n_splits=4)
  -> tune SVM / Random Forest / XGBoost / MLP
  -> OOF evaluation
  -> final grouped search + fit winner
```

## Supported Postures

| Label | Ý nghĩa |
|---|---|
| `correct` | Tư thế ngồi đúng/upright tương đối. |
| `forward_slouch` | Cúi hoặc gù người về phía trước. |
| `lean_left` | Nghiêng người/đầu sang trái. |
| `lean_right` | Nghiêng người/đầu sang phải. |

## Camera Protocol

Dataset hiện chủ yếu được thu với camera đặt khoảng **45° từ bên trái người dùng**. Webcam demo không mirror frame vì mirror có thể đảo ý nghĩa `lean_left` và `lean_right`.

`src/pose_detector.py` dùng Ultralytics YOLO Pose và lấy 6 keypoints:

- `nose`
- `left_eye`
- `right_eye`
- `left_ear`
- `left_shoulder`
- `right_shoulder`

## RAW12

`src/feature_extractor.py` chuyển pose thành 12 feature. Schema chính thức luôn lấy từ:

```python
FeatureExtractor.FEATURE_NAMES
```

| Nhóm | Feature |
|---|---|
| Angle/orientation | `shoulder_roll_deg`, `eye_roll_deg`, `head_shoulder_roll_diff_deg` |
| Neck | `neck_pitch_deg` |
| Spatial | `eye_center_x_px`, `eye_center_y_px`, `shoulder_center_x_px`, `eye_shoulder_vertical_gap_px`, `eye_shoulder_horizontal_offset_px` |
| Perspective/depth proxy | `inter_eye_distance_px`, `ear_nose_horizontal_span_px`, `shoulder_width_px` |

RAW12 chỉ mô tả hình học frame hiện tại; không chứa personal delta và không normalize bằng shoulder width hiện tại.

## Personal Calibration

Mỗi recording có baseline riêng:

```text
recording_id = person_id + "__" + session_id
CALIBRATION_SAMPLES = 30
```

Protocol offline/realtime dùng chung `PersonalCalibration`:

1. Thu các RAW12 sample ở tư thế `correct`.
2. Sắp theo `frame_index` trong offline research.
3. Lấy đúng 30 sample hợp lệ đầu tiên.
4. Non-angle feature dùng median.
5. Angle feature dùng circular median-like để xử lý đúng biên ±180°.
6. Calibration frames chỉ tạo baseline, không được dùng để train hoặc score.

## REP13

`src/representation_builder.py` biến đổi `RAW12 + baseline RAW12` thành REP13:

- 4 wrapped angle deltas;
- 5 spatial deltas chia baseline shoulder width;
- 3 scale log-ratios;
- `head_drift_magnitude`.

Schema chính thức:

```python
RepresentationBuilder.OUTPUT_FEATURE_NAMES
```

Model chỉ nhận REP13, không học trực tiếp từ RAW12.

## Dataset và training cohort

Dataset source tại lần FULL research gần nhất:

| Thống kê | Giá trị |
|---|---:|
| Valid samples | 4.014 |
| Source persons | 14 |
| Source recordings | 16 |
| `correct` | 1.051 |
| `forward_slouch` | 889 |
| `lean_left` | 1.033 |
| `lean_right` | 1.041 |

Theo audit chất lượng, `person08` và `person11` đang được loại tạm khỏi research cohort. Sau exclusion và calibration:

| Thống kê | Giá trị |
|---|---:|
| Persons dùng cho LOPO | 12 |
| Recordings | 14 |
| RAW samples sau exclusion | 3.530 |
| Calibration frames bị loại | 420 |
| REP13 samples được score | 3.110 |

Dataset trong `data/` được giữ local và ignore bởi Git.

## FULL Nested LOPO Research

Notebook source of truth:

```text
notebooks/02_training_experiments.ipynb
```

Cấu hình lần chạy hiện tại:

```text
FAST_MODE = False
SEARCH_N_JOBS = 4
INNER_SPLITS = 4
RANDOM_STATE = 42
outer CV = LeaveOneGroupOut(person_id)
inner CV = StratifiedGroupKFold(shuffle=True, random_state=42)
selection metric = Macro F1
```

Mỗi outer fold giữ toàn bộ recording của một người ở held-out side. Inner search chỉ thấy outer-train persons.

### Kết quả FULL MODE

| Model | LOPO Macro F1 mean | Std | Worst-person Macro F1 | Accuracy mean | Correct F1 |
|---|---:|---:|---:|---:|---:|
| **SVM RBF** | **0.909258** | **0.122354** | 0.642198 | **0.918690** | **0.909944** |
| MLP | 0.869364 | 0.124294 | 0.664921 | 0.885393 | 0.897436 |
| XGBoost | 0.848723 | 0.134024 | 0.664921 | 0.866371 | 0.866184 |
| Random Forest | 0.842999 | 0.159632 | 0.601659 | 0.864948 | 0.861818 |

Winner: **SVM RBF**

```text
C = 0.1
gamma = 0.1
class_weight = None
```

Điểm trên là outer OOF LOPO, không phải training accuracy. Worst fold của winner là `person02`, Macro F1 `0.642198`.

## Artifacts

Research model FULL MODE:

```text
models/best_model_v03_rep13.joblib
```

Metadata/config:

```text
models/training_metadata.json
models/calibration_config.json
```

`models/best_model.joblib` được giữ để tương thích với một số lệnh cũ. Artifact research FULL MODE hiện tại là `best_model_v03_rep13.joblib`.

Kết quả chi tiết:

```text
results/v03_lopo/
  lopo_model_summary.csv
  lopo_fold_metrics.csv
  lopo_oof_predictions.csv
  per_person_metrics.csv
  per_class_metrics.csv
  svm_search_results.csv
  random_forest_search_results.csv
  xgboost_search_results.csv
  mlp_search_results.csv
  confusion_matrix_<model>.png
  confusion_matrix_normalized_<model>.png
```

## Project Structure

```text
smart_posture_monitor/
|-- app/
|-- data/                              # Local dataset, ignored by Git
|-- docs/
|   `-- V03_RAW12_REP13_EDA_LOPO_Training_02-10-2026.md
|-- models/
|   |-- best_model.joblib             # Compatibility artifact
|   |-- best_model_v03_rep13.joblib   # Current FULL research winner
|   |-- calibration_config.json
|   |-- training_metadata.json
|   `-- yolo26n-pose.pt               # Local YOLO weights, ignored by Git
|-- notebooks/
|   |-- 01_eda_dataset.ipynb
|   `-- 02_training_experiments.ipynb
|-- results/
|   `-- v03_lopo/
|-- scripts/
|   |-- copy_rejected_images.py
|   `-- test_webcam_model.py
|-- src/
|   |-- dataset_builder.py
|   |-- feature_extractor.py
|   |-- inference.py
|   |-- personal_calibration.py
|   |-- pose_detector.py
|   |-- posture_predictor.py
|   |-- preprocessing.py
|   |-- representation_builder.py
|   |-- session_statistics.py
|   |-- temporal_monitor.py
|   `-- train.py                       # Legacy; chưa migrate sang final V03
|-- tests/
|-- requirements.txt
`-- README.md
```

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

YOLO pose weights cần tồn tại local:

```text
models/yolo26n-pose.pt
```

## EDA và Research

Chạy notebook theo thứ tự:

```text
1. notebooks/01_eda_dataset.ipynb
2. notebooks/02_training_experiments.ipynb
```

Notebook 02 hiện chạy FULL MODE mặc định. Đây là nested search đầy đủ và có thể tốn nhiều thời gian/CPU.

Không chạy `python -m src.train` để tái tạo model REP13 hiện tại: file này chưa được migrate sau khi research được chốt.

## Realtime Test

Kiểm tra model/metadata không cần webcam:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model_v03_rep13.joblib `
  --check-only
```

Chạy webcam:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model_v03_rep13.joblib `
  --camera 0
```

Tùy chọn:

```text
--no-pose
--calibration-samples 30
```

Controls:

```text
C      bắt đầu calibration mới
R      reset và calibration lại
Q/ESC  thoát
```

Khi calibration, người dùng cần ngồi ở tư thế `correct` tự nhiên cho đến khi đủ 30 RAW12 sample hợp lệ.

## Module Responsibilities

| Module | Vai trò |
|---|---|
| `src/pose_detector.py` | Frame → selected pose và 6 keypoints. |
| `src/feature_extractor.py` | Pose → RAW12. |
| `src/personal_calibration.py` | RAW12 correct samples → baseline RAW12. |
| `src/representation_builder.py` | RAW12 + baseline → REP13. |
| `src/posture_predictor.py` | REP13 → class label và probability. |
| `src/inference.py` | Điều phối realtime detection, calibration, representation và prediction. |
| `src/preprocessing.py` | Validate `features.csv`, tạo X_raw/y/groups/metadata. |
| `src/dataset_builder.py` | Raw images → metadata + RAW12 và rejected audit. |

## Realtime States

`PostureInferenceEngine` trả về:

```text
CALIBRATION_REQUIRED
CALIBRATING
CALIBRATED
LOW_CONFIDENCE
NO_PERSON
OK
```

- Không predict bằng RAW12 trước calibration.
- `NO_PERSON` không reset baseline.
- `LOW_CONFIDENCE` không được thêm vào calibration buffer.
- Baseline chỉ reset khi người dùng chủ động recalibrate.

## Validation

Kiểm tra trực tiếp artifact hiện tại:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model_v03_rep13.joblib `
  --check-only
```

Notebook FULL run gần nhất đã xác nhận:

- 48 outer-fold records: 12 persons × 4 models;
- mỗi model có 3.110 OOF predictions;
- person overlap bằng 0;
- không calibration frame nào bị score;
- model reload nhận đúng REP13 shape `(N, 13)`.

Một số unit test cũ vẫn mô tả API RAW29/DELTA29 và cần được migrate riêng; không dùng kết quả test legacy đó để kết luận về research notebook hiện tại.

## Current Limitations

- Camera protocol còn phụ thuộc góc đặt máy khoảng 45° bên trái.
- Calibration yêu cầu người dùng giữ tư thế correct ở đầu session.
- `person08` và `person11` đang bị exclude, cần làm sạch/thu lại dữ liệu trước khi đưa trở lại research cohort.
- Worst-person variance vẫn đáng kể dù mean LOPO cao.
- Model dùng pose 2D, chưa có depth/3D.
- Frame-level prediction có thể jitter; TemporalMonitor chưa tích hợp vào webcam demo.
- Production `src/train.py`, evaluation entry point và app final chưa được migrate sang REP13.

## Roadmap

1. Review/chốt kết quả FULL nested LOPO.
2. Viết lại `src/train.py` production theo RAW12 → REP13 và grouped protocol đã chốt.
3. Cập nhật evaluation entry point và canonical deployment artifact.
4. Thu lại hoặc sửa dữ liệu `person08`, `person11`, sau đó rerun FULL research.
5. Kiểm thử webcam trên nhiều người/camera setup.
6. Tích hợp TemporalMonitor và SessionStatistics.
7. Hoàn thiện UI và cảnh báo realtime.

## Tài liệu liên quan

- [Báo cáo V03 RAW12/REP13/LOPO](docs/V03_RAW12_REP13_EDA_LOPO_Training_02-10-2026.md)
- [EDA notebook](notebooks/01_eda_dataset.ipynb)
- [FULL training notebook](notebooks/02_training_experiments.ipynb)
- [Model comparison](results/v03_lopo/lopo_model_summary.csv)
- [LOPO fold metrics](results/v03_lopo/lopo_fold_metrics.csv)
