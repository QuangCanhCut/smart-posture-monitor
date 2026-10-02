# Smart Posture Monitor

Smart Posture Monitor là hệ thống nhận diện tư thế ngồi từ ảnh hoặc webcam bằng YOLO Pose, đặc trưng hình học thủ công và mô hình Machine Learning cổ điển.

Phiên bản hiện tại là **V03 — RAW12 + Personal Calibration + REP13**. Final FULL MODE đã hoàn tất trên dataset mới bằng **nested Leave-One-Person-Out (LOPO)**; model thắng là **SVM RBF**.

---

## Trạng thái hiện tại

- Geometry đầu vào: **RAW12**.
- Baseline cá nhân: **30 frame `correct` cho từng recording**.
- Representation cho model: **REP13**.
- Evaluation chính: **nested LOPO theo `person_id`**.
- Final cohort: **14 persons / 16 recordings**.
- Model tốt nhất: **SVM RBF**.
- LOPO Macro F1 mean: **0.919948**.
- OOF Macro F1: **0.936299**.
- Temporal smoothing chưa được tích hợp vào webcam demo.
- `src/train.py` và `src/evaluate.py` vẫn là pipeline cũ, **chưa phải entry point production của V03 hiện tại**.

---

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

### Research / Training

```text
data/processed/features.csv
  -> prepare_raw_dataset()
  -> validate 14 persons / 16 recordings
  -> baseline riêng cho từng recording_id
  -> loại 30 calibration frames khỏi tập ML
  -> RAW12 + baseline -> REP13
  -> outer LeaveOneGroupOut(person_id)
  -> inner StratifiedGroupKFold(n_splits=4)
  -> tune SVM / Random Forest / XGBoost / MLP
  -> OOF evaluation
  -> final grouped search + fit winner
```

Không còn exclusion `person08` / `person11` trong final retrain. Dataset mới sử dụng đầy đủ 14 người.

---

## Supported Postures

| Label | Ý nghĩa |
|---|---|
| `correct` | Tư thế ngồi đúng/upright tương đối. |
| `forward_slouch` | Cúi hoặc gù người về phía trước. |
| `lean_left` | Nghiêng người/đầu sang trái. |
| `lean_right` | Nghiêng người/đầu sang phải. |

---

## Camera Protocol

Dataset hiện chủ yếu được thu với camera đặt khoảng **45° từ bên trái người dùng**. Webcam demo không mirror frame vì mirror có thể đảo ý nghĩa `lean_left` và `lean_right`.

`src/pose_detector.py` dùng Ultralytics YOLO Pose và lấy 6 keypoints:

- `nose`
- `left_eye`
- `right_eye`
- `left_ear`
- `left_shoulder`
- `right_shoulder`

---

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

---

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

Final dataset có **16/16 recordings đủ 30 correct samples để calibration**.

---

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

### EDA signal sau calibration

Median normalized subject spread:

```text
RAW12  = 0.632149
REP13  = 0.418494
```

REP13 giảm đáng kể subject variation so với RAW12, hỗ trợ mục tiêu biểu diễn tư thế tương đối theo personal baseline.

Một số feature phân tách nổi bật trong EDA:

- `correct` vs `forward_slouch`: `delta_eye_center_y_body`
- `correct` vs `lean`: `head_drift_magnitude`
- `lean_left` vs `lean_right`: `delta_shoulder_center_x_body`

PCA 2D giải thích khoảng **52.715%** phương sai và chỉ được dùng cho mục đích chẩn đoán, không dùng làm input production.

---

## Dataset và Final Training Cohort

Dataset source của lần FINAL RETRAIN:

| Thống kê | Giá trị |
|---|---:|
| Total images | 4.814 |
| Valid samples | 4.456 |
| Rejected samples | 358 |
| Source persons | 14 |
| Source recordings | 16 |
| `correct` | 1.467 |
| `forward_slouch` | 937 |
| `lean_left` | 1.024 |
| `lean_right` | 1.028 |

Toàn bộ 358 rejected samples có lý do:

```text
low_keypoint_confidence
```

Sau calibration:

| Thống kê | Giá trị |
|---|---:|
| Persons dùng cho LOPO | 14 |
| Recordings | 16 |
| Calibration frames bị loại | 480 |
| REP13 samples được score | 3.976 |

Dataset trong `data/` được giữ local và ignore bởi Git.

### Recording structure

```text
person01__session01
person02__session01
person03__session01
person04__session01
person05__session01
person06__session01
person07__session01
person07__session02
person07__session03
person08__session01
person09__session01
person10__session01
person11__session01
person12__session01
person13__session01
person14__session01
```

`person07` có 3 recording nhưng cả 3 luôn thuộc cùng `person_id` group trong LOPO, nên không tạo subject leakage.

---

## FULL Nested LOPO Research

Notebook source of truth:

```text
notebooks/02_training_experiments.ipynb
```

Cấu hình final run:

```text
TRAINING_MODE = FULL
FAST_MODE = False
SEARCH_N_JOBS = 4
INNER_SPLITS = 4
RANDOM_STATE = 42

outer CV = LeaveOneGroupOut(person_id)
inner CV = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

selection metric = outer LOPO Macro F1 mean
```

Mỗi outer fold giữ toàn bộ recording của một người ở held-out side. Inner search chỉ thấy outer-train persons.

### Kết quả FULL MODE

| Model | LOPO Macro F1 Mean | Std | Worst-person Macro F1 | Accuracy Mean | Correct F1 |
|---|---:|---:|---:|---:|---:|
| **SVM RBF** | **0.919948** | **0.110044** | **0.671906** | **0.932851** | **0.938104** |
| Random Forest | 0.899258 | 0.137486 | 0.644433 | 0.915897 | 0.914313 |
| MLP | 0.899089 | 0.123974 | 0.618682 | 0.917051 | 0.939971 |
| XGBoost | 0.873577 | 0.146456 | 0.631538 | 0.894725 | 0.909948 |

Winner: **SVM RBF**

```text
C = 1
gamma = 0.01
class_weight = balanced
```

### Final SVM OOF Metrics

| Metric | Giá trị |
|---|---:|
| OOF Accuracy | 0.936871 |
| OOF Balanced Accuracy | 0.936982 |
| OOF Macro Precision | 0.937899 |
| OOF Macro Recall | 0.936982 |
| OOF Macro F1 | 0.936299 |

Lưu ý:

- **LOPO Macro F1 mean = 0.919948** là metric chính dùng để chọn model.
- **OOF Macro F1 = 0.936299** là Macro F1 tính trên toàn bộ OOF predictions gộp lại.

Hai metric này có ý nghĩa khác nhau và không nên dùng thay thế cho nhau.

---

## Per-Class Performance — Final SVM

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `correct` | 0.897317 | 0.982776 | 0.938104 | 987 |
| `forward_slouch` | 0.896050 | 0.919957 | 0.907846 | 937 |
| `lean_left` | 0.984064 | 0.964844 | 0.974359 | 1024 |
| `lean_right` | 0.974166 | 0.880350 | 0.924885 | 1028 |

### Confusion đáng chú ý

Largest off-diagonal error:

```text
actual lean_right -> predicted forward_slouch: 69 samples
```

Các confusion chính:

| Actual | Predicted | Count |
|---|---|---:|
| `lean_right` | `forward_slouch` | 69 |
| `lean_right` | `correct` | 52 |
| `forward_slouch` | `correct` | 48 |
| `lean_left` | `forward_slouch` | 25 |
| `forward_slouch` | `lean_right` | 18 |

`lean_left` là class ổn định nhất; `lean_right` hiện là class có recall thấp nhất trong final SVM.

---

## So sánh với artifact V03 trước

Artifact trước:

```text
SVM RBF LOPO Macro F1 mean = 0.909258
```

Final artifact:

```text
SVM RBF LOPO Macro F1 mean = 0.919948
```

Một số chênh lệch:

- LOPO Macro F1 mean: `+0.010690`
- Correct F1: khoảng `+0.028161`
- Worst-person Macro F1: khoảng `+0.029708`

Tuy nhiên đây **không phải A/B comparison hoàn toàn tương đương**, vì dataset/cohort final đã thay đổi và run cũ từng exclude một số person. Không nên diễn giải toàn bộ mức tăng là do một thay đổi duy nhất.

---

## Artifacts

Research model FULL MODE:

```text
models/best_model_v03_rep13.joblib
```

Compatibility artifact:

```text
models/best_model.joblib
```

Metadata/config:

```text
models/training_metadata.json
models/calibration_config.json
```

Artifact final đã được overwrite và smoke-tested với REP13 13 chiều.

Kết quả chi tiết:

```text
results/v03_lopo/
  calibration_audit.csv
  eda_dataset_summary.json
  lopo_model_summary.csv
  lopo_fold_metrics.csv
  lopo_oof_predictions.csv
  per_person_metrics.csv
  per_class_metrics.csv
  svm_search_results.csv
  random_forest_search_results.csv
  xgboost_search_results.csv
  mlp_search_results.csv
  raw_feature_statistics.csv
  rep13_feature_statistics.csv
  confusion_matrix_<model>.png
  confusion_matrix_normalized_<model>.png
```

---

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
|   `-- yolo26n-pose.pt
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

---

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

---

## EDA và Research

Chạy notebook theo thứ tự:

```text
1. notebooks/01_eda_dataset.ipynb
2. notebooks/02_training_experiments.ipynb
```

Notebook 02 hiện chạy FULL MODE theo protocol final. Đây là nested search đầy đủ và có thể tốn nhiều thời gian/CPU.

Không chạy:

```text
python -m src.train
```

để tái tạo model REP13 hiện tại vì `src/train.py` chưa được migrate sang final V03 protocol.

---

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

---

## Module Responsibilities

| Module | Vai trò |
|---|---|
| `src/pose_detector.py` | Frame → selected pose và 6 keypoints. |
| `src/feature_extractor.py` | Pose → RAW12. |
| `src/personal_calibration.py` | RAW12 correct samples → baseline RAW12. |
| `src/representation_builder.py` | RAW12 + baseline → REP13. |
| `src/posture_predictor.py` | REP13 → class label và probability. |
| `src/inference.py` | Điều phối realtime detection, calibration, representation và prediction. |
| `src/preprocessing.py` | Validate `features.csv`, tạo `X_raw/y/groups/metadata`. |
| `src/dataset_builder.py` | Raw images → metadata + RAW12 và rejected audit. |

---

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

Quy tắc:

- Không predict bằng RAW12 trước calibration.
- `NO_PERSON` không reset baseline.
- `LOW_CONFIDENCE` không được thêm vào calibration buffer.
- Baseline chỉ reset khi người dùng chủ động recalibrate.

---

## Validation

Final FULL run đã xác nhận:

- 14 persons / 16 recordings;
- 56 outer-fold model records: `14 persons × 4 models`;
- mỗi model có 3.976 OOF predictions;
- person overlap giữa train/test bằng 0;
- cả 3 recording của `person07` luôn cùng group;
- không calibration frame nào bị score;
- 16/16 recordings calibratable;
- model reload nhận đúng REP13 shape `(N, 13)`;
- final winner là SVM RBF.

Kiểm tra trực tiếp artifact:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model_v03_rep13.joblib `
  --check-only
```

Một số unit test cũ vẫn có thể mô tả API RAW29/DELTA29 và cần được migrate riêng; không dùng kết quả test legacy đó để kết luận về final V03 research pipeline.

---

## Current Limitations

- Camera protocol còn phụ thuộc góc đặt máy khoảng 45° bên trái.
- Calibration yêu cầu người dùng giữ tư thế `correct` ở đầu session.
- Số recording giữa các person chưa cân bằng; `person07` có 3 session.
- Worst-person Macro F1 hiện khoảng **0.6719**, cho thấy generalization giữa từng người vẫn còn biến thiên.
- `lean_right` có recall khoảng **0.8804**, thấp nhất trong 4 class của final SVM.
- Model dùng pose 2D, chưa có depth/3D.
- Frame-level prediction có thể jitter; `TemporalMonitor` chưa tích hợp vào webcam demo.
- Production `src/train.py`, evaluation entry point và app final chưa được migrate đầy đủ sang final REP13 protocol.

---

## Roadmap

1. Chốt và merge V03 Final Retrain vào `master`.
2. Viết lại `src/train.py` production theo RAW12 → Personal Calibration → REP13 và grouped protocol đã chốt.
3. Cập nhật evaluation entry point và canonical deployment workflow.
4. Kiểm thử webcam trên nhiều người và nhiều khoảng cách/camera setup hơn.
5. Mở rộng dataset theo hướng cân bằng số recording giữa các person.
6. Tích hợp `TemporalMonitor` và `SessionStatistics`.
7. Hoàn thiện UI, cảnh báo realtime và dashboard.

---

## Tài liệu liên quan

- [Báo cáo V03 RAW12/REP13/LOPO](docs/V03_RAW12_REP13_EDA_LOPO_Training_02-10-2026.md)
- [EDA notebook](notebooks/01_eda_dataset.ipynb)
- [FULL training notebook](notebooks/02_training_experiments.ipynb)
- [Model comparison](results/v03_lopo/lopo_model_summary.csv)
- [LOPO fold metrics](results/v03_lopo/lopo_fold_metrics.csv)
- [Per-class metrics](results/v03_lopo/per_class_metrics.csv)
- [Per-person metrics](results/v03_lopo/per_person_metrics.csv)
