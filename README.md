# Smart Posture Monitor

Smart Posture Monitor là project nhận diện tư thế ngồi từ ảnh hoặc webcam bằng YOLO Pose, feature hình học thủ công, Personal Baseline Delta và mô hình Machine Learning cổ điển.

Phiên bản hiện tại là **V03**. Điểm thay đổi chính so với pipeline raw-feature trước đây là model không học trực tiếp trên 29 đặc trưng tuyệt đối nữa, mà học trên độ lệch so với baseline cá nhân của từng người dùng/session.

## Overview

Pipeline realtime V03:

```text
Webcam frame
-> PoseDetector
-> 6 upper-body keypoints
-> FeatureExtractor
-> 29 RAW engineered features
-> PersonalCalibration
-> 29 DELTA features
-> PosturePredictor
-> XGBoost
-> frame-level posture prediction
```

Pipeline offline training:

```text
data/processed/features.csv
-> prepare_dataset()
-> person-based holdout split
-> Personal Baseline Delta per recording_id
-> train final XGBoost
-> models/best_model.joblib
```

## Current Version

```text
V03 - Personal Baseline Delta
```

V03 giải quyết một phần khác biệt hình thể, vị trí camera và setup giữa các người dùng bằng cách chuẩn hóa mỗi sample theo baseline correct của chính recording đó.

## Supported Postures

| Label | Ý nghĩa |
|---|---|
| `correct` | Tư thế ngồi đúng/upright tương đối. |
| `forward_slouch` | Cúi hoặc gù người về phía trước. |
| `lean_left` | Nghiêng người/đầu sang trái. |
| `lean_right` | Nghiêng người/đầu sang phải. |

## Camera Protocol

Dataset và realtime test hiện chủ yếu theo protocol:

```text
camera đặt khoảng 45 độ từ bên trái người dùng
```

Webcam test **không mirror frame**, vì mirror có thể đảo nghĩa `lean_left` và `lean_right`.

## Pose Keypoints

`src/pose_detector.py` dùng Ultralytics YOLO Pose và chỉ truyền 6 keypoints phần thân trên vào feature pipeline:

- `nose`
- `left_eye`
- `right_eye`
- `left_ear`
- `left_shoulder`
- `right_shoulder`

Nếu phát hiện nhiều người, detector chọn người phù hợp dựa trên điểm kết hợp giữa vị trí gần trung tâm và confidence.

## Feature Engineering

`src/feature_extractor.py` chuyển pose result thành vector 29 chiều. Danh sách chính thức nằm trong:

```python
FeatureExtractor.FEATURE_NAMES
```

Các nhóm feature chính:

- góc vai và góc đường mắt;
- tọa độ nose/eye/ear trong body frame;
- tỉ lệ khoảng cách đã chuẩn hóa theo shoulder width;
- độ bất đối xứng giữa hai vai;
- các proxy về head/body/gravity angle;
- offset so với trục dọc;
- thống kê height/spread của head landmarks.

FeatureExtractor chỉ tạo **RAW[29]**. Nó không calibration và không predict.

## Personal Baseline Delta

V03 dùng calibration cá nhân:

```text
K = 30 correct samples
baseline = median(correct_samples, axis=0)
delta = raw - baseline
```

Số chiều không đổi:

```text
29 RAW features -> 29 DELTA features
```

Trong realtime, user cần ngồi ở tư thế correct tự nhiên lúc bắt đầu session để hệ thống thu 30 vector RAW hợp lệ. Baseline chỉ reset khi user chủ động recalibrate.

Trong training/evaluation offline, baseline được tạo theo `recording_id`, không theo `session_id`, vì `session_id` có thể trùng giữa nhiều person.

## Model

Final model V03:

```text
Representation: Personal Baseline Delta
Classifier: XGBoost Tuned
Imbalance strategy: none
```

Best hyperparameters:

```text
n_estimators=200
max_depth=6
learning_rate=0.02
subsample=1.0
colsample_bytree=1.0
min_child_weight=1
gamma=0
reg_alpha=0
reg_lambda=5
```

Reference result từ Group/person-based CV trong notebook experiment:

```text
CV Macro F1 = 0.688936 +/- 0.140001
```

Đây là kết quả cross-validation theo person trên training persons, không phải realtime accuracy.

Official held-out evaluation hiện tại được chạy riêng bằng `src.evaluate` trên locked unseen-person holdout:

```text
Test persons: person01, person10, person12
Test samples sau calibration removal: 617
Macro F1: 0.644193
Accuracy: 0.632091
Balanced accuracy: 0.702947
```

Holdout test không được dùng để tune hoặc chọn model.

## Imbalance Experiment

Sau khi calibration loại 30 correct samples mỗi recording, class `correct` giảm đáng kể. Notebook đã benchmark imbalance strategy trên cùng representation V03:

| Strategy | Model | CV Macro F1 | Std |
|---|---|---:|---:|
| `none` | SVM RBF | 0.663675 | 0.186527 |
| `class_weight` | SVM RBF | 0.661674 | 0.178407 |
| `smote` | SVM RBF | 0.661379 | 0.172660 |
| `random_over` | SVM RBF | 0.657347 | 0.173149 |

Kết luận cho experiment hiện tại: không có bằng chứng cho thấy balancing cải thiện CV Macro F1, nên final training dùng `none`. Điều này không có nghĩa SMOTE luôn xấu; chỉ là trong thí nghiệm hiện tại nó không cải thiện representation V03.

## Project Structure

```text
smart_posture_monitor/
|-- app/
|   `-- app.py
|-- data/                         # Dataset/runtime data local, ignored by Git
|-- docs/                         # Báo cáo kỹ thuật và ghi chú tiến độ
|-- models/
|   |-- best_model.joblib          # Final classifier artifact nếu được tracking/local
|   |-- calibration_config.json
|   |-- split_manifest.json
|   |-- training_metadata.json
|   `-- yolo26n-pose.pt            # YOLO pose model local, ignored by Git
|-- notebooks/
|   |-- 01_eda_dataset.ipynb
|   `-- 02_training_experiments.ipynb
|-- results/
|   |-- calibration_audit.csv
|   `-- evaluation/
|-- scripts/
|   |-- copy_rejected_images.py
|   `-- test_webcam_model.py
|-- src/
|   |-- dataset_builder.py
|   |-- evaluate.py
|   |-- feature_extractor.py
|   |-- inference.py
|   |-- personal_calibration.py
|   |-- pose_detector.py
|   |-- posture_predictor.py
|   |-- preprocessing.py
|   |-- session_statistics.py
|   |-- temporal_monitor.py
|   `-- train.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Installation

Tạo virtual environment trên Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

YOLO pose weight cần có local:

```text
models/yolo26n-pose.pt
```

File `.pt`, dataset trong `data/`, cache và `.venv` được ignore theo `.gitignore`.

## Training

Chạy final production training:

```powershell
python -m src.train
```

`src.train` làm các bước:

- load dataset qua `src.preprocessing.prepare_dataset()`;
- reuse locked person split nếu `models/split_manifest.json` còn khớp dataset SHA256;
- nếu split manifest không hợp lệ thì tạo deterministic `GroupShuffleSplit`;
- build Personal Baseline Delta trên train recordings;
- remove calibration samples khỏi ML dataset;
- fit final XGBoost với params đã chốt;
- save artifacts.

Artifacts chính:

```text
models/best_model.joblib
models/split_manifest.json
models/training_metadata.json
models/calibration_config.json
results/calibration_audit.csv
```

`train.py` không chạy model selection lại và không dùng holdout test để predict/score.

## Evaluation

Chạy official held-out evaluation:

```powershell
python -m src.evaluate
```

`src.evaluate`:

- load model và metadata đã train;
- verify dataset SHA256;
- reconstruct locked unseen-person holdout từ `models/split_manifest.json`;
- nếu representation là `personal_baseline_delta`, rebuild delta cho holdout recordings;
- predict bằng model đã lưu;
- lưu metrics, classification report, confusion matrix, per-person metrics và misclassifications.

Output:

```text
results/evaluation/
```

Evaluation không retrain, không retune, không tạo split mới và không chọn model lại.

## Realtime Test

Chạy webcam integration test:

```powershell
python scripts/test_webcam_model.py
```

Tùy chọn:

```powershell
python scripts/test_webcam_model.py --camera 0
python scripts/test_webcam_model.py --no-pose
python scripts/test_webcam_model.py --calibration-samples 30
```

Controls:

```text
C      bắt đầu calibration mới
R      reset và calibration lại
Q/ESC  thoát
```

Script này test pipeline realtime V03 ở mức frame-level:

```text
Webcam
-> PostureInferenceEngine
-> PoseDetector
-> FeatureExtractor
-> PersonalCalibration
-> PosturePredictor
-> XGBoost prediction
```

Hiện script chưa dùng TemporalMonitor; mục tiêu hiện tại là quan sát raw prediction trước khi thêm smoothing.

## Module Responsibilities

| Module | Vai trò |
|---|---|
| `src/pose_detector.py` | Frame -> selected pose / 6 keypoints. |
| `src/feature_extractor.py` | Pose -> RAW[29]. |
| `src/personal_calibration.py` | Validate vector, collect calibration samples, median baseline, RAW -> DELTA, batch delta dataset. |
| `src/posture_predictor.py` | DELTA[29] -> class label + probability. |
| `src/inference.py` | Orchestrate frame-level realtime pipeline and calibration states. |
| `src/train.py` | Reproducible final V03 training. |
| `src/evaluate.py` | Official locked holdout evaluation. |
| `src/temporal_monitor.py` | Post-prediction smoothing/yaw gating module, not yet integrated into V03 webcam test. |
| `src/session_statistics.py` | Session-level statistics utility, future integration target. |

## Realtime States

`PostureInferenceEngine` trả về các state chính:

```text
CALIBRATION_REQUIRED
CALIBRATING
CALIBRATED
LOW_CONFIDENCE
NO_PERSON
OK
```

Important behavior:

- Không predict bằng RAW features trước calibration.
- `NO_PERSON` không reset baseline.
- `LOW_CONFIDENCE` không được thêm vào calibration buffer.
- Baseline chỉ reset khi user bấm recalibrate.

## Validation Commands

```powershell
python -m compileall src scripts
python -m pytest -q
python -m src.train --help
python -m src.evaluate --help
python scripts/test_webcam_model.py --help
```

Webcam hardware test cần camera thật nên không bắt buộc chạy trong môi trường CI/Codex.

## Current Limitations

- Camera protocol còn tương đối cố định, chủ yếu theo góc khoảng 45 độ từ bên trái.
- Personal Calibration yêu cầu user ngồi correct lúc bắt đầu session.
- Model dựa trên 2D pose, chưa dùng depth/3D.
- Raw frame prediction có thể jitter.
- TemporalMonitor đã tồn tại nhưng chưa được tích hợp vào webcam V03 test.
- SessionStatistics và alert logic chưa hoàn thiện end-to-end.
- Streamlit/app final integration vẫn là bước tiếp theo.
- Chưa có realtime acceptance test trên nhiều người dùng/setup khác nhau.

## Roadmap

1. Chạy realtime webcam integration trên nhiều người dùng hơn.
2. Ghi nhận prediction jitter sau calibration.
3. Tích hợp TemporalMonitor vào realtime pipeline.
4. Tích hợp SessionStatistics.
5. Thêm warning/alert logic.
6. Hoàn thiện Streamlit UI trong `app/`.
7. Kiểm thử robustness với camera distance/angle khác nhau.
8. Viết báo cáo BTL cuối cùng.

## Related Documentation

- [docs/V03_Personal_Baseline_Training_Evaluation_Realtime_30-09-2026.md](docs/V03_Personal_Baseline_Training_Evaluation_Realtime_30-09-2026.md)
- [docs/v03_personal_baseline_calibration_2026-09-29.md](docs/v03_personal_baseline_calibration_2026-09-29.md)
- [docs/feature_v2_29_features_2026-09-25.md](docs/feature_v2_29_features_2026-09-25.md)
- [notebooks/02_training_experiments.ipynb](notebooks/02_training_experiments.ipynb)
- [results/evaluation/evaluation_summary.md](results/evaluation/evaluation_summary.md)
