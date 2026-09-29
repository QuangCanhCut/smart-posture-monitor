# Smart Posture Monitor V03

## Báo cáo công việc ngày 30/09/2026

Tài liệu này ghi lại mốc hoàn thiện V03 của Smart Posture Monitor: chuyển representation sang Personal Baseline Delta, chốt model XGBoost, chuẩn hóa `train.py`, cập nhật held-out evaluation và refactor realtime inference theo trách nhiệm rõ ràng hơn.

## 1. Mục tiêu của V03

Pipeline V02 dùng trực tiếp 29 RAW engineered features. Cách này có một số rủi ro khi số subject còn ít:

- khác biệt hình thể giữa người dùng làm absolute geometry thay đổi;
- camera/session offsets làm cùng một posture có tọa độ/góc khác nhau;
- vị trí ghế, khoảng cách camera và setup ánh sáng làm feature raw khó tổng quát;
- random frame split không phù hợp vì nhiều frame cùng person có tương quan mạnh.

V03 thử hướng **Personal Baseline Delta**: mỗi người/session có một baseline tư thế correct riêng. Model học độ lệch so với correct posture cá nhân thay vì học geometry tuyệt đối.

## 2. Thay đổi representation

Representation mới:

```text
RAW[29] -> DELTA[29]
```

Với mỗi `recording_id`, lấy `K = 30` sample `correct` đầu tiên làm calibration window:

```text
baseline_j = median(f_1j, f_2j, ..., f_Kj)
```

Với sample còn lại:

```text
delta_t = raw_t - baseline
```

Baseline không làm thay đổi số chiều:

```text
29 RAW features -> 29 DELTA features
```

Calibration samples bị loại khỏi ML dataset sau khi tạo baseline, vì chính chúng đã được dùng để tính baseline. Nếu đưa lại vào train/score, class `correct` có thể có delta gần 0 bất thường và làm metric đẹp giả.

## 3. Training protocol

Protocol V03:

```text
RAW dataset
-> person-based holdout split
-> train persons / test persons
-> Personal Baseline Delta per recording_id
-> Group CV trên train persons
-> model comparison / tuning trong notebook
-> train.py tái tạo final winner
```

Các nguyên tắc chống leakage:

- split train/test theo `person_id`;
- audit person overlap và recording overlap phải rỗng;
- calibration theo `recording_id`, không theo `session_id`;
- tạo delta sau khi split, không tạo delta toàn dataset trước split;
- validation person trong Group CV được phép tự có baseline riêng, giống inference thật;
- calibration samples không xuất hiện trong training/validation scoring;
- scaler/resampling nếu có chỉ được nằm trong CV pipeline, nhưng final winner không cần scaler/resampling.

## 4. Imbalance experiments

Sau calibration, class `correct` giảm vì mỗi recording mất 30 frame correct. Notebook đã benchmark imbalance strategies trên V03 representation.

| Strategy | Model | CV Macro F1 | Std |
|---|---|---:|---:|
| none | SVM RBF | 0.663675 | 0.186527 |
| class_weight | SVM RBF | 0.661674 | 0.178407 |
| smote | SVM RBF | 0.661379 | 0.172660 |
| random_over | SVM RBF | 0.657347 | 0.173149 |

Kết luận:

- không có bằng chứng cho thấy balancing cải thiện CV Macro F1 trong experiment hiện tại;
- final imbalance strategy là `none`;
- final training pipeline không dùng SMOTE, RandomOverSampler hoặc class_weight.

Điều này không có nghĩa SMOTE luôn xấu; chỉ nói rằng trong setup V03 hiện tại nó không cải thiện kết quả CV.

## 5. Model experiments

Notebook `notebooks/02_training_experiments.ipynb` là nơi research:

- baseline models;
- imbalance benchmark;
- SVM GridSearchCV;
- XGBoost RandomizedSearchCV;
- model selection bằng CV Macro F1.

Final winner:

```text
Representation: Personal Baseline Delta
Classifier: XGBoost Tuned
Imbalance strategy: none
CV Macro F1: 0.688936 +/- 0.140001
```

Best parameters:

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

Official held-out evaluation đã chạy riêng bằng `src.evaluate` trên locked unseen-person holdout:

| Metric | Value |
|---|---:|
| Accuracy | 0.632091 |
| Balanced accuracy | 0.702947 |
| Macro precision | 0.688106 |
| Macro recall | 0.702947 |
| Macro F1 | 0.644193 |
| Weighted F1 | 0.624098 |

Held-out persons:

```text
person01, person10, person12
```

Holdout result không được dùng để chọn model.

## 6. `src/train.py`

`src/train.py` là production training entry point. Nó không phải notebook experiment thứ hai.

Trách nhiệm:

- load dataset qua `prepare_dataset()`;
- verify feature count = 29;
- reuse `models/split_manifest.json` nếu dataset SHA256/person/schema còn khớp;
- nếu manifest không hợp lệ thì tạo deterministic `GroupShuffleSplit`;
- tạo Personal Baseline Delta cho train recordings;
- remove calibration samples;
- fit final XGBoost với params đã chốt;
- save artifacts;
- reload smoke test trên train-delta samples.

Output:

```text
models/best_model.joblib
models/split_manifest.json
models/training_metadata.json
models/calibration_config.json
results/calibration_audit.csv
```

Safety checks chính:

- dataset tồn tại;
- feature count = 29;
- không person/recording leakage;
- mỗi recording đủ 30 correct samples;
- delta không NaN/Inf;
- đủ 4 classes sau calibration removal;
- saved model reload được và predict được.

## 7. `src/evaluate.py`

`src/evaluate.py` là official held-out evaluation.

Nó không:

- retrain;
- retune;
- reselect model;
- create new split.

Luồng:

```text
load artifacts
-> verify dataset SHA256
-> prepare_dataset()
-> reconstruct locked test persons
-> build Personal Baseline Delta cho holdout recordings
-> predict
-> save reports
```

Output trong `results/evaluation/`:

- `test_metrics.json`;
- `classification_report.csv`;
- `confusion_matrix.csv`;
- `confusion_matrix.png`;
- `confusion_matrix_normalized.csv`;
- `confusion_matrix_normalized.png`;
- `per_person_metrics.csv`;
- `per_person_macro_f1.png`;
- `test_predictions.csv`;
- `misclassifications.csv`;
- `test_calibration_audit.csv`;
- `evaluation_summary.md`.

Điểm quan trọng: nếu model metadata nói representation là `personal_baseline_delta`, evaluate phải rebuild delta features cho test recordings. Không được predict trực tiếp bằng RAW features.

## 8. `src/posture_predictor.py`

Refactor trách nhiệm:

OLD:

```text
frame
-> predictor
   -> YOLO
   -> extractor
   -> classifier
```

NEW:

```text
DELTA[29]
-> PosturePredictor
-> trained classifier
-> class label + probability
```

`PosturePredictor` hiện chỉ:

- load model và metadata;
- kiểm tra feature contract;
- nhận vector DELTA 29 chiều;
- trả `raw_label`, `label_id`, `probability`.

Nó không detect người, không extract keypoints, không feature engineering, không baseline calculation và không temporal smoothing. Đây là tách trách nhiệm theo Single Responsibility.

## 9. `src/inference.py`

`PostureInferenceEngine` là orchestrator frame-level realtime.

State machine:

```text
CALIBRATION_REQUIRED
    -> user nhấn C
CALIBRATING 1/30 ... 30/30
    -> baseline median
CALIBRATED
    -> RAW -> DELTA
OK prediction
```

Các status quan trọng:

```text
CALIBRATION_REQUIRED
CALIBRATING
CALIBRATED
LOW_CONFIDENCE
NO_PERSON
OK
```

Behavior:

- frame `NO_PERSON` không reset baseline;
- frame `LOW_CONFIDENCE` không được thêm vào calibration;
- baseline chỉ reset khi user chủ động recalibrate;
- trước calibration, engine không đưa RAW features vào model.

## 10. `scripts/test_webcam_model.py`

Script này là realtime integration/smoke test.

Mục tiêu:

- test webcam;
- test pose detector;
- test feature extraction;
- test Personal Baseline;
- test RAW -> DELTA;
- test XGBoost realtime;
- xem raw frame-level prediction.

Controls:

```text
C      bắt đầu calibration
R      reset và calibration lại
Q/ESC  thoát
```

Script không mirror frame để tránh đảo nghĩa `lean_left` và `lean_right`.

Hiện tại script chưa dùng `TemporalMonitor`. Điều này là cố ý: mục tiêu hiện tại là quan sát raw prediction trước khi thêm smoothing.

## 11. Kiến trúc tổng thể V03

```text
Webcam
   |
   v
PoseDetector
   |
   v
6 keypoints
   |
   v
FeatureExtractor
   |
   v
RAW[29]
   |
   v
PersonalCalibration
   |
   v
DELTA[29]
   |
   v
PosturePredictor
   |
   v
XGBoost
   |
   v
frame prediction
   |
   v
[future]
TemporalMonitor
   |
   v
SessionStatistics
   |
   v
App / warning / report
```

## 12. Những gì chưa hoàn thành

Các phần chưa nên mô tả là production-complete:

- TemporalMonitor chưa được tích hợp vào webcam V03 test;
- SessionStatistics chưa tích hợp end-to-end;
- alert/warning logic chưa hoàn thiện;
- Streamlit/app final integration chưa hoàn thiện;
- chưa có realtime acceptance test trên nhiều user/setup;
- chưa đánh giá camera robustness ở nhiều góc/khoảng cách khác nhau;
- chưa có packaging/deployment production.

## 13. Bước tiếp theo

1. Chạy realtime webcam integration với model V03.
2. Test đủ 4 posture sau calibration.
3. Test recalibration trong cùng session.
4. Ghi nhận prediction jitter.
5. Tích hợp `TemporalMonitor`.
6. Tích hợp `SessionStatistics`.
7. Hoàn thiện warning/alert logic.
8. Hoàn thiện Streamlit app.
9. Viết báo cáo BTL cuối cùng.

## 14. Validation đã chạy

Các lệnh đã chạy trong mốc V03:

```text
python -m src.train
python -m src.evaluate
python -m pytest -q
```

Kết quả test gần nhất:

```text
40 passed, 2 skipped
```

Webcam hardware test chưa chạy trong môi trường Codex vì không có camera thật.
