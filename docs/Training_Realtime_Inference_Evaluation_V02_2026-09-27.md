# Training, Realtime Inference và Evaluation V02 - 2026-09-27

## 1. Tổng quan công việc

Phiên làm việc này hoàn thiện các phần chính của pipeline V02:

- Training experiment cho Dataset V02.
- Model selection bằng person-group cross-validation.
- Realtime webcam inference prototype.
- Official held-out evaluation trên unseen persons.
- Phân tích performance theo class và theo person.

## 2. Training experiment

Notebook chính:

`notebooks/02_training_experiments.ipynb`

Pipeline training:

```text
data/processed/features.csv
-> src.preprocessing.prepare_dataset()
-> group split theo person_id
-> locked train/test persons
-> StratifiedGroupKFold trên training persons
-> baseline models
-> tuning SVM / XGBoost
-> chọn model theo CV Macro F1
-> fit final model trên toàn bộ training persons
-> save artifact
```

Model cuối được chọn:

`SVM RBF Tuned`

CV Macro F1:

`0.607130`

CV Macro F1 std:

`0.170`

Ghi chú quan trọng:

- Split theo `person_id`.
- Không random split theo frame.
- Test persons không được dùng để model selection.

## 3. Realtime webcam inference

File chính:

`scripts/test_webcam_model.py`

Luồng realtime:

```text
Webcam
-> PoseDetector
-> 6 keypoints
-> FeatureExtractor
-> 29 features
-> best_model.joblib
-> posture prediction
-> realtime display
```

Các class:

- `correct`
- `forward_slouch`
- `lean_left`
- `lean_right`

Ghi nhận hiện tại:

- Realtime inference hoạt động ổn định trong smoke test.
- Cả 4 class đều được nhận diện hợp lý trong thử nghiệm webcam.
- Tốc độ thực tế khoảng 12-14 FPS trong test hiện tại.
- Chưa dùng temporal smoothing.
- Chưa refactor qua `PosturePredictor`.
- File hiện tại vẫn là prototype/integration test.

## 4. Official evaluation

File chính:

`src/evaluate.py`

Luồng evaluation:

```text
features.csv
+ best_model.joblib
+ split_manifest.json
+ training_metadata.json
-> verify dataset SHA256
-> prepare_dataset()
-> reconstruct locked unseen-person test set
-> model.predict()
-> metrics
-> confusion matrix
-> per-person metrics
-> misclassification analysis
```

`evaluate.py` không thực hiện:

- Retrain.
- Tune.
- Resplit.
- Model selection.

Test persons:

- `person01`
- `person10`
- `person12`

Test samples:

`707`

Test recordings:

`3`

## 5. Kết quả held-out test

```text
Accuracy       : 0.793494
Balanced Acc.  : 0.788075
Macro Precision: 0.784227
Macro Recall   : 0.788075
Macro F1       : 0.779126
Weighted F1    : 0.796576
```

CV Macro F1:

`0.607130`

Test - CV Macro F1:

`+0.171996`

## 6. Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| correct | 0.875000 | 0.936047 | 0.904494 | 172 |
| forward_slouch | 0.563536 | 0.772727 | 0.651757 | 132 |
| lean_left | 0.994924 | 0.867257 | 0.926714 | 226 |
| lean_right | 0.703448 | 0.576271 | 0.633540 | 177 |

## 7. Per-person metrics

| Person | Samples | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| person01 | 148 | 0.797297 | 0.667631 | 0.768973 |
| person10 | 307 | 0.745928 | 0.744852 | 0.751592 |
| person12 | 252 | 0.849206 | 0.851527 | 0.851693 |

Nhận xét:

- Performance giữa subject vẫn có variance đáng kể.
- Đây là dấu hiệu quan trọng cho cross-subject generalization.

## 8. Confusion matrix

```text
                correct  forward_slouch  lean_left  lean_right
correct             161               2          0           9
forward_slouch        0             102          0          30
lean_left             9              17        196           4
lean_right           14              60          1         102
```

Nhận xét:

- `correct` và `lean_left` đang hoạt động tốt.
- Lỗi chính là `forward_slouch` <-> `lean_right`.
- Đặc biệt `lean_right` thường bị nhầm thành `forward_slouch`.

## 9. Đánh giá hiện tại

- V02 hiện tại được xem là baseline tốt.
- Held-out Macro F1 = `0.7791`.
- Accuracy = `0.7935`.
- Realtime inference có tín hiệu tích cực.
- Chưa nên tiếp tục tune trực tiếp dựa trên held-out test set này.
- Cần freeze V02 để làm baseline cho các phiên bản sau.

## 10. Hướng tiếp theo

- Phân tích cross-subject variation.
- Phân tích các fold CV yếu.
- Nghiên cứu overlap feature giữa `forward_slouch` và `lean_right`.
- Cân nhắc V03 scale/camera normalization.
- Giữ nguyên locked split khi so sánh V02 vs V03.
- Sau khi training logic ổn định mới refactor sang:
  - `src/train.py`
  - `src/posture_predictor.py`
  - `temporal_monitor.py`
  - app integration
