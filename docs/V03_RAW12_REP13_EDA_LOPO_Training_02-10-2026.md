# Báo cáo V03 — RAW12, REP13, EDA và Nested LOPO

**Ngày báo cáo:** 02-10-2026  
**Trạng thái:** Research notebook; `src/train.py` chưa được xây dựng lại trong nhiệm vụ này.

## 1. Mục tiêu nâng cấp V03

V03 tách hình học frame hiện tại (RAW12), baseline cá nhân theo recording và representation cho model (REP13). Research đánh giá khả năng generalize sang người mới bằng nested Leave-One-Person-Out, tránh subject leakage và hyperparameter leakage.

## 2. FeatureExtractor — RAW12

`src/feature_extractor.py` tạo đúng 12 đặc trưng thuộc bốn nhóm: angle/orientation, neck, spatial và perspective/depth. RAW12 không normalize bằng shoulder width hiện tại; tên feature lấy từ `FeatureExtractor.FEATURE_NAMES`.

## 3. PersonalCalibration

Mỗi `person_id + session_id` (recording) dùng 30 correct samples riêng. Non-angle dùng median; angle dùng circular median-like để không vỡ tại ±180°. Offline notebook và realtime cùng gọi `PersonalCalibration`.

## 4. RepresentationBuilder — REP13

`RAW12 + baseline RAW12 → REP13`: wrapped angle delta, spatial delta chia baseline shoulder width, scale log-ratio và `head_drift_magnitude`. Builder stateless, hỗ trợ `transform()` và `transform_batch()`.

## 5. DatasetBuilder

DatasetBuilder lưu metadata + RAW12, gồm `frame_index`, không tạo REP13. Rejected samples có audit riêng và logic không hard-code số người.

## 6. Dataset build hiện tại

- Tổng ảnh: 4341
- Valid: 4014
- Rejected: 327
- Persons nguồn: 14
- Recordings nguồn: 16
- Class distribution: {"correct": 1051, "forward_slouch": 889, "lean_left": 1033, "lean_right": 1041}
- Rejection reasons: {"low_keypoint_confidence": 327}

## 7. Preprocessing

`prepare_raw_dataset()` trả `PreparedRawDataset`: `X_raw`, encoded `y`, `groups=person_id`, metadata và recording_id; nó validate dữ liệu nhưng chưa calibration, chưa tạo REP13 và chưa train model.

## 8. EDA V03

EDA bao gồm schema/data quality, phân phối class/person/recording, rejected audit, descriptive RAW12, correlation, circular-angle audit, calibration stability, REP13 sanity, subject variation và PCA diagnostic. 14/14 recording đủ 30 correct samples. Đã loại 420 calibration frames; còn 3110 samples REP13 hữu hạn để score. Các góc gần ±180° được xử lý bằng circular baseline và wrapped delta.

## 9. Training experiments

Loại tạm khỏi training: `person08, person11` do audit chất lượng dữ liệu. Sau exclusion còn 12 người, 14 recording và 3110 mẫu REP13 để score.

Outer CV là `LeaveOneGroupOut(person_id)` (12 folds). Mỗi held-out person giữ toàn bộ recording ở test fold. Inner tuning chỉ thấy outer-train persons qua `StratifiedGroupKFold(4, shuffle=True, random_state=42)`. So sánh XGBoost, SVM RBF, Random Forest và MLP; scaler nằm trong pipeline SVM/MLP. Chạy hiện tại có `FAST_MODE=False`.

| Model         |   LOPO Macro F1 Mean |   LOPO Macro F1 Std |   Worst Person Macro F1 |   LOPO Accuracy Mean |   Correct F1 |   Training/Search time | Notes                                                    |
|:--------------|---------------------:|--------------------:|------------------------:|---------------------:|-------------:|-----------------------:|:---------------------------------------------------------|
| SVM_RBF       |             0.909258 |            0.122354 |                0.642198 |             0.91869  |     0.909944 |               109.031  | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| MLP           |             0.869364 |            0.124294 |                0.664921 |             0.885393 |     0.897436 |                72.3069 | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| XGBoost       |             0.848723 |            0.134024 |                0.664921 |             0.866371 |     0.866184 |               200.733  | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| Random_Forest |             0.842999 |            0.159632 |                0.601659 |             0.864948 |     0.861818 |               561.6    | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |

## 10. Best model

- Model: **SVM_RBF**
- Final grouped-search params: `{"model__C": 0.1, "model__class_weight": null, "model__gamma": 0.1}`
- LOPO Macro F1 mean/std: 0.909258 / 0.122354
- Worst-person Macro F1: 0.642198
- Correct F1 (OOF): 0.909944
- Model: `models/best_model_v03_rep13.joblib`
- Results: `results/v03_lopo/`

## 11. Vấn đề và rủi ro còn lại

- Số recording giữa người chưa đều; person07 có nhiều session hơn các person khác.
- Calibration phụ thuộc 30 correct frames đầu; cần review các recording có MAD/half-window shift cao trước khi cân nhắc stable-window alternative.
- Điểm có thể biến thiên theo search budget; nếu run này dùng FAST_MODE thì cần chạy full budget trước khi chốt production.
- Personal baseline là yêu cầu runtime; user mới phải calibration trước inference.

## 12. Bước tiếp theo

Review/chốt representation, protocol và winner từ research. Sau đó mới viết lại `src/train.py` final V03, rồi cập nhật PosturePredictor/realtime demo. Trong nhiệm vụ này không sửa bất kỳ file `src/` nào.
