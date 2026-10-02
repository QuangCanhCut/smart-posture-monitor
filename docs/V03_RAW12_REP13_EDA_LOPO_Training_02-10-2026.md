# V03 Final Retraining – Updated Dataset

**Ngày báo cáo:** 2026-10-03  
**Trạng thái:** FINAL RETRAIN; FULL MODE; artifacts đã được ghi đè.

## 1. Mục tiêu

V03 tách hình học frame hiện tại (RAW12), baseline cá nhân theo recording và representation cho model (REP13). Research đánh giá khả năng generalize sang người mới bằng nested Leave-One-Person-Out, tránh subject leakage và hyperparameter leakage.

### FeatureExtractor — RAW12

`src/feature_extractor.py` tạo đúng 12 đặc trưng thuộc bốn nhóm: angle/orientation, neck, spatial và perspective/depth. RAW12 không normalize bằng shoulder width hiện tại; tên feature lấy từ `FeatureExtractor.FEATURE_NAMES`.

### PersonalCalibration

Mỗi `person_id + session_id` (recording) dùng 30 correct samples riêng. Non-angle dùng median; angle dùng circular median-like để không vỡ tại ±180°. Offline notebook và realtime cùng gọi `PersonalCalibration`.

### RepresentationBuilder — REP13

`RAW12 + baseline RAW12 → REP13`: wrapped angle delta, spatial delta chia baseline shoulder width, scale log-ratio và `head_drift_magnitude`. Builder stateless, hỗ trợ `transform()` và `transform_batch()`.

### DatasetBuilder

DatasetBuilder lưu metadata + RAW12, gồm `frame_index`, không tạo REP13. Rejected samples có audit riêng và logic không hard-code số người.

## 2. Dataset mới

- Tổng ảnh: 4814
- Valid: 4456
- Rejected: 358
- Persons nguồn: 14
- Recordings nguồn: 16
- Session list: ["person01__session01", "person02__session01", "person03__session01", "person04__session01", "person05__session01", "person06__session01", "person07__session01", "person07__session02", "person07__session03", "person08__session01", "person09__session01", "person10__session01", "person11__session01", "person12__session01", "person13__session01", "person14__session01"]
- Class distribution: {"correct": 1467, "forward_slouch": 937, "lean_left": 1024, "lean_right": 1028}
- Samples/person: {"person01": 168, "person02": 301, "person03": 230, "person04": 304, "person05": 289, "person06": 269, "person07": 884, "person08": 263, "person09": 303, "person10": 340, "person11": 255, "person12": 284, "person13": 287, "person14": 279}
- Samples/session: {"person01__session01": 168, "person02__session01": 301, "person03__session01": 230, "person04__session01": 304, "person05__session01": 289, "person06__session01": 269, "person07__session01": 326, "person07__session02": 287, "person07__session03": 271, "person08__session01": 263, "person09__session01": 303, "person10__session01": 340, "person11__session01": 255, "person12__session01": 284, "person13__session01": 287, "person14__session01": 279}
- Rejection reasons: {"low_keypoint_confidence": 358}
- Dữ liệu person08–person11 đã được thu/thay lại; final run không kế thừa exclusion cũ.

### Preprocessing

`prepare_raw_dataset()` trả `PreparedRawDataset`: `X_raw`, encoded `y`, `groups=person_id`, metadata và recording_id; nó validate dữ liệu nhưng chưa calibration, chưa tạo REP13 và chưa train model.

## 3. EDA

EDA bao gồm schema/data quality, phân phối class/person/recording, rejected audit, descriptive RAW12, correlation, circular-angle audit, calibration stability, REP13 sanity, subject variation và PCA diagnostic. 16/16 recording đủ 30 correct samples. Đã loại 480 calibration frames; còn 3976 samples REP13 hữu hạn để score. Các góc gần ±180° được xử lý bằng circular baseline và wrapped delta.

Không có missing, NaN/Inf, duplicated row hay duplicated image path. Class max/min = 1,566; recording max/min = 2,024. Median normalized subject spread giảm từ 0,632 (RAW12) xuống 0,418 (REP13). Pairwise diagnostic nổi bật: `delta_eye_center_y_body` cho correct/forward_slouch, `head_drift_magnitude` cho correct/lean, và `delta_shoulder_center_x_body` cho lean_left/lean_right. PCA 2D giải thích 52,715% phương sai và chỉ được dùng để chẩn đoán.

## 4. Training Protocol

Final cohort không loại person nào: 14 người, 16 recording và 3976 mẫu REP13 để score.

Representation: REP13. Outer CV là `LeaveOneGroupOut(person_id)` (14 folds). Mỗi held-out person giữ toàn bộ recording ở test fold; ba session của person07 luôn cùng group. Inner tuning chỉ thấy outer-train persons qua `StratifiedGroupKFold(4, shuffle=True, random_state=42)`. Models: XGBoost, SVM RBF, Random Forest và MLP. Selection metric: outer LOPO Macro F1 mean. Chạy hiện tại có `TRAINING_MODE=FULL`, `FAST_MODE=False`.

## 5. Model Comparison

| Model         |   LOPO Macro F1 Mean |   LOPO Macro F1 Std |   Worst Person Macro F1 |   LOPO Accuracy Mean |   Correct F1 |   Training/Search time | Notes                                                    |
|:--------------|---------------------:|--------------------:|------------------------:|---------------------:|-------------:|-----------------------:|:---------------------------------------------------------|
| SVM_RBF       |             0.919948 |            0.110044 |                0.671906 |             0.932851 |     0.938104 |                300.749 | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| Random_Forest |             0.899258 |            0.137486 |                0.644433 |             0.915897 |     0.914313 |                789.698 | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| MLP           |             0.899089 |            0.123974 |                0.618682 |             0.917051 |     0.939971 |                108.509 | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |
| XGBoost       |             0.873577 |            0.146456 |                0.631538 |             0.894725 |     0.909948 |                368.391 | Nested LOPO; inner StratifiedGroupKFold; FAST_MODE=False |

## 6. Best Model

- Model: **SVM_RBF**
- Final grouped-search params: `{"model__C": 1, "model__class_weight": "balanced", "model__gamma": 0.01}`
- LOPO Macro F1 mean/std: 0.919948 / 0.110044
- Worst-person Macro F1: 0.671906
- OOF accuracy: 0.936871
- OOF balanced accuracy: 0.936982
- OOF macro precision/recall/F1: 0.937899 / 0.936982 / 0.936299
- Correct F1 (OOF): 0.938104
- Models: `models/best_model_v03_rep13.joblib`, `models/best_model.joblib`
- Results: `results/v03_lopo/`

## 7. Per-Class Performance

| model_name   | class          |   precision |   recall |       f1 |   support |
|:-------------|:---------------|------------:|---------:|---------:|----------:|
| SVM_RBF      | correct        |    0.897317 | 0.982776 | 0.938104 |       987 |
| SVM_RBF      | forward_slouch |    0.89605  | 0.919957 | 0.907846 |       937 |
| SVM_RBF      | lean_left      |    0.984064 | 0.964844 | 0.974359 |      1024 |
| SVM_RBF      | lean_right     |    0.974166 | 0.88035  | 0.924885 |      1028 |

## 8. Confusion Analysis

Largest off-diagonal pair: actual `lean_right` → predicted `forward_slouch` (69 samples).

| actual         | predicted      |   count |
|:---------------|:---------------|--------:|
| lean_right     | forward_slouch |      69 |
| lean_right     | correct        |      52 |
| forward_slouch | correct        |      48 |
| lean_left      | forward_slouch |      25 |
| forward_slouch | lean_right     |      18 |
| lean_left      | correct        |      11 |
| forward_slouch | lean_left      |       9 |
| correct        | forward_slouch |       6 |
| correct        | lean_right     |       6 |
| correct        | lean_left      |       5 |
| lean_right     | lean_left      |       2 |
| lean_left      | lean_right     |       0 |

## 9. So sánh V03 cũ

Previous artifact: SVM_RBF with LOPO Macro F1 mean 0.909258; the old run excluded persons 08 and 11, so the difference is contextual and not a like-for-like improvement claim.

Final cohort có LOPO Macro F1 mean 0,919948 (+0,010690), Correct F1 0,938104 (+0,028161) và worst-person Macro F1 0,671906 (+0,029708) so với artifact cũ. Các chênh lệch này không được diễn giải là cải thiện thuần túy vì cohort/dataset đã thay đổi.

## 10. Kết luận

Final retrain dùng đủ 14 persons/16 recordings, FULL MODE, nested person-grouped CV và không có subject overlap. Model thắng được chọn duy nhất theo outer LOPO Macro F1 mean; các artifact canonical đã được ghi đè và smoke-tested ở REP13 13 chiều.

## 11. Vấn đề và rủi ro còn lại

- Số recording giữa người chưa đều; person07 có nhiều session hơn các person khác.
- Calibration phụ thuộc 30 correct frames đầu; cần review các recording có MAD/half-window shift cao trước khi cân nhắc stable-window alternative.
- Điểm có thể biến thiên theo search budget; nếu run này dùng FAST_MODE thì cần chạy full budget trước khi chốt production.
- Personal baseline là yêu cầu runtime; user mới phải calibration trước inference.

## 12. Bước tiếp theo

Giữ nguyên methodology và realtime contract V03. Theo dõi worst-person/per-class metrics của final model; không thay feature hoặc split để tối ưu metric hậu nghiệm.
