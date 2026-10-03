# Smart Posture Monitor

Smart Posture Monitor là hệ thống giám sát tư thế ngồi theo thời gian thực, sử dụng **YOLO Pose** để trích xuất keypoints phần thân trên, kết hợp **Personal Calibration**, biểu diễn đặc trưng **REP13**, mô hình **SVM RBF** và **Temporal Smoothing** để nhận diện tư thế ổn định hơn trên webcam.

Phiên bản hiện tại là **V03 — RAW12 + Personal Calibration + REP13 + Realtime Web Dashboard**.

---

## 1. Trạng thái dự án

Dự án hiện đã hoàn thiện luồng chính của bài tập lớn:

- Nhận diện người và 6 keypoints phần thân trên bằng YOLO Pose.
- Trích xuất bộ đặc trưng hình học RAW12.
- Hiệu chuẩn tư thế chuẩn theo từng người bằng 30 frame `correct`.
- Chuyển `RAW12 + personal baseline` thành REP13.
- Phân loại 4 tư thế bằng SVM RBF.
- Temporal smoothing bằng sliding window + majority vote + hysteresis.
- Yaw gating để hạn chế nhiễu khi người dùng quay đầu.
- Dashboard web realtime bằng FastAPI + WebSocket.
- Hiển thị skeleton, tư thế, confidence, góc sinh trắc học và thống kê phiên.
- Cảnh báo sai tư thế dựa trên **thời gian thực**, không phụ thuộc FPS giả định.
- Lưu lịch sử phiên, thống kê vi phạm và xuất báo cáo.
- Runtime WebSocket sử dụng cơ chế **one-frame-in-flight / latest-frame-wins** để tránh backlog.

### Kết quả model final

| Metric | Giá trị |
|---|---:|
| Model | **SVM RBF** |
| LOPO Macro F1 Mean | **0.919948** |
| OOF Accuracy | **0.936871** |
| OOF Balanced Accuracy | **0.936982** |
| OOF Macro Precision | **0.937899** |
| OOF Macro Recall | **0.936982** |
| OOF Macro F1 | **0.936299** |

Hyperparameters của model thắng:

```text
C = 1
gamma = 0.01
class_weight = balanced
```

---

## 2. Bài toán

Hệ thống nhận webcam từ góc máy khoảng **45° bên trái người dùng** và phân loại tư thế ngồi thành 4 lớp:

| Label | Ý nghĩa |
|---|---|
| `correct` | Tư thế ngồi đúng / tương đối thẳng |
| `forward_slouch` | Cúi hoặc gù người về phía trước |
| `lean_left` | Nghiêng người / đầu sang trái |
| `lean_right` | Nghiêng người / đầu sang phải |

Mục tiêu của V03 không chỉ là dự đoán từng frame, mà còn giảm ảnh hưởng khác biệt hình thể giữa người dùng bằng **Personal Calibration** và giảm hiện tượng nhảy nhãn bằng **TemporalMonitor**.

---

## 3. Kiến trúc tổng thể

### 3.1. Realtime pipeline

```text
Webcam
  ↓
Browser capture
  ↓
WebSocket (one frame in-flight)
  ↓
PoseDetector
  ↓
6 upper-body keypoints
  ↓
FeatureExtractor
  ↓
RAW12
  ↓
PersonalCalibration
  ↓
RepresentationBuilder
  ↓
REP13
  ↓
PosturePredictor
  ↓
SVM RBF
  ↓
Raw prediction
  ↓
TemporalMonitor
  ├─ valid-label window
  ├─ majority vote
  ├─ hysteresis
  └─ yaw gating
  ↓
Stable posture
  ↓
SessionStatistics
  ↓
WebSocket response
  ↓
Realtime Dashboard
```

### 3.2. Research / training pipeline

```text
data/processed/features.csv
  ↓
prepare_raw_dataset()
  ↓
validate persons / recordings
  ↓
30 correct frames tạo baseline cho từng recording
  ↓
loại calibration frames khỏi tập ML
  ↓
RAW12 + baseline → REP13
  ↓
outer Leave-One-Person-Out
  ↓
inner StratifiedGroupKFold
  ↓
SVM / Random Forest / XGBoost / MLP
  ↓
OOF evaluation
  ↓
final grouped search
  ↓
fit winner
```

Evaluation chính sử dụng **nested Leave-One-Person-Out (LOPO)** theo `person_id` để hạn chế subject leakage.

---

## 4. Camera protocol và keypoints

Dataset và realtime demo được thiết kế chủ yếu cho camera đặt khoảng **45° từ bên trái người dùng**.

Frame không nên bị mirror trong pipeline inference vì thao tác mirror có thể làm đảo ý nghĩa `lean_left` và `lean_right`.

`PoseDetector` sử dụng 6 keypoints:

```text
nose
left_eye
right_eye
left_ear
left_shoulder
right_shoulder
```

Right ear không được sử dụng do dễ bị che khuất ở góc camera hiện tại.

---

## 5. Feature representation

### 5.1. RAW12

`src/feature_extractor.py` chuyển pose thành 12 feature hình học.

Schema chính thức lấy từ:

```python
FeatureExtractor.FEATURE_NAMES
```

Các nhóm feature chính:

| Nhóm | Feature |
|---|---|
| Angle / orientation | `shoulder_roll_deg`, `eye_roll_deg`, `head_shoulder_roll_diff_deg` |
| Neck | `neck_pitch_deg` |
| Spatial | `eye_center_x_px`, `eye_center_y_px`, `shoulder_center_x_px`, `eye_shoulder_vertical_gap_px`, `eye_shoulder_horizontal_offset_px` |
| Perspective / scale proxy | `inter_eye_distance_px`, `ear_nose_horizontal_span_px`, `shoulder_width_px` |

RAW12 mô tả hình học của frame hiện tại và chưa chứa personal delta.

### 5.2. Personal Calibration

Mỗi recording có baseline riêng:

```text
recording_id = person_id + "__" + session_id
CALIBRATION_SAMPLES = 30
```

Quy trình:

1. Người dùng ngồi ở tư thế `correct`.
2. Thu 30 RAW12 sample hợp lệ.
3. Tạo baseline cá nhân.
4. Các frame calibration chỉ dùng để tạo baseline, không dùng để train hoặc score.
5. Realtime chỉ bắt đầu classification sau khi calibration hoàn thành.

Trong final dataset, toàn bộ **16/16 recordings** đều đủ dữ liệu calibration.

### 5.3. REP13

`src/representation_builder.py` biến đổi:

```text
RAW12 + baseline RAW12 → REP13
```

REP13 gồm:

- 4 wrapped angle deltas;
- 5 spatial deltas được chuẩn hóa theo baseline shoulder width;
- 3 scale log-ratios;
- `head_drift_magnitude`.

Schema chính thức lấy từ:

```python
RepresentationBuilder.OUTPUT_FEATURE_NAMES
```

Model production chỉ nhận **REP13**, không học trực tiếp từ RAW12.

EDA cho thấy median normalized subject spread giảm từ:

```text
RAW12 = 0.632149
REP13 = 0.418494
```

---

## 6. Dataset

Dataset source của final retrain:

| Thống kê | Giá trị |
|---|---:|
| Total images | 4,814 |
| Valid samples | 4,456 |
| Rejected samples | 358 |
| Persons | 14 |
| Recordings | 16 |
| `correct` | 1,467 |
| `forward_slouch` | 937 |
| `lean_left` | 1,024 |
| `lean_right` | 1,028 |

Toàn bộ 358 rejected samples có nguyên nhân:

```text
low_keypoint_confidence
```

Sau khi loại 30 calibration frames của từng recording:

| Thống kê | Giá trị |
|---|---:|
| Persons dùng cho LOPO | 14 |
| Recordings | 16 |
| Calibration frames bị loại | 480 |
| REP13 samples được score | 3,976 |

Dataset trong `data/` được giữ local và không bắt buộc commit lên Git.

---

## 7. Model training và evaluation

Notebook source of truth:

```text
notebooks/02_training_experiments.ipynb
```

Cấu hình final:

```text
TRAINING_MODE = FULL
FAST_MODE = False
SEARCH_N_JOBS = 4
INNER_SPLITS = 4
RANDOM_STATE = 42

outer CV = LeaveOneGroupOut(person_id)
inner CV = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

selection metric = outer LOPO Macro F1 mean
```

### So sánh model

| Model | LOPO Macro F1 Mean | Std | Worst-person Macro F1 | Accuracy Mean | Correct F1 |
|---|---:|---:|---:|---:|---:|
| **SVM RBF** | **0.919948** | **0.110044** | **0.671906** | **0.932851** | **0.938104** |
| Random Forest | 0.899258 | 0.137486 | 0.644433 | 0.915897 | 0.914313 |
| MLP | 0.899089 | 0.123974 | 0.618682 | 0.917051 | 0.939971 |
| XGBoost | 0.873577 | 0.146456 | 0.631538 | 0.894725 | 0.909948 |

### Per-class performance — final SVM

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `correct` | 0.897317 | 0.982776 | 0.938104 | 987 |
| `forward_slouch` | 0.896050 | 0.919957 | 0.907846 | 937 |
| `lean_left` | 0.984064 | 0.964844 | 0.974359 | 1024 |
| `lean_right` | 0.974166 | 0.880350 | 0.924885 | 1028 |

Confusion đáng chú ý nhất:

```text
actual lean_right → predicted forward_slouch: 69 samples
```

---

## 8. Temporal smoothing

`src/temporal_monitor.py` xử lý hậu dự đoán, không yêu cầu retrain model.

Runtime V03 sử dụng:

```text
window_size = 7
min_samples = 3
hysteresis_frames = 2
yaw_mode = conservative
```

Chỉ 4 posture label hợp lệ được đưa vào smoothing window:

```text
correct
forward_slouch
lean_left
lean_right
```

Các trạng thái như `no_person`, `evaluating`, `head_turned`, `low_confidence` không được đưa vào majority vote.

Luồng xử lý:

```text
Raw prediction
  ↓
Valid-label deque
  ↓
Majority vote
  ↓
Hysteresis
  ↓
Stable posture
```

UI hiển thị `smoothed_label` thay vì raw prediction để hạn chế flickering.

---

## 9. Realtime WebSocket runtime

Web dashboard sử dụng **FastAPI + WebSocket**.

### Cơ chế chống backlog

Frontend không gửi frame theo interval cố định nữa.

```text
capture newest frame
  ↓
send to backend
  ↓
wait for response
  ↓
capture newest frame
  ↓
send next frame
```

Tại mọi thời điểm chỉ có tối đa **1 frame inference đang xử lý**.

Cơ chế này được gọi là:

```text
one-frame-in-flight
latest-frame-wins
```

Nếu inference chậm, hệ thống bỏ qua frame cũ thay vì xếp hàng khiến skeleton bị delay tích lũy.

### Tách capture và display

Frontend sử dụng:

- offscreen canvas để capture / encode frame gửi backend;
- output canvas để render video và skeleton mới nhất.

Video webcam render bằng `requestAnimationFrame()`, độc lập với tốc độ inference của model.

### Metrics runtime

Dashboard hiển thị:

- AI result FPS thực;
- WebSocket round-trip latency;
- `server_processing_ms`;
- stable posture;
- display confidence;
- 3 góc sinh trắc học;
- ergonomics score;
- số vi phạm;
- thời gian phiên.

---

## 10. Session Statistics

`src/session_statistics.py` sử dụng `time.monotonic()` cho logic thời gian realtime.

Các thông tin chính:

```text
correct_seconds
bad_seconds
alert_count
ergonomics_score
```

Cảnh báo sai tư thế sử dụng threshold theo **giây thực**.

Một bad episode liên tục chỉ được tính là **một violation**. Khi posture trở lại `correct`, episode được reset.

---

## 11. Dashboard

Web UI hiện hỗ trợ:

### Tổng quan
- webcam realtime;
- skeleton 6 keypoints;
- stable posture;
- confidence;
- FPS và latency;
- ergonomics score;
- thời gian phiên;
- số cảnh báo;
- 3 góc sinh trắc học;
- gợi ý công thái học.

### Analytics
- tỷ lệ tư thế đúng / sai;
- biểu đồ xu hướng theo phiên;
- thống kê công thái học.

### Session History
- lưu lịch sử phiên trên browser;
- số vi phạm;
- timeline vi phạm;
- lọc lịch sử;
- xuất CSV / Excel.

---

## 12. Realtime states

`PostureInferenceEngine` có các trạng thái chính:

```text
CALIBRATION_REQUIRED
CALIBRATING
CALIBRATED
CALIBRATION_SAMPLE_REJECTED
LOW_CONFIDENCE
NO_PERSON
OK
```

Nguyên tắc:

- Không predict bằng RAW12 khi chưa calibration.
- Calibration state không đi vào TemporalMonitor.
- `NO_PERSON` không reset personal baseline.
- `LOW_CONFIDENCE` không làm bẩn temporal window.
- Baseline chỉ reset khi người dùng chủ động recalibrate.
- Reset session chỉ reset temporal state và statistics.

---

## 13. Project structure

```text
smart_posture_monitor/
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/                              # Local dataset, ignored by Git
├── docs/
│   ├── V03_RAW12_REP13_EDA_LOPO_Training_02-10-2026.md
│   └── V03_Runtime_WebSocket_Temporal_Statistics_04-10-2026.md
├── models/
│   ├── best_model.joblib
│   ├── calibration_config.json
│   ├── training_metadata.json
│   └── yolo26n-pose.pt
├── notebooks/
│   ├── 01_eda_dataset.ipynb
│   └── 02_training_experiments.ipynb
├── results/
│   └── v03_lopo/
├── scripts/
│   ├── copy_rejected_images.py
│   └── test_webcam_model.py
├── src/
│   ├── dataset_builder.py
│   ├── evaluate.py
│   ├── feature_extractor.py
│   ├── inference.py
│   ├── personal_calibration.py
│   ├── pose_detector.py
│   ├── posture_predictor.py
│   ├── preprocessing.py
│   ├── representation_builder.py
│   ├── session_statistics.py
│   ├── temporal_monitor.py
│   └── train.py
├── tests/
│   ├── test_temporal_monitor.py
│   └── test_session_statistics.py
├── requirements.txt
└── README.md
```

---

## 14. Module responsibilities

| Module | Vai trò |
|---|---|
| `src/pose_detector.py` | Frame → selected person pose + 6 keypoints |
| `src/feature_extractor.py` | Pose → RAW12 |
| `src/personal_calibration.py` | 30 RAW12 correct samples → personal baseline |
| `src/representation_builder.py` | RAW12 + baseline → REP13 |
| `src/posture_predictor.py` | REP13 → raw class + probability |
| `src/temporal_monitor.py` | Raw prediction → stable posture |
| `src/session_statistics.py` | Time-based runtime statistics + violation episodes |
| `src/inference.py` | Điều phối toàn bộ realtime pipeline |
| `src/preprocessing.py` | Validate dataset và tạo input cho research pipeline |
| `src/dataset_builder.py` | Raw images → RAW12 dataset + rejected audit |
| `app/app.py` | FastAPI server + HTTP/WebSocket runtime |
| `app/templates/index.html` | Dashboard realtime và browser-side runtime |

---

## 15. Cài đặt

Khuyến nghị:

```text
Python 3.10 hoặc 3.11
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Các model cần tồn tại trong:

```text
models/best_model.joblib
models/training_metadata.json
models/yolo26n-pose.pt
```

---

## 16. Chạy Web Dashboard

Từ thư mục root:

```powershell
python -m uvicorn app.app:app --reload
```

Mở trình duyệt:

```text
http://127.0.0.1:8000
```

Quy trình sử dụng:

```text
1. Bật webcam
2. Ngồi đúng tư thế tự nhiên
3. Nhấn C để bắt đầu Personal Calibration
4. Giữ tư thế đến khi đủ 30 sample hợp lệ
5. Hệ thống bắt đầu realtime posture monitoring
```

Phím tắt:

| Phím | Chức năng |
|---|---|
| `C` | Bắt đầu calibration |
| `R` | Reset session |
| `Q` / `ESC` | Dừng webcam |

---

## 17. Realtime model test

Kiểm tra model/metadata:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model.joblib `
  --check-only
```

Chạy webcam test:

```powershell
python scripts/test_webcam_model.py `
  --model models/best_model.joblib `
  --camera 0
```

---

## 18. Tests

Runtime tests:

```powershell
python -m pytest -q -p no:cacheprovider `
  tests/test_temporal_monitor.py `
  tests/test_session_statistics.py
```

Kết quả của runtime patch:

```text
10 passed in 0.02s
```

Kiểm tra thêm:

```text
python -m compileall -q src app
```

Một số test legacy của repository vẫn mô tả API từ các phiên bản RAW29 / DELTA29 trước V03 và có thể không còn khớp với pipeline final RAW12 / REP13.

---

## 19. Research / retraining

Chạy notebook theo thứ tự:

```text
1. notebooks/01_eda_dataset.ipynb
2. notebooks/02_training_experiments.ipynb
```

`02_training_experiments.ipynb` là source of truth cho final nested LOPO training.

Không nên dùng:

```powershell
python -m src.train
```

để tái tạo final REP13 artifact nếu `src/train.py` vẫn chưa được migrate hoàn toàn sang protocol research cuối cùng.

---

## 20. Validation

Final V03 đã xác nhận:

- 14 persons / 16 recordings;
- 56 outer-fold model records = `14 persons × 4 models`;
- mỗi model có 3,976 OOF predictions;
- person overlap giữa train/test bằng 0;
- không calibration frame nào bị score;
- 16/16 recordings đủ dữ liệu calibration;
- model reload nhận đúng input REP13 shape `(N, 13)`;
- SVM RBF là model thắng;
- runtime TemporalMonitor chỉ nhận 4 posture class hợp lệ;
- runtime alert sử dụng thời gian thực;
- WebSocket không còn tích lũy backlog frame;
- reset session không xóa personal baseline.

---

## 21. Current limitations

- Camera protocol vẫn phụ thuộc tương đối vào góc đặt máy khoảng 45° bên trái.
- Personal Calibration yêu cầu người dùng giữ tư thế `correct` ở đầu lần sử dụng.
- Dataset giữa các subject chưa hoàn toàn cân bằng số recording.
- Worst-person Macro F1 khoảng **0.6719**.
- `lean_right` có recall khoảng **0.8804**, thấp nhất trong 4 class.
- Hệ thống sử dụng pose 2D, chưa có depth/3D.
- Runtime FPS vẫn phụ thuộc vào tốc độ YOLO Pose và phần cứng.
- Cơ chế latest-frame-wins loại bỏ backlog nhưng không làm inference model nhanh hơn.
- Một số legacy tests và entry point training cũ cần được migrate riêng sang V03 final protocol.

---

## 22. Hướng phát triển

1. Tối ưu YOLO inference bằng GPU / runtime acceleration.
2. Benchmark `imgsz` và resolution phù hợp cho realtime.
3. Thu thêm dữ liệu từ nhiều người và nhiều setup camera hơn.
4. Cân bằng số recording giữa các subject.
5. Nghiên cứu camera scale compensation.
6. Mở rộng dashboard với persistence phía server/database.
7. Đóng gói project bằng Docker.
8. Tách backend API để triển khai độc lập khi cần.
9. Mở rộng từ single-user sang multi-person posture monitoring.

---

## 23. Tài liệu liên quan

- `docs/V03_RAW12_REP13_EDA_LOPO_Training_02-10-2026.md`
- `docs/V03_Runtime_WebSocket_Temporal_Statistics_04-10-2026.md`
- `notebooks/01_eda_dataset.ipynb`
- `notebooks/02_training_experiments.ipynb`
- `results/v03_lopo/lopo_model_summary.csv`
- `results/v03_lopo/lopo_fold_metrics.csv`
- `results/v03_lopo/per_class_metrics.csv`
- `results/v03_lopo/per_person_metrics.csv`

---

## 24. Phiên bản hiện tại

```text
Smart Posture Monitor V03

YOLO Pose
+ RAW12
+ Personal Calibration
+ REP13
+ SVM RBF
+ Temporal Smoothing
+ Time-based Session Statistics
+ FastAPI / WebSocket
+ Realtime Dashboard
```

V03 là phiên bản hoàn chỉnh hiện tại của bài tập lớn, bao gồm cả **research/training pipeline** và **realtime web runtime**, đồng thời giữ nguyên model representation và evaluation protocol đã chốt.
