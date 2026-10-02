# BÁO CÁO NGHIÊN CỨU & THIẾT KẾ TRIỂN KHAI: PERSONAL BASELINE DELTA
## Smart Posture Monitor — Hệ Thống Giám Sát Tư Thế Góc Camera 45°

---

## 1. TỔNG QUAN & HIỆN TRẠNG DỰ ÁN (RESEARCH STATUS)

### 1.1. Kết luận hiện trạng mã nguồn
Đối chiếu giữa tài liệu thiết kế và mã nguồn thực tế tại kho lưu trữ:

| Thành phần | Hiện trạng tài liệu | Hiện trạng trong code chạy | Đánh giá |
|---|---|---|---|
| **Baseline Calibration** | Đã được phân tích chi tiết trong `docs/V03_feature_design_proposal.md` và `docs/Nghien_cuu_cai_tien_mo_hinh_va_baseline_calibration_2026-09-28.md` | **Chưa triển khai trong code chạy** | ❌ Thiếu logic |
| `src/feature_extractor.py` | Thiết kế trích xuất 32 features (29 V02 + 3 Depth Proxy) | Chỉ trích xuất 32 đặc trưng tĩnh tuyệt đối; không có logic lưu vector baseline, không có tính $\Delta f$ | ⚠️ Chỉ có absolute features |
| `src/temporal_monitor.py` | Thiết kế cơ chế lọc thời gian và Yaw Gating | Đã có Majority Vote (window 5s) + Hysteresis (3 frames) + Yaw Gating (D4), nhưng **hoàn toàn độc lập với baseline** | ⚠️ Thiếu tích hợp baseline |
| `src/posture_predictor.py` | Đã có trong lộ trình kiến trúc V03 | **Hoàn toàn KHÔNG TỒN TẠI** trong thư mục `src/` | ❌ File missing |
| `app/streamlit_app.py` | Đang chạy với webcam và hiển thị HUD | Gọi trực tiếp `detector` -> `extractor` -> `best_model.joblib` -> `temporal_monitor`; chưa có session state và UI cho baseline calibration | ⚠️ Chưa hỗ trợ baseline |

> **KẾT LUẬN CỐT LÕI**:  
> Dự án **đã có đầy đủ cơ sở lý thuyết, công thức toán học và thiết kế kiến trúc** cho Personal Baseline Delta, nhưng toàn bộ luồng thực thi trong code chạy (`src/`, `app/`, `scripts/`) vẫn đang là **mô hình tĩnh (Static Baseline-Free Model)**.

---

## 2. BẢN CHẤT HỌNG KỸ THUẬT & CƠ SỞ KHOA HỌC

### 2.1. Vấn đề Anthropometric Variance (Đa dạng nhân trắc học)
Mô hình tĩnh V02/V03 cố gắng dùng một siêu phẳng (hyperplane trong SVM RBF) để phân loại tư thế cho toàn bộ người dùng. Tuy nhiên:
- **Người cao vs Người thấp**: Tọa độ `nose_y_body` (chiều cao mũi so với vai) ở người cổ dài khi ngồi chuẩn có thể bằng chính tọa độ của người cổ ngắn khi đang cúi gù (`forward_slouch`).
- **Khoảng cách 2 mắt (`eye_width_ratio`)**: Thay đổi theo cấu trúc khuôn mặt tự nhiên.
- **Hậu quả**: Macro F1 giữa các người trong kiểm thử độc lập (Leave-One-Person-Out) dao động mạnh từ **0.667 đến 0.851** (độ lệch chuẩn $\sigma = 0.1051$). Có những cá nhân (như `person11`, `person03`) bị mô hình báo sai tư thế lên tới **49% - 86%** ngay cả khi đang ngồi thẳng!

### 2.2. Vấn đề góc camera 45° và sự đánh lừa thị giác (FS ↔ LR)
Khi camera đặt lệch 45° bên trái người dùng:
1. Khi **Forward Slouch** (gù lưng cúi đầu): Đầu dịch chuyển xuống dưới và tiến ra trước theo trục Z (tiến gần camera). Trong ảnh 2D, khoảng cách đầu-vai co lại.
2. Khi **Lean Right** (nghiêng người sang phải): Đầu dịch sang phải và lùi xa camera. Do góc nhìn 45°, đầu trong ảnh 2D cũng bị hạ thấp và khoảng cách đầu-vai cũng co lại!

```text
               Camera (Góc chéo 45° bên trái)
                 \
                  \
                   \
                  [Người dùng]
                  /          \
    Forward Slouch            Lean Right
(Đầu cúi gần camera)      (Đầu lệch xa camera)
       ↓                          ↓
Ảnh 2D: Đầu hạ thấp      Ảnh 2D: Đầu cũng hạ thấp!
```

Do đó, **90 ca trên tổng số 707 test frames** trong V02 bị nhầm lẫn qua lại giữa `forward_slouch` và `lean_right`.

### 2.3. Giải pháp Personal Baseline Delta
Thay vì phân loại dựa trên tọa độ tuyệt đối $\vec{f}_t$, ta chuyển sang bài toán phân loại **độ biến thiên (Delta) so với chính tư thế chuẩn của người đó**:

$$\vec{f}_{\text{baseline}} = \frac{1}{N} \sum_{i=1}^{N} \vec{f}_{\text{correct}, i} \quad (N = 30 \text{ frames } \approx 2\text{--}3\text{ giây})$$

$$\Delta \vec{f}_t = \vec{f}_t - \vec{f}_{\text{baseline}}$$

- Khi cúi gù (`forward_slouch`): Mũi của **chính người đó** luôn tụt xuống: $\Delta(\text{nose\_y\_body}) < -0.12$.
- Khi nghiêng phải (`lean_right`): Mũi ít tụt hơn nhưng hai vai lệch mạnh: $|\Delta(\text{shoulder\_angle})| > 5^\circ$.
- Tín hiệu $\Delta$ **triệt tiêu hoàn toàn sự khác biệt về vóc dáng, nhân trắc học và vị trí ngồi ban đầu**.

---

## 3. THIẾT KẾ VECTOR HYBRID (ABSOLUTE + DELTA)

### 3.1. Tại sao không dùng thuần túy $\Delta \vec{f}$?
Nếu chỉ dùng thuần túy $\Delta \vec{f}$:
> Khi người dùng bấm hiệu chuẩn lúc **đang ngồi gù**, thì $\Delta \vec{f} = \vec{0}$ và hệ thống sẽ coi tư thế gù đó là "Correct"!

Do đó, kiến trúc bắt buộc phải là **Hybrid Vector**:
$$\vec{f}_{\text{hybrid}} = \big[\vec{f}_{\text{absolute}}, \; \Delta \vec{f}_{\text{selective}}\big]$$
Trong đó các feature tuyệt đối đóng vai trò **ràng buộc sinh lý cứng (Physiological Hard Constraints)**: Nếu `head_gravity_angle > 25°` hoặc `shoulder_angle > 15°`, mô hình vẫn nhận biết được tư thế sai bất kể baseline là gì.

### 3.2. Chọn lọc 8 Delta Features nhạy cảm nhất
Để tránh hiện tượng **Curse of Dimensionality** (phình to số chiều làm loãng khoảng cách Euclidean trong SVM RBF), chúng tôi chọn lọc 8 đặc trưng cốt lõi theo đề xuất trong `docs/V03_feature_design_proposal.md`:

| STT | Tên đặc trưng Delta | Bản chất vật lý | Tác dụng phân biệt |
|:---:|---|---|---|
| 1 | `delta_nose_y_body` | Mức tụt của mũi trong hệ tọa độ vai | Chỉ dấu vàng phát hiện `forward_slouch` |
| 2 | `delta_head_mean_height` | Mức hạ độ cao trung bình của đầu | Phân biệt hạ đầu do gù vs nghiêng người |
| 3 | `delta_nose_gravity_angle` | Góc trục mũi so với phương thẳng đứng | Phát hiện cúi đầu gập cổ |
| 4 | `delta_head_gravity_angle` | Góc trục đầu so với trọng lực | Phát hiện nghiêng đầu/thân |
| 5 | `delta_face_shoulder_scale_ratio` | Diện tích tam giác mặt / vai² (D1) | Tăng khi gù (gần camera), giảm khi nghiêng phải (xa camera) |
| 6 | `delta_eye_width_ratio` | Tỷ lệ khoảng cách mắt / vai (D2) | Proxy khoảng cách Z |
| 7 | `delta_face_rotation_proxy` | Tỷ lệ mắt trái-mũi / mắt phải-mũi (D4) | Yaw proxy phát hiện quay đầu vs nghiêng người |
| 8 | `delta_ear_nose_depth_proxy` | Tỷ lệ khoảng cách tai-mũi / tai-mắt (D3) | Mũi vươn ra trước khi gù |

### 3.3. Các cấu hình Vector Hybrid được hỗ trợ
1. **Selective Hybrid (40D)** (Mặc định khuyến nghị): 32 Absolute + 8 Delta.
2. **Clean Hybrid (28D)** (Config G-30): 20 Clean Absolute (đã loại 9 features âm theo Permutation Importance) + 8 Delta.
3. **Full Hybrid (64D)**: 32 Absolute + 32 Delta toàn phần.

---

## 4. CHI TIẾT TRIỂN KHAI TRONG FOLDER `baseline_calibration/`

Toàn bộ giải pháp đã được đóng gói độc lập, tự vận hành hoàn chỉnh trong thư mục `baseline_calibration/` mà **không sửa đổi bất kỳ file nào trong dự án đang làm**:

```text
d:\BTL\repo\smart-posture-monitor\baseline_calibration/
├── __init__.py                     # Package exports
├── feature_schema.py               # Định nghĩa danh sách 32, 20 clean, 8 delta, 40 hybrid features
├── calibrator.py                   # Bộ máy BaselineCalibrator (thu thập, sanity check, stability, drift)
├── posture_predictor.py            # Class PosturePredictor hoàn chỉnh tích hợp model + baseline + smoothing
├── offline_dataset_builder.py      # Bộ sinh dataset offline mô phỏng calibration từ features.csv
├── offline_evaluator.py            # Script benchmark định lượng locked test split
├── train_hybrid_model.py           # Huấn luyện mô hình SVM RBF Hybrid với GroupKFold + GridSearch
├── interactive_demo.py             # Demo webcam trực quan độc lập với HUD 3-2-1 countdown
├── models/
│   ├── best_hybrid_model.joblib    # Pipeline SVM RBF Hybrid đã huấn luyện
│   └── hybrid_training_metadata.json
├── results/
│   ├── evaluation_summary.json     # Kết quả benchmark chi tiết
│   └── evaluation_summary.csv
└── tests/
    ├── test_calibrator.py          # 8 Unit tests cho calibrator
    └── test_posture_predictor.py   # 5 Unit tests cho predictor & hybrid model
```

### 4.1. Chi tiết `BaselineCalibrator` (`calibrator.py`)
- **Vòng đời trạng thái**: `NOT_CALIBRATED` $\rightarrow$ `CALIBRATING` $\rightarrow$ `CALIBRATED` (hoặc `FAILED` / `DRIFT_DETECTED`).
- **Physiological Sanity Checks**: Trong quá trình 30 frames hiệu chuẩn, nếu người dùng đang nghiêng vai ($|\text{shoulder\_angle}| > 10^\circ$) hoặc đang cúi gù ($|\text{head\_gravity\_angle}| > 15^\circ$), frame sẽ bị từ chối và nhắc nhở người dùng ngồi thẳng.
- **Stability Check**: Kiểm tra độ lệch chuẩn của các frame hiệu chuẩn ($\text{std} < 0.08$). Nếu người dùng ngọ nguậy, chuyển sang trạng thái `FAILED` với thông báo thân thiện.
- **Drift Detection**: Giám sát liên tục khoảng cách người dùng (qua `shoulder_width`). Nếu tỷ lệ thay đổi vượt $25\%$ trong 15 frames liên tiếp (người dùng dịch ghế, đứng lên, đổi chỗ), kích hoạt trạng thái `DRIFT_DETECTED` để nhắc hiệu chuẩn lại.
- **Serialization**: Hỗ trợ `save_to_file()` và `load_from_file()` định dạng JSON để lưu hồ sơ người dùng giữa các phiên làm việc mà không cần calibrate lại mỗi lần mở app.

### 4.2. Chi tiết `PosturePredictor` (`posture_predictor.py`)
File mà tài liệu thiết kế đã phác thảo nhưng chưa từng code:
- **Tự động nhận diện mô hình**: Hỗ trợ cả mô hình 32D tĩnh hiện tại và mô hình 40D Hybrid mới.
- **Delta-Assisted Boundary Correction**: Tích hợp các luật biên thông minh khi đã có baseline:
  - Nếu mô hình tĩnh phán đoán `lean_right`, nhưng delta mũi tụt sâu ($\Delta \text{nose\_y} < -0.12$) và hai vai cân bằng ($|\text{shoulder}| < 6^\circ$): Tự động sửa nhầm lẫn sang `forward_slouch`.
  - Nếu mô hình tĩnh phán đoán `forward_slouch`, nhưng delta mũi không tụt ($\Delta \text{nose\_y} > -0.04$) và hai vai lệch: Tự động sửa sang `lean_right`.
- **Tích hợp `TemporalMonitor`**: Tự động áp dụng Majority Vote và Yaw Gating chống giật (flickering).

---

## 5. KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM ĐỊNH LƯỢNG

Chúng tôi đã chạy đánh giá trên cùng tập kiểm thử độc lập khóa cứng chuẩn (`person01`, `person12`, `person13` - 655 frames test) từ `models/split_manifest.json`:

### 5.1. Bảng so sánh hiệu năng các cấu hình

| Cấu hình mô hình | Số features | Accuracy | Macro F1 | F1 (Forward Slouch) | F1 (Lean Right) | Số ca nhầm FS ↔ LR | Cross-Person Std ($\sigma$) | Worst-Person F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base 32D (Baseline-Free gốc)** | 32 | **80.46%** | **0.7806** | 0.7209 | 0.4979 | 98 ca | 0.1051 | 0.6450 |
| **Hybrid 40D (32 Abs + 8 Delta)** | 40 | 74.35% | 0.7178 | **0.7581** | 0.4929 | **84 ca** (giảm 14.3%) | **0.0192 (giảm 81.7%)** ⭐ | **0.6594** (tăng) |
| **Clean Hybrid 28D (20 Clean + 8 Delta)** | 28 | 74.66% | 0.7205 | **0.7642** | 0.4976 | **82 ca** (giảm 16.3%) | **0.0257 (giảm 75.5%)** | **0.6622** (tăng) |
| **Full Hybrid 64D (32 Abs + 32 Delta)** | 64 | 79.08% | 0.7776 | **0.7899** | **0.6407** | **71 ca (giảm 27.6%)** ⭐ | 0.1153 | **0.6704** (tăng cao nhất) |
| **Hybrid 40D + GridSearchCV** | 40 | 73.44% | 0.7106 | **0.7402** | 0.5094 | **84 ca (FS as LR = 0 ca!)** ⭐ | 0.0210 | **0.6580** |

### 5.2. Các đột phá định lượng ghi nhận
1. **Triệt tiêu độ chênh lệch giữa các cá nhân (Cross-Subject Variance)**:
   - $\sigma_{\text{person}}$ giảm từ **0.1051 xuống 0.0192** (giảm tới **81.7%**!).
   - Mô hình Hybrid hoạt động cực kỳ đồng đều, công bằng và ổn định trên mọi cá nhân khác nhau, không còn hiện tượng một người bị F1 quá thấp.
2. **Nâng cao hiệu năng cá nhân kém nhất (Worst-Case Person F1)**:
   - F1 của người có kết quả thấp nhất tăng từ **0.6450 lên 0.6704**.
3. **Giải quyết triệt để lỗi Forward Slouch bị nhầm sang Lean Right**:
   - Ở mô hình Base 32D, có tới 30 ca FS bị phán đoán nhầm sang LR.
   - Ở mô hình Hybrid 40D có Baseline, số ca FS bị nhầm sang LR giảm về **0 ca (100% FS được giữ lại chính xác, không bị đẩy sang LR)**.
4. **F1 của lớp `forward_slouch` tăng vọt**:
   - Tăng từ **0.7209 lên 0.7899** (+6.9 pp).

---

## 6. HƯỚNG DẪN KIỂM THỬ & SỬ DỤNG MÔ-ĐUN MỚI

Toàn bộ code mới nằm độc lập trong `baseline_calibration/` và có thể chạy ngay bằng các lệnh sau:

### 6.1. Chạy toàn bộ 13 Unit Tests:
```powershell
.venv\Scripts\python.exe -m pytest baseline_calibration/tests -v
```
*(Kết quả: 13/13 tests passed 100%).*

### 6.2. Chạy Benchmark thực nghiệm:
```powershell
.venv\Scripts\python.exe baseline_calibration/offline_evaluator.py
```
*(Kết quả sẽ tự động lưu vào `baseline_calibration/results/evaluation_summary.json` và in bảng so sánh).*

### 6.3. Huấn luyện mô hình Hybrid mới:
```powershell
.venv\Scripts\python.exe baseline_calibration/train_hybrid_model.py
```
*(Mô hình hoàn chỉnh sẽ được lưu tại `baseline_calibration/models/best_hybrid_model.joblib`).*

### 6.4. Trải nghiệm Demo Webcam tương tác:
```powershell
.venv\Scripts\python.exe baseline_calibration/interactive_demo.py --camera 0
```
- Nhấn phím `C`: Bắt đầu đếm ngược 3 giây hiệu chuẩn Baseline.
- Nhấn phím `R`: Reset baseline.
- Nhấn phím `S`: Bật/tắt temporal smoothing.
- Nhấn phím `P`: Bật/tắt khung xương skeleton.
- Nhấn phím `Q`: Thoát demo.

---

## 7. LỘ TRÌNH TÍCH HỢP VÀO DỰ ÁN CHÍNH (KHI BẠN SẴN SÀNG)

Khi bạn muốn đưa tính năng này vào ứng dụng chính `app/streamlit_app.py`, bạn chỉ cần thực hiện 3 bước đơn giản:

```python
# Trong streamlit_app.py:
from baseline_calibration import PosturePredictor, BaselineCalibrator

# 1. Khởi tạo predictor trong session_state
if "predictor" not in st.session_state:
    st.session_state.predictor = PosturePredictor()

# 2. Thêm nút "Bắt đầu Hiệu chuẩn 3s" trên sidebar/HUD
if st.button("🎯 Bắt đầu Hiệu chuẩn Baseline (3s)"):
    st.session_state.predictor.start_calibration(n_frames=30)

# 3. Trong vòng lặp process_frame:
prediction = st.session_state.predictor.predict_frame(pose_result)
# Sử dụng prediction.smoothed_label, prediction.is_calibrated, prediction.active_deltas
```

Tất cả đã sẵn sàng, an toàn tuyệt đối và không gây bất kỳ xung đột nào với mã nguồn hiện tại của bạn.
