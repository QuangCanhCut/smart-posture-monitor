# Báo Cáo Tiến Độ Dự Án Smart Posture Monitor — Phiên Bản V03

> **Ngày thực hiện**: 27/09/2026  
> **Nhánh Git**: `feature/v03-depth-proxy`  
> **Tác giả / Người thực hiện**: Manh (`hmanh2k61009@gmail.com`)  
> **Trạng thái**: Hoàn thành nâng cấp Phase 1 V03 (Depth Proxy), trích xuất Dataset, Huấn luyện mô hình và Kiểm thử Webcam thành công.

---

## Mục lục

1. [Bối Cảnh & Mục Tiêu Công Việc](#1-bối-cảnh--mục-tiêu-công-việc)
2. [Chi Tiết Nội Dung Thay Đổi Mã Nguồn](#2-chi-tiết-nội-dung-thay-đổi-mã-nguồn)
3. [Xây Dựng Lại Tập Dữ Liệu (Dataset Building)](#3-xây-dựng-lại-tập-dữ-liệu-dataset-building)
4. [Cơ Chế Bảo Vệ & Sao Lưu Dữ Liệu V02](#4-cơ-chế-bảo-vệ--sao-lưu-dữ-liệu-v02)
5. [Huấn Luyện & Tuyển Chọn Mô Hình V03](#5-huấn-luyện--tuyển-chọn-mô-hình-v03)
6. [Kiểm Thử Thời Gian Thực Qua Webcam (Webcam Smoke Test)](#6-kiểm-thử-thời-gian-thực-qua-webcam-webcam-smoke-test)
7. [Rà Soát Kiến Trúc Hệ Thống (Các File Trống Trong `src/`)](#7-rà-soát-kiến-trúc-hệ-thống-các-file-trống-trong-src)
8. [Nhật Ký Quản Lý Phiên Bản Git (Git History)](#8-nhật-ký-quản-lý-phiên-bản-git-git-history)
9. [Tổng Hợp Các Lệnh Đã Thực Thi & Kết Quả](#9-tổng-hợp-các-lệnh-đã-thực-thi--kết-quả)
10. [Lộ Trình Các Bước Tiếp Theo](#10-lộ-trình-các-bước-tiếp-theo)

---

## 1. Bối Cảnh & Mục Tiêu Công Việc

### 1.1. Vấn đề tồn đọng ở phiên bản V02
Trong phiên bản V02 (29 đặc trưng hình học), mô hình đạt độ chính xác khá tốt trên tập kiểm thử độc lập (Test Accuracy 79.3%, Macro F1 77.9%). Tuy nhiên, hệ thống gặp phải **thách thức hình học cốt lõi**:
- Do camera đặt ở góc chếch **45° bên trái**, cả hai tư thế **cúi gù lưng (`forward_slouch`)** và **nghiêng người sang phải (`lean_right`)** khi chiếu lên mặt phẳng ảnh 2D đều làm khoảng cách đầu - vai bị co ngắn và hạ thấp.
- Kết quả kiểm thử V02 ghi nhận có tới **90 mẫu bị nhầm lẫn giữa hai tư thế này** (60 mẫu nghiêng phải bị đoán thành gù lưng, 30 mẫu gù lưng thành nghiêng phải).

### 1.2. Mục tiêu ngày 27/09/2026
1. Nâng cấp bộ trích xuất đặc trưng [FeatureExtractor](file:///d:/BTL/repo/smart-posture-monitor/src/feature_extractor.py) từ **29 lên 32 đặc trưng**, bổ sung 3 đặc trưng giả lập chiều sâu (**Depth Proxies**): D1, D3, D4.
2. Cập nhật pipeline dữ liệu và chạy trích xuất lại toàn bộ `data/processed/features.csv` từ 4,341 ảnh gốc.
3. Đóng gói và sao lưu an toàn toàn bộ artifact của V02 để không bị mất mát dữ liệu đối chứng.
4. Huấn luyện lại mô hình máy học V03 trên không gian 32 đặc trưng bằng notebook [02_training_experiments.ipynb](file:///d:/BTL/repo/smart-posture-monitor/notebooks/02_training_experiments.ipynb).
5. Kiểm thử trực tiếp luồng suy diễn webcam thời gian thực với mô hình V03 mới.
6. Commit toàn bộ thành quả an toàn lên nhánh `feature/v03-depth-proxy`.

---

## 2. Chi Tiết Nội Dung Thay Đổi Mã Nguồn

### 2.1. Nâng cấp `src/feature_extractor.py` (29 → 32 Features)
- **Thêm 3 đặc trưng mới vào cuối danh sách `FEATURE_NAMES`** (đảm bảo tính tương thích ngược, không làm xáo trộn 29 đặc trưng cũ):
  1. `face_shoulder_scale_ratio` (D1):
     $$\text{Area}(\text{left\_eye}, \text{right\_eye}, \text{nose}) / \text{shoulder\_width}^2$$
     - *Ý nghĩa vật lý*: Khi gù lưng tiến lại gần camera, tam giác mặt to lên tương đối so với vai; khi nghiêng xa camera, tam giác mặt thu nhỏ.
  2. `ear_nose_depth_proxy` (D3):
     $$\text{dist}(\text{left\_ear}, \text{nose}) / \text{dist}(\text{left\_ear}, \text{eye\_center})$$
     - *Ý nghĩa vật lý*: Khi cúi đầu vươn ra trước, khoảng cách tai - mũi bị kéo giãn trên hình chiếu; khi nghiêng phải, đầu dịch chuyển nguyên khối nên tỷ lệ giữ ổn định.
  3. `face_rotation_proxy` (D4):
     $$\text{dist}(\text{left\_eye}, \text{nose}) / \text{dist}(\text{right\_eye}, \text{nose})$$
     - *Ý nghĩa vật lý*: Phản ánh góc quay của mặt. Tư thế cúi gù hầu như không quay mặt (tỷ lệ $\approx 1.0$), trong khi nghiêng người thường kéo theo xoay đầu (tỷ lệ $\neq 1.0$).
- Cập nhật hàm `extract()`: Trả về vector `numpy.ndarray shape (32,)`.

### 2.2. Khử hardcode trong `src/dataset_builder.py`
- Sửa giá trị cố định `EXPECTED_FEATURE_COUNT = 29` thành lấy động từ class trích xuất:
  ```python
  EXPECTED_FEATURE_COUNT = len(FeatureExtractor.FEATURE_NAMES)
  ```
- Giúp dataset builder tự động thích ứng khi số lượng feature thay đổi mà không cần sửa code thủ công.

### 2.3. Đồng bộ hóa kiểm thử trong `src/preprocessing.py` & `tests/test_preprocessing.py`
- Hệ thống kiểm tra tiền xử lý tự động nhận danh sách 32 features từ `FeatureExtractor`.
- Đã chạy kiểm thử tự động toàn diện: **28/28 tests passed (100%)**.

### 2.4. Cập nhật `scripts/test_webcam_model.py`
- Nâng cấp schema validator: Nhận diện và kiểm tra tính toàn vẹn của mô hình 32 features.
- Cập nhật banner giao diện hiển thị: `SMART POSTURE MONITOR - V03 REALTIME WEBCAM TEST`.

---

## 3. Xây Dựng Lại Tập Dữ Liệu (Dataset Building)

### 3.1. Câu lệnh thực thi
```powershell
.\.venv\Scripts\python.exe -m src.dataset_builder
```

### 3.2. Kết quả trích xuất
- **Tổng số ảnh quét được**: 4,341 ảnh (từ thư mục `data/raw/`).
- **Số mẫu hợp lệ được trích xuất**: **4,014 / 4,341 mẫu (92.47%)**.
- **Số mẫu bị loại (do che khuất hoặc thiếu keypoint)**: 327 mẫu (ghi vào `data/rejected/rejected_images.csv`).
- **Phạm vi đối tượng**: Đầy đủ 14 đối tượng (`person01` đến `person14`) qua 16 sessions.
- **Tập tin xuất ra**: [data/processed/features.csv](file:///d:/BTL/repo/smart-posture-monitor/data/processed/features.csv) với 4,014 dòng và 36 cột (4 cột metadata + 32 cột features).

---

## 4. Cơ Chế Bảo Vệ & Sao Lưu Dữ Liệu V02

Để đảm bảo không làm mất mô hình gốc V02 (29 features) phục vụ việc so sánh đối chứng khoa học, hệ thống đã được tổ chức sao lưu:

### 4.1. Thư mục sao lưu
- **[models/v02/](file:///d:/BTL/repo/smart-posture-monitor/models/v02)**:
  - `best_model.joblib`: Trọng số mô hình SVM V02 cũ (29 features).
  - `split_manifest.json`: Khóa phân định train/test cũ.
  - `training_metadata.json`: Metadata cấu hình 29 features cũ.
- **[results/v02/](file:///d:/BTL/repo/smart-posture-monitor/results/v02)**:
  - Toàn bộ bảng điểm Cross-Validation (`baseline_cv_results.csv`, `svm_search_results.csv`, `xgboost_search_results.csv`).
  - Thư mục báo cáo kiểm thử độc lập V02 (`results/v02/evaluation/` gồm classification report, confusion matrix, per-person metrics).

### 4.2. Công cụ so sánh tự động V02 vs V03
- Xây dựng script [scripts/compare_v02_v03.py](file:///d:/BTL/repo/smart-posture-monitor/scripts/compare_v02_v03.py).
- Khi chạy script này, hệ thống sẽ tự động so sánh side-by-side các chỉ số:
  - Accuracy, Macro F1, Balanced Accuracy.
  - **Số ca nhầm lẫn giữa `forward_slouch` và `lean_right`** (trước đây là 90 ca).
  - Điểm F1 chi tiết từng lớp và từng người kiểm thử độc lập (`person01`, `person10`, `person12`).

---

## 5. Huấn Luyện & Tuyển Chọn Mô Hình V03

### 5.1. Quy trình huấn luyện
Được thực hiện thông qua notebook [notebooks/02_training_experiments.ipynb](file:///d:/BTL/repo/smart-posture-monitor/notebooks/02_training_experiments.ipynb):
1. **Đọc dữ liệu chuẩn hóa**: Gọi hàm `prepare_dataset('data/processed/features.csv')`, nạp 4,014 mẫu với 32 đặc trưng.
2. **Khóa tập kiểm thử độc lập (Locked Holdout Split)**:
   - Dùng `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
   - Phân chia: 11 người tập huấn luyện (3,307 mẫu) và 3 người tập kiểm thử (`person01`, `person10`, `person12` - 707 mẫu).
   - Đảm bảo **Locked Test Set** hoàn toàn đồng nhất với V02 để việc đối chứng đạt độ tin cậy tuyệt đối.
3. **Cross-Validation**: 5-fold `StratifiedGroupKFold` trên 11 người tập train, chống rò rỉ danh tính đối tượng (`person leakage = []`).
4. **Tìm kiếm siêu tham số**: Chạy `GridSearchCV` trên mô hình SVM RBF (160 lần fit) và `RandomizedSearchCV` trên XGBoost (100 lần fit).

### 5.2. Kết quả tuyển chọn mô hình V03
- **Mô hình chiến thắng**: **SVM RBF Tuned**
  - Siêu tham số tối ưu: `C = 1`, `gamma = 0.001`, `class_weight = None`.
  - Điểm trung bình CV Macro F1: **0.5949 ± 0.1719**.
- **Artifacts xuất ra**:
  - [models/best_model.joblib](file:///d:/BTL/repo/smart-posture-monitor/models/best_model.joblib) (kích thước 777 KB, chứa Pipeline StandardScaler + SVM 32 features).
  - [models/training_metadata.json](file:///d:/BTL/repo/smart-posture-monitor/models/training_metadata.json) (đã cập nhật danh sách 32 features, SHA256 dataset mới).
  - [models/split_manifest.json](file:///d:/BTL/repo/smart-posture-monitor/models/split_manifest.json).

---

## 6. Kiểm Thử Thời Gian Thực Qua Webcam (Webcam Smoke Test)

### 6.1. Khắc phục lỗi kiểm tra Schema
- *Hiện tượng ban đầu*: Khi chạy webcam với code FeatureExtractor mới (32 features) trên mô hình cũ (29 features), hệ thống ném ngoại lệ:
  `RuntimeError: FeatureExtractor hiện tại KHÔNG khớp model đã train.`
- *Khắc phục*: Sau khi hoàn tất huấn luyện mô hình V03 ở Mục 5, cả model và FeatureExtractor đều đã thống nhất 32 đặc trưng.

### 6.2. Thực thi kiểm thử Webcam
```powershell
.\.venv\Scripts\python.exe scripts/test_webcam_model.py
```
- **Kết quả**:
  - Tải thành công mô hình `best_model.joblib` và metadata.
  - Nhận diện đủ 32 features và 4 classes: `{0: 'correct', 1: 'forward_slouch', 2: 'lean_left', 3: 'lean_right'}`.
  - Mở camera thành công, nhận diện khung xương người qua YOLO Pose, trích xuất 32 features và phân loại tư thế mượt mà.
  - Thoát chương trình thành công với mã **Exit Code: 0**.

---

## 7. Rà Soát Kiến Trúc Hệ Thống (Các File Trống Trong `src/`)

Hệ thống đã phân tích và làm rõ vai trò của 6 file đang có kích thước 0 bytes trong thư mục `src/`:

| Tên File | Vai Trò & Mục Đích Kiến Trúc | Kế Hoạch Triển Khai |
|---|---|---|
| `__init__.py` | Định danh Python package chuẩn cho `src/`. | Giữ nguyên trống (chuẩn Python). |
| `train.py` | CLI Script đóng gói toàn bộ logic train từ notebook thành lệnh chạy một dòng (`python -m src.train`). | Triển khai ở Phase hoàn thiện sản phẩm. |
| `posture_predictor.py` | High-level API bọc `PoseDetector + FeatureExtractor + Model` thành hàm đơn giản `predict(frame)`. Tích hợp `calibrate()` cho baseline cá nhân. | Triển khai để phục vụ App UI. |
| `temporal_monitor.py` | Bộ lọc mượt theo chuỗi thời gian (Rolling Window + Majority Voting). Chống nhảy nhãn (flickering) và lọc nhiễu tức thời. | Triển khai để chống giật nhãn khi stream webcam. |
| `session_statistics.py` | Theo dõi phiên ngồi, thống kê thời gian ngồi chuẩn/sai, tính điểm tư thế (Posture Score: 0-100) và cảnh báo ngồi quá lâu. | Triển khai để phục vụ Dashboard theo dõi sức khỏe. |
| `inference.py` | API suy diễn ngoại tuyến cho ảnh tĩnh hoặc file video lưu trữ trên ổ đĩa. | Triển khai phục vụ batch testing. |

---

## 8. Nhật Ký Quản Lý Phiên Bản Git (Git History)

Toàn bộ công việc trong ngày đã được commit an toàn trên nhánh tính năng:
- **Nhánh hiện tại**: `feature/v03-depth-proxy`
- **Các commit đã thực hiện**:
  1. `61d4a36` — *feat: hoan thanh V03 depth proxy (32 features), train model va backup v02*
     - Thay đổi 33 files (1,999 dòng thêm mới, 576 dòng sửa đổi).
     - Di chuyển các tài liệu thiết kế vào thư mục `docs/`.
     - Lưu trữ toàn bộ thư mục `models/v02/` và `results/v02/`.
     - Cập nhật mô hình `best_model.joblib` và metadata 32 features.
  2. `7508169` — *thay the 1 ky tu*
     - Cập nhật tiêu đề hiển thị trong `scripts/test_webcam_model.py` thành V03.
- **Tính an toàn**: Nhánh chính (`main`) hoàn toàn không bị ảnh hưởng.

---

## 9. Tổng Hợp Các Lệnh Đã Thực Thi & Kết Quả

| STT | Câu Lệnh Thực Thi | Mục Đích | Kết Quả |
|:---:|---|---|---|
| 1 | `.\.venv\Scripts\pytest tests/test_preprocessing.py` | Kiểm tra unit tests tiền xử lý dữ liệu | **28/28 passed (100%)** |
| 2 | `.\.venv\Scripts\python.exe -m src.dataset_builder` | Quét ảnh và trích xuất 32 features | **4,014 mẫu hợp lệ (92.47%)** |
| 3 | `.\.venv\Scripts\python.exe scripts/test_webcam_model.py` (Lần 1) | Thử nghiệm webcam với model cũ | Phát hiện schema mismatch (hành vi bảo vệ đúng) |
| 4 | Chạy notebook `02_training_experiments.ipynb` | Huấn luyện SVM/XGBoost trên 32 features | Tuyển chọn mô hình SVM RBF Tuned thành công |
| 5 | `.\.venv\Scripts\python.exe scripts/test_webcam_model.py` (Lần 2) | Kiểm thử webcam với model V03 mới | **Chạy thành công mượt mà, Exit Code 0** |
| 6 | `git commit` & `git push` | Lưu trữ lịch sử mã nguồn lên GitHub | Hoàn tất sạch sẽ trên `feature/v03-depth-proxy` |

---

## 10. Lộ Trình Các Bước Tiếp Theo

1. **Đánh giá định lượng V03 trên Locked Test Set**:
   - Chạy script đánh giá độc lập:
     ```powershell
     .\.venv\Scripts\python.exe -m src.evaluate
     ```
   - Lệnh này sẽ tính toán ma trận nhầm lẫn và metrics chính thức của V03 trên 707 mẫu của `person01`, `person10`, `person12`.
2. **Chạy so sánh đối chứng V02 vs V03**:
   - Chạy script so sánh:
     ```powershell
     .\.venv\Scripts\python.exe scripts/compare_v02_v03.py
     ```
   - Đo lường xem 3 đặc trưng Depth Proxy đã kéo giảm số lượng 90 ca nhầm lẫn giữa `forward_slouch` và `lean_right` xuống còn bao nhiêu.
3. **Hiện thực hóa tầng ứng dụng thời gian thực**:
   - Xây dựng [src/temporal_monitor.py](file:///d:/BTL/repo/smart-posture-monitor/src/temporal_monitor.py) để chống giật nhãn webcam.
   - Xây dựng [src/posture_predictor.py](file:///d:/BTL/repo/smart-posture-monitor/src/posture_predictor.py) và chuẩn bị cho Phase 2 (Personal Baseline Calibration).
