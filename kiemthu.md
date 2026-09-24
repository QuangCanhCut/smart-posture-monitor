# BÁO CÁO KIỂM THỬ TIỀN ĐIỀU KIỆN (BƯỚC 1) - LẦN 2
**Project:** Smart Posture Monitor  
**Mục tiêu:** Kiểm tra lại 3 điều kiện tiên quyết trước khi chạy `src/dataset_builder.py`  
**Thời gian kiểm tra:** 24/09/2026 - 17:58  
**Nhánh Git:** `dev-manh`  
**Kết quả chung:** ✅ **TẤT CẢ CÁC HẠNG MỤC ĐÃ ĐẠT CHUẨN (100% SẴN SÀNG)**

---

## 1. Bảng tổng hợp trạng thái kiểm tra

| STT | Hạng mục kiểm tra | Vị trí yêu cầu | Trạng thái | Chi tiết kiểm tra |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Thư mục ảnh đầu vào | `data/raw/` | ✅ **ĐẠT** | Đã có 4/4 nhãn, tổng cộng **3.286** file ảnh |
| **2** | File mô hình YOLO Pose | `models/yolo26n-pose.pt` | ✅ **ĐẠT** | File trọng số tồn tại (~7.51 MB) |
| **3** | Môi trường Python & Dependencies | `.venv/` & `requirements.txt` | ✅ **ĐẠT** | Đã cài đủ `ultralytics`, `opencv-python`, `numpy` |

---

## 2. Chi tiết kết quả kiểm tra từng hạng mục

### 2.1. Hạng mục 1: Thư mục dữ liệu ảnh `data/raw/`
* **Vị trí:** `d:\BTL\repo\smart-posture-monitor\data\raw`
* **Cấu trúc:** Có đủ 4 thư mục nhãn chuẩn theo quy ước dự án.
* **Số lượng và phân bố mẫu (Class Distribution):**
  * `correct`: **814** ảnh
  * `forward_slouch`: **791** ảnh
  * `lean_left`: **861** ảnh
  * `lean_right`: **820** ảnh
  * **Tổng cộng:** **3.286** ảnh
* **Đánh giá:** Tỷ lệ phân bố giữa 4 lớp rất cân bằng (~800 ảnh/nhãn), không bị mất cân bằng dữ liệu (imbalanced data). Các thư mục con lưu giữ đầy đủ thông tin `person_id` và `session_id`.

---

### 2.2. Hạng mục 2: File mô hình YOLO Pose
* **Vị trí:** `d:\BTL\repo\smart-posture-monitor\models\yolo26n-pose.pt`
* **Kích thước file:** 7.878.574 bytes (~7.51 MB).
* **Kiểm tra nạp mô hình:** Khởi tạo thành công class `PoseDetector` và load trọng số YOLO Pose vào bộ nhớ mà không gặp lỗi.

---

### 2.3. Hạng mục 3: Môi trường Python & Thư viện phụ thuộc
* **Môi trường:** `.venv` đã được tạo và kích hoạt.
* **Thư viện đã cài đặt:**
  * `ultralytics` (v8.4.161)
  * `opencv-python` (v5.0.0.93)
  * `numpy` (v2.5.3)
* **Kiểm thử tích hợp (Integration Test):**
  * Đã chạy lệnh test khởi tạo đồng thời cả 2 module:
    ```python
    from src.pose_detector import PoseDetector
    from src.feature_extractor import FeatureExtractor
    detector = PoseDetector()
    extractor = FeatureExtractor()
    ```
  * Kết quả: **Khởi tạo thành công hoàn toàn (`Exit code 0`).**

---

## 3. Kết luận & Bước tiếp theo

Toàn bộ các điều kiện cần thiết đã được đáp ứng đầy đủ và chuẩn xác.

Hiện tại bạn đã có thể tiến hành chạy lệnh xuất dữ liệu sang CSV:
```bash
python -m src.dataset_builder
```
Khi chạy xong, file bảng dữ liệu sạch sẽ được lưu tại:
`d:\BTL\repo\smart-posture-monitor\data\processed\features.csv`
