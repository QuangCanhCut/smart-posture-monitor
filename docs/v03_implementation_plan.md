# V03 Phase 1: Đánh giá toàn bộ codebase & Kế hoạch triển khai

## 1. Đánh giá từng file — Cần thay đổi gì?

### Tổng quan nhanh

| File | Trạng thái | Cần sửa? | Mức độ |
|---|---|---|---|
| [`src/feature_extractor.py`](file:///d:/BTL/repo/smart-posture-monitor/src/feature_extractor.py) | Core — cần sửa | ✅ **SỬA** | **LỚN** |
| [`src/dataset_builder.py`](file:///d:/BTL/repo/smart-posture-monitor/src/dataset_builder.py) | Hardcode `29` | ✅ **SỬA** | NHỎ |
| [`src/preprocessing.py`](file:///d:/BTL/repo/smart-posture-monitor/src/preprocessing.py) | Tự lấy từ `FeatureExtractor.FEATURE_NAMES` | ⚠️ **SỬA NHỎ** | NHỎ |
| [`src/evaluate.py`](file:///d:/BTL/repo/smart-posture-monitor/src/evaluate.py) | Dùng `prepare_dataset()` | ❌ Không sửa | — |
| [`src/pose_detector.py`](file:///d:/BTL/repo/smart-posture-monitor/src/pose_detector.py) | Không liên quan | ❌ Không sửa | — |
| [`scripts/test_webcam_model.py`](file:///d:/BTL/repo/smart-posture-monitor/scripts/test_webcam_model.py) | Hardcode `29` | ✅ **SỬA** | NHỎ |
| [`tests/test_preprocessing.py`](file:///d:/BTL/repo/smart-posture-monitor/tests/test_preprocessing.py) | Có thể hardcode `29` | ⚠️ **KIỂM TRA** | NHỎ |
| [`models/training_metadata.json`](file:///d:/BTL/repo/smart-posture-monitor/models/training_metadata.json) | V02 artifact — **KHÔNG SỬA** | ❌ | — |

---

## 2. Phân tích chi tiết từng file

### 2.1. `src/feature_extractor.py` — **THAY ĐỔI CHÍNH**

**Hiện trạng:**
- `FEATURE_NAMES`: danh sách 29 tên feature cố định
- `extract()`: trả về `np.array shape (29,)`
- Docstring ghi cứng `29 features`

**Cần làm:**
1. Thêm 4 feature mới vào `FEATURE_NAMES` (cuối danh sách để backward-compatible):
   - `face_shoulder_scale_ratio` (D1)
   - `ear_nose_depth_proxy` (D3)
   - `face_rotation_proxy` (D4)
   - `eye_width_ratio` (D2) — **ĐÃ CÓ SẴN** ở vị trí #12 → chỉ cần thêm 3 feature mới

2. Thêm logic tính toán 3 feature mới trong `extract()`:
   - **D1**: Diện tích tam giác (`left_eye`, `right_eye`, `nose`) / `shoulder_width²`
   - **D3**: `distance(left_ear, nose)` / `distance(left_ear, eye_center)`
   - **D4**: `distance(left_eye, nose)` / `distance(right_eye, nose)`

3. Cập nhật docstring: `29 → 32 features`

4. Cập nhật `features` array cuối cùng: thêm 3 giá trị mới

> [!IMPORTANT]
> Feature D2 (`eye_width_ratio`) đã tồn tại ở vị trí #12. Tín hiệu depth của nó chỉ mạnh khi kết hợp baseline calibration (Phase 2). Phase 1 không cần thêm gì cho D2.

**Thứ tự feature cuối cùng (32 features):**
```
Features 1-29: Giữ nguyên V02
Feature 30: face_shoulder_scale_ratio  (D1)
Feature 31: ear_nose_depth_proxy       (D3)
Feature 32: face_rotation_proxy        (D4)
```

---

### 2.2. `src/dataset_builder.py` — SỬA NHỎ

**Hiện trạng:**
- Dòng 61: `EXPECTED_FEATURE_COUNT = 29` — **HARDCODED**
- Dòng 111-118: Kiểm tra `len(self.feature_names) != self.EXPECTED_FEATURE_COUNT`

**Cần làm:**
- Thay `EXPECTED_FEATURE_COUNT = 29` thành lấy động từ `FeatureExtractor`:
  ```python
  EXPECTED_FEATURE_COUNT = len(FeatureExtractor.FEATURE_NAMES)
  ```
- Hoặc đơn giản hơn: đổi `29 → 32`

> [!TIP]
> Cách tốt nhất là lấy động từ `FeatureExtractor.FEATURE_NAMES` để tránh phải sửa lại mỗi khi thêm feature. Tuy nhiên hiện tại `DatasetBuilder` đã import `FeatureExtractor`, nên chỉ cần đổi hằng số.

---

### 2.3. `src/preprocessing.py` — SỬA NHỎ

**Hiện trạng:**
- Dòng 232-234: `EXPECTED_FEATURE_COLUMNS = list(FeatureExtractor.FEATURE_NAMES)` — **TỰ ĐỘNG**
- Dòng 236-238: `EXPECTED_NUM_FEATURES = len(...)` — **TỰ ĐỘNG**
- Docstring nhiều chỗ ghi `29 features` — cần cập nhật text

**Cần làm:**
- Logic code: **Không cần sửa** — vì nó đã lấy từ `FeatureExtractor.FEATURE_NAMES` tự động
- Docstring/comments: Cập nhật các chỗ ghi cứng `29` thành `32` hoặc viết generic hơn

---

### 2.4. `scripts/test_webcam_model.py` — SỬA NHỎ

**Hiện trạng:**
- Dòng 5: Docstring `FeatureExtractor (29 features)`
- Dòng 111-115: `if len(feature_columns) != 29` — **HARDCODED**

**Cần làm:**
- Đổi `29` thành `len(FeatureExtractor.FEATURE_NAMES)` hoặc `32`
- Cập nhật docstring

> [!WARNING]
> File này kiểm tra feature schema với `training_metadata.json`. Model V02 hiện tại vẫn ghi `n_features: 29`. Khi chạy webcam test với model V02 cũ + FeatureExtractor V03 mới (32 features) → **sẽ bị lỗi schema mismatch**. Đây là hành vi **đúng và mong muốn** — buộc phải train lại model mới trước khi test realtime.

---

### 2.5. `src/pose_detector.py` — KHÔNG SỬA

Pose detector chỉ trả về 6 keypoints. Tất cả 3 feature mới đều tính từ chính 6 keypoints này. Không cần thay đổi.

---

### 2.6. `src/evaluate.py` — KHÔNG SỬA

Evaluate dùng `prepare_dataset()` từ preprocessing, tự động nhận đúng số feature. Chỉ cần rebuild `features.csv` mới và train model mới.

---

## 3. Kế hoạch thực thi Phase 1

### Thứ tự thực hiện

```text
Bước 1: Sửa src/feature_extractor.py
        ├── Thêm 3 tên feature mới vào FEATURE_NAMES
        ├── Thêm logic tính D1, D3, D4 trong extract()
        ├── Thêm 3 giá trị vào array cuối cùng
        └── Cập nhật docstring (29 → 32)

Bước 2: Sửa src/dataset_builder.py
        └── EXPECTED_FEATURE_COUNT = 29 → dynamic

Bước 3: Sửa src/preprocessing.py
        └── Cập nhật docstring/comments (29 → 32)

Bước 4: Sửa scripts/test_webcam_model.py
        └── Bỏ hardcode 29, dùng dynamic

Bước 5: Kiểm tra tests/test_preprocessing.py
        └── Cập nhật nếu có hardcode 29

Bước 6 (sau khi có data): Rebuild features.csv
        └── python -m src.dataset_builder

Bước 7 (sau khi rebuild): Train & Evaluate
        └── Notebook mới hoặc train script
```

### Nguyên tắc an toàn

- **Không sửa** `models/training_metadata.json` hay `models/best_model.joblib` — đó là artifact V02
- **Không sửa** thứ tự 29 feature cũ — chỉ **append** 3 feature mới ở cuối
- **Giữ nguyên** toàn bộ logic tính toán 29 feature hiện có
- Feature mới dùng cùng `shoulder_width` làm khuôn chuẩn hóa (nhất quán với V02)

---

## 4. Chi tiết công thức 3 Feature mới

### D1: `face_shoulder_scale_ratio`
```python
# Diện tích tam giác (left_eye, right_eye, nose) / shoulder_width²
face_triangle_area = 0.5 * abs(
    (right_eye[0] - left_eye[0]) * (nose[1] - left_eye[1])
  - (nose[0] - left_eye[0]) * (right_eye[1] - left_eye[1])
)
face_shoulder_scale_ratio = face_triangle_area / (shoulder_width ** 2)
```
- Forward slouch: mặt gần camera → tam giác LỚN → ratio **TĂNG**
- Lean right: mặt xa camera → tam giác NHỎ → ratio **GIẢM**

### D3: `ear_nose_depth_proxy`
```python
ear_nose_dist = distance(left_ear, nose)
ear_eye_dist = distance(left_ear, eye_center)
ear_nose_depth_proxy = ear_nose_dist / max(ear_eye_dist, 1e-6)
```
- Forward slouch: mũi vươn ra trước → ear-nose tăng → ratio **TĂNG**
- Lean right: đầu dịch đồng bộ → ratio **ỔN ĐỊNH**

### D4: `face_rotation_proxy`
```python
left_eye_nose_dist = distance(left_eye, nose)
right_eye_nose_dist = distance(right_eye, nose)
face_rotation_proxy = left_eye_nose_dist / max(right_eye_nose_dist, 1e-6)
```
- Forward slouch: không xoay → ratio **≈ 1.0**
- Lean right: đầu xoay → ratio **≠ 1.0**
