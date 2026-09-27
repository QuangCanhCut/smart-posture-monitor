# V03 Feature Design: Baseline Calibration + Depth Proxy

## Bối cảnh hiện tại

Confusion matrix V02 cho thấy hai vấn đề **độc lập** nhau:

```
                correct  forward_slouch  lean_left  lean_right
correct             161               2          0           9
forward_slouch        0             102          0          30    ← 30 mẫu bị nhầm sang lean_right
lean_left             9              17        196           4
lean_right           14              60          1         102    ← 60 mẫu bị nhầm sang forward_slouch
```

| Vấn đề | Nguyên nhân gốc | Giải pháp |
|---|---|---|
| Cross-subject variance (F1 per-person dao động 0.667 → 0.851) | Nhân trắc học khác nhau giữa các người | **Baseline Calibration** |
| `lean_right` ↔ `forward_slouch` (90 mẫu nhầm / 707 test) | Ảnh chiếu 2D của hai chuyển động 3D khác nhau trùng nhau ở góc camera 45° | **Depth Proxy Features** |

**Hai giải pháp này bổ sung cho nhau, không thay thế nhau.**

---

## I. Thiết kế cụ thể: Depth Proxy Features

### Tại sao `lean_right` ≈ `forward_slouch` ở góc camera 45° bên trái?

```text
Camera đặt ở góc ~45° bên trái người dùng:

         Camera
           \  45°
            \
             \
              [Người ngồi]

Forward Slouch (gù/cúi):
  - Đầu dịch XUỐNG + RA TRƯỚC (theo trục Z, hướng vào camera)
  - Trong ảnh 2D: đầu hạ thấp, khoảng cách đầu-vai co lại
  - Eye width tăng nhẹ (mặt tiến gần camera hơn)

Lean Right (nghiêng phải):
  - Đầu dịch SANG PHẢI + RA XA camera (phải = xa ở góc 45° trái)
  - Trong ảnh 2D: đầu cũng hạ thấp, khoảng cách đầu-vai cũng co lại
  - Eye width giảm hoặc giữ nguyên (mặt xa camera hơn)
```

**Cốt lõi**: Cả hai đều khiến **đầu hạ thấp** trong ảnh 2D. Nhưng chúng khác nhau ở **hướng dịch chuyển trong không gian 3D** — mà dấu vết duy nhất còn lại trong ảnh 2D là **sự thay đổi kích thước biểu kiến** (apparent size change) của các landmark trên mặt.

### Nguyên lý Depth Proxy: kích thước biểu kiến = f(khoảng cách Z)

Với camera perspective, khi một vật thể dịch chuyển theo trục Z (gần/xa camera):

$$\text{kích thước pixel} \propto \frac{1}{Z}$$

Điều này có nghĩa:
- **Forward slouch**: mặt tiến gần camera → khoảng cách giữa các landmark mặt **tăng** trong pixel
- **Lean right**: mặt xa camera → khoảng cách giữa các landmark mặt **giảm** trong pixel

> [!IMPORTANT]
> Dấu hiệu phân biệt không nằm ở vị trí tuyệt đối của đầu (cả hai đều hạ), mà nằm ở **tỷ lệ kích thước khuôn mặt so với khung vai**.

### 4 Depth Proxy Features đề xuất

Tất cả đều tính được từ **6 keypoints hiện tại** — không cần thêm keypoint hay model mới:

#### Feature D1: `face_shoulder_scale_ratio`

```python
# Diện tích tam giác mặt (left_eye, right_eye, nose) / shoulder_width²
#
# Ý nghĩa vật lý:
#   - Tam giác (left_eye, right_eye, nose) là proxy cho "kích thước mặt trong ảnh"
#   - Chia cho shoulder_width² để chuẩn hóa (đã có sẵn)
#   - Forward slouch: mặt gần camera → tam giác LỚN → ratio TĂNG
#   - Lean right:     mặt xa camera  → tam giác NHỎ → ratio GIẢM
#
# Công thức (cross product / 2):

face_triangle_area = 0.5 * abs(
    (right_eye[0] - left_eye[0]) * (nose[1] - left_eye[1])
  - (nose[0] - left_eye[0]) * (right_eye[1] - left_eye[1])
)

face_shoulder_scale_ratio = face_triangle_area / (shoulder_width ** 2)
```

> [!TIP]
> Đây là feature **mạnh nhất** trong nhóm depth proxy vì nó dùng 3 điểm thay vì 2 → ít nhạy cảm với nhiễu keypoint đơn lẻ, và diện tích là đại lượng **bậc hai** nên khuếch đại sự khác biệt depth.

#### Feature D2: `eye_width_shoulder_change`

```python
# eye_width / shoulder_width — đã có sẵn (feature #12: eye_width_ratio)
# Nhưng khi kết hợp với baseline calibration:
#   Δ(eye_width_ratio) = eye_width_ratio_now - eye_width_ratio_baseline
#
# Forward slouch: eye_width tăng (mặt gần camera) → Δ > 0
# Lean right:     eye_width giảm (mặt xa camera)  → Δ < 0
#
# → Đã có trong V02, nhưng tín hiệu chỉ mạnh khi có baseline để tính delta.
# → Đây là ví dụ điển hình: depth proxy + baseline calibration bổ trợ nhau.
```

#### Feature D3: `ear_nose_depth_proxy`

```python
# Khoảng cách từ left_ear đến nose / shoulder_width
# so với khoảng cách từ left_ear đến eye_center / shoulder_width
#
# Ý nghĩa:
#   - Khi gù (forward slouch), mũi vươn ra trước →
#     nose xa ear hơn (cùng phía gần camera), nhưng ear-eye giữ nguyên
#   - Khi nghiêng phải (lean right), toàn bộ đầu dịch →
#     ear-nose và ear-eye thay đổi đồng bộ
#
# Feature = (ear_nose_distance / ear_eye_distance) — unitless ratio

ear_nose_dist = distance(left_ear, nose)
ear_eye_dist = distance(left_ear, eye_center)

ear_nose_depth_proxy = ear_nose_dist / max(ear_eye_dist, 1e-6)
```

#### Feature D4: `face_rotation_proxy`

```python
# Tỷ lệ khoảng cách left_eye→nose / right_eye→nose
#
# Ý nghĩa:
#   - Khi mặt hướng thẳng: tỷ lệ ≈ 1.0
#   - Forward slouch: đầu cúi xuống nhưng vẫn hướng về phía trước → tỷ lệ ≈ 1.0
#   - Lean right: đầu xoay sang phải → mắt phải gần mũi hơn, mắt trái xa hơn
#     → tỷ lệ > 1.0 (hoặc < 1.0 tùy hướng)
#
# Đây là proxy cho YAW rotation của đầu — cốt lõi của sự khác biệt 3D

left_eye_nose_dist = distance(left_eye, nose)
right_eye_nose_dist = distance(right_eye, nose)

face_rotation_proxy = left_eye_nose_dist / max(right_eye_nose_dist, 1e-6)
```

> [!NOTE]
> Feature D4 khai thác thông tin **asymmetry nội tại khuôn mặt** — thứ mà V02 chưa có. V02 có `nose_shoulder_asymmetry` và `eye_shoulder_asymmetry` (asymmetry đầu-vai), nhưng chưa có asymmetry **bên trong khuôn mặt** (eye-nose left vs right). Đây là tín hiệu rất khác biệt.

### Tại sao chỉ 4 features là đủ?

Bốn feature trên tấn công vấn đề từ 4 góc khác nhau:

| Feature | Đo cái gì | Forward Slouch | Lean Right |
|---|---|---|---|
| D1: `face_shoulder_scale_ratio` | Diện tích mặt / vai² | **Tăng** (mặt gần camera) | **Giảm** (mặt xa camera) |
| D2: `Δ(eye_width_ratio)` | Chiều ngang mắt / vai (delta) | **> 0** | **< 0** |
| D3: `ear_nose_depth_proxy` | Tỷ lệ ear-nose / ear-eye | **Tăng** (nose vươn ra) | **Ổn định** (dịch đồng bộ) |
| D4: `face_rotation_proxy` | Left-eye-nose / Right-eye-nose | **≈ 1.0** (không xoay) | **≠ 1.0** (đầu xoay) |

Nếu SVM RBF nhận được cả 4 tín hiệu này, nó sẽ tạo được một **hyperplane phi tuyến trong không gian 4D phụ** tách riêng `forward_slouch` ra khỏi `lean_right` — điều mà 29 feature V02 hiện tại không làm được vì thiếu thông tin depth.

---

## II. Thiết kế cụ thể: Baseline Calibration

### Cách mô phỏng offline đúng (tránh lạc quan giả tạo)

```python
def compute_baseline_offline(df, person_id, session_id, feature_cols, n_calib_frames=30):
    """
    Mô phỏng calibration thực tế:
    chỉ lấy N frame `correct` ĐẦU TIÊN trong session.
    
    KHÔNG lấy toàn bộ frame correct rải rác suốt session
    (vì production chỉ có 2-3 giây calibration ban đầu).
    """
    mask = (
        (df["person_id"] == person_id) &
        (df["session_id"] == session_id) &
        (df["label"] == "correct")
    )
    correct_frames = df.loc[mask, feature_cols]
    
    # Chỉ lấy N frame đầu tiên — giống production calibration window
    calib_frames = correct_frames.head(n_calib_frames)
    
    if len(calib_frames) < 10:
        # Không đủ frame calibration → fallback dùng toàn bộ correct
        # nhưng đánh dấu để tracking
        calib_frames = correct_frames
    
    baseline = calib_frames.mean()  # Vector trung bình f_base
    return baseline
```

### Feature scheme: Hybrid (Absolute + Delta)

```python
# Đừng chỉ dùng Δf — giữ cả feature tuyệt đối làm "safety net"

# Feature vector cuối cùng cho mỗi frame:
final_features = np.concatenate([
    f_absolute,      # 29 + 4 = 33 features (V02 + depth proxy)
    f_delta,         # 33 delta features (f_t - f_baseline)
])
# Tổng: 66 features

# Hoặc phiên bản conservative hơn — chỉ delta cho nhóm nhạy cảm:
delta_cols = [
    "nose_y_body",               # Head drop — ambiguous feature chính
    "head_mean_height",          # Cũng ambiguous
    "nose_gravity_angle",        # Cũng ambiguous
    "head_gravity_angle",        # Cũng ambiguous
    "face_shoulder_scale_ratio", # Depth proxy — cần delta mới mạnh
    "eye_width_ratio",           # Depth proxy — cần delta mới mạnh
    "face_rotation_proxy",       # Yaw proxy
    "ear_nose_depth_proxy",      # Depth proxy
]

final_features = np.concatenate([
    f_absolute_all_33,           # 33 absolute features
    f_delta_selected,            # 8 delta features cho nhóm ambiguous
])
# Tổng: 41 features — gọn hơn, ít overfitting hơn
```

> [!WARNING]
> **Tại sao giữ feature tuyệt đối?**
> Nếu baseline bị sai (người dùng calibrate lúc đang gù), thuần $\Delta \vec{f}$ sẽ báo "correct" khi thực tế đang gù. Feature tuyệt đối cung cấp **ràng buộc sinh lý cứng** — ví dụ `head_gravity_angle > 25°` thì dù baseline nói gì, cũng gần như chắc chắn đang gù. Model sẽ tự học cân bằng giữa hai luồng tín hiệu.

---

## III. Kế hoạch thực nghiệm đề xuất

### Phase 1: Thêm Depth Proxy vào V02 (không cần baseline)

```text
Mục tiêu:  Xác nhận depth proxy tách được lean_right / forward_slouch
Thay đổi:  FeatureExtractor: 29 → 33 features (thêm D1, D2, D3, D4)
Dataset:   Rebuild features.csv với 33 features
Training:  Cùng split manifest, cùng locked test persons
Metric:    So sánh F1(forward_slouch) và F1(lean_right) trước/sau
Expected:  F1 cặp này tăng 5-10 điểm, tổng Macro F1 tăng 2-5 điểm
```

Đây là bước **an toàn nhất** vì:
- Không cần thay đổi luồng inference (không cần calibration step)
- Không cần sửa UI/UX
- Backward-compatible hoàn toàn
- Cùng locked split → so sánh trực tiếp được

### Phase 2: Thêm Baseline Calibration (offline backtest)

```text
Mục tiêu:  Xác nhận baseline giảm cross-subject variance
Thay đổi:  Preprocessing: thêm delta features
           Training: dùng compute_baseline_offline() chỉ trên N frame đầu
Dataset:   Cùng features.csv từ Phase 1 (33 features)
Training:  33 absolute + 8-33 delta = 41-66 features
Metric:    So sánh F1 per-person variance trước/sau
Expected:  Per-person F1 std giảm, worst-case person F1 tăng đáng kể
```

### Phase 3: Tích hợp Calibration vào Realtime

```text
Mục tiêu:  PosturePredictor + calibration step trong production
Thay đổi:  src/posture_predictor.py: thêm calibrate() method
           UI: nút "Calibrate" + countdown 3 giây
Safety:    Sanity check baseline (head_gravity_angle phải < 15°, v.v.)
           Drift detection (bbox size thay đổi đột ngột > 20%)
```

---

## IV. Đánh giá rủi ro tổng hợp

| Rủi ro | Xác suất | Hậu quả | Giải pháp |
|---|---|---|---|
| Depth proxy features quá nhiễu vì keypoint jitter | Trung bình | Feature D1-D4 dao động mạnh → không giúp model | Dùng median filter 3-5 frames cho keypoint trước khi tính feature |
| Baseline calibration bị lệch do người calibrate sai | Cao | Model báo "correct" khi đang gù | Sanity check: `head_gravity_angle < 15°` và `shoulder_angle ~ 0°` mới chấp nhận baseline |
| Feature space 66D quá lớn → overfitting | Thấp-Trung bình | CV F1 cao nhưng test F1 thấp | Dùng phiên bản 41 features (selective delta), hoặc L1/L2 regularization |
| Camera ở góc khác 45° → depth proxy sai hướng | Thấp | D1-D4 mất tín hiệu | Depth proxy vẫn hoạt động ở mọi góc (chỉ khác cường độ), chỉ mất hoàn toàn ở góc 0° (chính diện) |

---

## V. Tóm tắt quyết định

```text
Hỏi: Nên bắt đầu từ đâu?
Trả lời: Phase 1 (Depth Proxy only).

Lý do:
1. Nó tấn công trực tiếp vấn đề đau nhất (90 mẫu nhầm lean_right ↔ forward_slouch)
2. Không cần thay đổi kiến trúc (chỉ thêm 4 feature vào FeatureExtractor)
3. Có thể đánh giá ngay trên cùng locked test set → so sánh minh bạch
4. Nếu Phase 1 đã tăng Macro F1 > 0.85, Phase 2 có thể trì hoãn
5. Nếu Phase 1 không hiệu quả, cần xem xét lại trước khi phức tạp hóa thêm
```
