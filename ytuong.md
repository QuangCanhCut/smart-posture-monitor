# Ý tưởng Personal Baseline Calibration — Phân tích đầy đủ

## Mục lục

1. [Ý tưởng cốt lõi là gì?](#1-ý-tưởng-cốt-lõi-là-gì)
2. [Tại sao nó hoạt động?](#2-tại-sao-nó-hoạt-động)
3. [Cách triển khai cụ thể](#3-cách-triển-khai-cụ-thể)
4. [Chọn "khuôn" nào làm gốc? — Phân tích các phương án](#4-chọn-khuôn-nào-làm-gốc--phân-tích-các-phương-án)
5. [Những trường hợp "nhìn thì đúng nhưng thực thi lại sai hoàn toàn"](#5-những-trường-hợp-nhìn-thì-đúng-nhưng-thực-thi-lại-sai-hoàn-toàn)
6. [Tại sao phải giữ theo đúng một khuôn cố định?](#6-tại-sao-phải-giữ-theo-đúng-một-khuôn-cố-định)
7. [Những điều baseline calibration KHÔNG giải quyết được](#7-những-điều-baseline-calibration-không-giải-quyết-được)
8. [Khuyến nghị triển khai cho Smart Posture Monitor](#8-khuyến-nghị-triển-khai-cho-smart-posture-monitor)

---

## 1. Ý tưởng cốt lõi là gì?

### Bài toán thực tế

Hai người khác nhau ngồi đúng tư thế, nhưng vector đặc trưng (feature vector) của họ lại khác nhau đáng kể:

```text
person01 ngồi thẳng: f = [2.1,  0.85, -3.2, 0.41, ...]
person12 ngồi thẳng: f = [4.7,  0.62, -1.8, 0.53, ...]
                           ↑       ↑      ↑     ↑
                        Khác nhau hoàn toàn!
```

Tại sao? Vì mỗi người có cổ dài/ngắn khác nhau, tỷ lệ đầu/vai khác nhau, khoảng cách mắt khác nhau, thói quen ngồi "thẳng" cũng hơi khác nhau. Đây là **hằng số nhân trắc học** — không liên quan đến tư thế, nhưng lại "bám" vào feature vector.

Khi model được train trên person01-person11 rồi test trên person12, nó gặp một "vùng" feature vector mà nó chưa bao giờ thấy. Kết quả: dự đoán sai.

### Phép trừ baseline

Ý tưởng: Yêu cầu mỗi người **ngồi đúng tư thế trong 2-3 giây** lúc bắt đầu sử dụng. Lấy trung bình feature vector trong khoảng đó làm "mốc cá nhân" (baseline):

$$\vec{f}_{\text{base}}^{\text{person}} = \frac{1}{N} \sum_{i=1}^{N} \vec{f}_i^{\text{correct}}$$

Sau đó, mọi frame tiếp theo được biến đổi:

$$\Delta \vec{f}_t = \vec{f}_t - \vec{f}_{\text{base}}^{\text{person}}$$

**Kết quả**: Khi cả person01 và person12 đều ngồi thẳng, $\Delta \vec{f} \approx \vec{0}$ cho cả hai. Khi cả hai gù lưng, $\Delta \vec{f}$ đều có cùng **hướng** (dù độ lớn có thể khác). Hằng số nhân trắc học bị triệt tiêu.

---

## 2. Tại sao nó hoạt động?

### Phân tách tín hiệu

Feature vector bất kỳ có thể phân tách thành:

$$\vec{f}_t = \underbrace{\vec{c}_{\text{person}}}_{\text{hằng số cá nhân}} + \underbrace{\vec{s}_t}_{\text{tín hiệu tư thế}} + \underbrace{\vec{\epsilon}_t}_{\text{nhiễu}}$$

Trong đó:
- $\vec{c}_{\text{person}}$: Cổ dài/ngắn, vai rộng/hẹp, tỷ lệ đầu — **không đổi** trong suốt phiên.
- $\vec{s}_t$: Tín hiệu tư thế thực sự — thay đổi khi người dùng gù/nghiêng.
- $\vec{\epsilon}_t$: Nhiễu keypoint detection, rung tay, v.v.

Khi ngồi đúng ($t$ thuộc calibration window): $\vec{s}_t \approx \vec{0}$, nên:

$$\vec{f}_{\text{base}} \approx \vec{c}_{\text{person}} + \bar{\vec{\epsilon}}$$

Trừ đi:

$$\Delta \vec{f}_t = \vec{f}_t - \vec{f}_{\text{base}} \approx \vec{s}_t + (\vec{\epsilon}_t - \bar{\vec{\epsilon}})$$

→ **Hằng số cá nhân biến mất**, chỉ còn tín hiệu tư thế + nhiễu.

### Kỹ thuật tương đương trong các lĩnh vực khác

Đây không phải ý tưởng mới. Nó là **subject-wise mean centering** — được dùng rộng rãi:

| Lĩnh vực | Tên gọi | Cùng nguyên lý |
|---|---|---|
| EEG / não bộ | Baseline correction | Trừ đi tín hiệu lúc nhắm mắt nghỉ |
| Nhận dạng giọng nói | Speaker normalization (CMVN) | Trừ mean phổ của từng người |
| Phân tích dáng đi | Gait normalization | Chuẩn hóa theo bước đi bình thường |
| Cảm biến gia tốc (IMU) | Zero-offset calibration | Trừ đi giá trị lúc đứng yên |

---

## 3. Cách triển khai cụ thể

### 3.1. Trong realtime (production)

```text
[Bắt đầu phiên làm việc]
    │
    ▼
[Hiện thông báo: "Hãy ngồi thẳng lưng, nhìn thẳng trong 3 giây"]
    │
    ▼
[Thu thập 30-90 frames (1-3 giây ở 30 FPS)]
    │
    ▼
[Tính trung bình: f_base = mean(f_1, f_2, ..., f_N)]
    │
    ▼
[Sanity check: head_gravity_angle < 15°? shoulder_angle ~ 0°?]
    │   Không → "Calibration không hợp lệ, vui lòng thử lại"
    ▼   Có
[Lưu f_base, bắt đầu giám sát]
    │
    ▼
[Mỗi frame: Δf = f_t - f_base → model.predict(Δf)]
```

### 3.2. Trong offline (backtest trên dataset)

```python
def simulate_calibration(df, person_id, session_id, feature_cols, n_calib=30):
    """
    Mô phỏng calibration đúng thực tế:
    Chỉ dùng N frame correct ĐẦU TIÊN trong session.
    """
    mask = (
        (df["person_id"] == person_id) &
        (df["session_id"] == session_id) &
        (df["label"] == "correct")
    )
    correct_frames = df.loc[mask, feature_cols].head(n_calib)
    baseline = correct_frames.mean()
    return baseline


def apply_calibration(df, feature_cols):
    """Áp dụng calibration cho từng (person, session)."""
    result = df.copy()
    delta_cols = [f"delta_{col}" for col in feature_cols]
    
    for (person, session), group in df.groupby(["person_id", "session_id"]):
        baseline = simulate_calibration(df, person, session, feature_cols)
        for col, dcol in zip(feature_cols, delta_cols):
            result.loc[group.index, dcol] = group[col] - baseline[col]
    
    return result
```

---

## 4. Chọn "khuôn" nào làm gốc? — Phân tích các phương án

Đây là câu hỏi quan trọng nhất. "Khuôn" (reference frame) quyết định baseline được neo vào cái gì.

### Phương án A: Dùng toàn bộ cơ thể (full body) làm khuôn

**Ý tưởng**: Normalize tất cả feature theo chiều cao toàn thân hoặc bbox toàn thân.

**Tại sao nó sai cho bài toán này:**

1. **Camera chỉ thấy phần trên**: Trong bố trí bàn làm việc thực tế, bàn che phần hông và chân. YOLO Pose trả về keypoints chân nhưng confidence rất thấp hoặc bị hallucinate vì bị che khuất. Dùng keypoints không đáng tin cậy làm gốc chuẩn hóa → tất cả feature bị nhiễu.

2. **Chân không mang thông tin tư thế ngồi**: Chiều dài chân, tư thế chân không liên quan đến việc đầu có gù hay vai có nghiêng không. Đưa thêm keypoints chân vào khuôn chỉ thêm nhiễu, không thêm tín hiệu.

3. **Bbox toàn thân thay đổi theo tư thế**: Khi gù lưng, bbox co lại theo chiều dọc. Nếu normalize feature theo bbox height, thì chính phép normalize đã "hấp thụ" một phần tín hiệu tư thế → model mất thông tin.

```text
Ngồi thẳng:          Gù lưng:
┌─────────┐          ┌─────────┐
│  ○ đầu  │          │         │
│  │      │ bbox     │  ○ đầu  │ bbox NHỎ HƠN
│ ─┼─ vai │ CAO      │ ─┼─ vai │
│  │      │          │         │
│ ─┼─ hông│          │ ─┼─ hông│
└─────────┘          └─────────┘

→ Nếu normalize theo bbox height:
  nose_y_body (thẳng) / bbox_height (lớn) ≈ nose_y_body (gù) / bbox_height (nhỏ)
  → Tín hiệu tư thế bị triệt tiêu chính bởi phép normalize!
```

### Phương án B: Dùng khoảng cách hai vai (shoulder_width) làm khuôn

**Đây chính là cách V02 hiện tại đang làm** — và nó đúng.

**Tại sao shoulder_width là khuôn tốt:**

1. **Ổn định qua mọi tư thế**: Khoảng cách vật lý giữa hai vai gần như không đổi dù bạn gù, nghiêng trái, hay nghiêng phải. Nó là **hằng số giải phẫu**, không phải tín hiệu tư thế.

2. **Phản ánh khoảng cách tới camera**: Nếu người ngồi xa camera hơn, shoulder_width (pixel) giảm, nhưng tất cả khoảng cách khác cũng giảm theo tỷ lệ → chia cho shoulder_width triệt tiêu hiệu ứng khoảng cách.

3. **Luôn nhìn thấy được**: Hai vai là keypoints có confidence cao nhất vì chúng hiếm khi bị che khuất ở góc camera 45° bên trái.

**Nhưng shoulder_width KHÔNG triệt tiêu hằng số nhân trắc học hoàn toàn**: Người vai rộng và người vai hẹp, sau khi chia shoulder_width, vẫn có tỷ lệ cổ/vai, đầu/vai khác nhau. Đây là lý do vẫn cần thêm baseline calibration.

### Phương án C: Dùng tư thế calibration (Personal Baseline) làm khuôn

**Đây là phương án đề xuất cho V03** — bổ sung thêm, không thay thế shoulder_width.

Khuôn ở đây là **vector feature trung bình lúc ngồi đúng** của chính người dùng đó. Tức là:
- Shoulder_width normalize giải quyết: khác biệt do **khoảng cách tới camera**.
- Baseline subtract giải quyết: khác biệt do **hình thể cơ thể** (nhân trắc học).

Hai phép biến đổi **chồng lên nhau**, không xung đột:

```text
Raw pixel → ÷ shoulder_width → features V02 → − baseline → Δ features V03
```

### Phương án D: Dùng "trung bình toàn dataset" làm khuôn (global mean centering)

**Ý tưởng**: Thay vì baseline riêng mỗi người, trừ đi mean của toàn bộ dataset.

**Tại sao nó sai:**

Trung bình toàn dataset = trung bình của tất cả các tỷ lệ cơ thể khác nhau. Nó không phải baseline của bất kỳ ai cụ thể. Sau khi trừ:
- Person có tỷ lệ "gần trung bình" → Δf nhỏ → tốt
- Person có tỷ lệ "xa trung bình" (cổ rất dài, vai rất rộng) → Δf lớn ngay cả khi ngồi đúng → sai

Hơn nữa, nếu fit global mean trên toàn dataset (bao gồm test set) trước khi split → **data leakage**. Nếu chỉ fit trên train set → person mới vẫn bị offset.

---

## 5. Những trường hợp "nhìn thì đúng nhưng thực thi lại sai hoàn toàn"

Đây là phần quan trọng nhất. Mỗi trường hợp dưới đây, khi nói ra, nghe rất hợp lý — nhưng khi implement và chạy trên dữ liệu thật, nó sẽ cho kết quả sai hoặc ảo.

---

### Bẫy 1: "Dùng TOÀN BỘ frame `correct` trong session làm baseline"

**Nghe hợp lý**: "Nhiều frame hơn → trung bình ổn định hơn → baseline chính xác hơn."

**Thực tế sai vì**:

Trong production, bạn chỉ có **2-3 giây calibration ban đầu** (~30-90 frames). Nhưng nếu khi backtest bạn lấy **toàn bộ** frame correct rải rác trong session (có thể hàng trăm frame, ở nhiều thời điểm khác nhau, nhiều điều kiện ánh sáng khác nhau) để tính baseline:

- Baseline "sạch" hơn vì được trung bình hóa trên nhiều biến thể.
- Kết quả backtest sẽ **lạc quan hơn thực tế** 3-8 điểm F1.
- Khi triển khai production với chỉ 30 frame đầu, F1 tụt xuống và bạn không hiểu tại sao.

```text
Backtest (dùng ~300 frame correct):  F1 = 0.91  ← "Wow, tuyệt vời!"
Production (dùng 30 frame đầu):      F1 = 0.84  ← "Sao tệ hơn nhiều?"

→ Kết quả backtest bị lạc quan giả tạo, không phải data leakage,
  nhưng là simulation mismatch.
```

**Cách đúng**: Khi backtest, chỉ lấy **N frame correct đầu tiên** (sắp xếp theo thứ tự thời gian / tên file) — chính xác như production sẽ làm.

---

### Bẫy 2: "Normalize theo bbox thay vì shoulder_width"

**Nghe hợp lý**: "Bbox bao quát toàn bộ người, nên nó phản ánh kích thước tổng thể tốt hơn."

**Thực tế sai vì**:

Bbox thay đổi kích thước **theo tư thế**:
- Gù lưng → bbox co chiều cao → bbox_height giảm.
- Nghiêng → bbox co chiều rộng → bbox_width giảm.

Khi chia feature cho một đại lượng **tương quan với target** (tư thế), bạn đang vô tình hấp thụ tín hiệu tư thế vào phép normalize. Model mất thông tin mà bạn muốn nó học.

Shoulder_width gần như **bất biến với tư thế** (vai không co/giãn khi bạn gù hay nghiêng) → đó là lý do nó là khuôn tốt.

**Minh họa bằng số**:

```text
                    nose_y (pixel)  bbox_height  nose_y/bbox_height  shoulder_width  nose_y/shoulder_width
Ngồi thẳng:         150            400          0.375               180             0.833
Gù lưng (cúi đầu): 200            320          0.625               180             1.111

→ nose_y/shoulder_width: thay đổi rõ ràng (0.833 → 1.111) → tín hiệu tư thế được giữ
→ nose_y/bbox_height: cũng thay đổi, NHƯNG bbox_height tự thay đổi theo nên
  tỷ lệ biến dạng không trung thực — ở một số tư thế tỷ lệ gần như không đổi
  dù tư thế thay đổi rõ rệt
```

---

### Bẫy 3: "Lấy 1 frame duy nhất làm baseline"

**Nghe hợp lý**: "Đơn giản nhất, nhanh nhất, người dùng chỉ cần đứng yên 1 frame."

**Thực tế sai vì**:

Keypoint detection có **jitter** (rung) ở mức 2-5 pixel giữa các frame liên tiếp, dù người ngồi hoàn toàn bất động. Nếu baseline chỉ dựa trên 1 frame, nó bắt được **nhiễu** chứ không phải tín hiệu trung bình.

Hậu quả:
- Frame calibration ngẫu nhiên bị nhiễu dương → baseline bị cao → mọi frame sau đó Δf bị lệch âm → model báo sai.
- Chạy lại calibration, lần này frame ngẫu nhiên bị nhiễu âm → kết quả ngược hoàn toàn.

```text
30 frame calibration (trung bình hóa):
  shoulder_angle_base = 0.12° ± 0.03°  ← Ổn định

1 frame calibration:
  shoulder_angle_base = -0.41°  ← Một lần
  shoulder_angle_base = +0.67°  ← Lần khác
  → Sai lệch 1.08° giữa hai lần calibrate!
```

**Tối thiểu**: 15-30 frames (~0.5-1 giây). Lý tưởng: 60-90 frames (~2-3 giây).

---

### Bẫy 4: "Trừ baseline xong thì bỏ hết feature tuyệt đối, chỉ dùng Δf"

**Nghe hợp lý**: "Δf đã chứa tín hiệu tư thế thuần túy, feature tuyệt đối chỉ thêm nhiễu nhân trắc học."

**Thực tế sai vì**:

Nếu người dùng calibrate sai (đang gù mà tưởng ngồi thẳng), thì:

$$\vec{f}_{\text{base}} = \vec{c}_{\text{person}} + \vec{s}_{\text{slouch}}$$

Khi họ tiếp tục gù:

$$\Delta \vec{f} = \vec{f}_t - \vec{f}_{\text{base}} = \vec{s}_{\text{slouch}} - \vec{s}_{\text{slouch}} = \vec{0}$$

→ Model báo "correct" vì Δf ≈ 0, nhưng thực tế đang gù.

**Giải pháp**: Giữ **cả hai luồng** — feature tuyệt đối cung cấp "ràng buộc cứng":

```text
Ví dụ: head_gravity_angle > 25° → gần như chắc chắn đang gù,
       DÙ Δ(head_gravity_angle) = 0 (vì baseline cũng đang gù).

Model học được: "Nếu absolute lớn VÀ delta nhỏ → baseline bị sai, vẫn là slouch."
```

Đây là lý do thiết kế **hybrid (absolute + delta)** luôn robust hơn thuần delta.

---

### Bẫy 5: "PCA trên delta features toàn dataset rồi mới chia train/test"

**Nghe hợp lý**: "PCA giúp giảm redundancy, nên fit trên nhiều data nhất có thể để PCA tốt nhất."

**Thực tế sai vì**:

Đây là **data leakage** kinh điển. Khi fit PCA (hoặc StandardScaler) trên toàn bộ dataset trước khi chia train/test:
- PCA "nhìn thấy" phân bố feature của test persons.
- Các thành phần chính (principal components) bị ảnh hưởng bởi test data.
- Model gián tiếp "biết" test data → kết quả lạc quan giả.

**Cách đúng**: PCA chỉ được fit **bên trong sklearn Pipeline**, chỉ trên train fold. Test fold chỉ được transform bằng PCA đã fit trên train.

```python
# SAI:
pca = PCA(n_components=10)
X_all_pca = pca.fit_transform(X_all)  # ← Leakage!
X_train, X_test = split(X_all_pca)

# ĐÚNG:
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("model", SVC(kernel="rbf")),
])
pipeline.fit(X_train, y_train)  # PCA chỉ fit trên train
pipeline.predict(X_test)        # PCA transform trên test
```

---

### Bẫy 6: "Calibrate một lần, dùng mãi"

**Nghe hợp lý**: "Cơ thể người không thay đổi qua các ngày, nên baseline vẫn đúng."

**Thực tế sai vì**:

Baseline phụ thuộc vào **toàn bộ bối cảnh vật lý**, không chỉ cơ thể:

| Yếu tố thay đổi | Ảnh hưởng đến baseline |
|---|---|
| Đổi ghế (cao/thấp hơn) | Góc vai, góc cổ thay đổi |
| Đổi vị trí camera | Toàn bộ tọa độ pixel thay đổi |
| Đổi khoảng cách ngồi | Shoulder_width (pixel) thay đổi → tỷ lệ feature thay đổi |
| Đổi áo (dày/mỏng/có cổ/không cổ) | Keypoint shoulder bị dịch |
| Mệt mỏi cuối ngày | "Ngồi thẳng" của buổi sáng ≠ "ngồi thẳng" của buổi chiều |

**Giải pháp**: 
- Re-calibrate mỗi phiên làm việc (mỗi lần mở app).
- Thêm **drift detection**: nếu shoulder_width (pixel) hoặc bbox size thay đổi đột ngột > 15-20% so với lúc calibrate → nhắc người dùng calibrate lại.

---

### Bẫy 7: "Dùng trung bình của TẤT CẢ các tư thế (kể cả sai) làm baseline"

**Nghe hợp lý**: "Trung bình toàn bộ → baseline ở giữa → Δf dương = nghiêng/gù, Δf âm = ngược lại."

**Thực tế sai HOÀN TOÀN vì**:

Nếu baseline = trung bình tất cả tư thế (correct + slouch + lean), thì:
- Δf khi ngồi đúng ≠ 0 (vì ngồi đúng lệch khỏi trung bình).
- Δf khi gù có thể gần 0 (vì gù gần trung bình nếu gù chiếm nhiều mẫu).
- **Ý nghĩa vật lý bị mất hoàn toàn**: Δf = 0 không còn nghĩa là "đang ở tư thế chuẩn".

Baseline **bắt buộc** phải là tư thế `correct`, vì nó là mốc tham chiếu có ý nghĩa vật lý: "trạng thái cột sống trung tính". Mọi lệch khỏi mốc này mới có ý nghĩa là "sai tư thế".

---

### Bẫy 8: "Feature đã chia shoulder_width rồi thì không cần baseline nữa"

**Nghe hợp lý**: "Shoulder_width normalize đã loại bỏ khác biệt kích thước cơ thể."

**Chỉ đúng một phần**:

Shoulder_width normalize loại bỏ: khác biệt do **khoảng cách camera** (ngồi gần ↔ ngồi xa).

Shoulder_width normalize KHÔNG loại bỏ: khác biệt do **tỷ lệ giải phẫu**.

Ví dụ cụ thể:

```text
Person A: cổ dài, vai hẹp
  → nose_y_body khi ngồi thẳng = 1.4 (đầu cao hơn vai rất nhiều)

Person B: cổ ngắn, vai rộng
  → nose_y_body khi ngồi thẳng = 0.9 (đầu gần vai hơn)
```

Cả hai đều đã được normalize theo shoulder_width, nhưng giá trị vẫn khác vì **tỷ lệ cổ/vai** là khác nhau. Model nhìn thấy `nose_y_body = 0.9` của Person B và có thể nhầm đó là "gù" vì giống với Person A lúc gù.

Baseline subtract giải quyết nốt phần này: trừ đi giá trị `correct` riêng của từng người → Δ(nose_y_body) = 0 cho cả hai khi ngồi thẳng.

---

## 6. Tại sao phải giữ theo đúng một khuôn cố định?

### "Khuôn" ở đây là gì?

Khuôn = bộ quy ước cố định về:
- **Dùng gì để chuẩn hóa** (shoulder_width, không phải bbox, không phải full body height).
- **Dùng hệ tọa độ nào** (Body Frame gắn với tâm vai + trục vai, không phải pixel thô).
- **Dùng keypoints nào** (6 keypoints thân trên, không thêm bớt tùy tiện).
- **Dùng tư thế nào làm mốc** (correct, không phải trung bình toàn bộ tư thế).

### Tại sao phải cố định?

**1. Tính nhất quán giữa train và inference**

Model được train với feature vector 29 chiều, theo thứ tự cố định, với cách tính cố định. Nếu ở inference bạn thay đổi bất kỳ điều gì:
- Đổi khuôn normalize (bbox thay vì shoulder_width)
- Đổi hệ tọa độ
- Thêm/bớt keypoint

→ Feature vector có **cùng kích thước** nhưng **khác ý nghĩa** → model predict rác.

**2. Tính khả tái lập (reproducibility)**

Nếu khuôn thay đổi tùy tiện giữa các lần chạy, bạn không thể:
- So sánh kết quả V02 vs V03 (feature space khác nhau)
- Debug khi model sai (không biết lỗi ở data hay ở khuôn)
- Audit dataset (features.csv không thể tái tạo)

**3. Tính ổn định qua các tư thế**

Khuôn tốt phải là đại lượng **bất biến với tư thế** (shoulder_width). Nếu khuôn thay đổi theo tư thế (bbox height), thì feature "tự co giãn" khi tư thế thay đổi → tín hiệu bị hấp thụ.

**4. Ý nghĩa vật lý rõ ràng**

Khi feature `nose_x_body = 0.15`, nó có nghĩa: "mũi nằm bên phải tâm vai một khoảng bằng 15% chiều rộng vai". Ý nghĩa này rõ ràng, debug được, và giải thích cho người không chuyên được. Nếu dùng bbox, `nose_x_body = 0.15` không có ý nghĩa cố định vì bbox thay đổi.

---

## 7. Những điều baseline calibration KHÔNG giải quyết được

Phải nói rõ để tránh ảo tưởng:

### 7.1. Không giải quyết nhầm lẫn `lean_right` ↔ `forward_slouch`

Baseline trừ đi **hằng số cá nhân** — giá trị không đổi theo tư thế. Nhưng vấn đề `lean_right` ↔ `forward_slouch` là do **ảnh chiếu 2D của hai chuyển động 3D khác nhau trùng nhau** — đây là vấn đề **trong cùng một frame, cùng một người**.

```text
Trước baseline:
  forward_slouch: f = [c + s_slouch]      ← s_slouch gần s_lean_right trong 2D
  lean_right:     f = [c + s_lean_right]

Sau baseline (trừ c):
  forward_slouch: Δf = [s_slouch]          ← VẪN gần nhau!
  lean_right:     Δf = [s_lean_right]

→ Phép trừ chỉ dịch gốc tọa độ, không tách các cụm nhãn đã chồng lấp.
```

Cần **depth proxy features** (diện tích tam giác mặt, tỷ lệ xoay mặt) để giải quyết.

### 7.2. Không giải quyết nhiễu keypoint detection

Nếu YOLO detect keypoint sai (nose bị nhảy sang vị trí khác vì bị tay che), baseline subtract không sửa được — nó chỉ trừ đi hằng số, không lọc nhiễu.

Cần **temporal smoothing** (lọc trung bình trượt) để giải quyết.

### 7.3. Không giải quyết baseline bị sai ngay từ đầu

Nếu người dùng calibrate lúc đang gù → toàn bộ session bị lệch. Đây là **điểm yếu cấu trúc**, không phải bug.

Cần **sanity check** trên baseline (ràng buộc hình học tối thiểu).

---

## 8. Khuyến nghị triển khai cho Smart Posture Monitor

### Thứ tự ưu tiên

```text
Phase 1: Thêm depth proxy features (D1-D4) vào FeatureExtractor
         → Giải quyết lean_right ↔ forward_slouch
         → Không cần thay đổi UX, không cần calibration step
         → Đánh giá ngay trên locked test set

Phase 2: Thêm baseline calibration (offline backtest)
         → Giải quyết cross-subject variance
         → Hybrid features: absolute + delta
         → Simulate đúng: chỉ N frame correct đầu tiên

Phase 3: Tích hợp calibration vào realtime
         → PosturePredictor.calibrate() method
         → UI: nút "Calibrate" + countdown + feedback
         → Sanity check + drift detection
```

### Checklist "không được quên"

- [ ] Khuôn normalize: giữ shoulder_width, KHÔNG đổi sang bbox
- [ ] Baseline CHỈ từ tư thế `correct`, KHÔNG trung bình toàn bộ tư thế
- [ ] Backtest CHỈ dùng N frame correct đầu tiên, KHÔNG toàn bộ correct
- [ ] Feature cuối cùng = absolute + delta (hybrid), KHÔNG thuần delta
- [ ] PCA/Scaler CHỈ fit trong Pipeline trên train fold, KHÔNG fit toàn dataset
- [ ] Calibration mỗi phiên, KHÔNG dùng baseline từ phiên trước
- [ ] Tối thiểu 15-30 frames calibration, KHÔNG dùng 1 frame
- [ ] Sanity check baseline (head_gravity_angle < 15°, shoulder_angle ~ 0°)
- [ ] So sánh V02 vs V03 trên CÙNG locked test set (person01, person10, person12)
