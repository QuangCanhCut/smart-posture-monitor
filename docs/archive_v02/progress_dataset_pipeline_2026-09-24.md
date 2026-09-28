# Bao cao tien do Dataset Pipeline - 2026-09-24

## 1. Muc tieu cong viec trong ngay

Muc tieu hom nay la hoan thien pipeline tao dataset feature tu tap anh raw:

```text
Raw Images
  -> PoseDetector
  -> 6 selected keypoints
  -> FeatureExtractor
  -> 18-dimensional feature vector
  -> DatasetBuilder
  -> features.csv
```

Dong thoi xay dung co che kiem tra va phan tich cac anh bi loai, de nhom co the xem lai nguyen nhan reject thay vi bo qua bang `continue` ma khong co thong tin.

## 2. Cau truc dataset

Dataset hien duoc to chuc theo format:

```text
data/raw/<label>/<person_id>_<session_id>/<frame>
```

Vi du:

```text
data/raw/correct/person01_session01/frame_0001.jpg
```

Bon label hien tai:

| Label |
| --- |
| `correct` |
| `forward_slouch` |
| `lean_left` |
| `lean_right` |

`person_id` duoc luu rieng de phuc vu viec split dataset theo nguoi, giam nguy co data leakage giua train/validation/test.

Vi du:

```text
person01_session01
person01_session02
person02_session01
```

## 3. Trang thai PoseDetector

Theo muc tieu pipeline, PoseDetector nen uu tien chu the chinh gan trung tam camera khi trong anh co nhieu nguoi. Logic mong muon:

```text
final_score = 0.7 * centrality_score + 0.3 * confidence_score
```

Sau do chon person co `final_score` lon nhat.

Muc dich:

- Giam truong hop chon nham nguoi o phia sau hoac ben canh.
- Uu tien chu the chinh thuong nam gan trung tam camera.

Trang thai code hien tai can luu y: `src/pose_detector.py` trong working tree khong co diff va van dang chon person co detection confidence cao nhat. File nay chua duoc tu dong sua trong phien lam viec nay vi yeu cau khong tu y thay doi logic code.

PoseDetector van chi tra ve 6 keypoints:

- `nose`
- `left_eye`
- `right_eye`
- `left_ear`
- `left_shoulder`
- `right_shoulder`

`right_ear` khong duoc su dung.

## 4. Cap nhat DatasetBuilder

`src/dataset_builder.py` hien dam nhan viec quet dataset raw va tao bang feature cho training.

Cac buoc chinh:

1. Kiem tra cau truc folder.
2. Parse `label`.
3. Parse `person_id`.
4. Parse `session_id`.
5. Doc anh bang OpenCV.
6. Chay `PoseDetector`.
7. Kiem tra pose/keypoints.
8. Chay `FeatureExtractor`.
9. Kiem tra vector feature.
10. Ghi mau hop le vao `data/processed/features.csv`.

CSV feature gom:

- `image_path`
- `session_id`
- `person_id`
- `label`
- 18 feature columns tu `FeatureExtractor.FEATURE_NAMES`

## 5. Xu ly rejected samples

DatasetBuilder hien tao:

```text
data/rejected/rejected_images.csv
```

File nay luu cac thong tin:

- `image_path`
- `session_id`
- `person_id`
- `label`
- `reason`
- `detail`
- `person_confidence`
- `keypoint_confidences`

Nhung reason hien co the bao gom:

| Reason |
| --- |
| `invalid_folder_structure` |
| `invalid_label` |
| `invalid_person_session_folder` |
| `read_error` |
| `no_person` |
| `invalid_pose_structure` |
| `missing_keypoints` |
| `invalid_keypoint` |
| `feature_extraction_failed` |
| `invalid_feature_length` |
| `non_finite_features` |
| `processing_error` |

Co che nay giup phan tich nguyen nhan anh bi loai thay vi chi bo anh va khong biet ly do.

## 6. Script truc quan hoa rejected images

Da them:

```text
scripts/copy_rejected_images.py
```

Vai tro:

- Doc `rejected_images.csv`.
- Doc lai anh raw.
- Chay lai `PoseDetector`.
- Ve bounding box cua person duoc chon.
- Ve 6 keypoints.
- Hien thi confidence tung keypoint.
- Hien thi person confidence.
- Hien thi reason anh bi reject.
- Luu anh truc quan hoa vao `data/rejected/rejected_img/`.

Muc dich la giup kiem tra truc quan tai sao mot frame bi loai.

Luu y: mau confidence tren anh chi phuc vu visualization, khong phai threshold chinh thuc cua `FeatureExtractor`.

## 7. Ket qua chay DatasetBuilder hien tai

| Chi so | Gia tri |
| --- | ---: |
| Tong so anh | 3286 |
| Mau hop le | 2888 |
| Ty le hop le | 87.89% |
| Mau bi loai | 398 |
| Ty le bi loai | 12.11% |

Chi tiet theo class:

| Class | So mau hop le |
| --- | ---: |
| correct | 720 |
| forward_slouch | 623 |
| lean_left | 782 |
| lean_right | 763 |

Chi tiet theo person:

| Person | So mau hop le |
| --- | ---: |
| person01 | 775 |
| person02 | 273 |
| person03 | 300 |
| person04 | 227 |
| person05 | 49 |
| person06 | 252 |
| person07 | 253 |
| person08 | 246 |
| person09 | 284 |
| person10 | 229 |

Nhan xet:

- `person01` co nhieu mau hon vi co 3 recording sessions/clips.
- Dataset hien chua can bang hoan toan theo person.
- Day la van de can xem xet truoc khi chia train/validation/test.

## 8. Thong ke rejected

| Reason | So mau |
| --- | ---: |
| `no_person` | 1 |
| `feature_extraction_failed` | 397 |

Nhan xet:

- YOLO Pose gan nhu luon phat hien duoc person.
- Van de chinh hien tai nam sau buoc person detection.
- 397 anh dang duoc DatasetBuilder gom vao `feature_extraction_failed`.
- Chua duoc phep ket luan rang ca 397 anh deu loi vi cung mot nguyen nhan.
- Can kiem tra `FeatureExtractor` va visualization cua rejected images de biet nguyen nhan that.

## 9. Viec chua hoan thanh / next step

Phan nay se tiep tuc trong phien lam viec sau.

TODO:

1. Quan sat cac rejected images da visualize.
2. Phan tich cac truong hop `FeatureExtractor` tra ve `None`.
3. Xac dinh chinh xac nguyen nhan cua 397 `feature_extraction_failed`.
4. Quyet dinh giu lai, loai bo, hoac dieu chinh threshold/preprocessing.
5. Kiem tra mat can bang so frame giua cac person.
6. Sau do moi chuan bi train/validation/test split theo person.
7. Tiep tuc training XGBoost sau khi dataset duoc xac nhan.

Khong ket luan rang 398 anh chac chan phai xoa. Hien trang la: dang duoc xem xet truoc khi dua ra quyet dinh cuoi cung.
