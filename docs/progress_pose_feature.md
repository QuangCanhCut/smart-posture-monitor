# Bao cao tien do Pose Detection va Feature Extraction

## 1. Muc tieu cong viec da thuc hien

Project `smart_posture_monitor` dang xay dung he thong nhan dien tu the ngoi tu webcam. Phan da hoan thanh den hien tai gom:

- Doc frame tu webcam bang OpenCV.
- Chay YOLO Pose de phat hien nguoi va keypoint.
- Chon person co bounding-box confidence cao nhat neu co nhieu nguoi trong frame.
- Lay dung 6 keypoint can thiet.
- Kiem tra chat luong keypoint bang confidence.
- Chuyen pose thanh vector 18 feature hinh hoc da normalize.
- Test realtime pipeline tu webcam den feature vector.

## 2. Kien truc pipeline hien tai

```text
Anh
  ↓
YOLO26n Pose
  ↓
Best person
  ↓
6 keypoints
  ↓
Quality checking
  ↓
Body coordinate system
  ↓
18 geometric features
  ↓
Dataset / preprocessing
  ↓
SVM / XGBoost
```

Pipeline trong project:

```text
Webcam -> PoseDetector -> FeatureExtractor -> 18 features -> Preprocessing -> ML Model -> Prediction -> Alert
```

## 3. Camera setup

Camera duoc dat co dinh khoang 45 do ve phia ben trai nguoi dung, khong phai goc truc dien. Goc nay anh huong den do on dinh cua mot so keypoint, dac biet la `right_ear`.

## 4. 4 posture class du kien

- `NORMAL`: ngoi thang.
- `HUNCH_DOWN`: ngoi gu / cui.
- `TILT_LEFT`: nghieng trai.
- `TILT_RIGHT`: nghieng phai.

## 5. 6 keypoint dang su dung

| Keypoint | COCO index |
| --- | --- |
| `nose` | 0 |
| `left_eye` | 1 |
| `right_eye` | 2 |
| `left_ear` | 3 |
| `left_shoulder` | 5 |
| `right_shoulder` | 6 |

## 6. Ly do bo `right_ear`

Voi camera dat lech khoang 45 do ben trai, `right_ear` co the bi che khuat hoac co confidence khong on dinh. Pipeline hien tai loai bo hoan toan `right_ear` de giu input nhat quan giua training va inference. Tat ca dataset, feature extraction va inference nen tiep tuc dung cung bo 6 keypoint nay.

## 7. Hoat dong chi tiet cua `pose_detector.py`

File `src/pose_detector.py` dinh nghia class `PoseDetector`:

- Load model YOLO Pose mot lan khi khoi tao.
- Mac dinh doc model tu `models/yolo26n-pose.pt`.
- Nhan mot frame OpenCV dang `numpy.ndarray`.
- Goi `YOLO.predict(...)` voi nguong `person_conf_threshold`.
- Neu khong co detection, tra ve `None`.
- Neu co nhieu person, lay person co bounding-box confidence cao nhat bang `np.argmax`.
- Lay bounding box cua person duoc chon.
- Lay toa do va confidence cua 6 keypoint can thiet.
- Tra ve dictionary gom confidence cua person, bounding box va keypoint.

## 8. Format output cua `PoseDetector`

```python
{
    "person_confidence": float,
    "bbox": np.ndarray([x1, y1, x2, y2]),
    "keypoints": {
        "nose": np.ndarray([x, y, confidence]),
        "left_eye": np.ndarray([x, y, confidence]),
        "right_eye": np.ndarray([x, y, confidence]),
        "left_ear": np.ndarray([x, y, confidence]),
        "left_shoulder": np.ndarray([x, y, confidence]),
        "right_shoulder": np.ndarray([x, y, confidence]),
    },
}
```

Neu khong phat hien nguoi, output la `None`.

## 9. Hoat dong chi tiet cua `feature_extractor.py`

File `src/feature_extractor.py` dinh nghia class `FeatureExtractor`:

- Nhan output cua `PoseDetector`.
- Kiem tra co du 6 keypoint bat buoc.
- Chuyen moi keypoint ve mang `[x, y, confidence]`.
- Kiem tra gia tri hop le va huu han.
- Loai frame neu mot keypoint co confidence thap hon `min_keypoint_confidence`.
- Dung hai vai de tao he toa do tuong doi cua co the.
- Dung `shoulder_width` de normalize khoang cach.
- Tao vector `numpy.ndarray` co shape `(18,)`.
- Tra ve `None` neu frame khong du chat luong.

## 10. Danh sach 18 feature theo dung thu tu

1. `shoulder_angle`
2. `eye_shoulder_angle`
3. `eye_vertical_difference`
4. `nose_x_body`
5. `nose_y_body`
6. `eye_center_x_body`
7. `eye_center_y_body`
8. `left_ear_x_body`
9. `left_ear_y_body`
10. `nose_eye_dx`
11. `nose_eye_dy`
12. `eye_width_ratio`
13. `ear_eye_ratio`
14. `nose_shoulder_center_distance`
15. `eye_shoulder_center_distance`
16. `nose_shoulder_asymmetry`
17. `eye_shoulder_asymmetry`
18. `nose_ear_ratio`

## 11. Y nghia tung feature

- `shoulder_angle`: do nghieng cua duong noi hai vai.
- `eye_shoulder_angle`: do nghieng cua mat so voi vai.
- `eye_vertical_difference`: chenh lech chieu cao hai mat theo truc co the.
- `nose_x_body`: vi tri ngang cua mui trong he toa do co the.
- `nose_y_body`: vi tri doc cua mui trong he toa do co the.
- `eye_center_x_body`: vi tri ngang cua trung tam hai mat.
- `eye_center_y_body`: vi tri doc cua trung tam hai mat.
- `left_ear_x_body`: vi tri ngang cua tai trai.
- `left_ear_y_body`: vi tri doc cua tai trai.
- `nose_eye_dx`: do lech ngang cua mui so voi trung tam mat.
- `nose_eye_dy`: do lech doc cua mui so voi trung tam mat, huu ich cho dau cui / pitch.
- `eye_width_ratio`: khoang cach hai mat chia cho do rong vai.
- `ear_eye_ratio`: khoang cach tai trai den mat trai chia cho do rong vai.
- `nose_shoulder_center_distance`: khoang cach mui den trung tam vai chia cho do rong vai.
- `eye_shoulder_center_distance`: khoang cach trung tam mat den trung tam vai chia cho do rong vai.
- `nose_shoulder_asymmetry`: chenh lech khoang cach mui den vai trai va vai phai.
- `eye_shoulder_asymmetry`: chenh lech khoang cach trung tam mat den vai trai va vai phai.
- `nose_ear_ratio`: khoang cach mui den tai trai chia cho do rong vai.

## 12. Normalize bang `shoulder_width`

Hai vai tao thanh truc ngang cua co the. Trung tam vai duoc dung lam goc toa do tuong doi. Moi khoang cach pixel duoc chia cho `shoulder_width`, giup giam anh huong cua:

- Nguoi ngoi gan / xa webcam.
- Kich thuoc co the khac nhau.
- Vi tri nguoi trong frame.

Neu `shoulder_width` gan bang 0, frame duoc xem la loi va bi bo.

## 13. Su dung keypoint confidence

Moi keypoint dau vao co format `[x, y, confidence]`. `FeatureExtractor` chi tao feature khi ca 6 keypoint deu co confidence lon hon hoac bang nguong `min_keypoint_confidence`. Mac dinh test realtime dang dung nguong `0.35`. Keypoint thap hon nguong lam frame bi bo de tranh dua pose kem chat luong vao dataset.

## 14. Test realtime da thuc hien

Da test realtime bang webcam:

- `tests/test_pose_detector.py` hien bounding box cua person duoc chon.
- Hien person confidence.
- Hien dung 6 keypoint va confidence tung keypoint.
- Ve mot so duong noi de quan sat pose.
- `tests/test_feature_extractor.py` hien `Features OK: 18` khi trich xuat thanh cong.
- Hien mot so feature quan trong tren webcam.
- Terminal in day du 18 feature dinh ky.

## 15. Cach chay

Can dat model local tai:

```text
models/yolo26n-pose.pt
```

Chay test pose detector:

```bash
python -m tests.test_pose_detector
```

Chay test feature extractor:

```bash
python -m tests.test_feature_extractor
```

## 16. Nhung phan chua lam

- Chua hoan thien dataset cho 4 class.
- Chua tao bang feature + label cho training.
- Chua co preprocessing/scaler chinh thuc.
- Chua train model phan loai posture.
- Chua co module prediction realtime cuoi cung.
- Chua co logic alert theo thoi gian.
- Chua danh gia bang metric ML.

## 17. Cong viec ban giao cho thanh vien ML tiep theo

Cong viec tiep theo nen lam:

- Hoan thien dataset 4 lop: `NORMAL`, `HUNCH_DOWN`, `TILT_LEFT`, `TILT_RIGHT`.
- Chuyen toan bo dataset qua `FeatureExtractor`.
- Tao bang feature + label.
- Thuc hien EDA.
- Kiem tra distribution tung feature theo class.
- Kiem tra correlation giua cac feature.
- Phat hien feature thua hoac qua nhieu nhieu.
- Train/test split theo person/session de tranh data leakage.
- Thu SVM.
- Thu XGBoost.
- Danh gia bang Accuracy, Precision, Recall, F1 va Confusion Matrix.
- Phan tich feature importance, permutation importance va ablation.
- Chon best model.
- Luu model/scaler bang Joblib.

Can dac biet luu y: khong nen random split cac frame gan nhau tu cung mot video/session vao ca train va test. Cach split nay co the gay data leakage va tao accuracy ao. Nen split theo person hoac session de ket qua danh gia gan voi thuc te hon.

