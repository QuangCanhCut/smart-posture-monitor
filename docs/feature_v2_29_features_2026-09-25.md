# Bao cao cap nhat Feature V2 - 2026-09-25

## 1. Muc tieu
- Nang cap vector feature tu 18 len 29 chieu.
- Bo sung cac dac trung hinh hoc nham cai thien kha nang phan biet tu the correct voi forward_slouch va cac tu the nghieng.

## 2. FeatureExtractor
- Giu nguyen thu tu 18 feature cu.
- Bo sung 11 feature moi:
  - head_body_angle
  - head_gravity_angle
  - nose_gravity_angle
  - face_pitch_angle
  - eye_vertical_axis_offset
  - nose_vertical_axis_offset
  - head_mean_height
  - head_height_spread
  - nose_body_angle
  - ear_body_angle
  - head_axis_angle_spread
- Output moi la numpy.ndarray shape (29,).
- Van su dung 6 keypoint cu:
  - nose
  - left_eye
  - right_eye
  - left_ear
  - left_shoulder
  - right_shoulder
- Khong can model da train de tao feature.

## 3. DatasetBuilder
- Dong bo tu vector 18 chieu sang 29 chieu.
- Van giu metadata:
  - image_path
  - session_id
  - person_id
  - label
- features.csv moi gom 4 metadata + 29 features = 33 columns.
- Logic rejected image duoc giu nguyen theo pipeline hien tai.

## 4. Test
- FeatureExtractor test da phan anh trang thai 29 features.
- PoseDetector test tiep tuc kiem tra 6 keypoint, khong them right_ear.
- Da thuc hien kiem tra:
  - FeatureExtractor.FEATURE_NAMES co 29 phan tu.
  - Python syntax/AST parse OK cho cac file dang thay doi.
  - FeatureExtractor.extract() voi pose hop le tra ve ndarray shape (29,).

## 5. Pipeline sau cap nhat

Image
-> PoseDetector
-> 6 keypoints
-> FeatureExtractor V2
-> 29 features
-> DatasetBuilder
-> features.csv
-> train/test tren Colab

## 6. Trang thai
- Branch hien tai: CanhCutDev.
- File source/test dang co thay doi trong working tree:
  - src/feature_extractor.py
  - src/dataset_builder.py
  - src/pose_detector.py
  - tests/test_feature_extractor.py
  - tests/test_pose_detector.py
- File bao cao duoc tao:
  - docs/feature_v2_29_features_2026-09-25.md
- Kiem tra da pass:
  - len(FeatureExtractor.FEATURE_NAMES) = 29.
  - FeatureExtractor.extract() tra ve ndarray shape (29,) voi input hop le.
  - Syntax/AST parse OK cho cac file Python dang thay doi.
- Luu y con ton tai:
  - src/pose_detector.py co thay doi logic chon person tu confidence cao nhat sang ket hop centrality va confidence. Can xac nhan day la thay doi hop le truoc khi commit neu muon tuan thu nghiem ngat yeu cau khong doi logic ngoai Feature V2.
  - Chua chay webcam test vi moi truong Codex khong dam bao ho tro webcam tuong tac.

## 7. Buoc tiep theo
- Build lai features.csv bang bo 29 features.
- Dua CSV moi len Colab.
- Train lai Random Forest / XGBoost / SVM.
- So sanh V1 18 features voi V2 29 features.
- Dac biet theo doi recall va F1 cua class correct.
