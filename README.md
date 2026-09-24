# Smart Posture Monitor

Smart Posture Monitor la project Python nhan dien tu the ngoi bang webcam. Pipeline hien tai tap trung vao hai phan da kiem chung: phat hien pose bang YOLO Pose va trich xuat 18 feature hinh hoc cho bai toan ML.

## Muc tieu

He thong du kien nhan dien 4 posture:

- `correct`: ngoi dung tu the
- `forward_slouch`: cui / gu nguoi ve phia truoc
- `lean_left`: nghieng nguoi sang trai
- `lean_right`: nghieng nguoi sang phai

Pipeline tong quat:

```text
Webcam -> YOLO26n Pose -> 6 keypoints -> Feature Extraction -> Preprocessing -> ML Model -> Prediction -> Alert
```

## Cau truc project

```text
smart_posture_monitor/
├── app/
├── data/
├── docs/
│   └── progress_pose_feature.md
├── models/
│   └── .gitkeep
├── notebooks/
├── src/
│   ├── pose_detector.py
│   └── feature_extractor.py
├── tests/
│   ├── test_pose_detector.py
│   └── test_feature_extractor.py
└── requirements.txt
```

## Setup moi truong

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies toi thieu:

- `ultralytics`
- `opencv-python`
- `numpy`

## Model YOLO Pose

Dat file model local tai:

```text
models/yolo26n-pose.pt
```

File `.pt` khong duoc commit len repository. Thu muc `models/` duoc giu bang `models/.gitkeep`.

## Cach chay test realtime

```bash
python -m tests.test_pose_detector
python -m tests.test_feature_extractor
```

Hai test tren dung webcam va hien thi cua so OpenCV. Bam `q` de thoat.

## Tai lieu ban giao

Bao cao chi tiet cho thanh vien Machine Learning nam tai:

[docs/progress_pose_feature.md](docs/progress_pose_feature.md)
