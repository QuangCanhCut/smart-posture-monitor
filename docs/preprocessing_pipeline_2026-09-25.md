# Preprocessing Pipeline - 2026-09-25

## 1. Muc tieu

Hoan thien tang preprocessing cho dataset posture classification sau khi da hoan thanh EDA tren `features.csv`.

Input hien tai:

`data/processed/features.csv`

Dataset su dung:
- 4 metadata columns:
  - image_path
  - session_id
  - person_id
  - label
- 29 engineered features tu FeatureExtractor V2
- 4 posture classes:
  - correct
  - forward_slouch
  - lean_left
  - lean_right

## 2. Luong preprocessing

Pipeline hien tai:

```text
features.csv
-> load_dataset()
-> get_feature_columns()
-> validate_schema()
-> validate_integrity()
-> add_recording_id()
-> filter_protocol_invalid()
-> prepare_model_data()
-> PreparedDataset
```

Output gom:

- `X`: 29 engineered features
- `y`: encoded labels
- `groups`: person_id
- `metadata`: image_path, session_id, person_id, recording_id, label

Label encoding hien tai:

- correct -> 0
- forward_slouch -> 1
- lean_left -> 2
- lean_right -> 3

## 3. Cac kiem tra duoc thuc hien

`preprocessing.py` kiem tra:

- file dataset ton tai
- dataset khong rong
- du metadata columns
- dung 29 features
- dung feature names tu `FeatureExtractor.FEATURE_NAMES`
- dung 4 labels
- features phai numeric
- khong co missing values
- khong co NaN / +Inf / -Inf
- khong co duplicated rows
- khong co duplicated image_path
- khong co conflicting label cho cung image_path

## 4. recording_id

Do `session_id` duoc reuse giua nhieu person nen tao:

`recording_id = person_id + "__" + session_id`

Vi du:

`person01__session01`

Muc dich:
- traceability
- audit
- debug
- kiem tra recording leakage

`recording_id` khong duoc dua vao model.

## 5. Protocol filtering

Hien tai `person05` duoc tam loai khoi modeling vi recording hien tai duoc quay sai camera acquisition protocol.

Person05 van duoc giu trong EDA/raw dataset de trace.

Sau khi person05 duoc quay lai dung protocol va rebuild `features.csv`, can doi:

`excluded_persons = ("person05",)`

thanh:

`excluded_persons = ()`

## 6. Nhung viec preprocessing.py KHONG thuc hien

Khong thuc hien:

- random train/test split theo frame
- StandardScaler tren toan dataset
- PCA tren toan dataset
- IQR outlier removal tu dong
- drop correlated features tu dong
- SMOTE/oversampling
- train SVM/XGBoost

Nhung thao tac tren se duoc xu ly o `train.py` hoac model pipeline de tranh data leakage.

## 7. Test

Da tao:

`tests/test_preprocessing.py`

Test suite kiem tra:
- load dataset
- missing file
- empty CSV
- feature schema
- metadata schema
- invalid labels
- non-numeric feature
- NaN
- +Inf / -Inf
- duplicated row
- duplicated image_path
- conflicting label
- recording_id
- protocol filtering
- X/y/groups/metadata
- end-to-end preprocessing pipeline

Ket qua test hien tai:

`25 passed`

Manual preprocessing run hien tai:

- Samples: 2897
- Features: 29
- Persons: 9
- Recordings: 11
- Classes: 4
- Excluded: person05

Output:

- X: (2897, 29)
- y: (2897,)
- groups: (2897,)
- metadata: (2897, 5)

## 8. Huong su dung

Manual test:

`python -m src.preprocessing`

Unit test:

`python -m pytest tests/test_preprocessing.py -v`

Toan bo project test:

`python -m pytest -v`

## 9. Huong phat trien tiep theo

Buoc tiep theo la trien khai `train.py`.

Training pipeline du kien:

```text
PreparedDataset
-> group split theo person_id
-> GroupKFold / Leave-One-Group-Out
-> baseline SVM va XGBoost
-> danh gia model
-> chon model tot nhat
```

Doi voi SVM:
StandardScaler phai nam trong sklearn Pipeline de tranh leakage.

Doi voi XGBoost:
Co the train truc tiep tren 29 engineered features lam baseline.
