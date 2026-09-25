# Báo cáo công việc - EDA data_v02 với 29 Features

## 1. Thông tin chung

- Ngày thực hiện: 25/09/2026
- Dataset: `data_v02`
- Số engineered features: 29
- Nhiệm vụ: Exploratory Data Analysis
- Project: Smart Posture Monitor
- Branch phát triển: `CanhCutDev`
- Notebook EDA: `notebooks/01_eda_dataset.ipynb`

## 2. Mục tiêu công việc

Quá trình EDA được thực hiện nhằm kiểm tra chất lượng dataset `data_v02` sau khi pipeline được nâng cấp lên `FeatureExtractor V2` với 29 engineered features.

Các mục tiêu chính trong notebook gồm:

- kiểm tra cấu trúc dataset và schema;
- kiểm tra missing values, infinity, duplicate và tính hợp lệ của metadata;
- kiểm tra phân bố 4 posture classes;
- kiểm tra phân bố dữ liệu theo `person_id` và `session_id`;
- đánh giá thống kê mô tả của 29 engineered features;
- phân tích phân phối feature theo từng posture class;
- phát hiện outlier bằng IQR;
- phân tích correlation và redundancy giữa các feature;
- đánh giá khả năng phân biệt posture của từng feature;
- trực quan hóa PCA;
- đánh giá nguy cơ data leakage và đề xuất split theo group/person.

Notebook chưa thực hiện preprocessing, chưa train model và chưa loại bỏ feature tự động.

## 3. Dataset sử dụng

Dataset được xây dựng theo pipeline:

`Image -> YOLO Pose -> 6 upper-body keypoints -> FeatureExtractor V2 -> 29 engineered features -> features.csv`

Thông tin dataset trong notebook:

- Số samples: 2945
- Số columns: 33
- Metadata columns: `image_path`, `session_id`, `person_id`, `label`
- Số engineered features: 29
- Số posture classes: 4
- Class labels: `correct`, `forward_slouch`, `lean_left`, `lean_right`
- Số persons: 10
- Số session IDs: 3
- Tất cả 29 features có kiểu numeric `float64`

Phân bố class:

| Class | Samples | Percentage |
| --- | ---: | ---: |
| `correct` | 743 | 25.23% |
| `forward_slouch` | 658 | 22.34% |
| `lean_left` | 782 | 26.55% |
| `lean_right` | 762 | 25.87% |

Phân bố theo person:

| Person | Samples |
| --- | ---: |
| `person01` | 796 |
| `person02` | 273 |
| `person03` | 307 |
| `person04` | 258 |
| `person05` | 48 |
| `person06` | 252 |
| `person07` | 255 |
| `person08` | 246 |
| `person09` | 284 |
| `person10` | 226 |

Notebook ghi nhận `person01` có 3 sessions (`session01`, `session02`, `session03`), trong khi các persons còn lại chủ yếu có `session01`. `session_id` không phải định danh recording duy nhất trên toàn dataset; một recording cụ thể nên được xác định bằng cặp `(person_id, session_id)`.

Notebook cũng ghi nhận `person05` chỉ còn 48 valid samples do lỗi acquisition protocol: camera được đặt/quay từ phía bên phải thay vì góc nhìn chuẩn khoảng 45 độ từ bên trái. `person10` có số lượng `forward_slouch` thấp bất thường, chỉ 17 samples.

## 4. Các bước EDA đã thực hiện

Notebook đã thực hiện các bước EDA sau:

- load `features.csv` của `data_v02`;
- kiểm tra shape, `head()`, danh sách columns và `info()`;
- xác nhận metadata columns và số lượng 29 feature columns;
- kiểm tra missing values trên toàn dataset và riêng 29 features;
- kiểm tra kiểu dữ liệu numeric của feature;
- kiểm tra `+inf`, `-inf`;
- kiểm tra duplicate rows và duplicate `image_path`;
- kiểm tra 4 labels hợp lệ;
- kiểm tra `person_id`, `session_id`, person rỗng và session rỗng;
- kiểm tra quan hệ session/person và image/label;
- phân tích class distribution bằng bảng và bar chart;
- phân tích person distribution, person x class, person x session;
- phân tích thống kê mô tả của 29 features;
- kiểm tra constant features, unique values, variance, range, giá trị âm và giá trị bằng 0;
- trực quan hóa độ lệch chuẩn của features;
- phân tích feature distribution theo posture class bằng mean, median, standard deviation, boxplot và histogram;
- tính normalized median spread và các chỉ số separation mô tả;
- phân tích outlier bằng IQR theo feature, class và person;
- kiểm tra các frame có nhiều feature outlier;
- phân tích Pearson correlation, Spearman correlation và candidate redundancy;
- đánh giá khả năng phân biệt class của từng feature, gồm pairwise separation và subject-aware univariate validation;
- thực hiện PCA sau `StandardScaler`;
- trực quan hóa PCA theo posture class và `person_id`;
- phân tích explained variance, cumulative variance và PCA loadings;
- đánh giá nguy cơ data leakage;
- tạo thử split theo group/person và kiểm tra train/validation/test không overlap person.

## 5. Kết quả chính

Kết quả data integrity:

- Dataset có 2945 samples và 29 engineered features.
- Không có missing values.
- Không có `+inf` hoặc `-inf`.
- Không có duplicate rows.
- Không có duplicate `image_path`.
- Không có constant feature.
- Các features có số lượng giá trị unique cao.

Phân bố class tương đối cân bằng ở mức toàn dataset. `lean_left` có số lượng sample lớn nhất, `forward_slouch` có số lượng sample thấp nhất, nhưng mức chênh lệch chưa đủ lớn để xem là class imbalance nghiêm trọng. Notebook chưa đề xuất oversampling, undersampling hoặc SMOTE tại giai đoạn này.

Phân bố theo person không đồng đều. `person01` có nhiều sample nhất do có 3 sessions. `person05` có ít valid samples nhất và được xác định là dữ liệu sai acquisition protocol. `person10` có bất thường ở class `forward_slouch`.

Về feature distribution, nhiều engineered features cho thấy khác biệt rõ giữa các posture classes. `shoulder_angle` là feature đơn biến nổi bật nhất với normalized median spread khoảng 2.04 và subject-aware univariate balanced accuracy khoảng `0.524 ± 0.169`.

Nhóm feature có tiềm năng phân biệt `lean_left` và `lean_right` gồm:

- `shoulder_angle`
- `eye_shoulder_angle`
- `eye_vertical_difference`
- `eye_vertical_axis_offset`
- `nose_vertical_axis_offset`
- `head_gravity_angle`
- `nose_gravity_angle`

Nhóm feature có tiềm năng phân biệt `correct` và `forward_slouch` gồm:

- `nose_eye_dy`
- `nose_y_body`
- `nose_ear_ratio`
- `head_body_angle`
- `nose_body_angle`
- `eye_shoulder_asymmetry`
- `nose_shoulder_asymmetry`
- `head_gravity_angle`
- `nose_gravity_angle`
- `head_mean_height`
- `head_height_spread`
- `eye_center_y_body`

Notebook kết luận không có một feature đơn lẻ nào đủ để giải quyết hoàn toàn bài toán phân loại 4 posture classes. Cần kết hợp nhiều engineered features trong model đa biến.

## 6. PCA

PCA được thực hiện trên 29 features sau khi chuẩn hóa bằng `StandardScaler`.

Kết quả explained variance:

| Principal Component | Explained variance | Cumulative variance |
| --- | ---: | ---: |
| PC1 | 45.92% | 45.92% |
| PC2 | 27.86% | 73.78% |
| PC3 | 12.05% | 85.84% |
| PC4 | 5.10% | 90.94% |

Notebook ghi nhận:

- PC1 + PC2 giữ khoảng 73.78% tổng variance.
- Cần 3 principal components để giữ hơn 80% variance.
- Cần 4 principal components để giữ hơn 90% variance.
- Các principal components cuối có explained variance gần bằng 0, cho thấy có redundancy đáng kể trong bộ 29 features.
- PCA 2D cho thấy 4 posture classes có centroid và cấu trúc khác nhau nhưng vẫn overlap đáng kể.
- PCA theo `person_id` cho thấy feature space chứa subject-specific structure khá rõ.

PCA trong notebook chỉ được dùng cho EDA. Chưa có quyết định dùng PCA làm preprocessing cho modeling.

## 7. Các vấn đề dữ liệu phát hiện được

Notebook không phát hiện vấn đề nghiêm trọng về mặt numerical integrity:

- Không có missing values.
- Không có infinity.
- Không có duplicate rows.
- Không có duplicate `image_path`.
- Không có constant feature.

Các vấn đề cần lưu ý:

- `person05` vi phạm acquisition protocol do camera được đặt/quay từ phía bên phải thay vì góc nhìn chuẩn từ bên trái. Notebook giữ subject này trong EDA để đảm bảo traceability nhưng khuyến nghị chưa dùng cho modeling chính thức cho đến khi quay lại đúng protocol.
- `person10` có số lượng `forward_slouch` thấp bất thường, chỉ 17 samples.
- Có outlier theo quy tắc IQR, đặc biệt ở các feature như `head_axis_angle_spread`, `head_gravity_angle`, `nose_gravity_angle`, `shoulder_angle`, `eye_center_y_body`, `head_mean_height`.
- Outlier không được xem tự động là bad data vì nhiều extreme values có thể phản ánh posture hợp lệ.
- Có redundancy cao giữa nhiều feature, ví dụ các cặp correlation gần tuyệt đối như `eye_center_y_body` - `head_mean_height`, `nose_y_body` - `eye_center_y_body`, `nose_shoulder_asymmetry` - `eye_shoulder_asymmetry`, `eye_vertical_axis_offset` - `nose_vertical_axis_offset`.
- Có nguy cơ subject leakage nếu random split theo frame.

Notebook không thực hiện automatic IQR filtering, không loại feature chỉ dựa trên correlation và không áp dụng resampling.

## 8. Kết luận EDA

Dataset `data_v02` hiện có chất lượng numerical tốt và phù hợp để làm cơ sở cho bước preprocessing/modeling tiếp theo.

FeatureExtractor V2 tạo ra 29 engineered features có posture-related signal rõ ràng. Nhiều feature thể hiện khả năng phân biệt posture, đặc biệt nhóm shoulder/asymmetry cho `lean_left` - `lean_right` và nhóm head/body geometry cho `correct` - `forward_slouch`.

Các điểm cần xử lý ở bước tiếp theo:

- loại hoặc thu lại dữ liệu `person05` theo đúng acquisition protocol trước khi modeling chính thức;
- xem xét bất thường `forward_slouch` của `person10`;
- thiết kế split theo `person_id` để tránh subject leakage;
- chuẩn hóa preprocessing bằng sklearn `Pipeline`;
- xem xét scaling cho SVM;
- đánh giá redundancy/feature selection dựa trên correlation, class separation, subject consistency và performance model.

Không thực hiện preprocessing trong nhiệm vụ EDA này.

## 9. Hướng phát triển tiếp theo

Xây dựng `preprocessing.py` dựa trên các kết quả EDA của `data_v02` trước khi train model.

Các hướng đề xuất ở mức thiết kế:

- xử lý `person05` bằng cách loại khỏi modeling dataset tạm thời hoặc thu lại dữ liệu đúng protocol;
- kiểm tra thêm dữ liệu raw/rejected của `person10` ở class `forward_slouch`;
- áp dụng split theo `person_id`, không random split theo frame;
- dùng GroupKFold hoặc Leave-One-Group-Out theo `person_id` cho model selection;
- fit `StandardScaler`, PCA hoặc các preprocessing transformations chỉ trên training set thông qua `Pipeline`;
- đánh giá feature redundancy và feature selection sau khi có kết quả cross-validation;
- tránh data leakage khi tuning XGBoost/SVM.

## 10. Các file liên quan

- Notebook EDA: `notebooks/01_eda_dataset.ipynb`
- Dataset tham chiếu trong notebook: `data/data_v02/features.csv`
- Báo cáo công việc: `docs/EDA_data_v02_29_features_2026-09-25.md`
