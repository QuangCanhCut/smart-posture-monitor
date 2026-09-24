**SMART POSTURE MONITOR**

**BẢN THẢO BÀI TẬP LỚN PYTHON \- TÀI LIỆU THỐNG NHẤT LÀM VIỆC NHÓM**

Hệ thống nhận diện tư thế ngồi bằng Webcam, YOLO Pose và Machine Learning

 

| Thuộc tính | Nội dung |
| ----- | ----- |
| Phiên bản | Working Draft v0.1 |
| Phạm vi nhận diện | 4 tư thế: correct, forward\_slouch, lean\_left, lean\_right |
| Camera | Góc nghiêng khoảng 45° bên trái người dùng |
| Pose model | YOLO26n Pose |
| Đầu vào ML | Vector 18 feature hình học từ 6 keypoint |
| Mục đích tài liệu | Khóa quy ước kỹ thuật, dữ liệu, source code, phân công và quy trình bàn giao giữa các thành viên |

 

**Nhóm thực hiện**

| Thành viên | Vai trò chính |
| ----- | ----- |
| Quang | Computer Vision & Data Interface Engineer |
| Mạnh | Machine Learning & Data Processing Engineer |
| Quân | Application & Integration Engineer |

 

 

# **0\. Cách sử dụng tài liệu này**

Tài liệu này không phải báo cáo cuối kỳ hoàn chỉnh. Đây là "working specification" để cả nhóm dùng chung trong quá trình phát triển. Khi có thay đổi quan trọng về nhãn, camera, keypoint, feature, schema dữ liệu hoặc interface giữa các module, nhóm cần cập nhật tài liệu này trước khi tiếp tục train/tích hợp.

·       Các tên label trong tài liệu là chuẩn chính thức của project; không tự đổi tên ở từng máy.

·       Thứ tự 18 feature phải giữ cố định giữa dataset building, training và inference.

·       Left/Right theo chuẩn COCO được hiểu theo cơ thể người được nhận diện, không theo phía trái/phải trên màn hình.

·       Không đưa frame gần nhau của cùng một video/session vào cả train và test.

·       Mỗi module chỉ đảm nhận đúng trách nhiệm đã mô tả để giảm xung đột khi làm việc song song.

# **1\. Mục tiêu và phạm vi bài toán**

Nhóm xây dựng một hệ thống nhận diện tư thế ngồi theo thời gian thực từ webcam. Hệ thống không phân loại trực tiếp từ ảnh bằng một mạng deep learning end-to-end; thay vào đó dùng YOLO Pose để lấy keypoint, sau đó trích xuất đặc trưng hình học và dùng mô hình Machine Learning để phân loại tư thế.

Pipeline mục tiêu:

Webcam / Image  
 	↓  
 YOLO26n Pose  
 	↓  
 Chọn 1 person mục tiêu  
 	↓  
 6 keypoints  
 	↓  
 FeatureExtractor  
 	↓  
 18 features  
 	↓  
 Preprocessing  
 	↓  
 SVM / XGBoost  
 	↓  
 Posture prediction  
 	↓  
 Temporal monitoring  
 	↓  
 Alert \+ Session statistics \+ Streamlit UI

MVP bắt buộc của đề tài gồm:

·       Nhận webcam realtime.

·       Phát hiện người và đúng 6 keypoint cần thiết.

·       Phân loại 4 tư thế theo bộ label thống nhất.

·       Cảnh báo khi tư thế xấu kéo dài trong một khoảng thời gian.

·       Hiển thị trạng thái hiện tại và một số thống kê phiên làm việc.

·       Có báo cáo đánh giá mô hình bằng các metric tiêu chuẩn và confusion matrix.

# **2\. Bộ nhãn chính thức của dataset**

Mỗi frame chỉ có đúng 1 label. Đây là bộ label thống nhất từ thời điểm hiện tại và phải được dùng giống nhau trong tên thư mục dữ liệu, file CSV, preprocessing, training, evaluation và inference.

| ID | Label chuẩn | Mô tả |
| ----- | ----- | ----- |
| 0 | correct | Ngồi đúng tư thế. |
| 1 | forward\_slouch | Cúi/gù phần đầu \- thân trên về phía trước. |
| 2 | lean\_left | Nghiêng sang trái. |
| 3 | lean\_right | Nghiêng sang phải. |

 

Quy ước semantic: lean\_left và lean\_right được hiểu theo hướng nghiêng của chính người dùng. Khi debug keypoint, quy ước Left/Right của YOLO/COCO vẫn theo cơ thể người, vì vậy có thể trông "ngược" khi xem ảnh camera đối diện.

# **3\. Quy ước camera và keypoint**

Camera được đặt cố định khoảng 45° về phía bên trái người dùng thay vì chính diện. Mục đích là giúp chuyển động cúi/gù về trước tạo ra thay đổi 2D rõ hơn trên ảnh (dịch theo cả trục ngang và dọc).

Do góc camera này khiến tai phải có thể lúc nhìn thấy, lúc bị che khuất, project chủ động loại bỏ right\_ear khỏi interface để giữ input nhất quán giữa mọi frame.

| COCO index | Keypoint | Trạng thái | Lý do |
| ----- | ----- | ----- | ----- |
| 0 | nose | Giữ | Đại diện vị trí đầu và pitch tương đối. |
| 1 | left\_eye | Giữ | Dùng cho eye line, eye center và head tilt. |
| 2 | right\_eye | Giữ | Dùng cùng left\_eye để xác định nghiêng đầu. |
| 3 | left\_ear | Giữ | Ổn định hơn ở góc camera hiện tại; bổ sung orientation của đầu. |
| 4 | right\_ear | Bỏ | Occlusion không ổn định ở góc camera 45°. |
| 5 | left\_shoulder | Giữ | Xây hệ tọa độ cơ thể. |
| 6 | right\_shoulder | Giữ | Xây hệ tọa độ cơ thể và shoulder width. |

 

Mỗi keypoint được biểu diễn dưới dạng \[x, y, confidence\]. Confidence dùng để kiểm soát chất lượng frame, không dùng như một feature posture.

# **4\. Kiến trúc source code và trách nhiệm từng module**

| Module | Vai trò | Chủ trì |
| ----- | ----- | ----- |
| src/pose\_detector.py | Ảnh/frame → YOLO Pose → chọn person confidence cao nhất → 6 keypoint. | Quang |
| src/feature\_extractor.py | 6 keypoint → hệ tọa độ theo vai → vector 18 feature. | Quang |
| src/dataset\_builder.py | Quét dataset ảnh → chạy PoseDetector \+ FeatureExtractor → tạo features.csv. | Mạnh (phối hợp Quang) |
| src/preprocessing.py | Đọc CSV, validate, split theo session/person, scale feature. | Mạnh |
| src/train.py | Train baseline/SVM/XGBoost, lưu model. | Mạnh |
| src/evaluate.py | Metric, classification report, confusion matrix, feature analysis. | Mạnh |
| src/posture\_predictor.py | Load scaler/model và dự đoán 1 vector feature realtime. | Mạnh \+ Quân |
| src/temporal\_monitor.py | Làm mượt prediction theo thời gian và quyết định cảnh báo. | Quân |
| src/session\_statistics.py | Tổng hợp thời gian từng posture, alert count, tỷ lệ tư thế đúng. | Quân |
| app/app.py | Ghép các module thành ứng dụng Streamlit/Webcam hoàn chỉnh. | Quân |

 

# **5\. Interface của PoseDetector**

PoseDetector là boundary của phần Computer Vision. Module này không tính angle/distance/ratio và không chứa logic ML. Nó chỉ chịu trách nhiệm biến một frame thành person được chọn \+ raw keypoints.

{  
 	"person\_confidence": float,  
 	"bbox": np.ndarray(\[x1, y1, x2, y2\]),  
 	"keypoints": {  
     	"nose": np.ndarray(\[x, y, conf\]),  
     	"left\_eye": np.ndarray(\[x, y, conf\]),  
     	"right\_eye": np.ndarray(\[x, y, conf\]),  
     	"left\_ear": np.ndarray(\[x, y, conf\]),  
     	"left\_shoulder": np.ndarray(\[x, y, conf\]),  
     	"right\_shoulder": np.ndarray(\[x, y, conf\])  
 	}  
 }

Quy tắc chọn chủ thể hiện tại: nếu YOLO phát hiện nhiều người, chọn person có bounding-box confidence cao nhất. Đây là rule MVP; nếu sau này cần robust hơn có thể bổ sung ưu tiên người gần tâm ảnh hoặc bbox lớn hơn, nhưng thay đổi này phải được thống nhất trước khi sửa interface.

# **6\. Interface của FeatureExtractor và vector 18 feature**

FeatureExtractor không biết gì về YOLO model. Nó chỉ nhận đúng output của PoseDetector, kiểm tra chất lượng keypoint, chuẩn hóa hình học và trả về numpy.ndarray shape (18,).

Các khoảng cách được chuẩn hóa theo shoulder\_width nhằm giảm ảnh hưởng của scale (gần/xa webcam). Hai vai đồng thời tạo hệ tọa độ tương đối theo cơ thể để giảm phụ thuộc vào vị trí người trong frame.

| \# | Feature | Ý nghĩa |
| ----- | ----- | ----- |
| 1 | shoulder\_angle | Góc đường hai vai; phản ánh độ nghiêng thân/vai. |
| 2 | eye\_shoulder\_angle | Góc đường hai mắt tương đối với vai; quan trọng cho lean\_left/lean\_right. |
| 3 | eye\_vertical\_difference | Chênh lệch độ cao hai mắt sau chuẩn hóa; bổ trợ phân biệt hướng nghiêng. |
| 4 | nose\_x\_body | Vị trí ngang của mũi trong hệ tọa độ theo vai. |
| 5 | nose\_y\_body | Độ cao tương đối của mũi so với hai vai; hữu ích cho forward\_slouch. |
| 6 | eye\_center\_x\_body | Vị trí ngang trung điểm hai mắt. |
| 7 | eye\_center\_y\_body | Độ cao trung điểm hai mắt; hữu ích cho cúi/gù. |
| 8 | left\_ear\_x\_body | Vị trí ngang tai trái. |
| 9 | left\_ear\_y\_body | Độ cao tai trái so với vai. |
| 10 | nose\_eye\_dx | Dịch ngang từ eye center tới nose trong hệ body. |
| 11 | nose\_eye\_dy | Dịch dọc từ eye center tới nose; phản ánh head pitch. |
| 12 | eye\_width\_ratio | Khoảng cách hai mắt / shoulder\_width. |
| 13 | ear\_eye\_ratio | Khoảng cách left\_ear \- left\_eye / shoulder\_width. |
| 14 | nose\_shoulder\_center\_distance | Khoảng cách nose \- shoulder center / shoulder\_width. |
| 15 | eye\_shoulder\_center\_distance | Khoảng cách eye center \- shoulder center / shoulder\_width. |
| 16 | nose\_shoulder\_asymmetry | Chênh khoảng cách nose tới hai vai; hỗ trợ hướng nghiêng. |
| 17 | eye\_shoulder\_asymmetry | Chênh khoảng cách eye center tới hai vai. |
| 18 | nose\_ear\_ratio | Khoảng cách nose \- left\_ear / shoulder\_width; bổ trợ orientation đầu. |

 

Nguyên tắc khóa interface: FEATURE\_NAMES trong FeatureExtractor là nguồn duy nhất xác định thứ tự feature. Dataset builder, preprocessing và predictor phải lấy danh sách này trực tiếp thay vì tự viết lại thứ tự bằng tay.

# **7\. Chuẩn dataset sau Feature Extraction**

Dữ liệu gốc hiện tại là các frame/ảnh đã cắt từ clip tự quay. DatasetBuilder phải chuyển các ảnh này thành bảng feature. CSV cuối không nên chỉ chứa feature \+ label; cần metadata để split đúng và truy vết lỗi.

| Nhóm cột | Cột đề xuất | Mục đích |
| ----- | ----- | ----- |
| Metadata | image\_path | Truy ngược frame nguồn khi cần debug. |
| Metadata | session\_id / clip\_id | Bắt buộc để tránh split frame cùng clip vào train và test. |
| Metadata | person\_id | Hữu ích khi có nhiều người tham gia dataset. |
| Target | label | correct / forward\_slouch / lean\_left / lean\_right. |
| Features | 18 cột FEATURE\_NAMES | Input chính cho ML. |

 

Cấu trúc thư mục raw được khuyến nghị:

data/raw/  
 ├── correct/  
 │   ├── session\_01/  
 │   └── session\_02/  
 ├── forward\_slouch/  
 ├── lean\_left/  
 └── lean\_right/

Output chuẩn: data/processed/features.csv

# **8\. Preprocessing và quy tắc chia dữ liệu**

Đây là khu vực dễ gây sai kết quả nhất. Vì frame được cắt từ video, các frame liên tiếp gần như giống nhau. Không được random split từng frame vào train/test nếu chúng đến từ cùng một clip/session.

Quy tắc bắt buộc:

·       Split theo session/clip hoặc theo person nếu dataset cho phép.

·       StandardScaler chỉ fit trên X\_train; sau đó transform X\_train/X\_test bằng cùng scaler.

·       Không fit scaler trên toàn bộ X trước khi split.

·       Không dùng test set để chọn feature/hyperparameter.

·       Lưu scaler đã fit để inference realtime dùng đúng biến đổi như training.

Sai: frame\_001(train), frame\_002(test), frame\_003(train) từ cùng một clip  
 Đúng: session\_01,02,03 → train; session\_04 → test

# **9\. Kế hoạch Machine Learning**

## **9.1 EDA trước khi train**

·       Đếm số sample hợp lệ theo 4 class.

·       Kiểm tra frame bị loại do keypoint confidence thấp.

·       Kiểm tra NaN/Inf/outlier.

·       Vẽ distribution/boxplot từng feature theo label.

·       Tính correlation để phát hiện feature trùng thông tin.

·       Quan sát feature có khả năng tách class, đặc biệt eye\_shoulder\_angle, nose\_y\_body, eye\_center\_y\_body, asymmetry.

## **9.2 Mô hình cần thử**

| Mô hình | Vai trò |
| ----- | ----- |
| Logistic Regression (khuyến nghị baseline) | Baseline đơn giản để biết feature có tín hiệu tuyến tính đến mức nào. |
| SVM | Mô hình chính; cần scale feature, thử linear/RBF và tune C/gamma. |
| XGBoost | Mô hình cây mạnh, hữu ích cho quan hệ phi tuyến và feature importance. |

 

## **9.3 Đánh giá**

·       Accuracy.

·       Precision, Recall, F1-score theo từng class và macro average.

·       Confusion Matrix.

·       Feature importance / permutation importance (tùy model).

·       Ablation: bỏ nhóm feature để kiểm tra feature nào thực sự đóng góp.

Không chọn model chỉ bằng Accuracy. Nếu forward\_slouch thường bị nhầm với correct, cần phân tích confusion matrix và quay lại dataset/feature engineering trước khi tăng độ phức tạp model.

# **10\. Inference realtime và ứng dụng**

Sau khi có model tốt nhất, phần ứng dụng không được train lại. Nó chỉ load artifacts đã khóa và chạy inference.

frame  
  ↓  
 PoseDetector  
  ↓  
 FeatureExtractor (18)  
  ↓  
 Scaler đã fit trên training  
  ↓  
 PosturePredictor  
  ↓  
 label hiện tại  
  ↓  
 TemporalMonitor  
  ↓  
 alert ổn định theo thời gian  
  ↓  
 SessionStatistics \+ Streamlit

TemporalMonitor cần chống cảnh báo nhấp nháy. Ví dụ chỉ cảnh báo khi posture xấu chiếm ≥80% trong cửa sổ gần nhất hoặc kéo dài liên tục trên một ngưỡng thời gian đã thống nhất.

# **11\. Phân công và điểm bàn giao**

| Thành viên | Phạm vi | Đầu ra phải bàn giao |
| ----- | ----- | ----- |
| Quang | Pose detection, chuẩn keypoint, feature engineering, interface dữ liệu. | pose\_detector.py; feature\_extractor.py; tests; FEATURE\_NAMES; quy ước camera/keypoint. |
| Mạnh | Dataset builder, EDA, preprocessing, train, evaluate. | features.csv; EDA; scaler; model; metric; confusion matrix; training notes. |
| Quân | Inference app, temporal logic, statistics, dashboard. | posture\_predictor.py; temporal\_monitor.py; session\_statistics.py; app.py. |

 

Điểm giao giữa Quang và Mạnh: output FeatureExtractor \+ FEATURE\_NAMES \+ metadata schema. Điểm giao giữa Mạnh và Quân: best model \+ scaler \+ label mapping \+ cách gọi predict.

# **12\. Quy ước Git/GitHub làm việc nhóm**

Repo chung dùng main làm nhánh ổn định. Khuyến nghị mỗi người làm trên branch riêng và tạo Pull Request vào main.

main  
 ├── feature/cv-feature    	\# Quang  
 ├── feature/ml-training  	\# Mạnh  
 └── feature/app-integration  \# Quân

·       Không commit .venv/, \_\_pycache\_\_, file tạm và IDE settings.

·       Không force push lên main.

·       Trước khi code: git pull; trước khi push: kiểm tra git status và chạy test liên quan.

·       Commit message gợi ý: feat:, fix:, docs:, test:, refactor:.

·       Mỗi Pull Request phải mô tả file đã sửa, interface có thay đổi không, test đã chạy và ảnh hưởng tới thành viên khác.

# **13\. Cấu trúc source code chuẩn**

smart\_posture\_monitor/  
 ├── app/  
 │   └── app.py  
 ├── data/  
 │   ├── raw/  
 │   └── processed/  
 ├── models/  
 │   ├── yolo26n-pose.pt      	\# local, không bắt buộc commit  
 │   ├── posture\_model.pkl    	\# sau training  
 │   └── scaler.pkl           	\# sau preprocessing  
 ├── notebooks/  
 │   └── EDA.ipynb  
 ├── src/  
 │   ├── \_\_init\_\_.py  
 │   ├── pose\_detector.py  
 │   ├── feature\_extractor.py  
 │   ├── dataset\_builder.py  
 │   ├── preprocessing.py  
 │   ├── train.py  
 │   ├── evaluate.py  
 │   ├── posture\_predictor.py  
 │   ├── temporal\_monitor.py  
 │   └── session\_statistics.py  
 ├── tests/  
 │   ├── \_\_init\_\_.py  
 │   ├── test\_pose\_detector.py  
 │   └── test\_feature\_extractor.py  
 ├── requirements.txt  
 └── README.md

# **14\. Lộ trình 5 tuần (bản cập nhật)**

| Tuần | Mục tiêu | Tiêu chí hoàn thành |
| ----- | ----- | ----- |
| 1 | Chứng minh pipeline CV \+ feature. | Webcam → YOLO26n → 6 keypoint → 18 features chạy realtime. |
| 2 | Hoàn thiện dataset feature và EDA. | features.csv có metadata; split strategy; EDA report; kiểm tra class balance. |
| 3 | Train và evaluate. | Baseline \+ SVM \+ XGBoost; confusion matrix; chọn model; lưu scaler/model. |
| 4 | Tích hợp app. | Realtime predictor \+ temporal monitoring \+ alert \+ session statistics \+ Streamlit. |
| 5 | Testing, tối ưu, báo cáo, demo. | Test nhiều người/camera, demo ổn định, hoàn thiện báo cáo và slide/video nếu cần. |

 

# **15\. Kế hoạch bàn giao cho ngày làm việc tiếp theo**

Mục tiêu gần nhất là để Mạnh có thể train model trên dataset hiện có mà không phải sửa lại phần CV/feature.

1\.     Tổ chức ảnh raw theo 4 label và theo session/clip.

2\.     Hoàn thiện dataset\_builder.py để gọi PoseDetector \+ FeatureExtractor cho từng ảnh.

3\.     Sinh data/processed/features.csv gồm metadata \+ 18 feature \+ label.

4\.     Chạy EDA và thống kê số frame bị loại do low confidence.

5\.     Chốt chiến lược split theo session/person.

6\.     Viết preprocessing.py, StandardScaler cho SVM và lưu scaler.

7\.     Train baseline, SVM, XGBoost.

8\.     Đánh giá bằng classification report \+ confusion matrix.

9\.     Lưu best model và thông tin label mapping để Quân tích hợp.

# **16\. Definition of Done theo từng mốc**

| Mốc | Được coi là hoàn thành khi |
| ----- | ----- |
| Pose Detection | Chọn đúng 1 person mục tiêu; trả đúng 6 keypoint \+ confidence; test webcam ổn định. |
| Feature Extraction | Trả vector shape (18,); loại frame low confidence; không NaN/Inf; test realtime OK. |
| Dataset Build | Có features.csv truy vết được source/session và đúng label mapping. |
| Preprocessing | Split không leakage; scaler chỉ fit train; pipeline reproducible. |
| Training | Có ít nhất SVM \+ XGBoost và artifacts được lưu. |
| Evaluation | Có metric theo class \+ confusion matrix \+ nhận xét lỗi chính. |
| Integration | Webcam → prediction → temporal alert chạy end-to-end. |
| Demo | Chạy được 4 posture trong điều kiện demo, không crash, có tài liệu hướng dẫn. |

 

# **17\. Decision Log \- các quyết định đang khóa**

| Quyết định | Trạng thái hiện tại |
| ----- | ----- |
| Số class | 4: correct, forward\_slouch, lean\_left, lean\_right |
| Camera | Khoảng 45° bên trái người dùng |
| Pose model | YOLO26n Pose |
| Keypoint dùng | nose, left\_eye, right\_eye, left\_ear, left\_shoulder, right\_shoulder |
| Keypoint bỏ | right\_ear |
| Feature vector | 18 feature theo FEATURE\_NAMES |
| Normalization | Theo shoulder\_width và hệ tọa độ hai vai |
| Person selection | BBox confidence cao nhất (MVP) |
| Split data | Theo session/clip/person, không random frame leakage |
| ML candidates | Baseline \+ SVM \+ XGBoost |

 

Nếu một quyết định trong bảng trên thay đổi, người sửa phải cập nhật tài liệu, source code liên quan và thông báo trong Pull Request trước khi merge.

# **Phụ lục A. Các lệnh kiểm tra nhanh**

\# Kích hoạt môi trường (nếu dùng CMD)  
 .venv\\Scripts\\activate.bat

 \# Test pose detector  
 python \-m tests.test\_pose\_detector

 \# Test feature extractor  
 python \-m tests.test\_feature\_extractor

 \# Kiểm tra import  
 python \-c "from src.pose\_detector import PoseDetector; from src.feature\_extractor import FeatureExtractor; print('Import OK')"

# **Phụ lục B. Checklist trước khi giao code cho thành viên khác**

·       Code đã Ctrl+S và test bằng đúng .venv.

·       Không còn reference model YOLO sai phiên bản.

·       Không đổi label hoặc FEATURE\_NAMES mà chưa báo nhóm.

·       README/tài liệu có hướng dẫn chạy và đường dẫn artifacts.

·       git status sạch hoặc hiểu rõ file nào đang thay đổi.

·       Không commit .venv và cache.

·       Pull Request ghi rõ interface có thay đổi hay không.

**\--- HẾT BẢN THẢO THỐNG NHẤT v0.1 \---**

