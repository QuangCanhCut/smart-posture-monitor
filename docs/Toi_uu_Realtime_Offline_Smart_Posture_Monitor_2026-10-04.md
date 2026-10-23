# Báo cáo tối ưu realtime và offline Smart Posture Monitor

- Ngày thực hiện: 2026-10-04
- Phạm vi: ứng dụng FastAPI chạy local cho một webcam và một người dùng.
- Mục tiêu: giữ webcam mượt, không backlog frame, giảm tải dashboard, giữ nguyên
  pipeline ML/calibration/statistics và cho phép chạy offline sau khi đã cài
  dependencies Python.

## 1. Luồng realtime đã kiểm tra

```text
Webcam (getUserMedia)
  -> canvas capture ở frontend
  -> WebSocket /ws/stream (base64 JPEG)
  -> FastAPI decode (base64 -> cv2.imdecode)
  -> PostureInferenceEngine.process_frame
  -> YOLO Pose -> FeatureExtractor RAW12
  -> PersonalCalibration -> RepresentationBuilder REP13 -> SVM
  -> TemporalMonitor -> SessionStatistics
  -> WebSocket response
  -> skeleton, posture, dashboard và chart
```

Model YOLO, SVM và metadata đã được khởi tạo singleton lúc server startup;
không có `YOLO(...)` hoặc `joblib.load(...)` trong `process_frame()`.

## 2. Root cause và bottleneck

### Kết quả đọc code thực tế

Phiên bản trước khi sửa **đã có** `frameInFlight`: frontend chỉ gọi
`sendFrame()` lại sau khi nhận frame result. Vì vậy, trong đường chạy bình
thường không có queue ảnh không giới hạn và không có bằng chứng rằng backend
đang xử lý tuần tự F1 -> F2 -> F3 cũ.

Các điểm còn gây rủi ro/chi phí là:

1. Reconnect WebSocket chưa có cơ chế nhận diện socket cũ hay guard cho timer
   reconnect khi trang bị đóng/khởi tạo lại.
2. Vòng `requestAnimationFrame` trước đây luôn chạy, kể cả sau khi tắt camera.
3. Dashboard có nhiều DOM update theo mỗi AI result, không cần thiết cho
   statistics vốn chỉ cần cập nhật khoảng một lần mỗi giây.
4. UI phụ thuộc Tailwind CDN, Lucide CDN, Chart.js CDN và Google Fonts, nên
   không đáp ứng yêu cầu offline.
5. Chart đã được tạo một lần và cập nhật theo phút (đúng hướng), nhưng dữ liệu
   hiển thị chưa có giới hạn cho các phiên rất dài.

## 3. Thay đổi đã thực hiện

### `app/app.py`

- Thêm `StaticFiles` và mount `/static`.
- Không thay đổi endpoint hiện có, WebSocket route, lifespan hay singleton
  inference engine.

### `app/templates/index.html`

- Giữ cơ chế one-frame-in-flight hiện có và bổ sung comment cho hành vi
  response-driven: backend trả kết quả xong thì mới chụp frame mới nhất.
- Bổ sung `reconnectTimer`, `isPageUnloading` và kiểm tra socket hiện hành:
  chỉ một WebSocket đang `OPEN/CONNECTING`, chỉ một reconnect timer và callback
  socket cũ không thể ghi đè trạng thái socket mới.
- Bổ sung cleanup `beforeunload`: dừng reconnect timer, animation frame,
  session/break interval, webcam tracks và WebSocket.
- Vòng vẽ canvas chỉ hoạt động khi camera bật, giới hạn khoảng 30 FPS. Luồng
  webcam/`getUserMedia` vẫn độc lập với inference YOLO.
- Skeleton vẫn vẽ bằng kết quả AI mới nhất; posture và biomechanics vẫn cập nhật
  theo AI result.
- Throttle phần statistics nặng (score, alert, thời lượng đúng, break tip) về
  tối đa 1 lần/giây. Session timer vẫn chạy 1 lần/giây.
- Chart vẫn aggregate theo phút, không recreate object Chart.js; chỉ giữ 120
  điểm gần nhất trên canvas (khoảng 2 giờ). Statistics backend không bị cắt.
- Hiển thị tách biệt `AI FPS`, `Server` (`server_processing_ms`) và `E2E`
  (`performance.now() - client_time`). E2E là latency end-to-end localhost,
  không phải Internet ping.
- Lịch sử session vẫn chỉ ghi `localStorage` khi kết thúc/reset session, không
  có write trong critical frame loop.
- Chuyển script/font URL sang `/static/vendor/...`.

### `app/static/vendor/`

Đã thêm asset frontend local để chạy offline:

- Tailwind CSS browser build 3.4.17.
- Lucide 1.52.0.
- Chart.js 4.5.1.
- Plus Jakarta Sans (400, 500, 600, 700, 800).

Nguồn, phiên bản và ghi chú license nằm tại
`app/static/vendor/README.md`. Không thêm Python package hay npm runtime.

### `tests/test_web_realtime_contract.py`

Thêm regression tests cho:

- Asset frontend phải local, không có resource URL HTTP(S) trong template.
- Frame sender không chạy bằng `setInterval`/`requestAnimationFrame` và có
  `frameInFlight`/response-driven scheduling.
- Reconnect và render loop có guard/cleanup.
- Dashboard throttle và chart bounded dataset.

## 4. Cơ chế chống frame backlog sau khi sửa

```text
AI rảnh
  -> chụp frame hiện tại F1, frameInFlight = true
  -> backend xử lý F1
  -> F2..Fn chỉ tồn tại trong video stream, không được gửi/queue
  -> nhận result F1, frameInFlight = false
  -> chụp frame mới nhất Fn
```

Vì không có array/deque ảnh ở frontend, backend hay `TemporalMonitor`, frame
backlog mục tiêu là 0. `TemporalMonitor` chỉ giữ `deque` prediction, có
`maxlen=7`; nó không giữ image frame.

## 5. Thành phần được giữ nguyên

Không sửa các phần sau:

- FastAPI và URL endpoint: `GET /`, `POST /api/reset_session`,
  `POST /api/calibrate`, `POST /api/set_alert_threshold`, `GET /api/info`,
  `WS /ws/stream`.
- Base64 JSON WebSocket protocol.
- YOLO model, SVM model, RAW12 feature extraction, REP13 representation,
  predictor, calibration pipeline và nhãn posture.
- `TemporalMonitor`: window 7, min samples 3, hysteresis 2, yaw gating.
- `SessionStatistics`: monotonic timing, counters, durations, ergonomics score
  và alert threshold.
- Skeleton, calibration/reset calibration, alert, chart, history, dark mode,
  sound và keyboard shortcuts.

Không có thay đổi API/data contract. Các metric `server_processing_ms` và
`client_time` vốn đã có trong response frame; frontend chỉ sử dụng chúng rõ
ràng hơn.

## 6. Offline và localhost

- Frontend không còn gọi CDN/Google Fonts khi chạy.
- URL WebSocket được xây dựng từ `window.location.host`, không hard-code địa
  chỉ Wi-Fi.
- Model vẫn load từ local:
  - `models/best_model.joblib`
  - `models/training_metadata.json`
  - `models/yolo26n-pose.pt`

Sau khi dependencies Python đã được cài, app có thể chạy khi tắt Wi-Fi.

## 7. Kiểm thử đã chạy

### Pass

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_web_realtime_contract.py `
  tests/test_temporal_monitor.py `
  tests/test_session_statistics.py `
  -p no:cacheprovider --basetemp .test-tmp-runtime
```

Kết quả: **14 passed**.

Ngoài ra đã pass:

- Parse syntax JavaScript inline bằng Node `--check`.
- `python -m compileall -q app src`.
- Smoke test Uvicorn không `--reload` ở `127.0.0.1:8765`.
- HTTP 200 cho `/`, `/api/info`, `/static/vendor/chart.js`.
- `POST /api/calibrate` trả `OK`.
- `POST /api/reset_session` trả `OK`.
- `POST /api/set_alert_threshold` với 5 giây trả `seconds: 5`.

### Chưa xác nhận tự động

- Webcam/prediction thực tế với camera vật lý.
- Di chuyển nhanh để đánh giá skeleton bằng mắt.
- Calibration đủ mẫu và reset calibration qua WebSocket.
- Chạy liên tục 10 phút.
- Stop/start camera, refresh và reconnect trong browser thật.
- Tắt Wi-Fi vật lý trong quá trình chạy.

### Vấn đề test có sẵn ngoài phạm vi nhiệm vụ

Chạy toàn bộ `pytest -q` hiện dừng ở collection vì
`tests/test_train.py` import `EXPECTED_FEATURE_COLUMNS` nhưng symbol này không
còn có trong `src.preprocessing`. Một số test calibration/preprocessing khác
cũng đang lệch API runtime hiện tại hoặc bị quyền thư mục temporary của môi
trường. Những vấn đề này không được sửa để tránh thay đổi training/calibration
ngoài phạm vi tối ưu realtime.

## 8. Cách chạy sau khi sửa

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`.

Khi benchmark, không dùng `--reload`.

## 9. Các tối ưu cố ý chưa làm

- Không chuyển binary WebSocket vì base64 không phải bottleneck đã được chứng
  minh sau khi scheduling đã response-driven; đổi protocol sẽ tăng rủi ro.
- Không resize/crop inference frame: resolution hiện tại chưa được benchmark
  là bottleneck và thay đổi có thể ảnh hưởng mapping keypoint/skeleton.
- Không thêm warm-up, multiprocessing, Redis/Celery, database hoặc kiến trúc
  production phức tạp; không phù hợp quy mô BTL một người dùng local.
- Không thay đổi model, feature engineering, training, temporal logic hay
  calibration để bảo toàn kết quả đang hoạt động.
