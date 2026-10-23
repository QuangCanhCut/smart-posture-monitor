# V03 Runtime WebSocket, Temporal Smoothing va Session Statistics - 04/10/2026

## 1. Tong quan cong viec

Phien lam viec nay hoan thien cac phan runtime cua Smart Posture Monitor V03 sau khi model REP13 da duoc chot:

- Sua luong WebSocket de chi co mot frame inference dang bay tai mot thoi diem.
- Tach canvas capture va canvas hien thi de tranh lay lai frame da ve skeleton lam input.
- Them do FPS/latency thuc te thay cho gia tri FPS co dinh.
- On dinh ket qua realtime bang temporal smoothing theo cua so co dinh.
- Chuyen thong ke session sang tinh theo thoi gian thuc.
- Tach reset session khoi reset calibration ca nhan.
- Bo sung test cho temporal monitor va session statistics.

Pham vi thay doi chi nam o runtime/inference/UI/test. Cac file training, feature representation, model artifact va metadata training khong bi thay doi.

## 2. Muc tieu

Runtime cu can duoc sua o cac diem chinh:

- Frontend gui frame theo interval co dinh, co nguy co backlog khi inference cham.
- Canvas output vua dung de hien thi vua dung de capture, de lam input bi tron voi skeleton/overlay.
- FPS hien thi la gia tri mac dinh, khong phan anh toc do thuc.
- Temporal smoothing co the bi nhieu boi `no_person`, `evaluating` hoac cac label trung gian.
- Thong ke vi pham dua nhieu vao so frame, khong phu hop khi FPS thuc te thay doi.
- Reset tren UI co nguy co xoa ca baseline calibration ca nhan.

Muc tieu sau khi sua la giu nguyen pipeline V03:

```text
RAW12
-> PersonalCalibration
-> REP13
-> PosturePredictor
-> TemporalMonitor
-> SessionStatistics
-> Web UI
```

## 3. Backend runtime

### `app/app.py`

WebSocket `/ws` duoc cap nhat de xu ly tung frame theo co che request/response:

- do `server_processing_ms` quanh `PostureInferenceEngine.process_frame()`;
- tra lai `client_time` trong ca response thanh cong va loi de frontend biet frame nao da ket thuc;
- xu ly action `calibrate` va `reset_calibration` truoc khi decode anh;
- guard truong hop payload khong co image hoac image rong;
- bo gia tri FPS hard-code khi khoi tao engine;
- endpoint set alert threshold dung don vi giay, khong nhan them voi FPS co dinh.

Ket qua la backend khong con gia dinh toc do webcam co dinh. Latency va FPS duoc do tu chinh luong runtime.

## 4. Inference engine

### `src/inference.py`

Inference runtime van giu nguyen thu tu feature/model cua V03:

```text
PoseDetector
-> FeatureExtractor RAW12
-> PersonalCalibration
-> RepresentationBuilder REP13
-> PosturePredictor
```

Nhung phan quanh model duoc sua de on dinh hon:

- TemporalMonitor duoc cau hinh cua so co dinh `window_size=7`, `min_samples=3`, `hysteresis_frames=2`.
- Chi 4 label hop le duoc dua vao smoothing:
  - `correct`
  - `forward_slouch`
  - `lean_left`
  - `lean_right`
- Frame `no_person`, calibration, low confidence hoac label khong hop le khong lam ban cua so smoothing.
- Che do yaw conservative giu lai posture on dinh gan nhat thay vi day `head_turned` vao cua so.
- Response bo sung `display_confidence`, dung cho UI hien thi confidence cua label da smooth.
- `reset_session` chi reset temporal state va statistics, khong xoa baseline calibration.
- `start_calibration` va `reset_calibration` van reset temporal state de tranh dung lai state cu.

## 5. Temporal smoothing

### `src/temporal_monitor.py`

TemporalMonitor duoc sua de phu hop voi runtime V03:

- Trang thai ban dau la `evaluating`, chua cong bo posture hop le khi chua du mau.
- Cua so smoothing chi chua cac posture hop le.
- Cac state trung gian nhu `no_person`, `evaluating`, `head_turned`, `low_confidence` khong duoc tinh vao vote.
- Khi du `min_samples`, monitor bootstrap posture dau tien bang majority vote.
- Sau khi da co posture on dinh, chuyen label can qua hysteresis de giam nhay label.
- `display_confidence` duoc tinh theo ty le label dang hien thi trong cua so hop le hien tai.
- `reset()` xoa window, pending transition va dua state ve `evaluating`.

Thay doi nay giup UI khong con hien cac trang thai dang danh gia nhu mot posture sai, dong thoi tranh tinh sai confidence khi label raw khac label smooth.

## 6. Session statistics

### `src/session_statistics.py`

Thong ke session duoc chuyen tu frame-based sang time-based bang `time.monotonic()`:

- Luu `correct_seconds` va `bad_seconds`.
- Alert threshold tinh theo giay, mac dinh 2.0 giay.
- Mot episode sai lien tuc chi tinh mot violation.
- Khi posture tro lai `correct`, episode sai duoc reset.
- Cac trang thai trung gian nhu no-person, calibration, low-confidence ket thuc episode sai nhung khong tinh la posture sai moi.
- Van giu frame counters de tuong thich voi UI va scoring cu.
- Ergonomics score uu tien duration thuc te, co fallback theo frame neu chua co duration.

Ket qua la thong ke khong con phu thuoc vao FPS ly thuyet. Neu inference cham, thoi gian sai van duoc tinh theo thoi gian thuc nguoi dung ngoi sai.

## 7. Frontend realtime UI

### `app/templates/index.html`

Frontend duoc sua de tranh backlog va hien thi du lieu runtime thuc:

- Them offscreen capture canvas rieng cho viec encode frame gui len server.
- Output canvas chi dung de ve video hien tai va skeleton moi nhat.
- Bo `setInterval(sendFrame, 66)`.
- Dung `requestAnimationFrame()` de render video lien tuc.
- Chi gui frame moi khi frame truoc da co response (`frameInFlight=false`).
- Tinh FPS tu so response thanh cong tren thoi gian thuc.
- Ping/RTT hien thi theo response websocket, khong cat bo cac gia tri lon hon 1500 ms.
- Confidence hien thi theo `display_confidence`, khop voi badge posture da smooth.
- Badge chi hien posture smooth hop le; `no_person` va `evaluating` khong bi hien thanh loi posture.
- Duration dung `stats.correct_seconds`, khong chia co dinh cho 15 FPS.
- Nut reset session chi reset session/runtime state, khong gui `reset_calibration`.

Thiet ke moi chap nhan viec model/YOLO co the cham, nhung khong de backlog tang dan. Skeleton co the tre mot response, nhung no khong bi cong don nhieu frame cu.

## 8. Tests da bo sung

Them hai file test moi:

- `tests/test_temporal_monitor.py`
- `tests/test_session_statistics.py`

Pham vi test:

- mot frame lean nhieu khong lam nhay posture;
- sustained lean moi duoc chuyen label sau majority + hysteresis;
- invalid state khong di vao smoothing window;
- head turn conservative giu state on dinh gan nhat;
- reset temporal dua state ve ban dau;
- bad posture duoi threshold khong tao alert;
- bad posture lien tuc chi tao mot violation;
- correct posture reset bad episode;
- neutral state ket thuc episode sai;
- reset statistics xoa counters va duration.

Tests dung fake clock cho `time.monotonic()` nen khong can sleep that.

## 9. Ket qua kiem tra

Da chay thanh cong cac lenh kiem tra lien quan den patch runtime:

```text
python -m pytest -q -p no:cacheprovider tests/test_temporal_monitor.py tests/test_session_statistics.py
```

Ket qua:

```text
10 passed in 0.02s
```

Kiem tra them:

- `python -m compileall -q src app`: thanh cong.
- Import smoke test cho `TemporalMonitor`, `SessionStatistics`, `PostureInferenceEngine`, `app.app`: thanh cong.
- Parse inline JavaScript trong `app/templates/index.html`: thanh cong.
- Static check khong con `setInterval(sendFrame, ...)`.
- Diff check cac file training/model representation quan trong khong co thay doi.

Full test suite cua repo hien van chua xanh do cac test cu dang lech voi API hien tai o preprocessing/personal calibration, vi du `EXPECTED_FEATURE_COLUMNS`, `is_calibrated`, `transform`, `fit_transform_offline`. Cac loi nay da ton tai ngoai pham vi patch runtime va khong duoc sua trong nhiem vu nay vi cac file training/representation can duoc giu nguyen.

## 10. File da thay doi

Runtime va UI:

- `app/app.py`
- `app/templates/index.html`
- `src/inference.py`
- `src/temporal_monitor.py`
- `src/session_statistics.py`

Tests:

- `tests/test_temporal_monitor.py`
- `tests/test_session_statistics.py`

Tai lieu:

- `docs/V03_Runtime_WebSocket_Temporal_Statistics_04-10-2026.md`

Khong thay doi cac artifact va file research/training chinh:

- `models/best_model.joblib`
- `models/best_model_v03_rep13.joblib`
- `models/training_metadata.json`
- `src/feature_extractor.py`
- `src/personal_calibration.py`
- `src/representation_builder.py`
- `src/preprocessing.py`
- `src/train.py`
- `src/evaluate.py`

## 11. Trang thai sau patch

Patch runtime da hoan thanh theo huong bao toan model V03 hien tai:

- khong doi feature order;
- khong doi label mapping;
- khong retrain;
- khong ghi de model artifact;
- khong doi logic RAW12/REP13/calibration cot loi.

Diem can luu y con lai la toc do thuc van phu thuoc vao YOLO va predictor. Neu backend mat 500-1000 ms cho mot frame, FPS UI se phan anh dung toc do nay. Thay doi hien tai khong lam model nhanh hon, nhung giup UI va thong ke khong bi sai do backlog hoac FPS gia dinh.
