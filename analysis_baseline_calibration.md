# Phân tích: Personal Baseline Calibration & Camera Integration

## 1. Tính năng camera trong `test_webcam_model.py` cần tích hợp

| Tính năng | Trong `test_webcam_model.py` | Trong `streamlit_app.py` | Cần thêm? |
|---|---|---|---|
| Skeleton overlay (bbox + keypoints) | ✅ `draw_pose()` | ✅ `draw_hud()` (đã có skeleton connections) | ❌ Đã có |
| Temporal Smoothing toggle | ✅ Phím S | ✅ Sidebar toggle | ❌ Đã có |
| Show/hide pose toggle | ✅ Phím P | ❌ Luôn bật, không toggle | ✅ Thêm toggle |
| FPS display | ✅ HUD panel | ✅ HUD overlay | ❌ Đã có |
| Person confidence | ✅ HUD | ❌ Không hiện | ✅ Thêm |
| Model probability | ✅ HUD | ✅ HUD | ❌ Đã có |
| Yaw D4 gating + indicator | ✅ HUD + icon | ⚠️ Gauge chỉ, không có HUD icon | ✅ Thêm vào HUD |
| Raw vs Smoothed label | ✅ HUD | ❌ Chỉ hiện smoothed | ✅ Thêm raw label |
| Color-coded skeleton by posture | ✅ `COLOR_MAP` | ✅ `COLOR_MAP_BGR` | ❌ Đã có |

## 2. Personal Baseline Delta — Trạng thái hiện tại

> **KẾT LUẬN: Project ĐÃ CÓ thiết kế & tài liệu cho Baseline Calibration nhưng CHƯA TRIỂN KHAI trong code chạy.**

- `src/feature_extractor.py` — Không có logic baseline/delta nào
- `src/temporal_monitor.py` — Không có baseline
- `src/posture_predictor.py` — **Không tồn tại** (đã được thiết kế trong docs nhưng chưa code)
- `docs/V03_feature_design_proposal.md` — Thiết kế chi tiết:
  - Thu thập N frame `correct` đầu (30 frames ≈ 2-3s)
  - Tính `f_baseline = mean(calib_frames)`
  - Delta features: `Δf = f_current - f_baseline`
  - Hybrid vector: absolute + delta (chọn 8 features nhạy cảm nhất)

## 3. Kế hoạch triển khai Baseline Calibration trong Streamlit

### Flow UX:
1. User bật webcam → hiện nút **"🎯 Bắt đầu Hiệu chuẩn Baseline (3s)"**
2. Click nút → Đếm ngược 3-2-1 trên HUD, thu thập features
3. Kết thúc 3s → Tính `baseline_vector`, lưu vào `session_state`
4. Từ đó trở đi, mỗi frame sẽ tính thêm 8 delta features
5. Hiện badge **"Baseline đã hiệu chuẩn ✅"** + nút reset

### Các thay đổi cần làm trong `streamlit_app.py`:
- Thêm session state: `baseline_vector`, `baseline_frames`, `calibration_state`
- Thêm calibration countdown overlay trên camera feed
- Sửa `process_frame()` để tính delta features khi có baseline
- Thêm UI elements: nút calibrate, badge trạng thái, thông tin baseline
- Thêm toggle bật/tắt skeleton overlay (từ `test_webcam_model.py`)
- Thêm person confidence + raw label vào HUD
