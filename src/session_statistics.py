from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionStatistics:
    """
    Quản lý thống kê phiên làm việc theo thời gian thực (chuẩn V03).
    Tương thích với FeatureExtractor (RAW12), RepresentationBuilder (REP13)
    và TemporalMonitor (gating head_turned).
    """
    start_time: Optional[float] = None
    total_frames: int = 0
    correct_frames: int = 0
    bad_frames: int = 0
    no_person_frames: int = 0
    alert_count: int = 0
    consecutive_bad_frames: int = 0
    last_alert_time: float = 0.0

    # Ngưỡng kích hoạt cảnh báo: 30 frame ở 15 FPS = 2 giây ngồi sai liên tục
    alert_threshold_frames: int = 30

    # Phân loại chi tiết vi phạm phục vụ biểu đồ và bộ lọc lỗi của dashboard
    slouch_frames: int = 0
    lean_left_frames: int = 0
    lean_right_frames: int = 0

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái phiên về mốc khởi tạo."""
        self.start_time = None
        self.total_frames = 0
        self.correct_frames = 0
        self.bad_frames = 0
        self.no_person_frames = 0
        self.alert_count = 0
        self.consecutive_bad_frames = 0
        self.last_alert_time = 0.0
        self.slouch_frames = 0
        self.lean_left_frames = 0
        self.lean_right_frames = 0

    def update(
        self,
        label: Optional[str],
        status: str = "OK",
        alert_threshold_frames: Optional[int] = None,
        debounce_seconds: float = 4.0,
    ) -> bool:
        """ 
        Cập nhật thống kê frame.
        - Bỏ qua khi hệ thống đang CALIBRATING hoặc CALIBRATION_REQUIRED.
        - Trả về True nếu kích hoạt cảnh báo ngồi sai liên tục.
        """
        # Nếu đang trong quá trình thu 30 mẫu Personal Baseline thì không tính vào phiên
        if status in ("CALIBRATING", "CALIBRATION_REQUIRED", "CALIBRATION_SAMPLE_REJECTED"):
            return False
        # Nếu đang trong quá trình thu 30 mẫu Personal Baseline thì không tính vào phiên
        if status in ("CALIBRATING", "CALIBRATION_REQUIRED", "CALIBRATION_SAMPLE_REJECTED"):
            return False

        # ĐẢM BẢO SỐ FRAME LUÔN ĐƯỢC CẬP NHẬT ĐỘNG
        if alert_threshold_frames is None:
            alert_threshold_frames = self.alert_threshold_frames
            
        # Chỉ bắt đầu tính thời gian phiên khi xuất hiện frame nhận diện người hợp lệ đầu tiên
        if self.start_time is None and label is not None and label not in ("no_person", "evaluating"):
            self.start_time = time.time()

        self.total_frames += 1
        trigger_alert = False
        threshold = alert_threshold_frames or self.alert_threshold_frames

        # Xử lý các trạng thái không người, quay đầu hoặc đang chờ
        if label is None or label in ("no_person", "evaluating", "head_turned"):
            if label == "no_person":
                self.no_person_frames += 1
            self.consecutive_bad_frames = 0

        elif label == "correct":
            self.correct_frames += 1
            self.consecutive_bad_frames = 0

        else:
            # Ngồi sai tư thế (forward_slouch, lean_left, lean_right)
            self.bad_frames += 1
            self.consecutive_bad_frames += 1

            lbl_lower = label.lower()
            if "slouch" in lbl_lower or "gù" in lbl_lower:
                self.slouch_frames += 1
            elif "left" in lbl_lower or "trái" in lbl_lower:
                self.lean_left_frames += 1
            elif "right" in lbl_lower or "phải" in lbl_lower:
                self.lean_right_frames += 1

            now = time.time()
            if self.consecutive_bad_frames >= threshold:
                if (now - self.last_alert_time) > debounce_seconds:
                    self.alert_count += 1
                    self.last_alert_time = now
                    trigger_alert = True

        return trigger_alert

    @property
    def elapsed_seconds(self) -> int:
        if self.start_time is None:
            return 0
        return int(max(0.0, time.time() - self.start_time))

    @property
    def ergonomics_score(self) -> float:
        valid_frames = self.correct_frames + self.bad_frames
        if valid_frames == 0:
            return 100.0
        return round((self.correct_frames / valid_frames) * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        """Đóng gói dữ liệu thống kê trả về qua WebSocket cho frontend."""
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "total_frames": self.total_frames,
            "correct_frames": self.correct_frames,
            "bad_frames": self.bad_frames,
            "no_person_frames": self.no_person_frames,
            "ergonomics_score": self.ergonomics_score,
            "alert_count": self.alert_count,
            "consecutive_bad_frames": self.consecutive_bad_frames,
            "slouch_frames": self.slouch_frames,
            "lean_left_frames": self.lean_left_frames,
            "lean_right_frames": self.lean_right_frames,
        }