from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SessionStatistics:
    start_time: Optional[float] = None
    total_frames: int = 0
    correct_frames: int = 0
    bad_frames: int = 0
    no_person_frames: int = 0
    alert_count: int = 0
    consecutive_bad_frames: int = 0
    last_alert_time: float = 0.0

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
        alert_threshold_frames: int = 20

    def update(self, label: str | None) -> bool:
        """Cập nhật thống kê frame. Trả về True nếu kích hoạt cảnh báo ngồi sai liên tục."""
        # Kích hoạt mốc thời gian xuất phát khi frame đầu tiên được gửi tới
        if self.start_time is None:
            self.start_time = time.time()

        self.total_frames += 1
        trigger_alert = False

        if label is None or label in ("no_person", "evaluating"):
            self.no_person_frames += 1
            self.consecutive_bad_frames = 0
        elif label == "correct":
            self.correct_frames += 1
            self.consecutive_bad_frames = 0
        else:
            self.bad_frames += 1
            self.consecutive_bad_frames += 1

            now = time.time()
            if self.consecutive_bad_frames >= self.alert_threshold_frames:
                if (now - self.last_alert_time) > 4.0:  # Debounce cảnh báo 4 giây
                    self.alert_count += 1
                    self.last_alert_time = now
                    trigger_alert = True

        return trigger_alert

    def to_dict(self) -> dict[str, Any]:
        # Nếu chưa có frame nào chạy thì thời gian trôi qua là 0 giây
        if self.start_time is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, time.time() - self.start_time)

        valid_frames = self.correct_frames + self.bad_frames
        score = (
            round((self.correct_frames / valid_frames) * 100, 1)
            if valid_frames > 0
            else 100.0
        )

        return {
            "elapsed_seconds": int(elapsed),
            "total_frames": self.total_frames,
            "correct_frames": self.correct_frames,
            "bad_frames": self.bad_frames,
            "no_person_frames": self.no_person_frames,
            "ergonomics_score": score,
            "alert_count": self.alert_count,
            "consecutive_bad_frames": self.consecutive_bad_frames,
        }