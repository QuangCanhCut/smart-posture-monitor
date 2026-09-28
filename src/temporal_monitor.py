from __future__ import annotations

from collections import Counter, deque
from typing import Optional


class TemporalMonitor:
    def __init__(
        self,
        fps: int = 10,
        window_seconds: float = 2.0,
        hysteresis_frames: int = 3,
    ):
        self.window_size = max(1, int(fps * window_seconds))
        self.hysteresis = hysteresis_frames

        self.history = deque(maxlen=self.window_size)
        self.current_smoothed_label: str = "correct"
        self.candidate_label: Optional[str] = None
        self.candidate_count: int = 0

    def reset(self) -> None:
        self.history.clear()
        self.current_smoothed_label = "correct"
        self.candidate_label = None
        self.candidate_count = 0

    def update(self, raw_prediction: str) -> str:
        self.history.append(raw_prediction)

        # Lấy nhãn chiếm ưu thế trong cửa sổ trượt
        counts = Counter(self.history)
        dominant_label, _ = counts.most_common(1)[0]

        # Bộ lọc Hysteresis chống chớp giật nhãn
        if dominant_label != self.current_smoothed_label:
            if dominant_label == self.candidate_label:
                self.candidate_count += 1
                if self.candidate_count >= self.hysteresis:
                    self.current_smoothed_label = dominant_label
                    self.candidate_label = None
                    self.candidate_count = 0
            else:
                self.candidate_label = dominant_label
                self.candidate_count = 1
        else:
            self.candidate_label = None
            self.candidate_count = 0

        return self.current_smoothed_label