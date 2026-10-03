from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


VALID_POSTURES = {"correct", "forward_slouch", "lean_left", "lean_right"}
BAD_POSTURES = VALID_POSTURES - {"correct"}
IGNORED_STATUSES = {
    "CALIBRATING",
    "CALIBRATION_REQUIRED",
    "CALIBRATION_SAMPLE_REJECTED",
    "CALIBRATED",
    "LOW_CONFIDENCE",
}


@dataclass
class SessionStatistics:
    """Frame counters plus monotonic, wall-clock posture durations."""

    alert_threshold_seconds: float = 2.0
    start_time: Optional[float] = None
    total_frames: int = 0
    correct_frames: int = 0
    bad_frames: int = 0
    no_person_frames: int = 0
    alert_count: int = 0
    consecutive_bad_frames: int = 0
    last_alert_time: float = 0.0
    slouch_frames: int = 0
    lean_left_frames: int = 0
    lean_right_frames: int = 0
    correct_seconds: float = 0.0
    bad_seconds: float = 0.0

    _last_update_time: Optional[float] = field(default=None, init=False, repr=False)
    _last_valid_label: Optional[str] = field(default=None, init=False, repr=False)
    _bad_since: Optional[float] = field(default=None, init=False, repr=False)
    _bad_episode_alerted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.alert_threshold_seconds <= 0:
            raise ValueError("alert_threshold_seconds phải lớn hơn 0.")

    def reset(self) -> None:
        """Reset session data without touching personal calibration."""
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
        self.correct_seconds = 0.0
        self.bad_seconds = 0.0
        self._last_update_time = None
        self._last_valid_label = None
        self._bad_since = None
        self._bad_episode_alerted = False

    def _accumulate_duration(self, now: float) -> None:
        if self._last_update_time is None:
            return
        delta = max(0.0, now - self._last_update_time)
        if self._last_valid_label == "correct":
            self.correct_seconds += delta
        elif self._last_valid_label in BAD_POSTURES:
            self.bad_seconds += delta

    def _end_bad_episode(self) -> None:
        self.consecutive_bad_frames = 0
        self._bad_since = None
        self._bad_episode_alerted = False

    def update(
        self,
        label: Optional[str],
        status: str = "OK",
        alert_threshold_seconds: Optional[float] = None,
    ) -> bool:
        """Update counters and return True once per continuous bad episode."""
        now = time.monotonic()
        self._accumulate_duration(now)

        threshold = (
            self.alert_threshold_seconds
            if alert_threshold_seconds is None
            else float(alert_threshold_seconds)
        )
        if threshold <= 0:
            raise ValueError("alert_threshold_seconds phải lớn hơn 0.")

        is_ignored = status in IGNORED_STATUSES
        is_no_person = status == "NO_PERSON" or label == "no_person"
        is_valid = status == "OK" and label in VALID_POSTURES

        if is_ignored or not is_valid:
            if is_no_person:
                self.total_frames += 1
                self.no_person_frames += 1
            self._last_valid_label = None
            self._last_update_time = now
            self._end_bad_episode()
            return False

        if self.start_time is None:
            self.start_time = now

        self.total_frames += 1
        self._last_valid_label = label
        self._last_update_time = now

        if label == "correct":
            self.correct_frames += 1
            self._end_bad_episode()
            return False

        self.bad_frames += 1
        self.consecutive_bad_frames += 1
        if label == "forward_slouch":
            self.slouch_frames += 1
        elif label == "lean_left":
            self.lean_left_frames += 1
        elif label == "lean_right":
            self.lean_right_frames += 1

        if self._bad_since is None:
            self._bad_since = now

        if not self._bad_episode_alerted and now - self._bad_since >= threshold:
            self._bad_episode_alerted = True
            self.alert_count += 1
            self.last_alert_time = now
            return True
        return False

    @property
    def elapsed_seconds(self) -> int:
        if self.start_time is None:
            return 0
        return int(max(0.0, time.monotonic() - self.start_time))

    @property
    def ergonomics_score(self) -> float:
        valid_seconds = self.correct_seconds + self.bad_seconds
        if valid_seconds > 0:
            return round((self.correct_seconds / valid_seconds) * 100.0, 1)
        valid_frames = self.correct_frames + self.bad_frames
        if valid_frames == 0:
            return 100.0
        return round((self.correct_frames / valid_frames) * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "total_frames": self.total_frames,
            "correct_frames": self.correct_frames,
            "bad_frames": self.bad_frames,
            "no_person_frames": self.no_person_frames,
            "correct_seconds": round(self.correct_seconds, 3),
            "bad_seconds": round(self.bad_seconds, 3),
            "ergonomics_score": self.ergonomics_score,
            "alert_count": self.alert_count,
            "consecutive_bad_frames": self.consecutive_bad_frames,
            "slouch_frames": self.slouch_frames,
            "lean_left_frames": self.lean_left_frames,
            "lean_right_frames": self.lean_right_frames,
        }
