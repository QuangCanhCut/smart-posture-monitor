from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.posture_predictor import PosturePredictor
from src.session_statistics import SessionStatistics
from src.temporal_monitor import TemporalMonitor


class PostureInferenceEngine:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        yolo_path: Optional[Path] = None,
    ):
        self.predictor = PosturePredictor(
            model_path=model_path,
            metadata_path=metadata_path,
            yolo_path=yolo_path,
        )
        self.monitor = TemporalMonitor(fps=10, window_seconds=2.0, hysteresis_frames=3)
        self.stats = SessionStatistics()

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        result = self.predictor.predict(frame_bgr)

        if not result["has_person"]:
            trigger_alert = self.stats.update("no_person")
            result["smoothed_label"] = "no_person"
            result["trigger_alert"] = trigger_alert
            result["stats"] = self.stats.to_dict()
            return result

        raw_label = result.get("raw_label")
        if raw_label is None:
            trigger_alert = self.stats.update(None)
            result["smoothed_label"] = "evaluating"
            result["trigger_alert"] = trigger_alert
            result["stats"] = self.stats.to_dict()
            return result

        smoothed_label = self.monitor.update(raw_prediction=raw_label)
        trigger_alert = self.stats.update(smoothed_label)

        result["smoothed_label"] = smoothed_label
        result["trigger_alert"] = trigger_alert
        result["stats"] = self.stats.to_dict()
        return result

    def reset_session(self) -> None:
        self.monitor.reset()
        self.stats.reset()