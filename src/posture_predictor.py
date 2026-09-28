from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.feature_extractor import FeatureExtractor
from src.pose_detector import PoseDetector


class PosturePredictor:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        yolo_path: Optional[Path] = None,
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        raw_id_to_label = self.metadata.get("id_to_label") or {
            str(v): k for k, v in self.metadata.get("label_to_id", {}).items()
        }
        self.id_to_label = {int(k): str(v) for k, v in raw_id_to_label.items()}
        self.feature_columns = [str(c) for c in self.metadata["feature_columns"]]

        self.model = joblib.load(model_path)
        self.detector = PoseDetector(
            model_path=yolo_path,
            person_conf_threshold=0.45,
        )
        self.extractor = FeatureExtractor(min_keypoint_confidence=0.30)

    def predict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        pose = self.detector.detect(frame_bgr)
        if pose is None:
            return {
                "has_person": False,
                "status": "NO_PERSON",
                "message": "Không tìm thấy người trong camera",
                "bbox": None,
                "keypoints": {},
                "angles": {},
                "raw_label": "no_person",
                "probability": None,
            }

        # Trích xuất keypoints sang dạng mảng [x, y, conf]
        keypoints_out: dict[str, list[float]] = {}
        for kp_name, kp_val in pose.get("keypoints", {}).items():
            arr = np.asarray(kp_val, dtype=float).reshape(-1)
            if len(arr) >= 2 and np.isfinite(arr[:2]).all():
                conf = float(arr[2]) if len(arr) >= 3 else 1.0
                keypoints_out[kp_name] = [round(float(arr[0]), 1), round(float(arr[1]), 1), round(conf, 2)]

        bbox_out = None
        if pose.get("bbox") is not None:
            raw_bbox = np.asarray(pose["bbox"], dtype=float).reshape(-1)
            if len(raw_bbox) >= 4 and np.isfinite(raw_bbox[:4]).all():
                bbox_out = [int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])]

        # Trích xuất 29 features
        features = self.extractor.extract(pose)
        if features is None:
            return {
                "has_person": True,
                "status": "LOW_CONFIDENCE",
                "message": "Keypoint bị che khuất hoặc không rõ",
                "bbox": bbox_out,
                "keypoints": keypoints_out,
                "angles": {},
                "raw_label": None,
                "probability": None,
            }

        features_array = np.asarray(features, dtype=np.float64).reshape(-1)
        feat_dict = dict(zip(FeatureExtractor.FEATURE_NAMES, features_array))

        # Phân loại tư thế bằng Classifier đã train
        X_live = pd.DataFrame([features_array], columns=self.feature_columns)
        pred_id = int(np.asarray(self.model.predict(X_live)).reshape(-1)[0])
        raw_label = self.id_to_label.get(pred_id, "unknown")

        probability = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs = np.asarray(self.model.predict_proba(X_live), dtype=float)
                classes = np.asarray(getattr(self.model, "classes_", []))
                if probs.ndim == 2 and probs.shape[0] == 1:
                    pos = np.where(classes == pred_id)[0]
                    probability = float(probs[0, pos[0]]) if pos.size == 1 else float(probs[0].max())
            except Exception:
                probability = None

        angles = {
            "shoulder_angle": round(feat_dict.get("shoulder_angle", 0.0), 1),
            "head_body_angle": round(feat_dict.get("head_body_angle", 0.0), 1),
            "head_gravity_angle": round(feat_dict.get("head_gravity_angle", 0.0), 1),
            "eye_shoulder_angle": round(feat_dict.get("eye_shoulder_angle", 0.0), 1),
            "face_pitch_angle": round(feat_dict.get("face_pitch_angle", 0.0), 1),
        }

        return {
            "has_person": True,
            "status": "OK",
            "message": "Phát hiện tư thế thành công",
            "bbox": bbox_out,
            "keypoints": keypoints_out,
            "angles": angles,
            "raw_label": raw_label,
            "probability": round(probability, 3) if probability is not None else None,
        }