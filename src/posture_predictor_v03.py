from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_extractor import FeatureExtractor
from src.personal_calibration import PersonalCalibration


class PosturePredictorV03:
    """Inference helper for V03 delta-feature models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        schema_path: str | Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_dir = project_root / "models" / "v03"

        self.model_path = Path(model_path) if model_path else model_dir / "best_model.joblib"
        self.metadata_path = (
            Path(metadata_path) if metadata_path else model_dir / "training_metadata.json"
        )
        self.schema_path = Path(schema_path) if schema_path else model_dir / "feature_schema.json"

        self.model = self._load_model(self.model_path)
        self.metadata = self._load_json(self.metadata_path)
        self.feature_schema = self._load_json(self.schema_path)

        self.feature_columns = self._validate_feature_schema()
        self.id_to_label = self._get_id_to_label()

    def predict_delta(
        self,
        delta_features: np.ndarray | list[float],
    ) -> tuple[str, float | None, int]:
        features = self._validate_vector(delta_features)
        X_live = pd.DataFrame([features], columns=self.feature_columns)

        pred_id = int(np.asarray(self.model.predict(X_live)).reshape(-1)[0])
        if pred_id not in self.id_to_label:
            raise RuntimeError(f"Model returned unknown class id: {pred_id}")

        probability = self._predict_probability(X_live, pred_id)
        return self.id_to_label[pred_id], probability, pred_id

    def predict_raw(
        self,
        raw_features: np.ndarray | list[float],
        calibration: PersonalCalibration,
    ) -> tuple[str, float | None, int]:
        delta_features = calibration.transform(raw_features)
        return self.predict_delta(delta_features)

    def _validate_vector(self, feature_vector: np.ndarray | list[float]) -> np.ndarray:
        try:
            vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError("Feature vector must be numeric.") from error

        if vector.shape != (len(self.feature_columns),):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Feature vector contains NaN or infinity.")

        return vector

    @staticmethod
    def _load_model(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"V03 model not found: {path}")
        return joblib.load(path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"V03 artifact not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _validate_feature_schema(self) -> list[str]:
        feature_columns = self.feature_schema.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError("feature_schema.json must contain feature_columns.")

        feature_columns = [str(name) for name in feature_columns]
        current_features = list(FeatureExtractor.FEATURE_NAMES)

        if feature_columns != current_features:
            raise RuntimeError(
                "Current FeatureExtractor order does not match V03 feature schema."
            )

        if len(feature_columns) != 29:
            raise RuntimeError(f"V03 expects 29 delta features, got {len(feature_columns)}.")

        metadata_columns = self.metadata.get("feature_columns")
        if metadata_columns is not None and [str(name) for name in metadata_columns] != feature_columns:
            raise RuntimeError("training_metadata feature_columns mismatch feature_schema.")

        model_feature_names = getattr(self.model, "feature_names_in_", None)
        if model_feature_names is not None:
            if [str(name) for name in model_feature_names] != feature_columns:
                raise RuntimeError("Model feature_names_in_ mismatch feature_schema.")

        return feature_columns

    def _get_id_to_label(self) -> dict[int, str]:
        raw = self.metadata.get("id_to_label")
        if isinstance(raw, dict) and raw:
            return {int(class_id): str(label) for class_id, label in raw.items()}

        raw = self.metadata.get("label_to_id")
        if isinstance(raw, dict) and raw:
            return {int(class_id): str(label) for label, class_id in raw.items()}

        raise ValueError("training_metadata.json needs id_to_label or label_to_id.")

    def _predict_probability(
        self,
        X_live: pd.DataFrame,
        pred_id: int,
    ) -> float | None:
        if not hasattr(self.model, "predict_proba"):
            return None

        try:
            probabilities = np.asarray(self.model.predict_proba(X_live), dtype=float)
            classes = np.asarray(getattr(self.model, "classes_", []))
        except (AttributeError, TypeError, ValueError):
            return None

        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            return None

        if classes.size == probabilities.shape[1]:
            positions = np.where(classes == pred_id)[0]
            if positions.size == 1:
                return float(probabilities[0, positions[0]])

        return float(probabilities[0].max())
