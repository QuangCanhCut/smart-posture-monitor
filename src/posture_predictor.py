from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class PosturePredictor:
    """
    Wrapper dùng để dự đoán tư thế từ REP13.

    Trách nhiệm:
    - Load model và training metadata.
    - Kiểm tra contract của feature đầu vào.
    - Nhận vector REP13 đúng thứ tự đã dùng khi training.
    - Trả về label và probability.

    Không chịu trách nhiệm:
    - Detect người.
    - Extract keypoints.
    - Extract raw features.
    - Personal calibration.
    - Temporal smoothing.
    """

    def __init__(
        self,
        model_path: Path | str,
        metadata_path: Path | str,
    ) -> None:
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata(metadata_path)

        feature_columns = (
            self.metadata.get("rep_feature_names")
            or self.metadata.get("feature_columns")
        )
        if not feature_columns:
            raise RuntimeError(
                "Metadata không chứa rep_feature_names/feature_columns."
            )
        self.feature_columns = [str(column) for column in feature_columns]

        self.feature_count = len(self.feature_columns)

        label_to_id = (
            self.metadata.get("class_mapping")
            or self.metadata.get("label_to_id")
            or {}
        )
        raw_id_to_label = self.metadata.get("id_to_label") or {
            str(value): key
            for key, value in label_to_id.items()
        }

        self.id_to_label = {
            int(key): str(value)
            for key, value in raw_id_to_label.items()
        }

        self.model = joblib.load(model_path)

        self._validate_training_contract()

    @staticmethod
    def _load_metadata(
        metadata_path: Path,
    ) -> dict[str, Any]:
        """Đọc training metadata từ file JSON."""
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy training metadata: {metadata_path}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if not isinstance(metadata, dict):
            raise ValueError(
                "training_metadata.json phải chứa JSON object."
            )

        return metadata

    def _validate_training_contract(self) -> None:
        """
        Kiểm tra model và metadata có đúng contract V03 hay không.
        """
        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Model đã load không có method predict()."
            )

        if self.feature_count != 13:
            raise RuntimeError(
                "PosturePredictor V03 yêu cầu đúng 13 REP features, "
                f"nhưng metadata chứa {self.feature_count}."
            )

        representation = self.metadata.get("representation")

        if str(representation).upper() != "REP13":
            raise RuntimeError(
                "Model hiện tại không được train bằng REP13."
            )

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if (
            model_feature_count is not None
            and int(model_feature_count) != self.feature_count
        ):
            raise RuntimeError(
                "Số feature model mong đợi không khớp metadata."
            )

        if not self.id_to_label:
            raise RuntimeError(
                "Không tìm thấy label mapping trong metadata."
            )

    def _validate_rep_features(
        self,
        rep_features: np.ndarray | list[float],
    ) -> np.ndarray:
        """
        Kiểm tra vector REP13 đầu vào.

        Vector hợp lệ:
        - Có đúng 1 chiều.
        - Có đúng 13 giá trị.
        - Không chứa NaN hoặc Infinity.
        """
        try:
            features = np.asarray(
                rep_features,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "REP13 features phải là dữ liệu dạng số."
            ) from error

        if features.ndim != 1:
            raise ValueError(
                "REP13 features phải là vector 1 chiều."
            )

        if features.shape != (self.feature_count,):
            raise ValueError(
                f"Expected {self.feature_count} REP13 features, "
                f"got shape {features.shape}."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "REP13 features chứa NaN hoặc Infinity."
            )

        return features

    def predict(
        self,
        rep_features: np.ndarray | list[float],
    ) -> dict[str, Any]:
        """
        Dự đoán tư thế từ vector REP13.
        """
        features = self._validate_rep_features(
            rep_features
        )

        # Giữ đúng feature names/order giống lúc training.
        X_live = pd.DataFrame(
            [features],
            columns=self.feature_columns,
        )

        pred_id = int(
            np.asarray(
                self.model.predict(X_live)
            ).reshape(-1)[0]
        )

        if pred_id not in self.id_to_label:
            raise RuntimeError(
                f"Model trả về class ID không xác định: {pred_id}"
            )

        raw_label = self.id_to_label[pred_id]

        probability = self._predict_probability(
            X_live,
            pred_id,
        )

        return {
            "status": "OK",
            "raw_label": raw_label,
            "label_id": pred_id,
            "probability": (
                round(probability, 3)
                if probability is not None
                else None
            ),
        }

    def _predict_probability(
        self,
        X_live: pd.DataFrame,
        pred_id: int,
    ) -> float | None:
        """
        Lấy xác suất của class được dự đoán nếu model hỗ trợ.
        """
        if not hasattr(self.model, "predict_proba"):
            return None

        try:
            probabilities = np.asarray(
                self.model.predict_proba(X_live),
                dtype=float,
            )

            classes = np.asarray(
                getattr(
                    self.model,
                    "classes_",
                    [],
                )
            )

            if (
                probabilities.ndim != 2
                or probabilities.shape[0] != 1
            ):
                return None

            positions = np.where(
                classes == pred_id
            )[0]

            if positions.size == 1:
                return float(
                    probabilities[
                        0,
                        positions[0],
                    ]
                )

            return float(
                probabilities[0].max()
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return None
