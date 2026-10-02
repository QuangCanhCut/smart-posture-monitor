from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.pose_detector import PoseDetector


class DatasetBuilder:
    """
    Xây features.csv dạng: metadata + RAW features.

    Pipeline:
        image -> PoseDetector -> FeatureExtractor -> RAW12 -> CSV

    Không xử lý calibration, REP13, preprocessing hay training.

    Cấu trúc raw:
        data/raw/<label>/<personXX_sessionYY>/<image>
    """

    VALID_LABELS = (
        "correct",
        "forward_slouch",
        "lean_left",
        "lean_right",
    )
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Không hard-code 14 người: person15/session03... vẫn dùng được.
    PERSON_SESSION_RE = re.compile(
        r"^(person\d+)_(session\d+)$",
        re.IGNORECASE,
    )
    FRAME_INDEX_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)

    def __init__(
        self,
        raw_data_dir: str | Path | None = None,
        output_csv_path: str | Path | None = None,
        rejected_csv_path: str | Path | None = None,
        detector: PoseDetector | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]

        self.raw_data_dir = Path(raw_data_dir or root / "data" / "raw")
        self.output_csv_path = Path(
            output_csv_path or root / "data" / "processed" / "features.csv"
        )
        self.rejected_csv_path = Path(
            rejected_csv_path
            or root / "data" / "rejected" / "rejected_images.csv"
        )

        self.detector = detector if detector is not None else PoseDetector()
        self.extractor = extractor if extractor is not None else FeatureExtractor()

        # FeatureExtractor là nguồn duy nhất định nghĩa schema RAW.
        self.feature_names = list(self.extractor.FEATURE_NAMES)
        self.feature_count = len(self.feature_names)

        if self.feature_count == 0:
            raise ValueError("FeatureExtractor.FEATURE_NAMES đang rỗng.")

        self.headers = [
            "image_path",
            "session_id",
            "person_id",
            "label",
            "frame_index",
            *self.feature_names,
        ]
        self.rejected_headers = [
            "image_path",
            "session_id",
            "person_id",
            "label",
            "frame_index",
            "reason",
            "detail",
            "person_confidence",
            "keypoint_confidences",
        ]

    # ==========================================================
    # 1. SCAN + PARSE METADATA
    # ==========================================================

    def _find_images(self) -> list[Path]:
        """Tìm ảnh hợp lệ và sort để thứ tự build ổn định."""
        return sorted(
            (
                p for p in self.raw_data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            ),
            key=lambda p: p.as_posix(),
        )

    def _parse_metadata(
        self,
        image_path: Path,
    ) -> tuple[str | None, str, str, int | None, str | None, str]:
        """
        Parse:
            <label>/<personXX_sessionYY>/<image>

        Return:
            label, session_id, person_id, frame_index, error, detail
        """
        try:
            parts = image_path.relative_to(self.raw_data_dir).parts
        except ValueError:
            return (
                None, "unknown", "unknown", None,
                "invalid_folder_structure",
                "Ảnh không nằm trong raw_data_dir.",
            )

        frame_index = self._frame_index(image_path)

        if len(parts) != 3:
            return (
                parts[0] if parts else None,
                "unknown",
                "unknown",
                frame_index,
                "invalid_folder_structure",
                "Cấu trúc phải là <label>/<person_id>_<session_id>/<file>.",
            )

        label, recording_folder, _ = parts

        if label not in self.VALID_LABELS:
            return (
                None, "unknown", "unknown", frame_index,
                "invalid_label",
                f"Label '{label}' không hợp lệ.",
            )

        match = self.PERSON_SESSION_RE.fullmatch(recording_folder)
        if match is None:
            return (
                label, "unknown", "unknown", frame_index,
                "invalid_person_session_folder",
                f"Folder '{recording_folder}' phải có dạng personXX_sessionYY.",
            )

        person_id, session_id = (
            value.lower() for value in match.groups()
        )
        return label, session_id, person_id, frame_index, None, ""

    @classmethod
    def _frame_index(cls, image_path: Path) -> int | None:
        """Lấy số thứ tự từ frame_0001.jpg; không có thì trả None."""
        match = cls.FRAME_INDEX_RE.search(image_path.stem)
        return int(match.group(1)) if match else None

    # ==========================================================
    # 2. KIỂM TRA POSE + RAW FEATURES
    # ==========================================================

    def _validate_pose(
        self,
        pose: Any,
    ) -> tuple[str | None, str, str]:
        """
        Kiểm tra đúng 6 keypoint mà FeatureExtractor cần.
        Return: reason, detail, chuỗi confidence để audit.
        """
        if not isinstance(pose, Mapping):
            return "invalid_pose_structure", "Pose không phải dictionary.", ""

        keypoints = pose.get("keypoints")
        if not isinstance(keypoints, Mapping):
            return (
                "invalid_pose_structure",
                "Pose không chứa dictionary 'keypoints'.",
                "",
            )

        missing = [
            name for name in self.extractor.KEYPOINT_NAMES
            if name not in keypoints
        ]
        if missing:
            return (
                "missing_keypoints",
                "Thiếu keypoint: " + ", ".join(missing),
                "",
            )

        confidence_text: list[str] = []

        for name in self.extractor.KEYPOINT_NAMES:
            parsed = self._parse_keypoint(keypoints[name])
            if parsed is None:
                return (
                    "invalid_keypoint",
                    f"Keypoint '{name}' không hợp lệ.",
                    ";".join(confidence_text),
                )

            _, _, confidence = parsed
            confidence_text.append(f"{name}={confidence:.4f}")

            if confidence < self.extractor.min_keypoint_confidence:
                return (
                    "low_keypoint_confidence",
                    (
                        f"{name}={confidence:.4f} < "
                        f"{self.extractor.min_keypoint_confidence:.4f}"
                    ),
                    ";".join(confidence_text),
                )

        return None, "", ";".join(confidence_text)

    @staticmethod
    def _parse_keypoint(value: Any) -> tuple[float, float, float] | None:
        """Chuẩn hóa keypoint về (x, y, confidence)."""
        try:
            if isinstance(value, Mapping):
                x = float(value["x"])
                y = float(value["y"])
                confidence = float(
                    value.get(
                        "confidence",
                        value.get("conf", value.get("score", 1.0)),
                    )
                )
            else:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size < 2:
                    return None
                x, y = float(arr[0]), float(arr[1])
                confidence = float(arr[2]) if arr.size >= 3 else 1.0
        except (TypeError, ValueError, KeyError):
            return None

        return (
            (x, y, confidence)
            if np.all(np.isfinite([x, y, confidence]))
            else None
        )

    def _validate_features(self, features: Any) -> np.ndarray | None:
        """RAW vector phải đúng dimension và không chứa NaN/Inf."""
        if features is None:
            return None

        try:
            vector = np.asarray(features, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None

        if vector.shape != (self.feature_count,):
            return None
        if not np.all(np.isfinite(vector)):
            return None

        return vector

    # ==========================================================
    # 3. REJECT LOG
    # ==========================================================

    @staticmethod
    def _person_confidence(pose: Any) -> float | None:
        if not isinstance(pose, Mapping) or pose.get("person_confidence") is None:
            return None

        try:
            value = float(pose["person_confidence"])
        except (TypeError, ValueError):
            return None

        return value if np.isfinite(value) else None

    def _reject(
        self,
        writer: csv.writer,
        counts: Counter[str],
        image_path: Path,
        session_id: str,
        person_id: str,
        label: str | None,
        frame_index: int | None,
        reason: str,
        detail: str = "",
        person_confidence: float | None = None,
        keypoint_confidences: str = "",
    ) -> None:
        """Đếm và ghi một sample bị loại vào rejected_images.csv."""
        counts[reason] += 1
        writer.writerow(
            [
                image_path.as_posix(),
                session_id,
                person_id,
                label or "unknown",
                "" if frame_index is None else frame_index,
                reason,
                detail,
                "" if person_confidence is None else f"{person_confidence:.6f}",
                keypoint_confidences,
            ]
        )

    # ==========================================================
    # 4. BUILD
    # ==========================================================

    def build(self) -> dict[str, Any]:
        """Quét toàn bộ raw data và ghi metadata + RAW features."""
        if not self.raw_data_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy raw data directory: {self.raw_data_dir}"
            )

        image_paths = self._find_images()
        if not image_paths:
            raise ValueError(f"Không tìm thấy ảnh trong: {self.raw_data_dir}")

        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rejected_csv_path.parent.mkdir(parents=True, exist_ok=True)

        rejected: Counter[str] = Counter()
        per_label: Counter[str] = Counter()
        per_person: Counter[str] = Counter()
        per_session: Counter[str] = Counter()
        success = 0

        print("=== BUILD RAW FEATURE DATASET V03 ===")
        print(f"Images       : {len(image_paths)}")
        print(f"RAW features : {self.feature_count}")
        print(f"Output       : {self.output_csv_path}")
        print(f"Rejected     : {self.rejected_csv_path}\n")

        with (
            self.output_csv_path.open("w", newline="", encoding="utf-8") as out,
            self.rejected_csv_path.open("w", newline="", encoding="utf-8") as rej,
        ):
            writer = csv.writer(out)
            rejected_writer = csv.writer(rej)
            writer.writerow(self.headers)
            rejected_writer.writerow(self.rejected_headers)

            for index, image_path in enumerate(image_paths, start=1):
                label = None
                session_id = "unknown"
                person_id = "unknown"
                frame_index = self._frame_index(image_path)

                try:
                    (
                        label, session_id, person_id, frame_index,
                        error, detail,
                    ) = self._parse_metadata(image_path)

                    if error:
                        self._reject(
                            rejected_writer, rejected, image_path,
                            session_id, person_id, label, frame_index,
                            error, detail,
                        )
                        continue

                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        self._reject(
                            rejected_writer, rejected, image_path,
                            session_id, person_id, label, frame_index,
                            "read_error", "cv2.imread() trả về None.",
                        )
                        continue

                    pose = self.detector.detect(frame)
                    if pose is None:
                        self._reject(
                            rejected_writer, rejected, image_path,
                            session_id, person_id, label, frame_index,
                            "no_person", "PoseDetector.detect() trả về None.",
                        )
                        continue

                    person_confidence = self._person_confidence(pose)
                    pose_error, pose_detail, kp_conf = self._validate_pose(pose)

                    if pose_error:
                        self._reject(
                            rejected_writer, rejected, image_path,
                            session_id, person_id, label, frame_index,
                            pose_error, pose_detail, person_confidence, kp_conf,
                        )
                        continue

                    features = self.extractor.extract(pose)
                    vector = self._validate_features(features)

                    if vector is None:
                        reason = (
                            "feature_extraction_failed"
                            if features is None
                            else "invalid_features"
                        )
                        self._reject(
                            rejected_writer, rejected, image_path,
                            session_id, person_id, label, frame_index,
                            reason,
                            f"RAW vector phải có shape ({self.feature_count},).",
                            person_confidence,
                            kp_conf,
                        )
                        continue

                    # CSV nền của V03 chỉ lưu metadata + RAW12.
                    writer.writerow(
                        [
                            image_path.as_posix(),
                            session_id,
                            person_id,
                            label,
                            "" if frame_index is None else frame_index,
                            *[f"{value:.6f}" for value in vector],
                        ]
                    )

                    success += 1
                    per_label[label] += 1
                    per_person[person_id] += 1
                    per_session[f"{person_id}__{session_id}"] += 1

                except Exception as exc:
                    # Không để một ảnh lỗi làm dừng cả lần build.
                    self._reject(
                        rejected_writer, rejected, image_path,
                        session_id, person_id, label, frame_index,
                        "processing_error",
                        f"{type(exc).__name__}: {exc}",
                    )

                if index % 100 == 0 or index == len(image_paths):
                    print(
                        f"[{index}/{len(image_paths)}] "
                        f"valid={success} | rejected={sum(rejected.values())}"
                    )

        stats = {
            "total_images_found": len(image_paths),
            "processed_success": success,
            "rejected_total": sum(rejected.values()),
            "rejection_counts": dict(rejected),
            "per_label_count": dict(per_label),
            "per_person_count": dict(per_person),
            "per_session_count": dict(per_session),
            "feature_count": self.feature_count,
        }

        self._print_summary(stats)
        return stats

    def _print_summary(self, stats: dict[str, Any]) -> None:
        """Báo cáo nhanh sau khi build."""
        total = stats["total_images_found"]
        success = stats["processed_success"]
        rate = 100.0 * success / total if total else 0.0

        print("\n=== BUILD COMPLETE ===")
        print(f"Valid       : {success}/{total} ({rate:.2f}%)")
        print(f"Rejected    : {stats['rejected_total']}")
        print(f"Persons     : {len(stats['per_person_count'])}")
        print(f"Sessions    : {len(stats['per_session_count'])}")
        print(f"RAW features: {stats['feature_count']}")

        print("\nPer label:")
        for label in self.VALID_LABELS:
            print(f"  {label}: {stats['per_label_count'].get(label, 0)}")

        if stats["rejection_counts"]:
            print("\nRejected reasons:")
            for reason, count in sorted(stats["rejection_counts"].items()):
                print(f"  {reason}: {count}")


if __name__ == "__main__":
    DatasetBuilder().build()
