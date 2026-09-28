import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.pose_detector import PoseDetector


"""
data/raw/
   ↓
1. Tìm tất cả ảnh
   ↓
2. Đọc đường dẫn ảnh
   ↓
3. Lấy label + person_id + session_id
   ↓
4. Kiểm tra cấu trúc folder có đúng không
   ↓
5. cv2.imread()
   ↓
6. PoseDetector.detect()
   ↓
7. Kiểm tra có đủ 6 keypoints không
   ↓
8. FeatureExtractor.extract()
   ↓
9. Có đủ 29 features không
   ↓
10A. HỢP LỆ                 10B. KHÔNG HỢP LỆ
     ↓                           ↓
 features.csv              rejected_images.csv
"""


class DatasetBuilder:
    """
    Quét thư mục ảnh raw, chạy PoseDetector và FeatureExtractor
    để trích xuất vector 29 đặc trưng và xuất ra file CSV phục vụ huấn luyện ML.

    Cấu trúc dữ liệu BẮT BUỘC:
        data/raw/<label>/<person_id>_<session_id>/<file>

    Ví dụ:
        data/raw/correct/person01_session01/frame_0001.jpg

    Quy ước nhãn hợp lệ:
        - correct
        - forward_slouch
        - lean_left
        - lean_right

    Các ảnh bị loại sẽ được ghi log vào:
        data/rejected/rejected_images.csv
    """

    EXPECTED_FEATURE_COUNT = 29

    VALID_LABELS = {
        "correct",
        "forward_slouch",
        "lean_left",
        "lean_right",
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        rejected_csv_path: Optional[str] = None,
        detector: Optional[PoseDetector] = None,
        extractor: Optional[FeatureExtractor] = None,
    ):
        project_root = Path(__file__).resolve().parents[1]

        self.raw_data_dir = (
            Path(raw_data_dir)
            if raw_data_dir
            else project_root / "data" / "raw"
        )

        self.output_csv_path = (
            Path(output_csv_path)
            if output_csv_path
            else project_root / "data" / "processed" / "features.csv"
        )

        self.rejected_csv_path = (
            Path(rejected_csv_path)
            if rejected_csv_path
            else project_root / "data" / "rejected" / "rejected_images.csv"
        )

        # Sử dụng detector và extractor được truyền vào hoặc tự khởi tạo.
        self.detector = detector if detector is not None else PoseDetector()
        self.extractor = extractor if extractor is not None else FeatureExtractor()

        # DatasetBuilder V2 yêu cầu đúng FeatureExtractor 29 features.
        self.feature_names = list(self.extractor.FEATURE_NAMES)

        if len(self.feature_names) != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "DatasetBuilder yêu cầu FeatureExtractor trả về "
                f"{self.EXPECTED_FEATURE_COUNT} features, nhưng FEATURE_NAMES "
                f"hiện có {len(self.feature_names)}."
            )

        # Headers của CSV dataset: Metadata + Label + 29 Features.
        self.headers = [
            "image_path",
            "session_id",
            "person_id",
            "label",
        ] + self.feature_names

        # Headers của CSV ghi lại các ảnh bị loại.
        self.rejected_headers = [
            "image_path",
            "session_id",
            "person_id",
            "label",
            "reason",
            "detail",
            "person_confidence",
            "keypoint_confidences",
        ]

    def _parse_path_info(
        self,
        image_path: Path,
    ) -> Tuple[Optional[str], str, str, Optional[str], str]:
        """
        Kiểm tra và trích xuất metadata từ đường dẫn ảnh.

        Cấu trúc hợp lệ duy nhất:
            data/raw/<label>/<person_id>_<session_id>/<file>

        Ví dụ:
            data/raw/correct/person01_session01/frame_0001.jpg

        Returns:
            label, session_id, person_id, error_reason, error_detail

        Nếu hợp lệ:
            error_reason = None
            error_detail = ""
        """

        try:
            rel_path = image_path.relative_to(self.raw_data_dir)
        except ValueError:
            return (
                None,
                "unknown",
                "unknown",
                "invalid_folder_structure",
                "Ảnh không nằm bên trong raw_data_dir.",
            )

        parts = rel_path.parts

        # Phải đúng dạng: <label>/<person_session>/<file>
        if len(parts) != 3:
            return (
                parts[0] if len(parts) >= 1 else None,
                "unknown",
                "unknown",
                "invalid_folder_structure",
                (
                    "Cấu trúc phải là "
                    "data/raw/<label>/<person_id>_<session_id>/<file>."
                ),
            )

        label = parts[0]
        person_session_folder = parts[1]

        if label not in self.VALID_LABELS:
            return (
                None,
                "unknown",
                "unknown",
                "invalid_label",
                f"Label '{label}' không thuộc VALID_LABELS.",
            )

        if "_" not in person_session_folder:
            return (
                label,
                "unknown",
                "unknown",
                "invalid_person_session_folder",
                (
                    f"Folder '{person_session_folder}' phải có dạng "
                    "<person_id>_<session_id>, ví dụ person01_session01."
                ),
            )

        person_id, session_id = person_session_folder.split("_", 1)

        if not person_id or not session_id:
            return (
                label,
                session_id if session_id else "unknown",
                person_id if person_id else "unknown",
                "invalid_person_session_folder",
                (
                    f"Folder '{person_session_folder}' không chứa đầy đủ "
                    "person_id và session_id."
                ),
            )

        # Với project hiện tại, quy ước tên nên là personXX_sessionXX.
        if not person_id.startswith("person") or not session_id.startswith("session"):
            return (
                label,
                session_id,
                person_id,
                "invalid_person_session_folder",
                (
                    f"Folder '{person_session_folder}' không đúng quy ước "
                    "personXX_sessionXX."
                ),
            )

        return label, session_id, person_id, None, ""

    def _validate_pose(
        self,
        pose: Any,
    ) -> Tuple[Optional[str], str, str]:
        """
        Kiểm tra cấu trúc output của PoseDetector trước khi FeatureExtractor chạy.

        Returns:
            reason, detail, keypoint_confidences

        Nếu pose hợp lệ:
            reason = None
        """

        if not isinstance(pose, dict):
            return (
                "invalid_pose_structure",
                "PoseDetector không trả về dictionary.",
                "",
            )

        keypoints = pose.get("keypoints")

        if not isinstance(keypoints, dict):
            return (
                "invalid_pose_structure",
                "Pose không chứa dictionary 'keypoints'.",
                "",
            )

        required_keypoints = list(PoseDetector.KEYPOINT_INDICES.keys())

        missing_keypoints = [
            name
            for name in required_keypoints
            if name not in keypoints
        ]

        if missing_keypoints:
            return (
                "missing_keypoints",
                "Thiếu keypoint: " + ", ".join(missing_keypoints),
                "",
            )

        confidence_parts = []

        for name in required_keypoints:
            keypoint = np.asarray(keypoints[name])

            if keypoint.size < 3:
                return (
                    "invalid_keypoint",
                    f"Keypoint '{name}' không đủ [x, y, confidence].",
                    "",
                )

            x = float(keypoint[0])
            y = float(keypoint[1])
            confidence = float(keypoint[2])

            if not np.isfinite([x, y, confidence]).all():
                return (
                    "invalid_keypoint",
                    f"Keypoint '{name}' chứa NaN hoặc Inf.",
                    "",
                )

            confidence_parts.append(
                f"{name}={confidence:.4f}"
            )

        return None, "", ";".join(confidence_parts)

    @staticmethod
    def _write_rejected(
        writer: csv.writer,
        image_path: Path,
        session_id: str,
        person_id: str,
        label: Optional[str],
        reason: str,
        detail: str = "",
        person_confidence: Optional[float] = None,
        keypoint_confidences: str = "",
    ) -> None:
        """
        Ghi một ảnh bị loại vào rejected_images.csv.
        """

        writer.writerow(
            [
                image_path.as_posix(),
                session_id,
                person_id,
                label if label is not None else "unknown",
                reason,
                detail,
                (
                    f"{person_confidence:.6f}"
                    if person_confidence is not None
                    else ""
                ),
                keypoint_confidences,
            ]
        )

    def build(self) -> Dict[str, Any]:
        """
        Quét toàn bộ thư mục raw, trích xuất 29 đặc trưng và ghi ra CSV.

        Đồng thời ghi toàn bộ ảnh bị loại cùng lý do vào
        data/rejected/rejected_images.csv.

        Trả về dictionary báo cáo thống kê.
        """

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Thư mục dữ liệu raw không tồn tại: {self.raw_data_dir}\n"
                "Cấu trúc yêu cầu:\n"
                "data/raw/<label>/<person_id>_<session_id>/<file>\n"
                f"Labels hợp lệ: {sorted(self.VALID_LABELS)}"
            )

        # Tạo thư mục output nếu chưa có.
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rejected_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Tìm toàn bộ ảnh và sort để thứ tự xử lý ổn định giữa các lần chạy.
        all_image_paths: List[Path] = sorted(
            [
                p
                for p in self.raw_data_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in self.IMAGE_EXTENSIONS
            ],
            key=lambda p: p.as_posix(),
        )

        stats: Dict[str, Any] = {
            "total_images_found": len(all_image_paths),
            "processed_success": 0,
            "rejected_total": 0,

            # Folder / metadata
            "skipped_invalid_folder_structure": 0,
            "skipped_invalid_label": 0,
            "skipped_invalid_person_session_folder": 0,

            # Image / pose
            "skipped_read_error": 0,
            "skipped_no_person": 0,
            "skipped_invalid_pose_structure": 0,
            "skipped_missing_keypoints": 0,
            "skipped_invalid_keypoint": 0,

            # Feature extraction
            "skipped_feature_extraction_failed": 0,
            "skipped_invalid_feature_length": 0,
            "skipped_non_finite_features": 0,

            # Unexpected runtime problems
            "skipped_processing_error": 0,

            "per_label_count": {
                label: 0
                for label in sorted(self.VALID_LABELS)
            },

            "per_person_count": {},
            "per_session_count": {},
        }

        print("=== BẮT ĐẦU XÂY DỰNG DATASET ===")
        print(f"Thư mục ảnh gốc: {self.raw_data_dir}")
        print(f"File CSV đầu ra: {self.output_csv_path}")
        print(f"File ảnh bị loại: {self.rejected_csv_path}")
        print(f"Số features mỗi mẫu: {len(self.feature_names)}")
        print(f"Tổng số file ảnh phát hiện: {len(all_image_paths)}\n")

        with (
            open(
                self.output_csv_path,
                mode="w",
                newline="",
                encoding="utf-8",
            ) as output_file,
            open(
                self.rejected_csv_path,
                mode="w",
                newline="",
                encoding="utf-8",
            ) as rejected_file,
        ):
            writer = csv.writer(output_file)
            rejected_writer = csv.writer(rejected_file)

            writer.writerow(self.headers)
            rejected_writer.writerow(self.rejected_headers)

            for idx, img_path in enumerate(all_image_paths, start=1):
                label = None
                session_id = "unknown"
                person_id = "unknown"

                try:
                    # =====================================================
                    # 1. Kiểm tra folder + parse label/person/session
                    # =====================================================
                    (
                        label,
                        session_id,
                        person_id,
                        path_error_reason,
                        path_error_detail,
                    ) = self._parse_path_info(img_path)

                    if path_error_reason is not None:
                        stats[
                            f"skipped_{path_error_reason}"
                        ] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            path_error_reason,
                            path_error_detail,
                        )
                        continue

                    # =====================================================
                    # 2. Đọc ảnh OpenCV
                    # =====================================================
                    frame = cv2.imread(str(img_path))

                    if frame is None:
                        stats["skipped_read_error"] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            "read_error",
                            "cv2.imread() trả về None.",
                        )
                        continue

                    # =====================================================
                    # 3. Phát hiện pose
                    # =====================================================
                    pose = self.detector.detect(frame)

                    if pose is None:
                        stats["skipped_no_person"] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            "no_person",
                            (
                                "PoseDetector.detect() trả về None. "
                                "Không phát hiện được person thỏa điều kiện."
                            ),
                        )
                        continue

                    person_confidence = pose.get(
                        "person_confidence"
                    )

                    if person_confidence is not None:
                        person_confidence = float(
                            person_confidence
                        )

                    # =====================================================
                    # 4. Kiểm tra đủ 6 keypoint + dữ liệu hợp lệ
                    # =====================================================
                    (
                        pose_error_reason,
                        pose_error_detail,
                        keypoint_confidences,
                    ) = self._validate_pose(pose)

                    if pose_error_reason is not None:
                        stats[
                            f"skipped_{pose_error_reason}"
                        ] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            pose_error_reason,
                            pose_error_detail,
                            person_confidence,
                            keypoint_confidences,
                        )
                        continue

                    # =====================================================
                    # 5. Trích xuất 29 features
                    # =====================================================
                    features = self.extractor.extract(pose)

                    if features is None:
                        stats[
                            "skipped_feature_extraction_failed"
                        ] += 1
                        stats["rejected_total"] += 1

                        confidence_values = [
                            float(pose["keypoints"][name][2])
                            for name in PoseDetector.KEYPOINT_INDICES
                        ]
                        min_confidence = min(confidence_values)

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            "feature_extraction_failed",
                            (
                                "FeatureExtractor.extract() trả về None. "
                                "Có thể do confidence keypoint thấp hoặc "
                                "hình học pose không đủ điều kiện. "
                                f"Min keypoint confidence={min_confidence:.4f}."
                            ),
                            person_confidence,
                            keypoint_confidences,
                        )
                        continue

                    features_array = np.asarray(
                        features,
                        dtype=np.float64,
                    ).reshape(-1)

                    if len(features_array) != self.EXPECTED_FEATURE_COUNT:
                        stats[
                            "skipped_invalid_feature_length"
                        ] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            "invalid_feature_length",
                            (
                                f"Nhận được {len(features_array)} features, "
                                f"nhưng yêu cầu {self.EXPECTED_FEATURE_COUNT}."
                            ),
                            person_confidence,
                            keypoint_confidences,
                        )
                        continue

                    if not np.isfinite(features_array).all():
                        stats[
                            "skipped_non_finite_features"
                        ] += 1
                        stats["rejected_total"] += 1

                        self._write_rejected(
                            rejected_writer,
                            img_path,
                            session_id,
                            person_id,
                            label,
                            "non_finite_features",
                            "Vector feature chứa NaN hoặc Inf.",
                            person_confidence,
                            keypoint_confidences,
                        )
                        continue

                    # =====================================================
                    # 6. Ghi mẫu hợp lệ vào features.csv
                    # =====================================================
                    row = [
                        img_path.as_posix(),
                        session_id,
                        person_id,
                        label,
                    ] + [
                        f"{value:.6f}"
                        for value in features_array
                    ]

                    writer.writerow(row)

                    stats["processed_success"] += 1
                    stats["per_label_count"][label] += 1

                    stats["per_person_count"].setdefault(
                        person_id,
                        0,
                    )

                    stats["per_person_count"][person_id] += 1

                    session_key = f"{person_id}_{session_id}"
                    stats["per_session_count"].setdefault(
                        session_key,
                        0,
                    )

                    stats["per_session_count"][session_key] += 1

                except Exception as exc:
                    # Không để một ảnh lỗi làm dừng toàn bộ quá trình build.
                    stats["skipped_processing_error"] += 1
                    stats["rejected_total"] += 1

                    self._write_rejected(
                        rejected_writer,
                        img_path,
                        session_id,
                        person_id,
                        label,
                        "processing_error",
                        f"{type(exc).__name__}: {exc}",
                    )

                if idx % 50 == 0 or idx == len(all_image_paths):
                    print(
                        f"Đã xử lý [{idx}/{len(all_image_paths)}] ảnh | "
                        f"Hợp lệ: {stats['processed_success']} | "
                        f"Bị loại: {stats['rejected_total']}"
                    )

        # =====================================================
        # 7. Báo cáo cuối
        # =====================================================
        total = stats["total_images_found"]
        success = stats["processed_success"]
        rejected = stats["rejected_total"]

        success_rate = (
            100.0 * success / total
            if total > 0
            else 0.0
        )

        print("\n=== HOÀN TẤT XÂY DỰNG DATASET ===")
        print(f"File feature: {self.output_csv_path}")
        print(f"File rejected: {self.rejected_csv_path}")
        print(f"Số mẫu hợp lệ: {success}/{total} ({success_rate:.2f}%)")
        print(f"Số mẫu bị loại: {rejected}/{total}")
        print(f"Số person có mẫu hợp lệ: {len(stats['per_person_count'])}")
        print(f"Số session có mẫu hợp lệ: {len(stats['per_session_count'])}")

        print("\nChi tiết số lượng hợp lệ theo từng nhãn:")
        for label, count in stats["per_label_count"].items():
            print(f"  - {label}: {count} mẫu")

        print("\nChi tiết số lượng hợp lệ theo từng người:")
        for person_id, count in sorted(
            stats["per_person_count"].items()
        ):
            print(f"  - {person_id}: {count} mẫu")

        print("\nChi tiết số lượng hợp lệ theo từng session:")
        for session_key, count in sorted(
            stats["per_session_count"].items()
        ):
            print(f"  - {session_key}: {count} mẫu")

        print("\nChi tiết lý do ảnh bị loại:")
        rejection_stat_keys = [
            key
            for key in stats
            if key.startswith("skipped_")
        ]

        for key in rejection_stat_keys:
            if stats[key] > 0:
                reason = key.removeprefix("skipped_")
                print(f"  - {reason}: {stats[key]}")

        return stats


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
