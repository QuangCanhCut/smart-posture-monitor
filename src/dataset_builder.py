import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.pose_detector import PoseDetector


class DatasetBuilder:
    """
    Quét thư mục ảnh raw, chạy PoseDetector và FeatureExtractor
    để trích xuất vector 18 đặc trưng và xuất ra file CSV phục vụ huấn luyện ML.
    
    Quy ước nhãn hợp lệ:
        - correct
        - forward_slouch
        - lean_left
        - lean_right
    """

    VALID_LABELS = {
        "correct",
        "forward_slouch",
        "lean_left",
        "lean_right"
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        detector: Optional[PoseDetector] = None,
        extractor: Optional[FeatureExtractor] = None
    ):
        project_root = Path(__file__).resolve().parents[1]

        self.raw_data_dir = (
            Path(raw_data_dir) if raw_data_dir else project_root / "data" / "raw"
        )
        self.output_csv_path = (
            Path(output_csv_path) if output_csv_path else project_root / "data" / "processed" / "features.csv"
        )

        # Sử dụng detector và extractor được truyền vào hoặc tự khởi tạo
        self.detector = detector if detector is not None else PoseDetector()
        self.extractor = extractor if extractor is not None else FeatureExtractor()

        # Headers của CSV: Metadata + Label + 18 Features
        self.headers = [
            "image_path",
            "session_id",
            "person_id",
            "label"
        ] + list(FeatureExtractor.FEATURE_NAMES)

    def _parse_path_info(self, image_path: Path) -> Tuple[Optional[str], str, str]:
        """
        Trích xuất label, session_id, person_id từ cấu trúc thư mục.
        Hỗ trợ:
            - data/raw/<label>/<session_id>/<file>
            - data/raw/<label>/<person_id>_<session_id>/<file>
            - data/raw/<label>/<file>
        """
        try:
            rel_path = image_path.relative_to(self.raw_data_dir)
        except ValueError:
            return None, "default", "unknown"

        parts = rel_path.parts

        if len(parts) < 2:
            return None, "default", "unknown"

        label = parts[0]
        if label not in self.VALID_LABELS:
            return None, "default", "unknown"

        if len(parts) >= 3:
            session_folder = parts[1]
            # Nếu tên folder có dạng personX_sessionY
            if "_" in session_folder:
                tokens = session_folder.split("_", 1)
                person_id = tokens[0]
                session_id = tokens[1]
            else:
                person_id = "unknown"
                session_id = session_folder
        else:
            person_id = "unknown"
            session_id = "default"

        return label, session_id, person_id

    def build(self) -> Dict[str, any]:
        """
        Quét toàn bộ thư mục raw, trích xuất đặc trưng và ghi ra CSV.
        Trả về dictionary báo cáo thống kê.
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Thư mục dữ liệu raw không tồn tại: {self.raw_data_dir}\n"
                f"Vui lòng tạo thư mục và đưa ảnh vào theo các nhãn: {sorted(list(self.VALID_LABELS))}"
            )

        # Tạo thư mục output nếu chưa có
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Tìm toàn bộ ảnh
        all_image_paths: List[Path] = [
            p for p in self.raw_data_dir.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        stats = {
            "total_images_found": len(all_image_paths),
            "processed_success": 0,
            "skipped_invalid_label": 0,
            "skipped_read_error": 0,
            "skipped_no_person": 0,
            "skipped_low_confidence": 0,
            "per_label_count": {label: 0 for label in self.VALID_LABELS}
        }

        print(f"=== BẮT ĐẦU XÂY DỰNG DATASET ===")
        print(f"Thư mục ảnh gốc: {self.raw_data_dir}")
        print(f"File CSV đầu ra: {self.output_csv_path}")
        print(f"Tổng số file ảnh phát hiện: {len(all_image_paths)}\n")

        with open(self.output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

            for idx, img_path in enumerate(all_image_paths, start=1):
                label, session_id, person_id = self._parse_path_info(img_path)

                if label is None:
                    stats["skipped_invalid_label"] += 1
                    continue

                # Đọc ảnh OpenCV
                frame = cv2.imread(str(img_path))
                if frame is None:
                    stats["skipped_read_error"] += 1
                    continue

                # 1. Phát hiện pose
                pose = self.detector.detect(frame)
                if pose is None:
                    stats["skipped_no_person"] += 1
                    continue

                # 2. Trích xuất 18 features
                features = self.extractor.extract(pose)
                if features is None:
                    stats["skipped_low_confidence"] += 1
                    continue

                # 3. Ghi dòng dữ liệu
                row = [
                    str(img_path.as_posix()),
                    session_id,
                    person_id,
                    label
                ] + [f"{val:.6f}" for val in features]

                writer.writerow(row)

                stats["processed_success"] += 1
                stats["per_label_count"][label] += 1

                if idx % 50 == 0 or idx == len(all_image_paths):
                    print(
                        f"Đã xử lý [{idx}/{len(all_image_paths)}] ảnh | "
                        f"Hợp lệ: {stats['processed_success']} | "
                        f"Bị loại (No person): {stats['skipped_no_person']} | "
                        f"Bị loại (Low conf): {stats['skipped_low_confidence']}"
                    )

        print("\n=== HOÀN TẤT XÂY DỰNG DATASET ===")
        print(f"File lưu tại: {self.output_csv_path}")
        print(f"Số mẫu hợp lệ: {stats['processed_success']}/{stats['total_images_found']}")
        print("Chi tiết số lượng theo từng nhãn:")
        for lbl, count in stats["per_label_count"].items():
            print(f"  - {lbl}: {count} mẫu")

        return stats


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
