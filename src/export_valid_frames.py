import csv
from pathlib import Path
from typing import Optional
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from src.pose_detector import PoseDetector


class ValidFrameExporter:
    """
    Đọc danh sách các mẫu hợp lệ từ data/processed/features.csv,
    chạy PoseDetector để lấy bounding box và 6 keypoints,
    vẽ trực quan (bbox màu xanh lá, skeleton, nhãn tư thế)
    và lưu vào thư mục data/anhdung/.
    """

    # Màu sắc đại diện cho 4 nhãn tư thế (BGR)
    LABEL_COLORS = {
        "correct": (0, 200, 0),        # Xanh lá đậm
        "forward_slouch": (0, 165, 255),# Màu cam
        "lean_left": (255, 0, 0),       # Xanh dương
        "lean_right": (255, 0, 255)     # Tím cánh sen
    }

    def __init__(
        self,
        features_csv_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.csv_path = (
            Path(features_csv_path)
            if features_csv_path
            else project_root / "data" / "processed" / "features.csv"
        )
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else project_root / "data" / "anhdung"
        )

        self.detector = PoseDetector()

    def _draw_valid_visualization(
        self,
        frame: np.ndarray,
        pose: dict,
        label: str,
        person_id: str,
        session_id: str
    ) -> np.ndarray:
        """
        Vẽ bounding box người, các keypoints và nhãn tư thế lên frame.
        """
        annotated = frame.copy()
        bbox = pose["bbox"].astype(int)
        keypoints = pose["keypoints"]
        box_color = self.LABEL_COLORS.get(label, (0, 255, 0))

        # 1. Vẽ bounding box
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)

        # 2. Vẽ nhãn thông tin trên đầu bounding box
        tag_text = f"[{label.upper()}] {person_id} ({session_id})"
        (text_w, text_h), baseline = cv2.getTextSize(
            tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        tag_y1 = max(0, bbox[1] - text_h - 10)
        tag_y2 = bbox[1]

        cv2.rectangle(
            annotated,
            (bbox[0], tag_y1),
            (bbox[0] + text_w + 12, tag_y2),
            box_color,
            -1
        )
        cv2.putText(
            annotated,
            tag_text,
            (bbox[0] + 6, tag_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 3. Vẽ đường nối vai (Shoulder line)
        ls = keypoints["left_shoulder"][:2].astype(int)
        rs = keypoints["right_shoulder"][:2].astype(int)
        cv2.line(annotated, tuple(ls), tuple(rs), (0, 255, 255), 2)

        # 4. Vẽ đường nối mắt (Eye line)
        le = keypoints["left_eye"][:2].astype(int)
        re = keypoints["right_eye"][:2].astype(int)
        cv2.line(annotated, tuple(le), tuple(re), (0, 255, 255), 2)

        # 5. Vẽ 6 keypoints
        for name, pt in keypoints.items():
            x, y = int(pt[0]), int(pt[1])
            # Chấm tròn bên trong xanh lá, viền trắng
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (x, y), 7, (255, 255, 255), 1)

        # 6. Tag trạng thái VALID ở góc trên bên trái màn hình
        cv2.rectangle(annotated, (10, 10), (280, 50), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"VALID SAMPLE: {label}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        return annotated

    def export(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file {self.csv_path}!\n"
                f"Vui lòng chạy 'python -m src.dataset_builder' trước."
            )

        # Đọc danh sách ảnh hợp lệ từ CSV
        valid_records = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_records.append({
                    "image_path": row["image_path"],
                    "session_id": row["session_id"],
                    "person_id": row["person_id"],
                    "label": row["label"]
                })

        total = len(valid_records)
        print(f"=== BẮT ĐẦU XUẤT ẢNH ĐÚNG KÈM BBOX ===")
        print(f"File nguồn: {self.csv_path}")
        print(f"Thư mục lưu: {self.output_dir}")
        print(f"Tổng số ảnh cần xuất: {total}\n")

        saved_count = 0

        for idx, item in enumerate(valid_records, start=1):
            img_path = Path(item["image_path"])
            if not img_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            pose = self.detector.detect(frame)
            if pose is None:
                continue

            # Vẽ bbox + skeleton + nhãn
            annotated = self._draw_valid_visualization(
                frame,
                pose,
                label=item["label"],
                person_id=item["person_id"],
                session_id=item["session_id"]
            )

            # Phân loại lưu vào thư mục con theo nhãn: data/anhdung/<label>/
            label_dir = self.output_dir / item["label"]
            label_dir.mkdir(parents=True, exist_ok=True)

            out_filename = f"{item['person_id']}_{item['session_id']}_{img_path.name}"
            out_path = label_dir / out_filename

            cv2.imwrite(str(out_path), annotated)
            saved_count += 1

            if idx % 100 == 0 or idx == total:
                print(f"Đã xử lý [{idx}/{total}] ảnh...")

        print(f"\n=== HOÀN TẤT ===")
        print(f"Đã lưu thành công {saved_count} ảnh đúng vào:")
        print(f"👉 {self.output_dir}")


if __name__ == "__main__":
    exporter = ValidFrameExporter()
    exporter.export()
