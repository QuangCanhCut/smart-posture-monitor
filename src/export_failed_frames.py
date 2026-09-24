from pathlib import Path
from typing import Optional
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.pose_detector import PoseDetector


class FailedFrameExporter:
    """
    Quét lại data/raw, phát hiện những frame bị loại (không có người hoặc low confidence),
    vẽ khung xương + bounding box + lý do lỗi trực tiếp lên ảnh
    và xuất ra thư mục data/anhloi/ để nhóm quan sát, debug.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_keypoint_confidence: float = 0.35
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.raw_data_dir = (
            Path(raw_data_dir) if raw_data_dir else project_root / "data" / "raw"
        )
        self.output_dir = (
            Path(output_dir) if output_dir else project_root / "data" / "anhloi"
        )

        self.detector = PoseDetector()
        self.extractor = FeatureExtractor(min_keypoint_confidence=min_keypoint_confidence)
        self.min_conf = min_keypoint_confidence

    def _draw_failure_visualization(self, frame: np.ndarray, pose: Optional[dict]) -> np.ndarray:
        """
        Vẽ bounding box, 6 keypoints và lý do lỗi lên frame.
        """
        annotated = frame.copy()

        # Trường hợp 1: Không phát hiện thấy người
        if pose is None:
            cv2.rectangle(annotated, (10, 10), (450, 60), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                "FAIL: NO PERSON DETECTED",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return annotated

        # Trường hợp 2: Có người nhưng keypoint bị lỗi
        bbox = pose["bbox"].astype(int)
        keypoints = pose["keypoints"]

        # Vẽ bounding box người (màu vàng)
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)

        # Kiểm tra xem những keypoint nào bị lỗi (< min_conf)
        failed_kps = []
        for name, pt in keypoints.items():
            conf = float(pt[2])
            if conf < self.min_conf:
                failed_kps.append(f"{name}:{conf:.2f}")

        # Vẽ đường nối vai (Left Shoulder -> Right Shoulder)
        ls = keypoints["left_shoulder"][:2].astype(int)
        rs = keypoints["right_shoulder"][:2].astype(int)
        cv2.line(annotated, tuple(ls), tuple(rs), (255, 255, 0), 2)

        # Vẽ đường nối 2 mắt
        le = keypoints["left_eye"][:2].astype(int)
        re = keypoints["right_eye"][:2].astype(int)
        cv2.line(annotated, tuple(le), tuple(re), (255, 255, 0), 2)

        # Vẽ 6 keypoints
        for name, pt in keypoints.items():
            x, y, conf = int(pt[0]), int(pt[1]), float(pt[2])
            is_valid = conf >= self.min_conf

            # Đỏ nếu lỗi, Xanh lá nếu đạt chuẩn
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.circle(annotated, (x, y), 6, color, -1)
            cv2.circle(annotated, (x, y), 8, (255, 255, 255), 1)

            # Ghi text tên điểm và conf
            label_text = f"{name} ({conf:.2f})"
            cv2.putText(
                annotated,
                label_text,
                (x + 8, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        # Vẽ thanh thông báo lỗi ở góc trên
        cv2.rectangle(annotated, (10, 10), (600, 65), (0, 0, 0), -1)
        err_msg = f"FAIL: Low conf (<{self.min_conf})"
        if failed_kps:
            err_msg += f" [{', '.join(failed_kps[:2])}]"

        cv2.putText(
            annotated,
            err_msg,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

        return annotated

    def export(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_images = [
            p for p in self.raw_data_dir.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        print(f"=== BẮT ĐẦU TRÍCH XUẤT ẢNH LỖI ===")
        print(f"Thư mục nguồn: {self.raw_data_dir}")
        print(f"Thư mục lưu ảnh lỗi: {self.output_dir}\n")

        saved_count = 0
        failed_per_label = {}

        for img_path in all_images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            pose = self.detector.detect(frame)
            is_failed = False

            if pose is None:
                is_failed = True
            else:
                features = self.extractor.extract(pose)
                if features is None:
                    is_failed = True

            if is_failed:
                annotated_frame = self._draw_failure_visualization(frame, pose)

                # Xác định nhãn từ thư mục cha
                rel_parts = img_path.relative_to(self.raw_data_dir).parts
                label = rel_parts[0] if len(rel_parts) > 0 else "unknown"

                # Tạo thư mục con theo từng class: data/anhloi/<label>/
                label_dir = self.output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                # Tên file bên trong: session_filename.jpg
                out_filename = "_".join(rel_parts[1:]) if len(rel_parts) > 1 else img_path.name
                out_path = label_dir / out_filename

                cv2.imwrite(str(out_path), annotated_frame)
                saved_count += 1
                failed_per_label[label] = failed_per_label.get(label, 0) + 1

                if saved_count % 50 == 0:
                    print(f"Đã lưu {saved_count} ảnh lỗi...")

        print(f"\n=== HOÀN TẤT ===")
        print(f"Đã trích xuất và phân loại thành công: {saved_count} ảnh lỗi vào:")
        print(f"👉 {self.output_dir}")
        print("Chi tiết số lượng ảnh lỗi theo từng class:")
        for lbl, count in failed_per_label.items():
            print(f"  - {lbl}: {count} ảnh")


if __name__ == "__main__":
    exporter = FailedFrameExporter()
    exporter.export()
