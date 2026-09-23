from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """
    Phát hiện pose của người có detection confidence cao nhất
    và chỉ trả về 6 keypoint cần dùng cho bài toán posture.

    6 keypoints:
        0 - Nose
        1 - Left Eye
        2 - Right Eye
        3 - Left Ear
        5 - Left Shoulder
        6 - Right Shoulder

    Right Ear (index 4) bị loại bỏ hoàn toàn.
    """

    # COCO keypoint indices
    KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,

        # 4 = right_ear -> KHÔNG DÙNG

        "left_shoulder": 5,
        "right_shoulder": 6
    }

    def __init__(
        self,
        model_path=None,
        person_conf_threshold=0.5
    ):
        """
        model_path:
            Đường dẫn tới model YOLO Pose.

            Nếu không truyền vào thì mặc định dùng:
            models/yolo26n-pose.pt

        person_conf_threshold:
            Ngưỡng confidence tối thiểu của person.
        """

        # =====================================================
        # 1. Xác định đường dẫn model
        # =====================================================

        if model_path is None:

            # pose_detector.py nằm trong:
            # smart_posture_monitor/src/
            #
            # parents[1] -> smart_posture_monitor/

            project_root = Path(__file__).resolve().parents[1]

            model_path = (
                project_root
                / "models"
                / "yolo26n-pose.pt"
            )

        self.model_path = Path(model_path)

        # =====================================================
        # 2. Kiểm tra model có tồn tại không
        # =====================================================

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Không tìm thấy YOLO model tại:\n"
                f"{self.model_path}\n\n"
                f"Hãy đảm bảo file yolo26n-pose.pt "
                f"nằm trong thư mục models/"
            )

        # =====================================================
        # 3. Load YOLO model
        # =====================================================

        self.model = YOLO(
            str(self.model_path)
        )

        self.person_conf_threshold = (
            person_conf_threshold
        )

    # =========================================================
    # Main function
    # =========================================================

    def detect(self, frame):
        """
        Input:
            frame:
                Ảnh OpenCV dạng numpy.ndarray.

        Output:
            {
                "person_confidence": float,

                "bbox": np.array(
                    [x1, y1, x2, y2]
                ),

                "keypoints": {

                    "nose":
                        np.array([x, y, confidence]),

                    "left_eye":
                        np.array([x, y, confidence]),

                    "right_eye":
                        np.array([x, y, confidence]),

                    "left_ear":
                        np.array([x, y, confidence]),

                    "left_shoulder":
                        np.array([x, y, confidence]),

                    "right_shoulder":
                        np.array([x, y, confidence])
                }
            }

        Nếu không phát hiện được person:
            return None
        """

        # =====================================================
        # 1. Kiểm tra input
        # =====================================================

        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "frame phải là numpy.ndarray"
            )

        # =====================================================
        # 2. Chạy YOLO Pose
        # =====================================================

        results = self.model.predict(
            source=frame,
            conf=self.person_conf_threshold,
            verbose=False
        )

        if len(results) == 0:
            return None

        result = results[0]

        # =====================================================
        # 3. Kiểm tra YOLO có phát hiện person không
        # =====================================================

        if result.boxes is None:
            return None

        if len(result.boxes) == 0:
            return None

        if result.keypoints is None:
            return None

        # =====================================================
        # 4. Lấy confidence của tất cả person
        # =====================================================

        person_confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        if len(person_confidences) == 0:
            return None

        # =====================================================
        # 5. Chọn person có confidence CAO NHẤT
        # =====================================================

        best_person_index = int(
            np.argmax(person_confidences)
        )

        best_person_confidence = float(
            person_confidences[
                best_person_index
            ]
        )

        # =====================================================
        # 6. Bounding box của person được chọn
        # =====================================================

        bbox = (
            result.boxes.xyxy[
                best_person_index
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # =====================================================
        # 7. Lấy 17 keypoint YOLO của person đó
        # =====================================================

        all_xy = (
            result.keypoints.xy[
                best_person_index
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # =====================================================
        # 8. Confidence của từng keypoint
        # =====================================================

        if result.keypoints.conf is not None:

            all_conf = (
                result.keypoints.conf[
                    best_person_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

        else:

            # Trường hợp hiếm model không trả confidence
            all_conf = np.ones(
                len(all_xy),
                dtype=np.float32
            )

        # =====================================================
        # 9. Chỉ lấy đúng 6 keypoint cần thiết
        # =====================================================

        selected_keypoints = {}

        for name, index in self.KEYPOINT_INDICES.items():

            x = float(
                all_xy[index][0]
            )

            y = float(
                all_xy[index][1]
            )

            confidence = float(
                all_conf[index]
            )

            selected_keypoints[name] = np.array(
                [
                    x,
                    y,
                    confidence
                ],
                dtype=np.float32
            )

        # =====================================================
        # 10. Output cuối cùng
        # =====================================================

        return {

            "person_confidence":
                best_person_confidence,

            "bbox":
                bbox,

            "keypoints":
                selected_keypoints
        }