from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """
    Phát hiện pose của người phù hợp nhất với chủ thể chính,
    ưu tiên người gần tâm ảnh và có detection confidence cao.
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
        # 5. Chọn person phù hợp nhất:
        #    ưu tiên người gần tâm ảnh + confidence cao
        # =====================================================

        all_bboxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        frame_height, frame_width = frame.shape[:2]

        # Tâm của ảnh
        image_center_x = frame_width / 2.0
        image_center_y = frame_height / 2.0

        # Khoảng cách lớn nhất có thể từ tâm ảnh tới góc ảnh
        max_center_distance = np.sqrt(
            image_center_x ** 2
            + image_center_y ** 2
        )

        person_scores = []

        for i, bbox_candidate in enumerate(all_bboxes):

            x1, y1, x2, y2 = bbox_candidate

            # Tâm bounding box của person
            person_center_x = (x1 + x2) / 2.0
            person_center_y = (y1 + y2) / 2.0

            # Khoảng cách từ tâm person tới tâm ảnh
            center_distance = np.sqrt(
                (person_center_x - image_center_x) ** 2
                + (person_center_y - image_center_y) ** 2
            )

            # Chuẩn hóa về khoảng 0 -> 1
            normalized_distance = (
                center_distance / max_center_distance
                if max_center_distance > 0
                else 0.0
            )

            # Centrality:
            # gần tâm ảnh -> gần 1
            # xa tâm ảnh  -> gần 0
            centrality_score = max(
                0.0,
                1.0 - normalized_distance
            )

            confidence_score = float(
                person_confidences[i]
            )

            # Ưu tiên vị trí trung tâm hơn confidence
            final_score = (
                0.7 * centrality_score
                + 0.3 * confidence_score
            )

            person_scores.append(final_score)


        best_person_index = int(
            np.argmax(person_scores)
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