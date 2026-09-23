import numpy as np


class FeatureExtractor:
    """
    Biến output của PoseDetector thành vector feature.

    Input:
        pose_result = {
            "person_confidence": float,
            "bbox": [...],
            "keypoints": {
                "nose": [x, y, confidence],
                "left_eye": [x, y, confidence],
                "right_eye": [x, y, confidence],
                "left_ear": [x, y, confidence],
                "left_shoulder": [x, y, confidence],
                "right_shoulder": [x, y, confidence]
            }
        }

    Output:
        numpy.ndarray shape (18,)

    Nếu keypoint không đủ chất lượng:
        return None
    """

    FEATURE_NAMES = [
        "shoulder_angle",
        "eye_shoulder_angle",
        "eye_vertical_difference",

        "nose_x_body",
        "nose_y_body",

        "eye_center_x_body",
        "eye_center_y_body",

        "left_ear_x_body",
        "left_ear_y_body",

        "nose_eye_dx",
        "nose_eye_dy",

        "eye_width_ratio",
        "ear_eye_ratio",

        "nose_shoulder_center_distance",
        "eye_shoulder_center_distance",

        "nose_shoulder_asymmetry",
        "eye_shoulder_asymmetry",

        "nose_ear_ratio"
    ]

    REQUIRED_KEYPOINTS = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "left_shoulder",
        "right_shoulder"
    ]

    def __init__(
        self,
        min_keypoint_confidence=0.35
    ):
        """
        min_keypoint_confidence:
            Confidence tối thiểu của từng keypoint.

        Nếu một trong 6 keypoint thấp hơn ngưỡng
        thì frame đó không được dùng tạo feature.
        """

        self.min_keypoint_confidence = (
            min_keypoint_confidence
        )

    # =========================================================
    # Utility
    # =========================================================

    @staticmethod
    def _distance(point_a, point_b):
        """
        Khoảng cách Euclidean giữa hai điểm.
        """

        return float(
            np.linalg.norm(
                point_a - point_b
            )
        )

    @staticmethod
    def _normalize_line_angle(angle):
        """
        Chuẩn hóa góc đường thẳng về khoảng:

            [-90, 90]

        Vì một đường thẳng góc 0° và 180°
        thực chất có cùng hướng.
        """

        while angle > 90:
            angle -= 180

        while angle <= -90:
            angle += 180

        return float(angle)

    @classmethod
    def _line_angle_deg(cls, vector):
        """
        Tính góc của một đường thẳng theo độ.
        """

        angle = np.degrees(
            np.arctan2(
                vector[1],
                vector[0]
            )
        )

        return cls._normalize_line_angle(
            angle
        )

    # =========================================================
    # MAIN
    # =========================================================

    def extract(self, pose_result):
        """
        Chuyển pose_result thành vector 18 features.
        """

        # =====================================================
        # 1. Kiểm tra input
        # =====================================================

        if pose_result is None:
            return None

        keypoints = pose_result.get(
            "keypoints"
        )

        if keypoints is None:
            return None

        # =====================================================
        # 2. Kiểm tra đủ 6 keypoint
        # =====================================================

        for name in self.REQUIRED_KEYPOINTS:

            if name not in keypoints:
                return None

        # =====================================================
        # 3. Kiểm tra confidence + format
        # =====================================================

        points = {}

        for name in self.REQUIRED_KEYPOINTS:

            point = np.asarray(
                keypoints[name],
                dtype=np.float64
            )

            if point.shape[0] < 3:
                return None

            x = point[0]
            y = point[1]
            confidence = point[2]

            if not np.all(
                np.isfinite(point)
            ):
                return None

            if (
                confidence
                <
                self.min_keypoint_confidence
            ):
                return None

            points[name] = np.array(
                [x, y],
                dtype=np.float64
            )

        # =====================================================
        # 4. Lấy từng điểm
        # =====================================================

        nose = points["nose"]

        left_eye = points["left_eye"]
        right_eye = points["right_eye"]

        left_ear = points["left_ear"]

        left_shoulder = points[
            "left_shoulder"
        ]

        right_shoulder = points[
            "right_shoulder"
        ]

        # =====================================================
        # 5. Shoulder center
        # =====================================================

        shoulder_center = (
            left_shoulder
            +
            right_shoulder
        ) / 2.0

        shoulder_vector = (
            right_shoulder
            -
            left_shoulder
        )

        shoulder_width = (
            np.linalg.norm(
                shoulder_vector
            )
        )

        # Hai vai gần như trùng nhau
        # => frame lỗi
        if shoulder_width < 1e-6:
            return None

        # =====================================================
        # 6. Tạo hệ tọa độ theo cơ thể
        # =====================================================

        # Trục ngang:
        #
        # Left Shoulder -> Right Shoulder

        shoulder_axis = (
            shoulder_vector
            /
            shoulder_width
        )

        # Trục vuông góc với vai

        up_axis = np.array(
            [
                shoulder_axis[1],
                -shoulder_axis[0]
            ],
            dtype=np.float64
        )

        # Trong OpenCV:
        #
        # y tăng xuống dưới.
        #
        # Ta muốn up_axis luôn hướng lên trên ảnh.

        if up_axis[1] > 0:
            up_axis = -up_axis

        # =====================================================
        # Hàm chuyển một điểm pixel sang
        # hệ tọa độ cơ thể
        # =====================================================

        def body_coordinate(point):

            relative = (
                point
                -
                shoulder_center
            )

            # Chuẩn hóa theo shoulder width
            relative = (
                relative
                /
                shoulder_width
            )

            body_x = np.dot(
                relative,
                shoulder_axis
            )

            body_y = np.dot(
                relative,
                up_axis
            )

            return (
                float(body_x),
                float(body_y)
            )

        # =====================================================
        # 7. Eye center
        # =====================================================

        eye_center = (
            left_eye
            +
            right_eye
        ) / 2.0

        eye_vector = (
            right_eye
            -
            left_eye
        )

        eye_width = np.linalg.norm(
            eye_vector
        )

        if eye_width < 1e-6:
            return None

        # =====================================================
        # FEATURE 1
        # Shoulder angle
        # =====================================================

        shoulder_angle = (
            self._line_angle_deg(
                shoulder_vector
            )
        )

        # =====================================================
        # FEATURE 2
        # Eye line relative to shoulder line
        # =====================================================

        eye_angle = (
            self._line_angle_deg(
                eye_vector
            )
        )

        eye_shoulder_angle = (
            self._normalize_line_angle(
                eye_angle
                -
                shoulder_angle
            )
        )

        # =====================================================
        # FEATURE 3
        # Độ chênh chiều cao hai mắt
        # =====================================================

        eye_vertical_difference = float(
            np.dot(
                eye_vector,
                up_axis
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 4 - 5
        # Nose trong hệ tọa độ cơ thể
        # =====================================================

        (
            nose_x_body,
            nose_y_body
        ) = body_coordinate(
            nose
        )

        # =====================================================
        # FEATURE 6 - 7
        # Eye center
        # =====================================================

        (
            eye_center_x_body,
            eye_center_y_body
        ) = body_coordinate(
            eye_center
        )

        # =====================================================
        # FEATURE 8 - 9
        # Left ear
        # =====================================================

        (
            left_ear_x_body,
            left_ear_y_body
        ) = body_coordinate(
            left_ear
        )

        # =====================================================
        # FEATURE 10 - 11
        # Nose relative to eye center
        # =====================================================

        nose_eye_vector = (
            nose
            -
            eye_center
        )

        nose_eye_dx = float(
            np.dot(
                nose_eye_vector,
                shoulder_axis
            )
            /
            shoulder_width
        )

        nose_eye_dy = float(
            np.dot(
                nose_eye_vector,
                up_axis
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 12
        # Eye width / shoulder width
        # =====================================================

        eye_width_ratio = float(
            eye_width
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 13
        # Left ear -> Left eye
        # =====================================================

        ear_eye_ratio = (
            self._distance(
                left_ear,
                left_eye
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 14
        # Nose -> shoulder center
        # =====================================================

        nose_shoulder_center_distance = (
            self._distance(
                nose,
                shoulder_center
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 15
        # Eye center -> shoulder center
        # =====================================================

        eye_shoulder_center_distance = (
            self._distance(
                eye_center,
                shoulder_center
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 16
        # Nose shoulder asymmetry
        # =====================================================

        nose_left_shoulder_distance = (
            self._distance(
                nose,
                left_shoulder
            )
        )

        nose_right_shoulder_distance = (
            self._distance(
                nose,
                right_shoulder
            )
        )

        nose_shoulder_asymmetry = float(
            (
                nose_left_shoulder_distance
                -
                nose_right_shoulder_distance
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 17
        # Eye center shoulder asymmetry
        # =====================================================

        eye_left_shoulder_distance = (
            self._distance(
                eye_center,
                left_shoulder
            )
        )

        eye_right_shoulder_distance = (
            self._distance(
                eye_center,
                right_shoulder
            )
        )

        eye_shoulder_asymmetry = float(
            (
                eye_left_shoulder_distance
                -
                eye_right_shoulder_distance
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 18
        # Nose -> Left Ear
        # =====================================================

        nose_ear_ratio = (
            self._distance(
                nose,
                left_ear
            )
            /
            shoulder_width
        )

        # =====================================================
        # Final feature vector
        # =====================================================

        features = np.array(
            [
                shoulder_angle,
                eye_shoulder_angle,
                eye_vertical_difference,

                nose_x_body,
                nose_y_body,

                eye_center_x_body,
                eye_center_y_body,

                left_ear_x_body,
                left_ear_y_body,

                nose_eye_dx,
                nose_eye_dy,

                eye_width_ratio,
                ear_eye_ratio,

                nose_shoulder_center_distance,
                eye_shoulder_center_distance,

                nose_shoulder_asymmetry,
                eye_shoulder_asymmetry,

                nose_ear_ratio
            ],
            dtype=np.float32
        )

        # =====================================================
        # Final validation
        # =====================================================

        if not np.all(
            np.isfinite(features)
        ):
            return None

        return features

    # =========================================================
    # Debug helper
    # =========================================================

    def to_dict(self, features):
        """
        Chuyển vector thành dictionary để dễ debug.
        """

        if features is None:
            return None

        if (
            len(features)
            !=
            len(self.FEATURE_NAMES)
        ):

            raise ValueError(
                "Số feature không hợp lệ"
            )

        return dict(
            zip(
                self.FEATURE_NAMES,
                features.tolist()
            )
        )