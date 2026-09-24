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
        numpy.ndarray shape (29,)

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

        "nose_ear_ratio",

        "head_body_angle",
        "head_gravity_angle",
        "nose_gravity_angle",
        "face_pitch_angle",

        "eye_vertical_axis_offset",
        "nose_vertical_axis_offset",

        "head_mean_height",
        "head_height_spread",

        "nose_body_angle",
        "ear_body_angle",
        "head_axis_angle_spread"
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
        """Khoảng cách Euclidean giữa hai điểm."""
        return float(np.linalg.norm(point_a - point_b))

    @staticmethod
    def _normalize_line_angle(angle):
        """Chuẩn hóa góc đường thẳng về [-90, 90]."""
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        return float(angle)

    @classmethod
    def _line_angle_deg(cls, vector):
        """Góc của một đường thẳng theo độ."""
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        return cls._normalize_line_angle(angle)

    @staticmethod
    def _signed_angle_from_image_up(vector):
        """
        Góc có dấu giữa vector và phương thẳng đứng đi lên của ảnh.

        OpenCV: +x sang phải, +y xuống dưới.
        image-up = (0, -1).

        0 độ  : vector hướng thẳng lên.
        > 0   : vector lệch về bên phải ảnh.
        < 0   : vector lệch về bên trái ảnh.
        """
        dx = float(vector[0])
        up_component = float(-vector[1])
        return float(np.degrees(np.arctan2(dx, up_component)))

    # =========================================================
    # MAIN
    # =========================================================

    def extract(self, pose_result):
        """
        Chuyển pose_result thành vector 29 features.

        Returns:
            np.ndarray shape (29,) hoặc None nếu frame không hợp lệ.
        """

        # -----------------------------------------------------
        # 1. Kiểm tra input
        # -----------------------------------------------------
        if pose_result is None:
            return None

        keypoints = pose_result.get("keypoints")
        if keypoints is None:
            return None

        # -----------------------------------------------------
        # 2. Kiểm tra đủ 6 keypoint + confidence + format
        # -----------------------------------------------------
        points = {}

        for name in self.REQUIRED_KEYPOINTS:
            if name not in keypoints:
                return None

            point = np.asarray(keypoints[name], dtype=np.float64)

            if point.ndim != 1 or point.shape[0] < 3:
                return None

            if not np.all(np.isfinite(point)):
                return None

            x, y, confidence = point[:3]

            if confidence < self.min_keypoint_confidence:
                return None

            points[name] = np.array([x, y], dtype=np.float64)

        # -----------------------------------------------------
        # 3. Lấy từng điểm
        # -----------------------------------------------------
        nose = points["nose"]
        left_eye = points["left_eye"]
        right_eye = points["right_eye"]
        left_ear = points["left_ear"]
        left_shoulder = points["left_shoulder"]
        right_shoulder = points["right_shoulder"]

        # -----------------------------------------------------
        # 4. Shoulder center + width
        # -----------------------------------------------------
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_width = float(np.linalg.norm(shoulder_vector))

        if shoulder_width < 1e-6:
            return None

        # -----------------------------------------------------
        # 5. Hệ tọa độ BODY gắn với hai vai
        # -----------------------------------------------------
        shoulder_axis = shoulder_vector / shoulder_width

        # Vector vuông góc với vai
        up_axis = np.array(
            [shoulder_axis[1], -shoulder_axis[0]],
            dtype=np.float64,
        )

        # OpenCV y tăng xuống dưới -> ép up_axis hướng lên ảnh
        if up_axis[1] > 0:
            up_axis = -up_axis

        def body_coordinate(point):
            relative = (point - shoulder_center) / shoulder_width
            body_x = np.dot(relative, shoulder_axis)
            body_y = np.dot(relative, up_axis)
            return float(body_x), float(body_y)

        # -----------------------------------------------------
        # 6. Eye center
        # -----------------------------------------------------
        eye_center = (left_eye + right_eye) / 2.0
        eye_vector = right_eye - left_eye
        eye_width = float(np.linalg.norm(eye_vector))

        if eye_width < 1e-6:
            return None

        # =====================================================
        # FEATURE 1 - 18: giữ nguyên V1
        # =====================================================

        # 1. Shoulder angle
        shoulder_angle = self._line_angle_deg(shoulder_vector)

        # 2. Eye line relative to shoulder line
        eye_angle = self._line_angle_deg(eye_vector)
        eye_shoulder_angle = self._normalize_line_angle(
            eye_angle - shoulder_angle
        )

        # 3. Eye vertical difference in BODY frame
        eye_vertical_difference = float(
            np.dot(eye_vector, up_axis) / shoulder_width
        )

        # 4 - 5. Nose BODY coordinate
        nose_x_body, nose_y_body = body_coordinate(nose)

        # 6 - 7. Eye center BODY coordinate
        eye_center_x_body, eye_center_y_body = body_coordinate(eye_center)

        # 8 - 9. Left ear BODY coordinate
        left_ear_x_body, left_ear_y_body = body_coordinate(left_ear)

        # 10 - 11. Nose relative to eye center in BODY frame
        nose_eye_vector = nose - eye_center

        nose_eye_dx = float(
            np.dot(nose_eye_vector, shoulder_axis) / shoulder_width
        )
        nose_eye_dy = float(
            np.dot(nose_eye_vector, up_axis) / shoulder_width
        )

        # 12. Eye width / shoulder width
        eye_width_ratio = float(eye_width / shoulder_width)

        # 13. Left ear -> left eye / shoulder width
        ear_eye_ratio = (
            self._distance(left_ear, left_eye) / shoulder_width
        )

        # 14. Nose -> shoulder center / shoulder width
        nose_shoulder_center_distance = (
            self._distance(nose, shoulder_center) / shoulder_width
        )

        # 15. Eye center -> shoulder center / shoulder width
        eye_shoulder_center_distance = (
            self._distance(eye_center, shoulder_center) / shoulder_width
        )

        # 16. Nose-shoulder asymmetry
        nose_left_shoulder_distance = self._distance(nose, left_shoulder)
        nose_right_shoulder_distance = self._distance(nose, right_shoulder)

        nose_shoulder_asymmetry = float(
            (
                nose_left_shoulder_distance
                - nose_right_shoulder_distance
            )
            / shoulder_width
        )

        # 17. Eye-center-shoulder asymmetry
        eye_left_shoulder_distance = self._distance(
            eye_center, left_shoulder
        )
        eye_right_shoulder_distance = self._distance(
            eye_center, right_shoulder
        )

        eye_shoulder_asymmetry = float(
            (
                eye_left_shoulder_distance
                - eye_right_shoulder_distance
            )
            / shoulder_width
        )

        # 18. Nose -> left ear / shoulder width
        nose_ear_ratio = self._distance(nose, left_ear) / shoulder_width

        # =====================================================
        # FEATURE 19
        # Head axis relative to body up-axis
        # =====================================================

        head_body_angle = float(
            np.degrees(
                np.arctan2(
                    eye_center_x_body,
                    eye_center_y_body
                )
            )
        )

        # =====================================================
        # FEATURE 20
        # Head axis relative to image vertical / gravity proxy
        # =====================================================

        head_vector = (
            eye_center
            -
            shoulder_center
        )

        if np.linalg.norm(head_vector) < 1e-6:
            return None

        head_gravity_angle = (
            self._signed_angle_from_image_up(
                head_vector
            )
        )

        # =====================================================
        # FEATURE 21
        # Nose axis relative to image vertical / gravity proxy
        # =====================================================

        nose_vector = (
            nose
            -
            shoulder_center
        )

        if np.linalg.norm(nose_vector) < 1e-6:
            return None

        nose_gravity_angle = (
            self._signed_angle_from_image_up(
                nose_vector
            )
        )

        # =====================================================
        # FEATURE 22
        # Face projected pitch angle
        # =====================================================

        if np.linalg.norm(nose_eye_vector) < 1e-6:
            return None

        face_pitch_angle = float(
            np.degrees(
                np.arctan2(
                    nose_eye_dx,
                    -nose_eye_dy
                )
            )
        )

        # =====================================================
        # FEATURE 23
        # Eye center offset from camera vertical axis
        # through shoulder center
        # =====================================================

        eye_vertical_axis_offset = float(
            (
                eye_center[0]
                -
                shoulder_center[0]
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 24
        # Nose offset from camera vertical axis
        # through shoulder center
        # =====================================================

        nose_vertical_axis_offset = float(
            (
                nose[0]
                -
                shoulder_center[0]
            )
            /
            shoulder_width
        )

        # =====================================================
        # FEATURE 25
        # Mean normalized height of head landmarks
        # =====================================================

        head_heights = np.array(
            [
                nose_y_body,
                eye_center_y_body,
                left_ear_y_body
            ],
            dtype=np.float64
        )

        head_mean_height = float(
            np.mean(
                head_heights
            )
        )

        # =====================================================
        # FEATURE 26
        # Spread of normalized head landmark heights
        # =====================================================

        head_height_spread = float(
            np.std(
                head_heights
            )
        )

        # =====================================================
        # FEATURE 27
        # Nose axis relative to body up-axis
        # =====================================================

        nose_body_angle = float(
            np.degrees(
                np.arctan2(
                    nose_x_body,
                    nose_y_body
                )
            )
        )

        # =====================================================
        # FEATURE 28
        # Left ear axis relative to body up-axis
        # =====================================================

        ear_body_angle = float(
            np.degrees(
                np.arctan2(
                    left_ear_x_body,
                    left_ear_y_body
                )
            )
        )

        # =====================================================
        # FEATURE 29
        # Angular spread of eye / nose / ear head axes
        # =====================================================

        def angle_difference(angle_a, angle_b):
            return (
                (angle_a - angle_b + 180.0)
                % 360.0
                - 180.0
            )

        nose_angle_delta = angle_difference(
            nose_body_angle,
            head_body_angle
        )

        ear_angle_delta = angle_difference(
            ear_body_angle,
            head_body_angle
        )

        head_axis_angle_spread = float(
            np.std(
                np.array(
                    [
                        0.0,
                        nose_angle_delta,
                        ear_angle_delta
                    ],
                    dtype=np.float64
                )
            )
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

                nose_ear_ratio,

                head_body_angle,
                head_gravity_angle,
                nose_gravity_angle,
                face_pitch_angle,

                eye_vertical_axis_offset,
                nose_vertical_axis_offset,

                head_mean_height,
                head_height_spread,

                nose_body_angle,
                ear_body_angle,
                head_axis_angle_spread
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
