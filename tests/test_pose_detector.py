import cv2

from src.pose_detector import PoseDetector


# ============================================================
# 1. Khởi tạo PoseDetector
# ============================================================

# Không cần truyền model_path nữa.
#
# PoseDetector sẽ tự dùng:
#
# models/yolo26n-pose.pt

detector = PoseDetector(
    person_conf_threshold=0.5
)


# ============================================================
# 2. Mở webcam
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Không thể mở webcam."
    )


# ============================================================
# Một số cấu hình chỉ dùng để hiển thị
# ============================================================

KEYPOINT_DISPLAY_THRESHOLD = 0.20


# ============================================================
# 3. Webcam loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Không đọc được frame từ webcam"
        )

        break


    # ========================================================
    # IMAGE -> POSE DETECTOR
    # ========================================================

    pose = detector.detect(frame)


    if pose is not None:

        # ====================================================
        # 4. Bounding box
        # ====================================================

        bbox = pose["bbox"]

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # 5. Person confidence
        # ====================================================

        person_confidence = (
            pose["person_confidence"]
        )


        cv2.putText(
            frame,

            f"Person: {person_confidence:.2f}",

            (
                x1,
                max(y1 - 10, 25)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        # ====================================================
        # 6. Lấy 6 keypoints
        # ====================================================

        keypoints = pose["keypoints"]


        # ====================================================
        # 7. Vẽ 6 keypoints
        # ====================================================

        for name, point in keypoints.items():

            x = int(point[0])
            y = int(point[1])

            confidence = float(
                point[2]
            )


            # Nếu confidence quá thấp
            # thì không vẽ điểm đó.

            if (
                confidence
                <
                KEYPOINT_DISPLAY_THRESHOLD
            ):
                continue


            # Vẽ điểm

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 0, 255),
                -1
            )


            # Tên điểm + confidence

            cv2.putText(
                frame,

                f"{name}: {confidence:.2f}",

                (
                    x + 7,
                    y - 7
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.4,

                (255, 255, 0),

                1
            )


        # ====================================================
        # 8. Vẽ một số đường nối để dễ quan sát
        # ====================================================

        def get_xy(name):

            point = keypoints[name]

            confidence = float(
                point[2]
            )

            if (
                confidence
                <
                KEYPOINT_DISPLAY_THRESHOLD
            ):
                return None

            return (
                int(point[0]),
                int(point[1])
            )


        nose = get_xy(
            "nose"
        )

        left_eye = get_xy(
            "left_eye"
        )

        right_eye = get_xy(
            "right_eye"
        )

        left_ear = get_xy(
            "left_ear"
        )

        left_shoulder = get_xy(
            "left_shoulder"
        )

        right_shoulder = get_xy(
            "right_shoulder"
        )


        # ---------------------------------------------
        # Eye line
        # ---------------------------------------------

        if (
            left_eye is not None
            and right_eye is not None
        ):

            cv2.line(
                frame,
                left_eye,
                right_eye,
                (255, 0, 255),
                2
            )


        # ---------------------------------------------
        # Left eye -> Left ear
        # ---------------------------------------------

        if (
            left_eye is not None
            and left_ear is not None
        ):

            cv2.line(
                frame,
                left_eye,
                left_ear,
                (255, 0, 255),
                2
            )


        # ---------------------------------------------
        # Shoulder line
        # ---------------------------------------------

        if (
            left_shoulder is not None
            and right_shoulder is not None
        ):

            cv2.line(
                frame,
                left_shoulder,
                right_shoulder,
                (255, 0, 255),
                2
            )


        # ---------------------------------------------
        # Nose -> eyes
        # ---------------------------------------------

        if (
            nose is not None
            and left_eye is not None
        ):

            cv2.line(
                frame,
                nose,
                left_eye,
                (255, 0, 255),
                1
            )


        if (
            nose is not None
            and right_eye is not None
        ):

            cv2.line(
                frame,
                nose,
                right_eye,
                (255, 0, 255),
                1
            )


    else:

        # ====================================================
        # Không tìm thấy person
        # ====================================================

        cv2.putText(
            frame,

            "No person detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 0, 255),

            2
        )


    # ========================================================
    # 9. Hiển thị webcam
    # ========================================================

    cv2.imshow(
        "Pose Detector Test - YOLO26n Pose",
        frame
    )


    # ========================================================
    # 10. Bấm q để thoát
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# 11. Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()
