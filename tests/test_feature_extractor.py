import cv2

from src.pose_detector import PoseDetector
from src.feature_extractor import FeatureExtractor


# ============================================================
# 1. Khởi tạo modules
# ============================================================

detector = PoseDetector(
    person_conf_threshold=0.5
)

extractor = FeatureExtractor(
    min_keypoint_confidence=0.35
)


# ============================================================
# 2. Webcam
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Không thể mở webcam"
    )


frame_count = 0


# ============================================================
# 3. Main loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1


    # ========================================================
    # IMAGE -> POSE
    # ========================================================

    pose = detector.detect(
        frame
    )


    if pose is not None:

        # ====================================================
        # POSE -> FEATURES
        # ====================================================

        features = extractor.extract(
            pose
        )


        if features is not None:

            # ================================================
            # Hiện trạng thái
            # ================================================

            cv2.putText(
                frame,
                "Features OK: 18",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


            # ================================================
            # Chuyển sang dictionary
            # ================================================

            feature_dict = (
                extractor.to_dict(
                    features
                )
            )


            # ================================================
            # Hiện một vài feature quan trọng
            # ================================================

            display_features = [

                "shoulder_angle",

                "eye_shoulder_angle",

                "nose_y_body",

                "eye_center_y_body",

                "nose_eye_dy",

                "nose_shoulder_asymmetry"
            ]


            y_position = 70


            for name in display_features:

                value = feature_dict[
                    name
                ]

                cv2.putText(
                    frame,

                    f"{name}: {value:.3f}",

                    (
                        20,
                        y_position
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (0, 255, 255),

                    1
                )

                y_position += 25


            # ================================================
            # In toàn bộ features mỗi 30 frame
            # ================================================

            if frame_count % 30 == 0:

                print(
                    "\n"
                    +
                    "=" * 60
                )

                print(
                    "FEATURE VECTOR"
                )

                print(
                    "=" * 60
                )


                for name, value in (
                    feature_dict.items()
                ):

                    print(
                        f"{name:35s}: "
                        f"{value:.4f}"
                    )


        else:

            cv2.putText(
                frame,

                "Keypoints quality too low",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )


    else:

        cv2.putText(
            frame,

            "No person detected",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2
        )


    # ========================================================
    # 4. Show webcam
    # ========================================================

    cv2.imshow(
        "Feature Extractor Test",
        frame
    )


    if (
        cv2.waitKey(1)
        &
        0xFF
        ==
        ord("q")
    ):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()