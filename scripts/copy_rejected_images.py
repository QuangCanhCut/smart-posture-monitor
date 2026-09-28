import csv
import shutil
import sys
from pathlib import Path

import cv2


# =========================================================
# 1. XÁC ĐỊNH PROJECT ROOT
# =========================================================
#
# File hiện tại nằm tại:
#
# smart_posture_monitor/
# └── scripts/
#     └── copy_rejected_images.py
#
# parents[1] -> smart_posture_monitor/
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# 2. THÊM PROJECT ROOT VÀO PYTHON PATH
# =========================================================
# Giúp import được src.pose_detector khi chạy trực tiếp:
# python scripts/copy_rejected_images.py
# =========================================================
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_detector import PoseDetector


"""
Trực quan hóa các ảnh bị loại trong quá trình build dataset.

Script sẽ:
1. Đọc rejected_images.csv.
2. Đọc lại ảnh raw.
3. Chạy lại PoseDetector.
4. Vẽ bounding box, 6 keypoints, confidence, reason và detail.
5. Lưu ảnh trực quan hóa vào rejected_img/.
"""


# =========================================================
# 3. ĐƯỜNG DẪN FILE
# =========================================================
REJECTED_CSV = (
    PROJECT_ROOT
    / "data"
    / "rejected"
    / "rejected_images.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "rejected"
    / "rejected_img"
)


# =========================================================
# 4. MÀU HIỂN THỊ KEYPOINT
# =========================================================
# LƯU Ý:
# Đây chỉ là màu trực quan hóa, KHÔNG phải threshold chính thức
# của FeatureExtractor.
#
# confidence >= 0.75  -> xanh
# confidence >= 0.50  -> vàng
# confidence <  0.50  -> đỏ
# =========================================================
def get_confidence_color(confidence):
    if confidence >= 0.75:
        return (0, 255, 0)       # Xanh

    if confidence >= 0.50:
        return (0, 255, 255)     # Vàng

    return (0, 0, 255)           # Đỏ


# =========================================================
# 5. VẼ POSE LÊN ẢNH
# =========================================================
def draw_pose(frame, pose, reason, detail=""):
    """
    Vẽ thông tin debug lên ảnh rejected.

    Hiển thị:
    - bounding box của person được chọn
    - person confidence
    - 6 keypoints và confidence từng keypoint
    - một số đường nối để dễ quan sát pose
    - reason ảnh bị reject
    - detail lỗi nếu có

    Parameters
    ----------
    frame:
        Ảnh OpenCV.
    pose:
        Output từ PoseDetector.detect().
    reason:
        Reason lấy từ rejected_images.csv.
    detail:
        Detail lấy từ rejected_images.csv.

    Returns
    -------
    annotated:
        Ảnh đã được trực quan hóa.
    """
    annotated = frame.copy()

    detail_text = str(detail).strip()
    if len(detail_text) > 110:
        detail_text = detail_text[:107] + "..."

    # -----------------------------------------------------
    # 5.1 Trường hợp không detect được person
    # -----------------------------------------------------
    if pose is None:
        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 120),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            f"REJECTED: {reason}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        if detail_text:
            cv2.putText(
                annotated,
                f"DETAIL: {detail_text}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            annotated,
            "NO PERSON DETECTED",
            (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return annotated

    # -----------------------------------------------------
    # 5.2 Bounding box
    # -----------------------------------------------------
    x1, y1, x2, y2 = map(int, pose["bbox"])

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
    )

    # -----------------------------------------------------
    # 5.3 Person confidence
    # -----------------------------------------------------
    person_confidence = float(pose["person_confidence"])

    cv2.putText(
        annotated,
        f"Person conf: {person_confidence:.3f}",
        (x1, max(60, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    # -----------------------------------------------------
    # 5.4 Lấy 6 keypoints
    # -----------------------------------------------------
    keypoints = pose["keypoints"]

    # -----------------------------------------------------
    # 5.5 Vẽ các đường nối
    # -----------------------------------------------------
    # Chỉ dùng để dễ nhìn hình dạng pose.
    # Không ảnh hưởng tới FeatureExtractor.
    connections = [
        ("left_eye", "nose"),
        ("right_eye", "nose"),
        ("left_eye", "left_ear"),
        ("left_shoulder", "right_shoulder"),
    ]

    for point_a, point_b in connections:
        if point_a not in keypoints or point_b not in keypoints:
            continue

        xa, ya, _ = keypoints[point_a]
        xb, yb, _ = keypoints[point_b]

        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(
                annotated,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                (255, 255, 255),
                1,
            )

    # -----------------------------------------------------
    # 5.6 Vẽ từng keypoint
    # -----------------------------------------------------
    for name, point in keypoints.items():
        x = int(point[0])
        y = int(point[1])
        confidence = float(point[2])
        color = get_confidence_color(confidence)

        if x <= 0 or y <= 0:
            continue

        cv2.circle(
            annotated,
            (x, y),
            6,
            color,
            -1,
        )

        cv2.putText(
            annotated,
            f"{name}: {confidence:.2f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    # -----------------------------------------------------
    # 5.7 Header hiển thị reason + detail
    # -----------------------------------------------------
    cv2.rectangle(
        annotated,
        (0, 0),
        (annotated.shape[1], 78),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        f"REJECTED: {reason}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    if detail_text:
        cv2.putText(
            annotated,
            f"DETAIL: {detail_text}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    return annotated


# =========================================================
# 6. MAIN
# =========================================================
def main():
    # -----------------------------------------------------
    # 6.1 Kiểm tra rejected_images.csv
    # -----------------------------------------------------
    if not REJECTED_CSV.exists():
        raise FileNotFoundError(
            f"Khong tim thay rejected CSV:\n{REJECTED_CSV}"
        )

    # -----------------------------------------------------
    # 6.2 Xóa rejected_img cũ
    # -----------------------------------------------------
    # Tránh trường hợp lần chạy trước có nhiều ảnh hơn lần mới,
    # khiến ảnh cũ còn sót lại trong folder kết quả.
    if OUTPUT_DIR.exists():
        print("Dang xoa rejected_img cu...")
        shutil.rmtree(OUTPUT_DIR)

    # -----------------------------------------------------
    # 6.3 Tạo rejected_img mới
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 6.4 Load PoseDetector
    # -----------------------------------------------------
    print("Dang load PoseDetector...")
    detector = PoseDetector()
    print("PoseDetector da san sang.\n")

    # -----------------------------------------------------
    # 6.5 Đọc rejected CSV
    # -----------------------------------------------------
    with open(REJECTED_CSV, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    total = len(rows)
    print(f"Tong so anh rejected: {total}\n")

    # -----------------------------------------------------
    # 6.6 Counter
    # -----------------------------------------------------
    processed = 0
    missing = 0
    read_failed = 0
    pose_failed = 0
    write_failed = 0

    # =====================================================
    # 7. XỬ LÝ TỪNG ẢNH
    # =====================================================
    for index, row in enumerate(rows, start=1):
        # Metadata từ rejected_images.csv
        image_path = Path(row["image_path"])
        label = row["label"]
        person_id = row["person_id"]
        session_id = row["session_id"]
        reason = row["reason"]
        detail = row.get("detail", "")

        # -------------------------------------------------
        # 7.1 Folder output
        # -------------------------------------------------
        # rejected_img/
        # └── correct/
        #     └── person01_session02/
        #         └── frame_xxxx_rejected.jpg
        destination_dir = (
            OUTPUT_DIR
            / label
            / f"{person_id}_{session_id}"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 7.2 Kiểm tra ảnh có tồn tại không
        # -------------------------------------------------
        if not image_path.exists():
            print(f"[MISSING] {image_path}")
            missing += 1
            continue

        # -------------------------------------------------
        # 7.3 Đọc ảnh
        # -------------------------------------------------
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[READ ERROR] {image_path}")
            read_failed += 1
            continue

        # -------------------------------------------------
        # 7.4 Chạy lại PoseDetector
        # -------------------------------------------------
        try:
            pose = detector.detect(frame)
        except Exception as error:
            print(f"[POSE ERROR] {image_path}: {error}")
            pose_failed += 1
            continue

        # -------------------------------------------------
        # 7.5 Vẽ bbox + keypoints + confidence
        # -------------------------------------------------
        annotated = draw_pose(
            frame,
            pose,
            reason,
            detail,
        )

        # -------------------------------------------------
        # 7.6 Tạo tên file output
        # -------------------------------------------------
        # frame_0018.jpg -> frame_0018_rejected.jpg
        output_name = f"{image_path.stem}_rejected.jpg"
        destination_path = destination_dir / output_name

        # -------------------------------------------------
        # 7.7 Lưu ảnh
        # -------------------------------------------------
        success = cv2.imwrite(str(destination_path), annotated)

        if not success:
            print(f"[WRITE ERROR] {destination_path}")
            write_failed += 1
            continue

        processed += 1

        # -------------------------------------------------
        # 7.8 Log tiến độ
        # -------------------------------------------------
        if index % 25 == 0 or index == total:
            print(f"[{index}/{total}] Da truc quan hoa: {processed}")

    # =====================================================
    # 8. BÁO CÁO CUỐI
    # =====================================================
    print("\n================================")
    print("HOAN TAT TRUC QUAN HOA REJECTED")
    print("================================")
    print(f"Tong rejected: {total}")
    print(f"Da tao anh: {processed}")
    print(f"Khong tim thay anh: {missing}")
    print(f"Loi doc anh: {read_failed}")
    print(f"Loi PoseDetector: {pose_failed}")
    print(f"Loi ghi anh: {write_failed}")
    print(f"\nFolder ket qua:\n{OUTPUT_DIR}")


# =========================================================
# 9. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
