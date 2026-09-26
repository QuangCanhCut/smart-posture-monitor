import cv2
from src.temporal_monitor import TemporalMonitor
from state import state


# Tạo một TemporalMonitor.
# Hiện tại cảnh báo sau 5 giây BAD liên tục.
monitor = TemporalMonitor(alert_time=5)


def process_frame(frame):

    # Fake prediction.
    # Hiện tại ta vẫn chưa có model thật của B.
    posture = "BAD POSTURE"
    confidence = 0.91
    # Gửi kết quả tư thế vào Temporal Monitor.
    warning = monitor.update(posture)
    state["posture"] = posture
    state["warning"] = warning
    state["confidence"] = 0.91
    # Hiển thị tư thế lên webcam.
    cv2.putText(
        frame,
        posture,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Nếu Temporal Monitor yêu cầu cảnh báo
    if warning:

        cv2.putText(
            frame,
            "WARNING: BAD POSTURE!",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    return frame