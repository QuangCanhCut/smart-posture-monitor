import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

from camera_service import process_frame
from state import state
from src.pomodoro_timer import Pomodoro_timer

from streamlit_autorefresh import st_autorefresh

# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Sitting Posture Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 95%;
}


/* Main title */

.title-container {
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 22px 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}


/* Posture status */

.status-box {
    padding: 10px 0;
}

.status-label {
    font-size: 16px;
    margin-bottom: 10px;
}

.status-badge {
    color: white;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}


/* Pomodoro timer */

.timer-container {
    text-align: center;
    padding: 18px;
    border-radius: 12px;
    background-color: #f5f5f5;
    margin-bottom: 15px;
}

.timer-label {
    font-size: 14px;
    color: #666;
}

.timer-value {
    font-size: 42px;
    font-weight: bold;
    margin-top: 5px;
}


/* Timer button colors */

div.stButton > button {
    height: 45px;
    border-radius: 8px;
    font-weight: 600;
}


/* Pause button - yellow */

.pause-button button {
    background-color: #f59e0b !important;
    color: white !important;
    border: none !important;
}

.pause-button button:hover {
    background-color: #d97706 !important;
    color: white !important;
}


/* Resume button - green */

.resume-button button {
    background-color: #10b981 !important;
    color: white !important;
    border: none !important;
}

.resume-button button:hover {
    background-color: #059669 !important;
    color: white !important;
}


/* Stop button - red */

.stop-button button {
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
}

.stop-button button:hover {
    background-color: #dc2626 !important;
    color: white !important;
}


/* Disabled button */

div.stButton > button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("""
<div class="title-container">
    <h1>🖥️ AI Sitting Posture Monitoring System</h1>
    <p>Real-time Pose Estimation & Intelligent Ergonomic Alerting</p>
</div>
""", unsafe_allow_html=True)

# SESSION TIMER

if "pomodoro" not in st.session_state:
    st.session_state.pomodoro = Pomodoro_timer()
pomodoro = st.session_state.pomodoro

# Tự refresh mỗi 1 giây khi timer đang chạy
if pomodoro.running and not pomodoro.paused:
    st_autorefresh(
        interval=1000,
        key="pomodoro_refresh"
    )

# SIDEBAR
with st.sidebar:
    st.header("⚙️ System Control")
    st.write("Cấu hình và tùy chỉnh hệ thống")
    st.divider()
    st.subheader("Threshold Settings")
    alert_time = st.slider(
        "Cảnh báo khi ngồi sai (giây)",
        3,
        30,
        5
    )
    confidence_threshold = st.slider(
        "Độ tin cậy AI (Confidence)",
        0.5,
        1.0,
        0.75,
        0.05
    )
    st.divider()
    st.subheader("Information")
    st.info("""
**AI Sitting Posture Monitor**

Webcam → YOLO-Pose → ML Model → Temporal Monitoring → Alert
""")

# MAIN LAYOUT
col_video, col_stats = st.columns(
    [1.8, 1.2],
    gap="medium"
)

# CAMERA PROCESSOR

class CameraProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(
            format="bgr24"
        )
        img = process_frame(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# LIVE STREAM

with col_video:
    st.subheader("📷 Live Stream")
    with st.container(border=True):
        webrtc_streamer(
            key="posture-camera",
            video_processor_factory=CameraProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720}
                },
                "audio": False
            }
        )

# REAL-TIME MONITORING

with col_stats:
    st.subheader("📊 Real-time Monitoring")
    current_posture = state["posture"]
    current_warning = state["warning"]
    current_confidence = state["confidence"]
    warnings_count = state["warnings_count"]

    # POSTURE STATUS

    if current_warning:
        status_text = "⚠️ BAD POSTURE"
        status_color = "#ef4444"

    else:
        status_text = "✓ GOOD POSTURE"
        status_color = "#10b981"

    st.markdown(
        f"""
<div class="status-box">
<div class="status-label">Trạng thái hiện tại</div>

<div class="status-badge"
     style="background-color:{status_color};">
{status_text}
</div>

</div>
""",
        unsafe_allow_html=True
    )

    # METRICS

    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            "Posture Score",
            f"{current_confidence * 100:.0f}%"
        )
    with m2:
        st.metric(
            "Warnings Count",
            warnings_count
        )

    # CONFIDENCE
    st.progress(
        min(current_confidence, 1.0),
        text=f"AI Confidence: {current_confidence * 100:.0f}%"
    )

    # WARNING MESSAGE
    if current_warning:
        st.error(
            "⚠️ **Cảnh báo tư thế**\n\n"
            "Bạn đang ngồi sai tư thế."
        )

    else:
        st.success(
            "✓ **Tư thế tốt**\n\n"
            "Hãy tiếp tục duy trì tư thế hiện tại."
        )

# WORK SESSION / POMODORO

st.divider()
st.subheader("⏱️ Work Session")

# Lấy thời gian hiện tại
elapsed_time = pomodoro.get_elapsed_time()

# TIMER DISPLAY

st.markdown(
    f"""
<div class="timer-container">

<div class="timer-label">
SESSION TIME
</div>

<div class="timer-value">
{Pomodoro_timer.format_time(elapsed_time)}
</div>

</div>
""",
    unsafe_allow_html=True
)


# TIMER BUTTONS
b1, b2, b3 = st.columns(3)

# START

with b1:
    if not pomodoro.running:
        if st.button(
            "▶ Start",
            use_container_width=True,
            type="primary",
            key="start_button"
        ):
            pomodoro.start()
            st.rerun()

    else:
        st.button(
            "▶ Start",
            use_container_width=True,
            disabled=True,
            key="start_disabled"
        )

# PAUSE / RESUME

with b2:
    # Đang chạy → Pause
    if pomodoro.running and not pomodoro.paused:
        st.markdown(
            '<div class="pause-button">',
            unsafe_allow_html=True
        )
        if st.button(
            "⏸ Pause",
            use_container_width=True,
            key="pause_button"
        ):
            pomodoro.pause()
            st.rerun()
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # Đang pause → Resume
    elif pomodoro.running and pomodoro.paused:

        st.markdown(
            '<div class="resume-button">',
            unsafe_allow_html=True
        )

        if st.button(
            "▶ Resume",
            use_container_width=True,
            key="resume_button"
        ):
            pomodoro.resume()
            st.rerun()

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
    # Chưa Start
    else:
        st.button(
            "⏸ Pause",
            use_container_width=True,
            disabled=True,
            key="pause_disabled"
        )

# STOP

with b3:
    if pomodoro.running:
        st.markdown(
            '<div class="stop-button">',
            unsafe_allow_html=True
        )
        if st.button(
            "⏹ Stop",
            use_container_width=True,
            key="stop_button"
        ):
            pomodoro.stop()
            st.rerun()
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=True,
            key="stop_disabled"
        )

# FOOTER

st.divider()
st.caption(
    "AI Posture Monitor v1.0 • "
    "Built with Streamlit, WebRTC & YOLO-Pose"
)