"""
Smart Posture Monitor - Shadcn UI Dashboard
Web Application built with FastAPI, YOLO Pose, SVM RBF, and Temporal Smoothing (V03).
"""

from __future__ import annotations

import base64
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ------------------------------------------------------------
# 1. Project paths setup
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import PostureInferenceEngine  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n-pose.pt"

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"

# ------------------------------------------------------------
# 2. Engine Singleton & Lifespan Handler
# ------------------------------------------------------------
engine: Optional[PostureInferenceEngine] = None


def get_engine() -> PostureInferenceEngine:
    global engine
    if engine is None:
        engine = PostureInferenceEngine(
            model_path=MODEL_PATH,
            metadata_path=METADATA_PATH,
            yolo_path=YOLO_MODEL_PATH,
            calibration_samples=30,
        )
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo mô hình AI 1 lần duy nhất lúc bật server
    get_engine()
    yield


# ------------------------------------------------------------
# 3. FastAPI App Initialization
# ------------------------------------------------------------
app = FastAPI(
    title="Smart Posture Monitor - Dashboard",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AlertThresholdPayload(BaseModel):
    seconds: int


# ------------------------------------------------------------
# 4. HTTP & WebSocket Routes
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.post("/api/reset_session")
async def reset_session_api():
    eng = get_engine()
    if hasattr(eng, "reset_session"):
        eng.reset_session()
    return JSONResponse({"status": "OK", "message": "Session reset successful"})


@app.post("/api/calibrate")
async def start_calibration_api():
    """Endpoint HTTP hỗ trợ kích hoạt hiệu chuẩn dáng ngồi chuẩn."""
    eng = get_engine()
    if hasattr(eng, "start_calibration"):
        eng.start_calibration()
    return JSONResponse({"status": "OK", "message": "Calibration started"})


@app.post("/api/set_alert_threshold")
async def set_alert_threshold(payload: AlertThresholdPayload):
    eng = get_engine()
    # 1. Giới hạn số giây hợp lệ (từ 1 đến 60 giây)
    seconds = max(1, min(payload.seconds, 60))
    # The setting is wall-clock time; inference throughput is intentionally
    # irrelevant here.
    if hasattr(eng, "stats"):
        eng.stats.alert_threshold_seconds = float(seconds)
    return {
        "status": "OK",
        "seconds": seconds,
        "message": f"Cảnh báo sau {seconds} giây",
    }


@app.get("/api/info")
async def get_system_info():
    eng = get_engine()
    metadata = getattr(eng.predictor, "metadata", {})
    return {
        "model_name": metadata.get("model_name", "SVM Classifier (RBF) - REP13"),
        "yolo_model": "YOLO Pose (yolo26n-pose.pt)",
        "features_count": len(getattr(eng.predictor, "feature_columns", [])),
        "classes": metadata.get("classes", list(getattr(eng.predictor, "id_to_label", {}).values())),
        "is_calibrated": getattr(eng, "is_calibrated", False),
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    eng = get_engine()
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            client_time = None
            try:
                msg = json.loads(data)

                # 1.Lấy mốc thời gian client gửi lên
                client_time = msg.get("client_time")

                # Bắt tín hiệu Calibration từ frontend (khi ấn nút hoặc ấn phím C)
                action = msg.get("action", "")
                if action:
                    if action == "calibrate":
                        eng.start_calibration()
                        await websocket.send_json({
                            "status": "CALIBRATING",
                            "message": "Đang bắt đầu thu thập mốc chuẩn cá nhân...",
                            "calibrated": False,
                            "calibration_progress": 0.0,
                        })
                    elif action == "reset_calibration":
                        eng.reset_calibration()
                        await websocket.send_json({
                            "status": "CALIBRATION_REQUIRED",
                            "message": "Đã xóa mốc chuẩn cá nhân",
                            "calibrated": False,
                            "calibration_progress": 0.0,
                        })
                    else:
                        await websocket.send_json({
                            "status": "ERROR",
                            "message": f"Unknown action: {action}",
                        })
                    # Action-only messages are not image frames.
                    continue

                img_data = msg.get("image")
                if not isinstance(img_data, str) or not img_data.strip():
                    await websocket.send_json({
                        "status": "ERROR",
                        "message": "Missing image data",
                        "client_time": client_time,
                    })
                    continue

                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    processing_started = time.perf_counter()
                    result = eng.process_frame(frame)
                    result["server_processing_ms"] = round(
                        (time.perf_counter() - processing_started) * 1000.0,
                        1,
                    )

                    # 2.Gắn ngược lại client_time vào kết quả trước khi gửi đi
                    if client_time is not None:
                        result["client_time"] = client_time

                    await websocket.send_json(result)
                else:
                    await websocket.send_json({
                        "status": "ERROR",
                        "message": "Frame decode failed",
                        "client_time": client_time,
                    })
            except Exception as inner_e:
                error_result = {"status": "ERROR", "message": str(inner_e)}
                if client_time is not None:
                    error_result["client_time"] = client_time
                await websocket.send_json(error_result)
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------
# 5. Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Khởi chạy ứng dụng tại http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
