"""
Smart Posture Monitor - Shadcn UI Dashboard
Web Application built with FastAPI, YOLO Pose, SVM RBF, and Temporal Smoothing.
"""

from __future__ import annotations

import base64
import json
import sys
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

# FPS xử lý frame của hệ thống
FPS = 10

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
        )
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo mô hình lúc bật server
    get_engine()
    yield


# ------------------------------------------------------------
# 3. FastAPI App Initialization
# ------------------------------------------------------------
app = FastAPI(
    title="Smart Posture Monitor - Dashboard",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FramePayload(BaseModel):
    image: str
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
    eng.reset_session()
    return JSONResponse({"status": "OK", "message": "Session reset successful"})

@app.post("/api/set_alert_threshold")
async def set_alert_threshold(payload: AlertThresholdPayload):
    eng = get_engine()
    # Giới hạn giá trị hợp lệ từ 1 đến 60 giây
    seconds = max(1, min(payload.seconds, 60))
    # Đổi số giây thành số frame
    threshold_frames = seconds * FPS
    # Cập nhật ngưỡng cảnh báo cho session hiện tại
    eng.stats.alert_threshold_frames = threshold_frames
    return JSONResponse(
        {
            "status": "OK",
            "seconds": seconds,
            "threshold_frames": threshold_frames,
            "message": f"Cảnh báo sau {seconds} giây ({threshold_frames} frame)",
        }
    )

@app.get("/api/info")
async def get_system_info():
    eng = get_engine()
    metadata = eng.predictor.metadata
    return {
        "model_name": metadata.get("model_name", "SVM Classifier (RBF)"),
        "yolo_model": "YOLO Pose (yolo26n-pose.pt)",
        "features_count": len(eng.predictor.feature_columns),
        "classes": metadata.get("classes", list(eng.predictor.id_to_label.values())),
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    eng = get_engine()
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                img_data = msg.get("image", "")
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    result = eng.process_frame(frame)
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"status": "ERROR", "message": "Frame decode failed"})
            except Exception as inner_e:
                await websocket.send_json({"status": "ERROR", "message": str(inner_e)})
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------
# 5. Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Khởi chạy ứng dụng tại http://localhost:8000 ...")
    # Truyền trực tiếp app instance để tránh lỗi resolve path của uvicorn reloader
    uvicorn.run(app, host="0.0.0.0", port=8000)