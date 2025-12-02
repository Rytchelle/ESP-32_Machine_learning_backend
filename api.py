import math
import numpy as np
from scipy import stats as scipy_stats
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Union, Set
from datetime import datetime
from collections import deque
from pathlib import Path
import asyncio
import json
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sanitize_float(value: Union[float, int, None], default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return default
        return float(value)
    return default


class AnomalyDetector:
    def __init__(self, model_path: str):
        model = np.load(model_path, allow_pickle=True)
        self.mu = model["mu"]
        self.cov = model["cov"]
        
        threshold_val = model["threshold"]
        self.threshold = float(threshold_val.item()) if isinstance(threshold_val, np.ndarray) else float(threshold_val)
        
        # Cache da matriz inversa para performance
        epsilon = 1e-6
        cov_reg = self.cov + epsilon * np.eye(self.cov.shape[0])
        scale = np.median(np.diag(cov_reg))
        self.inv_covmat = np.linalg.inv(cov_reg / scale) / scale
        
        # Detecção mais rápida - só precisa de 1 confirmação
        self.last_anomaly = False

    def predict(self, data):
        # Preprocessamento inline
        data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)
        data = data - np.mean(data, axis=0)
        
        # Extração de features otimizada
        features = []
        for axis_idx in range(data.shape[1]):
            axis_data = data[:, axis_idx]
            features.extend([
                np.std(axis_data),
                scipy_stats.kurtosis(axis_data),
                np.max(np.abs(axis_data)),
                np.sqrt(np.mean(np.square(axis_data))),
                np.max(axis_data) - np.min(axis_data),
            ])
        features = np.array(features)
        
        # Distância de Mahalanobis com matriz pré-calculada
        x_mu = features - self.mu
        distance = float(np.sqrt(np.dot(np.dot(x_mu, self.inv_covmat), x_mu)))
        
        # Detecção imediata - sem delay
        is_anomaly = distance > self.threshold
        
        # Confiança simplificada
        ratio = distance / self.threshold
        if ratio < 0.7:
            confidence = 0.1
        elif ratio < 1.0:
            confidence = 0.3
        elif ratio < 1.5:
            confidence = 0.6
        else:
            confidence = min(0.95, 0.6 + (ratio - 1.5) * 0.1)
        
        # Cor do status
        if is_anomaly:
            status_color = "red"
        elif ratio > 0.7:
            status_color = "yellow"
        else:
            status_color = "green"
        
        self.last_anomaly = is_anomaly
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": sanitize_float(confidence),
            "distance": sanitize_float(distance),
            "threshold": sanitize_float(self.threshold),
            "status_color": status_color,
            "timestamp": datetime.now().isoformat(),
        }


class AccelerometerData(BaseModel):
    data: List[List[float]]
    sensor_id: str = "default"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AnomalyDetector("models/mahalanobis_model.npz")

# Buffer para o gráfico
MAX_SAMPLES = 200
recent_samples = deque(maxlen=MAX_SAMPLES)

latest_status: Dict[str, Any] = {
    "is_anomaly": False,
    "confidence": 0.0,
    "distance": 0.0,
    "threshold": detector.threshold,
    "status_color": "green",
    "timestamp": None,
}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: str):
        if not self.active_connections:
            return
        disconnected = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except:
                disconnected.add(conn)
        self.active_connections -= disconnected


ws_manager = ConnectionManager()


@app.post("/predict")
async def predict_anomaly(data: AccelerometerData):
    global latest_status
    
    array_data = np.array(data.data)
    
    # Guarda todos os pontos para o gráfico
    now_ms = int(datetime.now().timestamp() * 1000)
    if array_data.ndim == 2 and array_data.shape[0] > 0:
        for i in range(array_data.shape[0]):
            x, y, z = float(array_data[i, 0]), float(array_data[i, 1]), float(array_data[i, 2])
            recent_samples.append({"x": x, "y": y, "z": z, "t": now_ms + i})
    
    # Predição
    result = detector.predict(array_data)
    latest_status = result
    
    # Broadcast via WebSocket
    samples_list = list(recent_samples)[-200:]
    msg = json.dumps({
        "type": "update",
        "status_color": result["status_color"],
        "confidence": result["confidence"],
        "distance": result["distance"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "timestamp": result["timestamp"],
        "samples": samples_list
    })
    await ws_manager.broadcast(msg)
    
    return result


@app.get("/realtime/state")
async def get_state():
    return latest_status


@app.get("/realtime/samples")
async def get_samples(limit: int = 50):
    return {"samples": list(recent_samples)[-limit:]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Envia estado inicial
        await websocket.send_text(json.dumps({
            "type": "init",
            "s": latest_status.get("status_color", "green"),
            "c": latest_status.get("confidence", 0),
            "d": latest_status.get("distance", 0),
            "t": detector.threshold
        }))
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
            except:
                break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return Response(content="1", media_type="text/plain")


@app.get("/status")
async def get_status():
    return {
        "ok": True,
        "clients": len(ws_manager.active_connections),
        "threshold": detector.threshold
    }


web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
