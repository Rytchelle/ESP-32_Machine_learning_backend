import math
import numpy as np
from scipy import stats as scipy_stats
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Deque, Dict, Any, Union, Set
from datetime import datetime
from collections import deque
from pathlib import Path
import asyncio
import json
import logging


# ============================================================
# SANITIZAÇÃO DE VALORES PARA JSON
# ============================================================
def sanitize_float(value: Union[float, int, None], default: float = 0.0, max_value: float = 1e10) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return default
        if math.isinf(value):
            return max_value if value > 0 else -max_value
        return float(value)
    return default


def sanitize_dict(data: Dict[str, Any], default: float = 0.0, max_value: float = 1e10) -> Dict[str, Any]:
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = sanitize_dict(value, default, max_value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                sanitize_float(v, default, max_value) if isinstance(v, (int, float)) else v
                for v in value
            ]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = sanitize_float(value, default, max_value)
        else:
            result[key] = value
    return result


def sanitize_sample(x: float, y: float, z: float, timestamp: int) -> Dict[str, Any]:
    return {
        "timestamp": timestamp,
        "x": sanitize_float(x),
        "y": sanitize_float(y),
        "z": sanitize_float(z),
    }


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, model_path: str):
        model = np.load(model_path, allow_pickle=True)
        self.mu = model["mu"]
        self.cov = model["cov"]
        
        threshold_val = model["threshold"]
        if isinstance(threshold_val, np.ndarray):
            self.threshold = float(threshold_val.item())
        else:
            self.threshold = float(threshold_val)
        
        self.has_scaler = False
        self.last_predictions = [False, False, False]
        self.recent_distances = []
        
        model_type = str(model.get("model_type", "standard"))
        logger.info("Model loaded - Type: %s, Threshold: %.3f", model_type, self.threshold)

    def preprocess(self, data, remove_dc=True):
        data = data - np.mean(data, axis=0)
        return data

    def extract_features(self, sample):
        features = []
        for axis_idx in range(sample.shape[1]):
            axis_data = sample[:, axis_idx]
            features.extend([
                np.std(axis_data),
                scipy_stats.kurtosis(axis_data),
                np.max(np.abs(axis_data)),
                np.sqrt(np.mean(np.square(axis_data))),
                np.max(axis_data) - np.min(axis_data),
            ])
        return np.array(features)

    def mahalanobis_distance(self, x):
        x_mu = x - self.mu
        epsilon = 1e-6
        cov_reg = self.cov + epsilon * np.eye(self.cov.shape[0])
        try:
            scale = np.median(np.diag(cov_reg))
            cov_scaled = cov_reg / scale
            inv_covmat = np.linalg.inv(cov_scaled) / scale
            if x_mu.ndim == 1:
                mahal = np.sqrt(np.dot(np.dot(x_mu, inv_covmat), x_mu))
            else:
                mahal = np.sqrt(np.sum(np.dot(x_mu, inv_covmat) * x_mu, axis=1))
            return mahal
        except np.linalg.LinAlgError:
            return np.inf

    def calculate_confidence(self, distance):
        self.recent_distances.append(distance)
        self.recent_distances = self.recent_distances[-10:]
        
        if distance < self.threshold * 0.5:
            confidence = 0.05
        elif distance < self.threshold * 0.8:
            confidence = 0.15
        elif distance < self.threshold:
            confidence = 0.25
        elif distance < self.threshold * 1.2:
            confidence = 0.45
        elif distance < self.threshold * 1.5:
            confidence = 0.65
        elif distance < self.threshold * 2.0:
            confidence = 0.80
        else:
            confidence = 0.90
        
        if len(self.recent_distances) >= 3:
            recent_mean = np.mean(self.recent_distances[-3:])
            recent_std = np.std(self.recent_distances[-3:])
            if recent_std > recent_mean * 0.3:
                confidence *= 0.7
            if abs(distance - recent_mean) > recent_mean * 0.5:
                confidence *= 0.8
        
        return float(np.clip(confidence, 0.0, 1.0))

    def predict(self, data):
        processed_data = self.preprocess(data)
        features = self.extract_features(processed_data)
        distance = float(self.mahalanobis_distance(features))
        is_anomaly_candidate = distance > self.threshold

        self.last_predictions.pop(0)
        self.last_predictions.append(is_anomaly_candidate)
        stable_anomaly = sum(self.last_predictions) >= 2
        confidence = self.calculate_confidence(distance)

        feature_names = ["std", "kurtosis", "peak_amplitude", "rms", "peak_to_peak"]
        feature_stats = {}
        n_features_per_axis = len(feature_names)
        n_axes = len(features) // n_features_per_axis

        for axis_idx in range(n_axes):
            start_idx = axis_idx * n_features_per_axis
            axis_features = features[start_idx : start_idx + n_features_per_axis]
            feature_stats[f"axis_{axis_idx}"] = {
                name: float(value) for name, value in zip(feature_names, axis_features)
            }

        safe_feature_stats = {}
        for axis_name, stats in feature_stats.items():
            safe_feature_stats[axis_name] = {name: sanitize_float(val) for name, val in stats.items()}

        return {
            "is_anomaly": bool(stable_anomaly),
            "confidence": sanitize_float(confidence),
            "distance": sanitize_float(distance),
            "threshold": sanitize_float(self.threshold),
            "feature_values": safe_feature_stats,
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

MAX_SAMPLES: int = 1000
recent_samples: Deque[Dict[str, Any]] = deque(maxlen=MAX_SAMPLES)
latest_status: Dict[str, Any] = {
    "is_anomaly": False,
    "confidence": 0.0,
    "distance": 0.0,
    "threshold": 0.0,
    "timestamp": None,
}

sensor_connection_status = {
    "connected": False,
    "last_data_time": None,
    "disconnect_time": None,
    "total_disconnections": 0,
    "connection_start_time": None,
}

SENSOR_TIMEOUT_SECONDS = 10
subscribers: List[asyncio.Queue] = []
websocket_clients: Set[WebSocket] = set()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket desconectado. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
        safe_message = sanitize_dict(message)
        json_message = json.dumps(safe_message)
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception:
                disconnected.add(connection)
        for conn in disconnected:
            self.active_connections.discard(conn)


ws_manager = ConnectionManager()


def update_sensor_connection_status():
    global sensor_connection_status
    now = datetime.now()
    
    if sensor_connection_status["last_data_time"] is None:
        sensor_connection_status["connected"] = False
        return
    
    time_since_last_data = (now - sensor_connection_status["last_data_time"]).total_seconds()
    
    if time_since_last_data > SENSOR_TIMEOUT_SECONDS:
        if sensor_connection_status["connected"]:
            sensor_connection_status["connected"] = False
            sensor_connection_status["disconnect_time"] = now
            sensor_connection_status["total_disconnections"] += 1
            logger.warning(f"SENSOR DESCONECTADO! Última mensagem há {time_since_last_data:.1f}s")
    else:
        if not sensor_connection_status["connected"]:
            sensor_connection_status["connected"] = True
            sensor_connection_status["connection_start_time"] = now
            logger.info("SENSOR RECONECTADO!")


def make_status_payload(pred: Dict[str, Any]) -> Dict[str, Any]:
    confidence = pred.get("confidence", 0.0)
    distance = pred.get("distance", 0.0)
    threshold = pred.get("threshold", 1.0)
    is_anomaly = pred.get("is_anomaly", False)
    
    if is_anomaly:
        status_color = "red"
    elif distance > threshold * 0.7:
        status_color = "yellow"
    else:
        status_color = "green"
    
    return {
        "is_anomaly": bool(is_anomaly),
        "confidence": sanitize_float(confidence),
        "distance": sanitize_float(distance),
        "threshold": sanitize_float(threshold),
        "timestamp": pred.get("timestamp"),
        "status_color": status_color,
    }


@app.post("/predict")
async def predict_anomaly(data: AccelerometerData):
    try:
        global sensor_connection_status
        now = datetime.now()
        
        if sensor_connection_status["last_data_time"] is None:
            sensor_connection_status["connection_start_time"] = now
            logger.info("SENSOR CONECTADO pela primeira vez!")
        
        sensor_connection_status["last_data_time"] = now
        update_sensor_connection_status()
        
        array_data = np.array(data.data)
        array_data = np.nan_to_num(array_data, nan=0.0, posinf=1e10, neginf=-1e10)
        
        logger.info("Received data shape: %s from sensor %s", array_data.shape, data.sensor_id)

        now_ms = int(datetime.now().timestamp() * 1000)
        if array_data.ndim == 2 and array_data.shape[0] > 0:
            n = array_data.shape[0]
            for i in range(n):
                x, y, z = map(float, array_data[i, :3])
                ts = now_ms - (n - 1 - i)
                recent_samples.append(sanitize_sample(x, y, z, ts))

        result = detector.predict(array_data)
        result = sanitize_dict(result)

        global latest_status
        latest_status = make_status_payload(result)
        
        for q in list(subscribers):
            try:
                q.put_nowait(latest_status)
            except Exception:
                pass
        
        # Envia samples junto com prediction para evitar request extra
        recent_list = list(recent_samples)[-100:]
        await ws_manager.broadcast({
            "type": "prediction",
            "status": latest_status,
            "samples_count": len(recent_samples),
            "samples": recent_list,
            "result": result
        })

        return result
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


@app.get("/realtime/state")
async def get_state():
    return sanitize_dict(latest_status)


@app.get("/realtime/samples")
async def get_samples(limit: int = 300):
    data = list(recent_samples)[-limit:]
    sanitized_data = [sanitize_dict(sample) for sample in data]
    return {"samples": sanitized_data}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        initial_state = sanitize_dict({
            "type": "connected",
            "status": latest_status,
            "samples_count": len(recent_samples),
            "message": "Conectado ao servidor de anomalias"
        })
        await websocket.send_text(json.dumps(initial_state))
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif message.get("type") == "get_state":
                        await websocket.send_text(json.dumps(sanitize_dict({
                            "type": "state",
                            "status": latest_status,
                            "samples_count": len(recent_samples)
                        })))
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return Response(content="1", media_type="text/plain")


@app.get("/status")
async def get_status():
    return sanitize_dict({
        "api_running": True,
        "sensor_connected": sensor_connection_status["connected"],
        "samples_count": len(recent_samples),
        "websocket_clients": len(ws_manager.active_connections),
        "latest_status": latest_status,
        "threshold": float(detector.threshold),
        "timestamp": datetime.now().isoformat()
    })


web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
