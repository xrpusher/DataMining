# backend_server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

# URL inference server
INFERENCE_SERVER_URL = "http://127.0.0.1:8001/predict"  # Предполагаем, что inference server работает на порту 8001

# Модель данных для точки
class Point(BaseModel):
    x: float
    y: float

# Эндпоинт для клиента
@app.post("/predict_cluster")
def predict_cluster(point: Point):
    # Перенаправляем запрос на inference server
    response = requests.post(INFERENCE_SERVER_URL, json=point.dict())
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Inference server error"}
