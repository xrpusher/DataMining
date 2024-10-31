# inference_server.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Модель данных для точки
class Point(BaseModel):
    x: float
    y: float

# Загрузка модели при старте сервера
MODEL_PATH = 'gmm_model.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}. Пожалуйста, обучите и сохраните модель перед запуском сервера.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

means = model['means']
covariances = model['covariances']
pi_k = model['pi_k']

# Функция для вычисления плотности многомерного нормального распределения
def multivariate_gaussian(x, mean, cov):
    d = len(x)
    cov_inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    return np.exp(exponent) / denominator

# Эндпоинт для предсказания
@app.post("/predict")
def predict(point: Point):
    x_point = np.array([point.x, point.y])
    K = len(means)
    probs = np.zeros(K)

    for k in range(K):
        probs[k] = pi_k[k] * multivariate_gaussian(x_point, means[k], covariances[k])

    probs /= np.sum(probs)

    return {"probabilities": probs.tolist()}
