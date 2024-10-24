from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Модель данных для точки
class Point(BaseModel):
    x: float
    y: float

# Параметры обученной GMM модели
means = np.array([[-1.0, -1.0], [1.0, 1.0]])
covariances = [np.array([[0.5, 0.2], [0.2, 0.5]]), np.array([[0.5, -0.2], [-0.2, 0.5]])]
pi_k = [0.5, 0.5]

# Функция для вычисления плотности многомерного нормального распределения
def multivariate_gaussian(x, mean, cov):
    d = len(x)
    cov_inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    return np.exp(exponent) / denominator

# POST-запрос для предсказания кластера
@app.post("/predict_cluster")
def predict_cluster(point: Point):
    x_point = np.array([point.x, point.y])
    K = len(means)
    probs = np.zeros(K)
    
    for k in range(K):
        probs[k] = pi_k[k] * multivariate_gaussian(x_point, means[k], covariances[k])
    
    probs /= np.sum(probs)
    
    return {"probabilities": probs.tolist()}
