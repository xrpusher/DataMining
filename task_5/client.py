import requests
import numpy as np
import time

# URL вашего FastAPI сервера
url = "http://127.0.0.1:8000/predict_cluster"

# Генерация множества случайных точек
num_points = 100
test_points = np.random.uniform(low=-3.0, high=3.0, size=(num_points, 2))

# Функция для отправки запросов с задержкой
def test_cluster_prediction_with_delay(points, delay=0.01):
    for i, point in enumerate(points):
        data = {"x": point[0], "y": point[1]}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Point {i+1} ({point[0]:.2f}, {point[1]:.2f}): {response.json()}")
        else:
            print(f"Error for point {i+1} ({point[0]:.2f}, {point[1]:.2f})")
        time.sleep(delay)  # Задержка между запросами

# Запуск тестов с задержкой 10 миллисекунд между запросами
test_cluster_prediction_with_delay(test_points, delay=0.01)
