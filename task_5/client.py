# client.py

import requests
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

# URL вашего бэкенд-сервера
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
if __name__ == "__main__":
    test_cluster_prediction_with_delay(test_points, delay=0.01)

    # Загрузка обученной модели
    with open('gmm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    means = model['means']
    covariances = model['covariances']

    # Генерация данных для визуализации (если необходимо)
    # Или используйте реальные данные

    # Пример: загрузка обучающих данных
    # Здесь создадим случайные данные для примера
    mean1 = [-1.0, -1.0]
    mean2 = [1.0, 1.0]
    cov1 = [[0.5, 0.1], [0.1, 0.5]]
    cov2 = [[0.3, -0.2], [-0.2, 0.3]]

    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, cov1, 100)
    X2 = np.random.multivariate_normal(mean2, cov2, 100)
    X = np.vstack([X1, X2])

    # Визуализация данных
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=30, alpha=0.6, label='Data Points')

    # Визуализация кластеров
    colors = ['red', 'blue']
    for k in range(len(means)):
        plt.scatter(means[k][0], means[k][1], c=colors[k], s=200, marker='X', label=f'Cluster {k+1} Mean')
        
        # Визуализация эллипсов ковариации
        eigenvalues, eigenvectors = np.linalg.eigh(covariances[k])
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        
        # Передаём параметры как именованные аргументы
        ellipse = Ellipse(xy=means[k], width=width, height=height, angle=angle, edgecolor=colors[k], facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.legend()
    plt.title('GMM Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
