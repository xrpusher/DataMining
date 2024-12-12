import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider

# Function to compute multivariate normal distribution probability density function
# Функция для вычисления функции плотности вероятности многомерного нормального распределения
def multivariate_gaussian(x, mean, cov):
    d = len(x)
    cov_inv = np.linalg.inv(cov)  # Inverse of covariance matrix
    # Обратная матрица ковариации
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)  # Exponent term
    # Экспоненциальный член
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))  # Normalization denominator
    # Нормализующий знаменатель
    return np.exp(exponent) / denominator

def em_gmm(X, K, max_iter=100, tol=1e-6):
    N, d = X.shape  # Number of data points and dimensions
    # Количество точек данных и размерностей

    # Initialize means by selecting random data points
    # Инициализация средних значений выбором случайных точек данных
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    means = X[indices]
    covariances = np.array([np.eye(d) * 0.3] * K)  # Initial covariance matrices
    # Начальные матрицы ковариации
    pi_k = np.ones(K) / K  # Equal priors for each cluster
    # Равные априорные вероятности для каждого кластера

    log_likelihoods = []
    params = []

    for iteration in range(max_iter):
        # E-step: Compute responsibilities (gamma)
        # E-шаг: вычисление ответственностей (гамма)
        gamma = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                gamma[n, k] = pi_k[k] * multivariate_gaussian(X[n], means[k], covariances[k])
            gamma[n, :] /= np.sum(gamma[n, :])  # Normalize responsibilities
            # Нормализация ответственностей

        # M-step: Update parameters based on the responsibilities
        # M-шаг: обновление параметров на основе ответственностей
        N_k = np.sum(gamma, axis=0)

        for k in range(K):
            # Update means
            # Обновление средних значений
            means[k] = np.sum(gamma[:, k].reshape(-1, 1) * X, axis=0) / N_k[k]

            # Update covariances
            # Обновление ковариационных матриц
            covariances[k] = np.zeros((d, d))
            for n in range(N):
                diff = X[n] - means[k]
                covariances[k] += gamma[n, k] * np.outer(diff, diff)
            covariances[k] /= N_k[k]

            # Update mixture coefficients
            # Обновление коэффициентов смеси
            pi_k[k] = N_k[k] / N

        # Compute log likelihood
        # Вычисление логарифма правдоподобия
        log_likelihood = 0
        for n in range(N):
            temp = 0
            for k in range(K):
                temp += pi_k[k] * multivariate_gaussian(X[n], means[k], covariances[k])
            log_likelihood += np.log(temp)
        log_likelihoods.append(log_likelihood)

        # Store parameters for visualization
        # Сохранение параметров для визуализации
        params.append((means.copy(), covariances.copy(), pi_k.copy(), gamma.copy()))

        # Check for convergence based on log likelihood change
        # Проверка сходимости на основе изменения логарифма правдоподобия
        if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    # Print the final parameters after convergence
    # Вывод финальных параметров после сходимости
    print("\nFinal Parameters After EM Algorithm:")
    print(f"Means:\n {means}")
    print(f"Covariances:\n {covariances}")
    print(f"Mixture Coefficients:\n {pi_k}")

    return params, log_likelihoods

# Function to predict cluster membership probabilities
# Функция для предсказания вероятностей принадлежности кластерам
def predict_proba(X_point, means, covariances, pi_k):
    K = len(means)  # Number of clusters
    # Количество кластеров
    probs = np.zeros(K)

    # Compute the numerator for each cluster's responsibility
    # Вычисление числителя для ответственности каждого кластера
    for k in range(K):
        probs[k] = pi_k[k] * multivariate_gaussian(X_point, means[k], covariances[k])

    # Normalize to get probabilities
    # Нормализация для получения вероятностей
    probs /= np.sum(probs)

    return probs

# Function to plot an ellipse representing a Gaussian distribution
# Функция для отображения эллипса, представляющего нормальное распределение
def plot_gaussian_ellipse(mean, cov, ax, color):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # Eigenvalues and eigenvectors
    # Собственные значения и собственные векторы
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Angle of the ellipse (rotation)
    # Угол эллипса (поворот)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and height of the ellipse
    # Ширина и высота эллипса
    width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations
    # 2 стандартных отклонения

    # Plot the ellipse
    # Построение эллипса
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, color=color, fill=False, linewidth=2)
    ax.add_artist(ell)

# Function to visualize GMM results with a slider
# Функция для визуализации результатов GMM с помощью слайдера
def visualize_gmm_slider(X, params):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    # Slider axis and slider
    # Ось слайдера и сам слайдер
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Iteration', 1, len(params), valinit=1, valstep=1)

    def update(iteration):
        ax.clear()
        means, covariances, pi_k, gamma = params[int(iteration) - 1]

        # Assign each point to the cluster with the highest responsibility
        # Присвоение каждой точки кластеру с наибольшей ответственностью
        cluster_assignments = np.argmax(gamma, axis=1)

        # Scatter plot of the data points colored by cluster assignment
        # Точечный график точек данных, окрашенных по принадлежности к кластеру
        colors = ['red', 'blue']
        for k in range(len(means)):
            points_in_cluster = X[cluster_assignments == k]
            ax.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], color=colors[k], s=30, marker='o', alpha=0.6)

        # Plot the Gaussian ellipses
        # Построение гауссовских эллипсов
        for k in range(len(means)):
            plot_gaussian_ellipse(means[k], covariances[k], ax, colors[k])
            # Mark the mean of the cluster
            # Отметить среднее значение кластера
            ax.scatter(means[k][0], means[k][1], c=colors[k], s=200, marker='x', label=f'Cluster {k+1} mean')

        ax.text(-1.5, 2.0, f'L = {int(iteration)}', fontsize=16, verticalalignment='top', horizontalalignment='left')
        ax.set_title('GMM Clustering with EM Algorithm')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-2.45, 2.45)  # Set axis limits
        # Установка пределов осей
        ax.set_ylim(-2.45, 2.45)
        plt.draw()

    slider.on_changed(update)
    update(1)
    plt.show()

# Main execution
# Основная часть программы
if __name__ == "__main__":
    # Adjusted mean values
    # Скорректированные средние значения
    mean1 = [-1.0, -1.0]
    mean2 = [1.0, 1.0]
    
    # Adjusted covariance matrices for wider clusters
    # Скорректированные ковариационные матрицы для более широких кластеров
    cov1 = [[0.5, 0.2], [0.2, 0.5]]  # Wider covariance matrix
    # Более широкая ковариационная матрица
    cov2 = [[0.5, -0.2], [-0.2, 0.5]]  # Another wider covariance matrix
    # Другая широкая ковариационная матрица

    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, cov1, 100)  # Generate data for cluster 1
    # Генерация данных для кластера 1
    X2 = np.random.multivariate_normal(mean2, cov2, 100)  # Generate data for cluster 2
    # Генерация данных для кластера 2
    X = np.vstack([X1, X2])  # Combine the data
    # Объединение данных

    # Fit GMM with EM algorithm
    # Обучение GMM с помощью EM алгоритма
    K = 2  # Number of clusters
    # Количество кластеров
    params, log_likelihoods = em_gmm(X, K)

    # Visualize the GMM clustering and Gaussian contours with a slider
    # Визуализация кластеризации GMM и гауссовских контуров с помощью слайдера
    visualize_gmm_slider(X, params)

    # Define a new point to classify (between the clusters)
    # Определение новой точки для классификации (между кластерами)
    X_new_point = np.array([0, 0])

    # Predict the probabilities of the point belonging to each cluster
    # Предсказание вероятностей принадлежности точки к каждому кластеру
    means, covariances, pi_k, gamma = params[-1]
    probs = predict_proba(X_new_point, means, covariances, pi_k)

    # Print the results
    # Вывод результатов
    print(f"Probabilities for point {X_new_point}:")
    for i, p in enumerate(probs):
        print(f"Cluster {i+1}: {p:.4f}")

    # Optionally, create a probability contour plot
    # Опционально: создание контурного графика вероятностей
    # Create a grid of points
    # Создание сетки точек
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    pos = np.dstack((X_grid, Y_grid))

    # Compute probabilities for each point in the grid
    # Вычисление вероятностей для каждой точки сетки
    probs_grid = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            point = np.array([X_grid[i, j], Y_grid[i, j]])
            probs = predict_proba(point, means, covariances, pi_k)
            probs_grid[i, j] = probs[1]  # Probability of belonging to Cluster 2
            # Вероятность принадлежности кластеру 2

    # Plot the contour
    # Построение контура
    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, probs_grid, levels=20, cmap='RdBu')
    plt.colorbar(label='Probability of Cluster 2')
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    plt.title('Probability Contour Plot for Cluster 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
