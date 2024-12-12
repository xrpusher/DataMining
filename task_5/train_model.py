# train_model.py

import numpy as np
import pickle

# Функция для вычисления плотности многомерного нормального распределения
def multivariate_gaussian(x, mean, cov):
    d = len(x)
    cov_inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    return np.exp(exponent) / denominator

# EM-алгоритм для GMM
def em_gmm(X, K, max_iter=100, tol=1e-6):
    N, d = X.shape
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    means = X[indices]
    covariances = np.array([np.eye(d)] * K)
    pi_k = np.ones(K) / K

    for iteration in range(max_iter):
        # E-step
        gamma = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                gamma[n, k] = pi_k[k] * multivariate_gaussian(X[n], means[k], covariances[k])
            gamma[n, :] /= np.sum(gamma[n, :])

        # M-step
        N_k = np.sum(gamma, axis=0)
        for k in range(K):
            means[k] = np.sum(gamma[:, k].reshape(-1, 1) * X, axis=0) / N_k[k]
            covariances[k] = np.zeros((d, d))
            for n in range(N):
                diff = X[n] - means[k]
                covariances[k] += gamma[n, k] * np.outer(diff, diff)
            covariances[k] /= N_k[k]
            pi_k[k] = N_k[k] / N

        # Проверка сходимости (опционально)
        if iteration > 0:
            diff_means = np.linalg.norm(means - prev_means)
            if diff_means < tol:
                print(f"Сходимость достигнута на итерации {iteration}")
                break
        prev_means = means.copy()

    # Сохранение модели на диск
    model = {'means': means, 'covariances': covariances, 'pi_k': pi_k}
    with open('gmm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Модель успешно сохранена в 'gmm_model.pkl'.")

if __name__ == "__main__":
    # Генерация обучающих данных
    mean1 = [-1.0, -1.0]
    mean2 = [1.0, 1.0]
    cov1 = [[0.5, 0.1], [0.1, 0.5]]  # Разные ковариационные матрицы//
    cov2 = [[0.3, -0.2], [-0.2, 0.3]]

    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean1, cov1, 100)
    X2 = np.random.multivariate_normal(mean2, cov2, 100)
    X = np.vstack([X1, X2])

    # Обучение модели
    K = 2
    em_gmm(X, K)
