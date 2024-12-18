import math

def chernoff_bounds(p, n, delta=0.05):
    # Пример приближенного доверительного интервала на основе Черноффа.
    # Для упрощения: если n слишком мало, вернем просто p, без доверительного интервала.
    if n == 0:
        return p, p
    # Используем простой приближенный доверительный интервал для биномиальной пропорции:
    # p ± sqrt((ln(1/delta))/(2*n))
    # Это не каноническая формула Черноффа, но приближение для демонстрации.
    bound = math.sqrt((math.log(1/delta)) / (2*n))
    low = max(0.0, p - bound)
    high = min(1.0, p + bound)
    return low, high