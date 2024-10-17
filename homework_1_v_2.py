import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

# Определяем архитектуру целевой сети согласно преподавателю
class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        l1 = nn.Linear(2, 4)
        nn.init.uniform_(l1.weight, 0, 3.14)
        r1 = nn.Tanh()
        l2 = nn.Linear(4, 4)
        nn.init.uniform_(l2.weight, 0, 3.14)
        r2 = nn.Tanh()
        l3 = nn.Linear(4, 4)
        nn.init.uniform_(l3.weight, 0, 3.14)
        r3 = nn.Tanh()
        l4 = nn.Linear(4, 1)
        nn.init.uniform_(l4.weight, 0, 3.14)
        layers = [l1, r1, l2, r2, l3, r3, l4]
        self.module_list = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

# Создаем экземпляр целевой сети
fixed_mlp = Target()

# Генерируем входные данные согласно рекомендациям преподавателя
n_points = 1000  # Количество точек
input_shape = 2  # Размерность входа
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0  # Масштабируем и смещаем данные

# Получаем выходы целевой сети
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Разделяем на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    input_data.numpy(), output_data.numpy(), test_size=0.2, random_state=42
)

# Нормализация данных
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Определяем новую модель MLP2 согласно архитектуре преподавателя
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        l1 = nn.Linear(2, 4)
        r1 = nn.Tanh()
        l2 = nn.Linear(4, 4)
        r2 = nn.Tanh()
        l3 = nn.Linear(4, 4)
        r3 = nn.Tanh()
        l4 = nn.Linear(4, 1)
        self.module_list = nn.ModuleList([l1, r1, l2, r2, l3, r3, l4])
    
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

# Создаем экземпляр MLP2
mlp2 = MLP2()

# Определяем функцию потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp2.parameters(), lr=0.001, weight_decay=1e-5)  # Повышенная скорость обучения

# Преобразуем данные в тензоры
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)

# Добавляем списки для сохранения значений функции потерь
train_loss_values = []
val_loss_values = []

# Обучение модели с ранней остановкой
num_epochs = 1000  # Увеличено количество эпох
best_val_loss = float('inf')
patience = 20  # Увеличена терпимость для ранней остановки
trigger_times = 0

# Для сохранения лучшей модели
best_model_wts = copy.deepcopy(mlp2.state_dict())

for epoch in range(num_epochs):
    # Обучение
    mlp2.train()
    optimizer.zero_grad()  # Обнуляем градиенты
    outputs = mlp2(X_train_tensor)  # Получаем предсказания на обучающей выборке
    train_loss = criterion(outputs, y_train_tensor)  # Вычисляем потерю на обучающей выборке
    train_loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса

    # Оценка на валидационной выборке
    mlp2.eval()
    with torch.no_grad():
        val_outputs = mlp2(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    # Сохраняем значения функции потерь
    train_loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())

    # Реализация ранней остановки и сохранение лучшей модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(mlp2.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Загрузка лучшей модели
mlp2.load_state_dict(best_model_wts)

# График функции потерь по эпохам
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss', color='blue')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Оценка модели на валидационных данных
with torch.no_grad():
    predicted = mlp2(X_val_tensor)
    val_loss = criterion(predicted, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.6f}')

# Преобразуем тензоры в numpy для построения графиков
predicted = scaler_y.inverse_transform(predicted.numpy())
y_val = scaler_y.inverse_transform(y_val_tensor.numpy())

# График реальных против предсказанных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predicted, alpha=0.5, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Добавляем идеальную линию предсказаний
min_val = min(y_val.min(), predicted.min())
max_val = max(y_val.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
plt.legend()
plt.grid(True)
plt.show()

# График распределения ошибок
residuals = y_val - predicted
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Дополнительная визуализация: сравнение функций FixedMLP и MLP2
# Генерация диапазона входных данных для визуализации
# Для 2D входа необходимо зафиксировать один из признаков
fixed_feature = 5.0  # Можно выбрать среднее значение или любое другое фиксированное значение
input_range = torch.linspace(0, 10, 100).unsqueeze(1)
fixed_features = torch.full((100, 1), fixed_feature)
input_range_combined = torch.cat((input_range, fixed_features), dim=1)

# Масштабирование входных данных
input_range_combined_scaled = scaler_X.transform(input_range_combined.numpy())
input_range_tensor = torch.from_numpy(input_range_combined_scaled).float()

with torch.no_grad():
    fixed_outputs = fixed_mlp(input_range_tensor).numpy()
    mlp2_outputs = mlp2(input_range_tensor).numpy()

# Обратное масштабирование выходных данных для визуализации
fixed_outputs = scaler_y.inverse_transform(fixed_outputs)
mlp2_outputs = scaler_y.inverse_transform(mlp2_outputs)

plt.figure(figsize=(10, 6))
plt.plot(input_range.numpy(), fixed_outputs, label='FixedMLP Output', color='blue')
plt.plot(input_range.numpy(), mlp2_outputs, label='MLP2 Output', color='green')
plt.xlabel('Input Feature 1')
plt.ylabel('Output')
plt.title('Сравнение выходов FixedMLP и MLP2 при фиксированном Feature 2')
plt.legend()
plt.grid(True)
plt.show()
