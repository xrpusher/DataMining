import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

# Define the Target network architecture
# Определяем архитектуру
class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        # Define the first linear layer with 2 inputs and 4 outputs
        # Определение первого линейного слоя с 2 входами и 4 выходами
        l1 = nn.Linear(2, 4)
        # Initialize weights uniformly between 0 and π
        # Инициализация весов равномерно в диапазоне от 0 до π
        nn.init.uniform_(l1.weight, 0, 3.14)
        # Define the first activation function as Tanh
        # Определение первой функции активации как Tanh
        r1 = nn.Tanh()
        
        # Define the second linear layer with 4 inputs and 4 outputs
        # Определение второго линейного слоя с 4 входами и 4 выходами
        l2 = nn.Linear(4, 4)
        # Initialize weights uniformly between 0 and π
        # Инициализация весов равномерно в диапазоне от 0 до π
        nn.init.uniform_(l2.weight, 0, 3.14)
        # Define the second activation function as Tanh
        # Определение второй функции активации как Tanh
        r2 = nn.Tanh()
        
        # Define the third linear layer with 4 inputs and 4 outputs
        # Определение третьего линейного слоя с 4 входами и 4 выходами
        l3 = nn.Linear(4, 4)
        # Initialize weights uniformly between 0 and π
        # Инициализация весов равномерно в диапазоне от 0 до π
        nn.init.uniform_(l3.weight, 0, 3.14)
        # Define the third activation function as Tanh
        # Определение третьей функции активации как Tanh
        r3 = nn.Tanh()
        
        # Define the fourth linear layer with 4 inputs and 1 output
        # Определение четвертого линейного слоя с 4 входами и 1 выходом
        l4 = nn.Linear(4, 1)
        # Initialize weights uniformly between 0 and π
        # Инициализация весов равномерно в диапазоне от 0 до π
        nn.init.uniform_(l4.weight, 0, 3.14)
        
        # Create a list of all layers
        # Создание списка всех слоев
        layers = [l1, r1, l2, r2, l3, r3, l4]
        # Use ModuleList to store layers
        # Использование ModuleList для хранения слоев
        self.module_list = nn.ModuleList(layers)
    
    def forward(self, x):
        # Pass the input through each layer in sequence
        # Пропуск входных данных через каждый слой по порядку
        for layer in self.module_list:
            x = layer(x)
        return x

# Create an instance of the Target network
# Создаем экземпляр целевой сети
fixed_mlp = Target()

# Generate input data
# Генерируем входные данные
n_points = 1000  # Number of data points
# Количество точек
input_shape = 2  # Input dimension
# Размерность входа
# Generate data from a normal distribution, scaled by 3 and shifted by 5
# Генерация данных из нормального распределения, масштабированных на 3 и смещенных на 5
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0  # Масштабируем и смещаем данные

# Obtain outputs from the Target network
# Получаем выходы целевой сети
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Split into training and validation datasets
# Разделяем на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    input_data.numpy(), output_data.numpy(), test_size=0.2, random_state=42
)

# Data normalization (optional but can improve training)
# Нормализация данных (опционально, но может улучшить обучение)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Define the MLP2 model architecture
# Определяем новую модель MLP2 согласно архитектуре 
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        # Define each layer and activation function
        # Определение каждого слоя и функции активации
        l1 = nn.Linear(2, 4)
        r1 = nn.Tanh()
        l2 = nn.Linear(4, 4)
        r2 = nn.Tanh()
        l3 = nn.Linear(4, 4)
        r3 = nn.Tanh()
        l4 = nn.Linear(4, 1)
        # Store layers in ModuleList
        # Сохранение слоев в ModuleList
        self.module_list = nn.ModuleList([l1, r1, l2, r2, l3, r3, l4])
    
    def forward(self, x):
        # Pass the input through each layer in sequence
        # Пропуск входных данных через каждый слой по порядку
        for layer in self.module_list:
            x = layer(x)
        return x

# Create an instance of MLP2
# Создаем экземпляр MLP2
mlp2 = MLP2()

# Define the loss function and optimizer
# Определяем функцию потерь и оптимизатор
criterion = nn.MSELoss()  # Mean Squared Error loss
# Функция потерь: среднеквадратичная ошибка
optimizer = torch.optim.Adam(mlp2.parameters(), lr=0.001, weight_decay=1e-5)  # Increased learning rate
# Оптимизатор: Adam с повышенной скоростью обучения и регуляризацией L2

# Convert data to tensors
# Преобразуем данные в тензоры
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)

# Initialize lists to store loss values
# Инициализируем списки для сохранения значений функции потерь
train_loss_values = []
val_loss_values = []

# Training the model with early stopping
# Обучение модели с ранней остановкой
num_epochs = 1000  # Increased number of epochs
# Увеличено количество эпох
best_val_loss = float('inf')  # Initialize best validation loss to infinity
# Инициализация наилучшей валидационной потери как бесконечность
patience = 20  # Increased patience for early stopping
# Увеличена терпимость для ранней остановки
trigger_times = 0  # Counter for patience
# Счетчик для терпимости

# To save the best model weights
# Для сохранения лучшей модели
best_model_wts = copy.deepcopy(mlp2.state_dict())

for epoch in range(num_epochs):
    # Training phase
    # Фаза обучения
    mlp2.train()
    optimizer.zero_grad()  # Zero the gradients
    # Обнуляем градиенты
    outputs = mlp2(X_train_tensor)  # Get predictions on training data
    # Получаем предсказания на обучающей выборке
    train_loss = criterion(outputs, y_train_tensor)  # Compute training loss
    # Вычисляем потерю на обучающей выборке
    train_loss.backward()  # Backpropagate
    # Обратное распространение ошибки
    optimizer.step()  # Update weights
    # Обновляем веса
    
    # Validation phase
    # Фаза валидации
    mlp2.eval()
    with torch.no_grad():
        val_outputs = mlp2(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Store loss values
    # Сохраняем значения функции потерь
    train_loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())
    
    # Implement early stopping and save the best model
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
    
    # Print loss every 100 epochs
    # Вывод потерь каждые 100 эпох
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Load the best model weights
# Загрузка лучшей модели
mlp2.load_state_dict(best_model_wts)

# Plot Training and Validation Loss over epochs
# График функции потерь по эпохам
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss', color='blue')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='orange')
plt.xlabel('Epochs')  # Label for X-axis
plt.ylabel('Loss')     # Label for Y-axis
plt.title('Training and Validation Loss Over Epochs')  # Title of the plot
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()

# Evaluate the model on validation data
# Оценка модели на валидационных данных
with torch.no_grad():
    predicted = mlp2(X_val_tensor)
    val_loss = criterion(predicted, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.6f}')

# Inverse transform the data to original scale
# Обратное преобразование данных к исходному масштабу
predicted = scaler_y.inverse_transform(predicted.numpy())
y_val = scaler_y.inverse_transform(y_val_tensor.numpy())

# Plot Actual vs Predicted Values
# График реальных против предсказанных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predicted, alpha=0.5, label='Predicted Values')
plt.title('Actual vs Predicted Values')  # Title of the plot
plt.xlabel('Actual Values')  # Label for X-axis
plt.ylabel('Predicted Values')  # Label for Y-axis

# Add the ideal prediction line
# Добавляем идеальную линию предсказаний
min_val = min(y_val.min(), predicted.min())
max_val = max(y_val.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()

# Plot Residuals Distribution
# График распределения ошибок
residuals = y_val - predicted

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Residuals Distribution')  # Title of the plot
plt.xlabel('Error')  # Label for X-axis
plt.ylabel('Frequency')  # Label for Y-axis
plt.grid(True)  # Show grid
plt.show()

# Additional Visualization: Comparing FixedMLP and MLP2 Outputs
# Дополнительная визуализация: сравнение функций FixedMLP и MLP2
# Generate a range of input data for visualization
# Генерация диапазона входных данных для визуализации
# For 2D input, fix one of the features
# Для 2D входа необходимо зафиксировать один из признаков
fixed_feature = 5.0  # You can choose the mean value or any other fixed value
# Можно выбрать среднее значение или любое другое фиксированное значение
input_range = torch.linspace(0, 10, 100).unsqueeze(1)
fixed_features = torch.full((100, 1), fixed_feature)
input_range_combined = torch.cat((input_range, fixed_features), dim=1)

# Scale the input data
# Масштабирование входных данных
input_range_combined_scaled = scaler_X.transform(input_range_combined.numpy())
input_range_tensor = torch.from_numpy(input_range_combined_scaled).float()

with torch.no_grad():
    fixed_outputs = fixed_mlp(input_range_tensor).numpy()
    mlp2_outputs = mlp2(input_range_tensor).numpy()

# Inverse transform the outputs for visualization
# Обратное масштабирование выходных данных для визуализации
fixed_outputs = scaler_y.inverse_transform(fixed_outputs)
mlp2_outputs = scaler_y.inverse_transform(mlp2_outputs)

# Plot the outputs of FixedMLP and MLP2
# Построение выходов FixedMLP и MLP2
plt.figure(figsize=(10, 6))
plt.plot(input_range.numpy(), fixed_outputs, label='FixedMLP Output', color='blue')
plt.plot(input_range.numpy(), mlp2_outputs, label='MLP2 Output', color='green')
plt.xlabel('Input Feature 1')  # Label for X-axis
plt.ylabel('Output')  # Label for Y-axis
plt.title('Сравнение выходов FixedMLP и MLP2 при фиксированном Feature 2')  # Title of the plot
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()
