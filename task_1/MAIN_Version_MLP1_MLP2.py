import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Определяем более сложную архитектуру фиксированной сети
# Define a more complex architecture of the fixed network
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
        l1 = nn.Linear(2, 8)
        nn.init.uniform_(l1.weight, 0, 3.14)
        r1 = nn.Tanh()
        l2 = nn.Linear(8, 16)
        nn.init.uniform_(l2.weight, 0, 3.14)
        r2 = nn.Tanh()
        l3 = nn.Linear(16, 16)
        nn.init.uniform_(l3.weight, 0, 3.14)
        r3 = nn.Tanh()
        l4 = nn.Linear(16, 1)
        nn.init.uniform_(l4.weight, 0, 3.14)
        layers = [l1, r1, l2, r2, l3, r3, l4]
        self.module_list = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

# Создаем экземпляр фиксированной сети
# Create an instance of the fixed network
fixed_mlp = FixedMLP()

# Генерируем входные данные в соответствии с рекомендациями
# Generate input data as per recommendations
n_points = 10000  # Количество точек
input_shape = 2  # Размерность входа
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0  # Масштабируем и смещаем данные

# Получаем выходы фиксированной сети
# Obtain outputs from the fixed network
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Разделяем на обучающую и валидационную выборки
# Split into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(
    input_data.numpy(), output_data.numpy(), test_size=0.2, random_state=42
)

# Нормализация данных
# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Определяем новую модель MLP, которую будем обучать
# Define the new MLP model to be trained
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Создаем экземпляр MLP2
# Create an instance of MLP2
mlp2 = MLP2()

# Определяем функцию потерь и оптимизатор
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp2.parameters(), lr=0.0001)

# Преобразуем данные в тензоры
# Convert data to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)

# Добавляем списки для сохранения значений функции потерь
# Add lists to store loss values
train_loss_values = []
val_loss_values = []

# Обучение модели с ранней остановкой
# Train the model with early stopping
num_epochs = 500
best_val_loss = float('inf')
patience = 10
trigger_times = 0

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

    # Реализация ранней остановки
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Ранняя остановка на эпохе {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# График функции потерь по эпохам
# Plot Loss vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss', color='blue')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Оценка модели на валидационных данных
# Evaluate the model on validation data
with torch.no_grad():
    predicted = mlp2(X_val_tensor)
    val_loss = criterion(predicted, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.4f}')

# Преобразуем тензоры в numpy для построения графиков
# Convert tensors to numpy for plotting
predicted = predicted.numpy()
y_val = y_val_tensor.numpy()

# График реальных против предсказанных значений
# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predicted, alpha=0.5, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Добавляем идеальную линию предсказаний
# Add ideal prediction line
min_val = min(y_val.min(), predicted.min())
max_val = max(y_val.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
plt.legend()
plt.grid(True)
plt.show()

# График распределения ошибок
# Plot Residuals Distribution
residuals = y_val - predicted
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
