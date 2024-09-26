import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Определяем архитектуру MLP с фиксированными весами
# Define the architecture of MLP with fixed weights
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),  # Вход из 10 признаков в 20 нейронов
                                # Input from 10 features to 20 neurons
            nn.ReLU(),
            nn.Linear(20, 1)    # Выходной слой
                                # Output layer
        )
        # Инициализируем фиксированные веса
        # Initialize fixed weights
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1, 1)  # Инициализируем веса в диапазоне [-1, 1]
                                        # Initialize weights in the range [-1, 1]

    def forward(self, x):
        return self.layers(x)

# Создаем экземпляр фиксированного MLP
# Create an instance of the fixed MLP
fixed_mlp = FixedMLP()

# Генерируем случайные входные данные
# Generate random input data
input_data = torch.randn(1000, 10)  # 1000 образцов, каждый с 10 признаками
                                    # 1000 samples, each with 10 features

# Получаем выходы фиксированного MLP
# Obtain outputs from the fixed MLP
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Разделяем на обучающую и тестовую выборки
# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    input_data.numpy(), output_data.numpy(), test_size=0.2, random_state=42
)

# Определяем новую модель MLP
# Define the new MLP model
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Создаем экземпляр MLP2
# Create an instance of MLP2
mlp2 = MLP2()

# Определяем функцию потерь и оптимизатор
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp2.parameters(), lr=0.001)

# Преобразуем данные в тензоры
# Convert data to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Добавляем список для сохранения значений функции потерь
# Add a list to store loss values
loss_values = []

# Обучение модели
# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Обнуляем градиенты
                           # Zero the gradients
    outputs = mlp2(X_train_tensor)  # Получаем предсказания
                                     # Get predictions
    loss = criterion(outputs, y_train_tensor)  # Вычисляем потерю
                                               # Compute loss
    loss.backward()  # Вычисляем градиенты
                     # Compute gradients
    optimizer.step()  # Обновляем веса
                      # Update weights

    # Сохраняем значение функции потерь
    # Save the loss value
    loss_values.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# График функции потерь по эпохам
# Plot Loss vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss / Функция потерь', color='blue')
plt.title('График функции потерь по эпохам / Loss vs Epochs')
plt.xlabel('Эпоха / Epoch')
plt.ylabel('Loss / Функция потерь')
plt.legend()
plt.grid(True)
plt.show()

# Оценка модели на тестовых данных
# Evaluate the model on test data
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

with torch.no_grad():
    predicted = mlp2(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Преобразуем тензоры в numpy для построения графиков
# Convert tensors to numpy for plotting
predicted = predicted.numpy()
y_test = y_test_tensor.numpy()

# График реальных против предсказанных значений
# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted, alpha=0.5, label='Предсказанные значения / Predicted Values')
plt.title('Реальные против Предсказанных значений / Actual vs Predicted Values')
plt.xlabel('Реальные значения / Actual Values')
plt.ylabel('Предсказанные значения / Predicted Values')

# Добавляем идеальную линию предсказаний
# Add ideal prediction line
min_val = min(y_test.min(), predicted.min())
max_val = max(y_test.max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное предсказание / Ideal Prediction')
plt.legend()
plt.grid(True)
plt.show()

# График распределения ошибок
# Plot Residuals Distribution
residuals = y_test - predicted
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Распределение ошибок (Residuals) / Residuals Distribution')
plt.xlabel('Ошибка / Error')
plt.ylabel('Частота / Frequency')
plt.grid(True)
plt.show()
