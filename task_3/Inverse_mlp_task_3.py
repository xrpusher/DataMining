import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the FixedMLP (Target Network)
# Определение FixedMLP (Целевая нн)
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
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

# Create an instance of the FixedMLP
# Создание экземпляра FixedMLP
fixed_mlp = FixedMLP()

# Generate input data
# Генерация входных данных
n_points = 1000  # Number of data points
# Количество точек данных
input_shape = 2  # Input dimension
# Размерность входных данных
# Generate data from a normal distribution, scaled by 3 and shifted by 5
# Генерация данных из нормального распределения, масштабированных на 3 и сдвинутых на 5
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0

# Obtain outputs from the FixedMLP
# Получение выходных данных из FixedMLP
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Prepare the training data: inputs are outputs from FixedMLP, targets are original inputs
# Подготовка обучающих данных: входы — выходы из FixedMLP, цели — оригинальные входы
X = output_data.numpy()
y = input_data.numpy()

# Split into training and validation datasets
# Разделение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Data normalization (optional but can improve training)
# Нормализация данных (опционально, но может улучшить обучение)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Define the Inverse MLP model
# Определение модели Inverse MLP
class InverseMLP(nn.Module):
    def __init__(self):
        super(InverseMLP, self).__init__()
        # Define a sequential model with multiple layers
        # Определение последовательной модели с несколькими слоями
        self.layers = nn.Sequential(
            nn.Linear(1, 16),    # First layer: 1 input to 16 neurons
            nn.ReLU(),           # ReLU activation
            nn.Dropout(0.2),     # Dropout with probability 0.2
            nn.Linear(16, 32),   # Second layer: 16 inputs to 32 neurons
            nn.ReLU(),           # ReLU activation
            nn.Dropout(0.2),     # Dropout with probability 0.2
            nn.Linear(32, 32),   # Third layer: 32 inputs to 32 neurons
            nn.ReLU(),           # ReLU activation
            nn.Linear(32, 2)     # Output layer: 32 inputs to 2 outputs (matches input_data's dimension)
        )
        
    def forward(self, x):
        # Pass the input through the sequential layers
        # Пропуск входных данных через последовательные слои
        return self.layers(x)

# Create an instance of InverseMLP
# Создание экземпляра InverseMLP
inverse_mlp = InverseMLP()

# Define the loss function and optimizer
# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()  # Mean Squared Error loss
# Функция потерь: среднеквадратичная ошибка
optimizer = torch.optim.Adam(inverse_mlp.parameters(), lr=0.0001)  # Adam optimizer with learning rate 0.0001
# Оптимизатор: Adam с скоростью обучения 0.0001

# Convert data to tensors
# Преобразование данных в тензоры
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()

# Lists to store loss values
# Списки для хранения значений потерь
train_loss_values = []
val_loss_values = []

# Train the model with early stopping
# Обучение модели с ранней остановкой
num_epochs = 500  # Maximum number of epochs
# Максимальное количество эпох
best_val_loss = float('inf')  # Initialize best validation loss to infinity
# Инициализация наилучшей валидационной потери как бесконечность
patience = 15  # Number of epochs to wait for improvement before stopping
# Количество эпох ожидания улучшения перед остановкой
trigger_times = 0  # Counter for patience

for epoch in range(num_epochs):
    # Training phase
    # Фаза обучения
    inverse_mlp.train()
    optimizer.zero_grad()  # Zero the gradients
    # Обнуление градиентов
    outputs = inverse_mlp(X_train_tensor)  # Predict inputs from outputs
    # Предсказание входов из выходов
    train_loss = criterion(outputs, y_train_tensor)  # Compute training loss
    # Вычисление потерь на обучающей выборке
    train_loss.backward()  # Backpropagate
    # Обратное распространение ошибки
    optimizer.step()  # Update weights
    # Обновление весов
    
    # Validation phase
    # Фаза валидации
    inverse_mlp.eval()
    with torch.no_grad():
        val_outputs = inverse_mlp(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Store loss values
    # Сохранение значений потерь
    train_loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())
    
    # Early stopping
    # Ранняя остановка
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print loss every 10 epochs
    # Вывод потерь каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Plot Loss vs Epochs
# Построение графика потерь по эпохам
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
inverse_mlp.eval()
with torch.no_grad():
    predicted_inputs = inverse_mlp(X_val_tensor)
    val_loss = criterion(predicted_inputs, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.6f}')

# Inverse transform the data to original scale
# Обратное преобразование данных к исходному масштабу
predicted_inputs_np = scaler_y.inverse_transform(predicted_inputs.numpy())
actual_inputs_np = scaler_y.inverse_transform(y_val_tensor.numpy())

# Plot Actual vs Predicted Inputs
# Построение графика фактических vs предсказанных входов
plt.figure(figsize=(10, 6))
plt.scatter(actual_inputs_np[:, 0], predicted_inputs_np[:, 0], alpha=0.5, label='Dimension 1')  # Dimension 1
plt.scatter(actual_inputs_np[:, 1], predicted_inputs_np[:, 1], alpha=0.5, label='Dimension 2')  # Dimension 2
plt.title('Actual vs Predicted Inputs')  # Title of the plot
plt.xlabel('Actual Inputs')  # Label for X-axis
plt.ylabel('Predicted Inputs')  # Label for Y-axis
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()

# Plot Residuals Distribution for both dimensions
# Построение распределения остатков для обеих размерностей
residuals = actual_inputs_np - predicted_inputs_np

plt.figure(figsize=(10, 6))
plt.hist(residuals[:, 0], bins=30, alpha=0.7, label='Dimension 1')  # Histogram for Dimension 1
plt.hist(residuals[:, 1], bins=30, alpha=0.7, label='Dimension 2')  # Histogram for Dimension 2
plt.title('Residuals Distribution')  # Title of the plot
plt.xlabel('Error')  # Label for X-axis
plt.ylabel('Frequency')  # Label for Y-axis
plt.legend()  # Display legend
plt.grid(True)  # Show grid
plt.show()
