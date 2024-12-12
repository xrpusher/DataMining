import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Определение FixedMLP (Целевая нн) / Definition of FixedMLP (Target Neural Network)
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
        l1 = nn.Linear(2, 4)  # Линейный слой с 2 входами и 4 выходами / Linear layer with 2 inputs and 4 outputs
        nn.init.uniform_(l1.weight, 0, 3.14)  # Инициализация весов равномерно от 0 до π / Weight initialization uniformly from 0 to π
        r1 = nn.Tanh()  # Функция активации Tanh / Tanh activation function
        l2 = nn.Linear(4, 4)  # Линейный слой с 4 входами и 4 выходами / Linear layer with 4 inputs and 4 outputs
        nn.init.uniform_(l2.weight, 0, 3.14)  # Инициализация весов / Weight initialization
        r2 = nn.Tanh()  # Функция активации Tanh / Tanh activation function
        l3 = nn.Linear(4, 4)  # Линейный слой с 4 входами и 4 выходами / Linear layer with 4 inputs and 4 outputs
        nn.init.uniform_(l3.weight, 0, 3.14)  # Инициализация весов / Weight initialization
        r3 = nn.Tanh()  # Функция активации Tanh / Tanh activation function
        l4 = nn.Linear(4, 1)  # Линейный слой с 4 входами и 1 выходом / Linear layer with 4 inputs and 1 output
        nn.init.uniform_(l4.weight, 0, 3.14)  # Инициализация весов / Weight initialization
        layers = [l1, r1, l2, r2, l3, r3, l4]  # Список слоев / List of layers
        self.module_list = nn.ModuleList(layers)  # Использование ModuleList для хранения слоев / Using ModuleList to store layers
    
    def forward(self, x):
        for layer in self.module_list:  # Пропуск входных данных через каждый слой / Pass the input through each layer
            x = layer(x)
        return x

fixed_mlp = FixedMLP()  # Создание экземпляра FixedMLP / Creating an instance of FixedMLP

# Генерация входных данных / Generating input data
n_points = 50000  # Значительно увеличиваем количество точек данных / Significantly increasing the number of data points
input_shape = 2
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0  # Генерация данных с нормальным распределением / Generating data from normal distribution

# Получение выходных данных из FixedMLP / Getting output data from FixedMLP
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Подготовка обучающих данных / Preparing training data
X = output_data.numpy()  # Преобразование выходных данных в numpy массив / Converting output data to numpy array
y = input_data.numpy()  # Преобразование входных данных в numpy массив / Converting input data to numpy array

# Разделение на обучающую и валидационную выборки / Splitting into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Преобразование данных в тензоры (без нормализации) / Converting data to tensors (without normalization)
X_train_tensor = torch.from_numpy(X_train).float()  # Преобразование обучающих входов / Converting training inputs
y_train_tensor = torch.from_numpy(y_train).float()  # Преобразование обучающих целей / Converting training targets
X_val_tensor = torch.from_numpy(X_val).float()  # Преобразование валидационных входов / Converting validation inputs
y_val_tensor = torch.from_numpy(y_val).float()  # Преобразование валидационных целей / Converting validation targets

# Определение модели InverseMLP с уменьшенной сложностью / Definition of InverseMLP with reduced complexity
class InverseMLP(nn.Module):
    def __init__(self):
        super(InverseMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 6),   # Уменьшаем количество нейронов / Reducing the number of neurons
            nn.Tanh(),
            nn.Linear(6, 12),  # Увеличиваем количество нейронов / Increasing the number of neurons
            nn.Tanh(),
            nn.Linear(12, 2)
        )
        
    def forward(self, x):
        return self.layers(x)  # Пропуск входных данных через модель / Passing input through the model

inverse_mlp = InverseMLP()  # Создание экземпляра InverseMLP / Creating an instance of InverseMLP

# Определение функции потерь и оптимизатора / Defining loss function and optimizer
criterion = nn.MSELoss()  # Функция потерь: среднеквадратичная ошибка / Loss function: Mean Squared Error
optimizer = torch.optim.Adam(inverse_mlp.parameters(), lr=0.001)  # Оптимизатор Adam с заданной скоростью обучения / Adam optimizer with specified learning rate

# Списки для хранения значений потерь / Lists to store loss values
train_loss_values = []
val_loss_values = []

# Обучение модели с ранней остановкой / Training the model with early stopping
num_epochs = 500
best_val_loss = float('inf')  # Инициализация наилучшей валидационной потери / Initialize best validation loss
patience = 15  # Количество эпох для ожидания улучшения перед остановкой / Number of epochs to wait for improvement before stopping
trigger_times = 0  # Счетчик для patience / Counter for patience

for epoch in range(num_epochs):
    # Фаза обучения / Training phase
    inverse_mlp.train()  # Переключение в режим обучения / Switching to training mode
    optimizer.zero_grad()  # Обнуление градиентов / Zero the gradients
    outputs = inverse_mlp(X_train_tensor)  # Предсказание на обучающей выборке / Predictions on training data
    train_loss = criterion(outputs, y_train_tensor)  # Вычисление потерь на обучающей выборке / Compute training loss
    train_loss.backward()  # Обратное распространение / Backpropagation
    optimizer.step()  # Обновление параметров модели / Update model parameters
    
    # Фаза валидации / Validation phase
    inverse_mlp.eval()  # Переключение в режим оценки / Switching to evaluation mode
    with torch.no_grad():  # Отключение градиентов / Disable gradient calculation
        val_outputs = inverse_mlp(X_val_tensor)  # Предсказание на валидационной выборке / Predictions on validation data
        val_loss = criterion(val_outputs, y_val_tensor)  # Вычисление потерь на валидационной выборке / Compute validation loss
    
    # Сохранение значений потерь / Storing loss values
    train_loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())
    
    # Ранняя остановка / Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")  # Сообщение о ранней остановке / Early stopping message
            break
    
    # Вывод потерь каждые 10 эпох / Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Построение графика потерь по эпохам / Plotting loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss', color='blue')  # График потерь на обучении / Training loss plot
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='orange')  # График потерь на валидации / Validation loss plot
plt.xlabel('Epochs')  # Подпись для оси X / Label for X-axis
plt.ylabel('Loss')  # Подпись для оси Y / Label for Y-axis
plt.title('Training and Validation Loss Over Epochs')  # Заголовок графика / Title of the plot
plt.legend()  # Отображение легенды / Display legend
plt.grid(True)  # Показать сетку / Show grid
plt.show()

# Оценка модели на валидационных данных / Evaluating the model on validation data
inverse_mlp.eval()
with torch.no_grad():
    predicted_inputs = inverse_mlp(X_val_tensor)  # Предсказания на валидационных данных / Predictions on validation data
    val_loss = criterion(predicted_inputs, y_val_tensor)  # Вычисление потерь на валидационной выборке / Compute validation loss
    print(f'Validation Loss: {val_loss.item():.6f}')  # Вывод потерь на валидационной выборке / Print validation loss

# Преобразование тензоров в numpy массивы / Convert tensors to numpy arrays
predicted_inputs_np = predicted_inputs.numpy()
actual_inputs_np = y_val_tensor.numpy()

# Построение графика фактических vs предсказанных входов / Plotting Actual vs Predicted Inputs
plt.figure(figsize=(10, 6))
plt.scatter(actual_inputs_np[:, 0], predicted_inputs_np[:, 0], alpha=0.5, label='Dimension 1')  # Размерность 1 / Dimension 1
plt.scatter(actual_inputs_np[:, 1], predicted_inputs_np[:, 1], alpha=0.5, label='Dimension 2')  # Размерность 2 / Dimension 2
plt.title('Actual vs Predicted Inputs')  # Заголовок графика / Title of the plot
plt.xlabel('Actual Inputs')  # Подпись для оси X / Label for X-axis
plt.ylabel('Predicted Inputs')  # Подпись для оси Y / Label for Y-axis
plt.legend()  # Отображение легенды / Display legend
plt.grid(True)  # Показать сетку / Show grid
plt.show()

# Построение распределения остатков для обеих размерностей / Plotting Residuals Distribution for both dimensions
residuals = actual_inputs_np - predicted_inputs_np  # Вычисление остатков / Calculate residuals

plt.figure(figsize=(10, 6))
plt.hist(residuals[:, 0], bins=30, alpha=0.7, label='Dimension 1')  # Гистограмма для размерности 1 / Histogram for Dimension 1
plt.hist(residuals[:, 1], bins=30, alpha=0.7, label='Dimension 2')  # Гистограмма для размерности 2 / Histogram for Dimension 2
plt.title('Residuals Distribution')  # Заголовок графика / Title of the plot
plt.xlabel('Error')  # Подпись для оси X / Label for X-axis
plt.ylabel('Frequency')  # Подпись для оси Y / Label for Y-axis
plt.legend()  # Отображение легенды / Display legend
plt.grid(True)  # Показать сетку / Show grid
plt.show()
