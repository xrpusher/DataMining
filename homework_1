import torch
import torch.nn as nn

# Define the architecture of the MLP
# Определяем архитектуру MLP
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),  # Input from 10 features to 20 neurons
                                 # Вход из 10 признаков в 20 нейронов
            nn.ReLU(),
            nn.Linear(20, 1)    # Output layer
                                 # Выходной слой
        )
        # Initialize fixed weights
        # Инициализируем фиксированные веса
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1, 1)  # Initialize weights in the range [-1, 1]
                                        # Инициализируем веса в диапазоне [-1, 1]

    def forward(self, x):
        return self.layers(x)

# Create an instance of the fixed MLP
# Создаем экземпляр фиксированного MLP
fixed_mlp = FixedMLP()

# Generate random input data
# Генерируем случайные входные данные
input_data = torch.randn(1000, 10)  # 1000 samples, each with 10 features
                                     # 1000 образцов, каждый с 10 признаками

# Obtain outputs from the fixed MLP
# Получаем выходы фиксированного MLP
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Split into training and testing datasets
# Разделяем на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    input_data.numpy(), output_data.numpy(), test_size=0.2, random_state=42
)

# Define the new MLP model
# Определяем новую модель MLP
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

# Create an instance of MLP2
# Создаем экземпляр MLP2
mlp2 = MLP2()

# Define the loss function and optimizer
# Определяем функцию потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp2.parameters(), lr=0.001)

# Convert data to tensors
# Преобразуем данные в тензоры
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Train the model
# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp2(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data
# Оценка модели на тестовых данных
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

with torch.no_grad():
    predicted = mlp2(X_test_tensor)
    test_loss = criterion(predicted, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
