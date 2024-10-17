import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the FixedMLP (Target Network) as given by the professor
class FixedMLP(nn.Module):
    def __init__(self):
        super(FixedMLP, self).__init__()
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

# Create an instance of the FixedMLP
fixed_mlp = FixedMLP()

# Generate input data as per the professor's recommendation
n_points = 1000  # Number of data points
input_shape = 2  # Input dimension
input_data = 3.0 * torch.randn(n_points, input_shape) + 5.0  # Scale and shift the data

# Obtain outputs from the FixedMLP
with torch.no_grad():
    output_data = fixed_mlp(input_data)

# Prepare the training data: inputs are outputs from FixedMLP, targets are original inputs
X = output_data.numpy()
y = input_data.numpy()

# Split into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Data normalization (optional but can improve training)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Define the Inverse MLP model
class InverseMLP(nn.Module):
    def __init__(self):
        super(InverseMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output dimension matches input_data's dimension
        )
        
    def forward(self, x):
        return self.layers(x)

# Create an instance of InverseMLP
inverse_mlp = InverseMLP()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(inverse_mlp.parameters(), lr=0.0001)

# Convert data to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()

# Lists to store loss values
train_loss_values = []
val_loss_values = []

# Train the model with early stopping
num_epochs = 500
best_val_loss = float('inf')
patience = 15
trigger_times = 0

for epoch in range(num_epochs):
    # Training
    inverse_mlp.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = inverse_mlp(X_train_tensor)  # Predict inputs from outputs
    train_loss = criterion(outputs, y_train_tensor)  # Compute training loss
    train_loss.backward()  # Backpropagate
    optimizer.step()  # Update weights

    # Validation
    inverse_mlp.eval()
    with torch.no_grad():
        val_outputs = inverse_mlp(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    # Store loss values
    train_loss_values.append(train_loss.item())
    val_loss_values.append(val_loss.item())

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')

# Plot Loss vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, label='Training Loss', color='blue')
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on validation data
inverse_mlp.eval()
with torch.no_grad():
    predicted_inputs = inverse_mlp(X_val_tensor)
    val_loss = criterion(predicted_inputs, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.6f}')

# Inverse transform the data to original scale
predicted_inputs_np = scaler_y.inverse_transform(predicted_inputs.numpy())
actual_inputs_np = scaler_y.inverse_transform(y_val_tensor.numpy())

# Plot Actual vs Predicted Inputs
plt.figure(figsize=(10, 6))
plt.scatter(actual_inputs_np[:, 0], predicted_inputs_np[:, 0], alpha=0.5, label='Dimension 1')
plt.scatter(actual_inputs_np[:, 1], predicted_inputs_np[:, 1], alpha=0.5, label='Dimension 2')
plt.title('Actual vs Predicted Inputs')
plt.xlabel('Actual Inputs')
plt.ylabel('Predicted Inputs')
plt.legend()
plt.grid(True)
plt.show()

# Plot Residuals Distribution for both dimensions
residuals = actual_inputs_np - predicted_inputs_np

plt.figure(figsize=(10, 6))
plt.hist(residuals[:, 0], bins=30, alpha=0.7, label='Dimension 1')
plt.hist(residuals[:, 1], bins=30, alpha=0.7, label='Dimension 2')
plt.title('Residuals Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
