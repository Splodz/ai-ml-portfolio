# breast_cancer_nn.py

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------
# Loading and Preparing the dataset
# ------------------------------------------------

data = load_breast_cancer()

X = data.data  # features
y = data.target  # labels  <-- FIXED

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

input_dim = X_train_tensor.shape[1]


# ------------------------------------------------
# Defining the Neural Network
# ------------------------------------------------

class BreastCancerNet(nn.Module):  # FIXED MODULE NAME
    def __init__(self, input_dim):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Instantiate model (OUTSIDE the class)
model = BreastCancerNet(input_dim)

# ------------------------------------------------
# Loss + Optimizer
# ------------------------------------------------

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------
# Training Loop
# ------------------------------------------------

num_epochs = 100

for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train_tensor)  # FIXED variable name
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            predicted = (outputs >= 0.5).float()
            correct = (predicted == y_train_tensor).sum().item()
            total = y_train_tensor.size(0)
            acc = correct / total * 100.0

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f} | Train Accuracy: {acc:.2f}%")

# ------------------------------------------------
# Evaluation
# ------------------------------------------------

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predicted = (test_outputs >= 0.5).float()
    correct = (test_predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    test_acc = correct / total * 100.0

print(f"\nTest Accuracy: {test_acc:.2f}%")
