"""
NOTES TO INTERVIEWER:

This project trains a feedforward neural network with two hidden layers
to classify breast cancer tumors as benign or malignant.

The model learns patterns from 30 numeric features using backpropagation
and gradient descent, and is evaluated on unseen test data.
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim

"""
NOTES TO INTERVIEWER:

I use PyTorch's nn module to define neural network layers and loss
functions, and optim to update model weights during training using Adam.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
NOTES TO INTERVIEWER:

I use scikit-learn to load the dataset, split it into training and test
sets, and normalize the features so the neural network trains efficiently.
"""

# ------------------------------------------------
# Loading and Preparing the Dataset
# ------------------------------------------------

data = load_breast_cancer()

# Features (569 samples, 30 numeric features)
X = data.data

# Labels: 0 = malignant, 1 = benign
y = data.target

"""
NOTES TO INTERVIEWER:

Each row represents one patient, and each column is a measurement derived
from tumor imaging. The labels indicate whether the tumor is malignant
or benign.
"""

# Train / test split (80/20), stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""
NOTES TO INTERVIEWER:

I split the data into training and test sets and used stratification to
ensure both sets maintain the same malignant/benign class distribution.
"""

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
NOTES TO INTERVIEWER:

I normalized the input features because neural networks converge faster
and more reliably when inputs are on similar scales.
"""

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

input_dim = X_train_tensor.shape[1]

"""
NOTES TO INTERVIEWER:

I converted the NumPy arrays to PyTorch tensors and reshaped the labels
to match the model output shape for binary classification.
"""

# ------------------------------------------------
# Model Definition
# ------------------------------------------------

class BreastCancerNet(nn.Module):
    """
    NOTES TO INTERVIEWER:

    This class defines a feedforward neural network with two hidden layers.
    Hidden layers allow the model to learn nonlinear feature interactions,
    while keeping the architecture simple to reduce overfitting.
    """

    def __init__(self, input_dim):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        NOTES TO INTERVIEWER:

        The forward pass defines how data flows through the network,
        applying linear transformations followed by nonlinear activations.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


model = BreastCancerNet(input_dim)

# ------------------------------------------------
# Loss and Optimizer
# ------------------------------------------------

criterion = nn.BCELoss()

"""
NOTES TO INTERVIEWER:

I used binary cross-entropy loss because this is a binary classification
task with probabilistic outputs from a sigmoid activation.
"""

optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
NOTES TO INTERVIEWER:

I used the Adam optimizer because it adapts learning rates per parameter
and converges efficiently for neural networks.
"""

# ------------------------------------------------
# Training Loop
# ------------------------------------------------

"""
NOTES TO INTERVIEWER:

During training, I perform a forward pass to compute predictions,
calculate the loss, then use backpropagation to compute gradients
and update the model weights using Adam.

Training accuracy is computed every 10 epochs to monitor progress
without cluttering the output.
"""

num_epochs = 100

for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train_tensor)
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

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {loss.item():.4f} | "
            f"Train Accuracy: {acc:.2f}%"
        )

# ------------------------------------------------
# Evaluation
# ------------------------------------------------

"""
NOTES TO INTERVIEWER:

After training, I evaluate the model on unseen test data to measure
generalization. Gradient tracking is disabled, and the model is switched
to evaluation mode.
"""

model.eval()

with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predicted = (test_outputs >= 0.5).float()
    correct = (test_predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    test_acc = correct / total * 100.0

print(f"\nTest Accuracy: {test_acc:.2f}%")
