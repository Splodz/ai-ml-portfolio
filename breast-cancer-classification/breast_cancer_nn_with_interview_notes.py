"""
NOTES TO INTERVIEWER:

This project trains a feedforward neural network with two hidden layers
to classify breast cancer tumors as benign or malignant.

The model learns patterns from 30 numeric features using backpropagation
and gradient descent, and is evaluated on unseen test data.

Key design decisions include:
- BCEWithLogitsLoss for numerical stability
- Dropout for regularization on a small dataset
- A Trainer class for clean, production-style code structure
- A reproducibility seed for consistent results
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

"""
NOTES TO INTERVIEWER:

I use PyTorch's nn module to define neural network layers and loss
functions, and optim to update model weights during training using Adam.

matplotlib is used to plot training vs validation loss over time,
which helps visualize whether the model is overfitting.

scikit-learn provides the dataset, train/test splitting, feature scaling,
and evaluation metrics including the confusion matrix and classification report.
"""

# Reproducibility
torch.manual_seed(42)

"""
NOTES TO INTERVIEWER:

Setting a manual seed ensures that results are reproducible across runs.
Without this, weight initialization and data splitting introduce randomness
that makes it harder to compare experiments fairly.
"""

# ------------------------------------------------
# Data Loading and Preparation
# ------------------------------------------------

def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    to_label  = lambda arr: torch.tensor(arr, dtype=torch.float32).view(-1, 1)

    return to_tensor(X_train), to_tensor(X_test), to_label(y_train), to_label(y_test)

"""
NOTES TO INTERVIEWER:

Each row in the dataset represents one patient, and each column is a
measurement derived from tumor imaging. Labels are 0 = malignant, 1 = benign.

I used stratify=y in train_test_split to preserve the class balance between
malignant and benign samples in both the training and test sets. Without
stratification, random splits can produce imbalanced subsets, especially
on smaller datasets.

I called fit_transform on the training set and transform only on the test set.
This is critical — fitting the scaler on test data would cause data leakage,
where the model indirectly learns statistics from data it should never have seen.

Labels are reshaped to (-1, 1) to match the model's single output neuron shape.
"""

# ------------------------------------------------
# Model Definition
# ------------------------------------------------

class BreastCancerNet(nn.Module):
    """
    NOTES TO INTERVIEWER:

    This class defines a feedforward neural network with two hidden layers
    and dropout regularization.

    Hidden layers allow the model to learn nonlinear feature interactions.
    The architecture is intentionally small (32 → 16) to reduce overfitting
    on a dataset with only 455 training samples.

    Dropout randomly zeros a fraction of neuron outputs during training,
    which forces the network to learn redundant representations and reduces
    reliance on any single neuron — a form of regularization.

    The output layer produces a raw logit (no sigmoid), which is handled
    by BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(BreastCancerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            # No sigmoid here — BCEWithLogitsLoss handles it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NOTES TO INTERVIEWER:

        The forward pass defines how data flows through the network.
        Using nn.Sequential simplifies this to a single call, making
        the architecture easy to read and modify.
        """
        return self.network(x)


# ------------------------------------------------
# Trainer Class
# ------------------------------------------------

class Trainer:
    """
    NOTES TO INTERVIEWER:

    Encapsulating training logic in a Trainer class separates concerns
    cleanly — the model defines the architecture, and the Trainer handles
    the training loop, loss tracking, and evaluation. This is a common
    pattern in production ML codebases.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    """
    NOTES TO INTERVIEWER:

    I used BCEWithLogitsLoss instead of BCELoss + Sigmoid for two reasons:
    1. Numerical stability — it uses the log-sum-exp trick internally,
       avoiding floating point issues that occur when sigmoid outputs
       approach exactly 0 or 1.
    2. Cleaner design — the loss function handles the sigmoid internally,
       so the model only needs to output raw logits.

    As a result, the decision threshold shifts from 0.5 (probability space)
    to 0.0 (logit space) during evaluation.

    Adam is used as the optimizer because it adapts learning rates per
    parameter and converges efficiently for neural networks.
    """

    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        predicted = (outputs > 0.0).float()  # threshold is 0.0 for raw logits
        return (predicted == labels).sum().item() / labels.size(0) * 100.0

    """
    NOTES TO INTERVIEWER:

    Because the model outputs raw logits (not probabilities), the decision
    boundary is 0.0 rather than 0.5. A logit above 0.0 maps to a probability
    above 0.5 after sigmoid, meaning the model predicts benign (class 1).
    """

    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
    ) -> tuple[float, float]:

        # --- Training ---
        self.model.train()
        self.optimizer.zero_grad()

        """
        NOTES TO INTERVIEWER:

        optimizer.zero_grad() must be called BEFORE the forward pass.
        PyTorch accumulates gradients by default — if you don't zero them,
        gradients from the previous step are added to the current step,
        corrupting the weight updates.
        """

        train_outputs = self.model(X_train)
        train_loss = self.criterion(train_outputs, y_train)
        train_loss.backward()
        self.optimizer.step()

        # --- Validation ---
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_loss = self.criterion(val_outputs, y_val)

        """
        NOTES TO INTERVIEWER:

        Validation loss is computed after optimizer.step() using a fresh
        forward pass. This ensures the logged metrics reflect the model
        AFTER the weight update, not before it.

        model.eval() disables dropout during validation so results are
        deterministic. torch.no_grad() disables gradient tracking, which
        saves memory and speeds up inference.
        """

        return train_loss.item(), val_loss.item()

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
        num_epochs: int = 100,
        log_every:  int = 10,
    ) -> None:
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7}")
        print("-" * 55)

        for epoch in range(num_epochs):
            train_loss, val_loss = self.train_epoch(X_train, y_train, X_val, y_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % log_every == 0:
                self.model.eval()
                with torch.no_grad():
                    train_acc = self._compute_accuracy(self.model(X_train), y_train)
                    val_acc   = self._compute_accuracy(self.model(X_val),   y_val)
                print(f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | {train_acc:>8.2f}% | {val_acc:>6.2f}%")

    def plot_losses(self, save_path: str = "loss_curve.png") -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"\nLoss curve saved to '{save_path}'")

    """
    NOTES TO INTERVIEWER:

    Plotting train vs validation loss is a key diagnostic tool.
    If validation loss starts rising while training loss keeps falling,
    the model is overfitting. In this project, both curves decrease
    together, which indicates the model generalizes well.
    """


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        test_outputs   = model(X_test)
        test_predicted = (test_outputs > 0.0).float()
        acc = (test_predicted == y_test).sum().item() / y_test.size(0) * 100.0

    print(f"\nTest Accuracy: {acc:.2f}%")

    y_pred = test_predicted.cpu().numpy().ravel()
    y_true = y_test.cpu().numpy().ravel()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["malignant", "benign"]))

"""
NOTES TO INTERVIEWER:

The confusion matrix shows:
- True Negatives  (top-left):  correctly predicted malignant
- False Positives (top-right): malignant predicted as benign
- False Negatives (bottom-left): benign predicted as malignant
- True Positives  (bottom-right): correctly predicted benign

In a medical context, false negatives (missing a malignant tumor) are
more dangerous than false positives. The classification report's recall
score for the malignant class is therefore the most clinically important metric.
"""


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    """
    NOTES TO INTERVIEWER:

    The if __name__ == '__main__' guard ensures this script only runs
    when executed directly, not when imported as a module. This is a
    Python best practice that makes code reusable and testable.
    """

    X_train, X_test, y_train, y_test = load_data()

    model   = BreastCancerNet(input_dim=X_train.shape[1], dropout_rate=0.3)
    trainer = Trainer(model, lr=0.001)

    trainer.fit(X_train, y_train, X_test, y_test, num_epochs=100, log_every=10)
    trainer.plot_losses("loss_curve.png")
    evaluate(model, X_test, y_test)
