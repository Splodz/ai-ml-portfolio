# breast_cancer_nn.py
"""
Breast Cancer Classification — Feedforward Neural Network
==========================================================
Binary classification of breast tumors as malignant or benign
using a feedforward neural network trained on the UCI Breast Cancer
Wisconsin dataset.

Pipeline:
    1. Load dataset from scikit-learn
    2. Stratified train/test split
    3. Feature scaling with StandardScaler
    4. Two-stage feedforward network with dropout regularization
    5. Training with BCEWithLogitsLoss and Adam optimizer
    6. Evaluation (accuracy, F1, confusion matrix, classification report)
    7. Training/validation loss curve visualization
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
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS    = 100
DROPOUT_RATE  = 0.3
LOG_EVERY     = 10

torch.manual_seed(RANDOM_STATE)


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the UCI Breast Cancer Wisconsin dataset and prepare tensors.

    Dataset:
        - 569 samples, 30 numeric features
        - Labels: 0 = malignant, 1 = benign
        - Class balance: ~37% malignant, ~63% benign

    Preprocessing:
        - Stratified split preserves class balance in both sets
        - StandardScaler fitted on training data only (no leakage)
        - Labels reshaped to (-1, 1) to match single output neuron

    Returns:
        X_train, X_test, y_train, y_test as float32 tensors
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features — fit on training data only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    to_label  = lambda arr: torch.tensor(arr, dtype=torch.float32).view(-1, 1)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Features:      {X_train.shape[1]}")

    return to_tensor(X_train), to_tensor(X_test), to_label(y_train), to_label(y_test)


# ------------------------------------------------
# Model
# ------------------------------------------------

class BreastCancerNet(nn.Module):
    """
    Feedforward neural network for binary breast cancer classification.

    Architecture:
        Input (30 features)
            → Linear(30, 32) → ReLU → Dropout(0.3)
            → Linear(32, 16) → ReLU → Dropout(0.3)
            → Linear(16, 1)  [raw logit output]

    The output is a raw logit — no sigmoid applied in forward().
    BCEWithLogitsLoss handles the sigmoid internally for numerical stability,
    using the log-sum-exp trick to avoid floating point issues when outputs
    approach exactly 0 or 1.

    Dropout randomly zeros a fraction of outputs during training, forcing
    the network to learn redundant representations and reducing overfitting
    on this small dataset (455 training samples).
    """

    def __init__(self, input_dim: int, dropout_rate: float = DROPOUT_RATE):
        super(BreastCancerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            # No sigmoid — BCEWithLogitsLoss handles it internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ------------------------------------------------
# Trainer
# ------------------------------------------------

class Trainer:
    """
    Encapsulates the training loop, loss tracking, and visualization.

    Separating training logic from model definition follows the single
    responsibility principle and mirrors patterns used in production
    ML codebases.
    """

    def __init__(self, model: nn.Module, lr: float = LEARNING_RATE):
        self.model     = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    def _accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy using logit threshold of 0.0 (equivalent to prob > 0.5)."""
        predicted = (outputs > 0.0).float()
        return (predicted == labels).sum().item() / labels.size(0) * 100.0

    def _train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val:   torch.Tensor,
        y_val:   torch.Tensor,
    ) -> tuple[float, float]:
        """Run one training epoch and return train/val loss."""

        # Training pass
        self.model.train()
        self.optimizer.zero_grad()
        train_outputs = self.model(X_train)
        train_loss    = self.criterion(train_outputs, y_train)
        train_loss.backward()
        self.optimizer.step()

        # Validation pass — no gradient tracking, dropout disabled
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_loss    = self.criterion(val_outputs, y_val)

        return train_loss.item(), val_loss.item()

    def fit(
        self,
        X_train:    torch.Tensor,
        y_train:    torch.Tensor,
        X_val:      torch.Tensor,
        y_val:      torch.Tensor,
        num_epochs: int = NUM_EPOCHS,
        log_every:  int = LOG_EVERY,
    ) -> None:
        """Train the model and log progress every log_every epochs."""
        print(f"\n{'='*55}")
        print(f"Training — {num_epochs} epochs")
        print(f"{'='*55}")
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7}")
        print("-" * 55)

        for epoch in range(num_epochs):
            train_loss, val_loss = self._train_epoch(X_train, y_train, X_val, y_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % log_every == 0:
                self.model.eval()
                with torch.no_grad():
                    train_acc = self._accuracy(self.model(X_train), y_train)
                    val_acc   = self._accuracy(self.model(X_val),   y_val)
                print(
                    f"{epoch+1:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
                    f"{train_acc:>8.2f}% | {val_acc:>6.2f}%"
                )

    def plot_losses(self, save_path: str = "loss_curve.png") -> None:
        """Plot and save training vs validation loss curves."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.train_losses, label="Train Loss", linewidth=2)
        plt.plot(self.val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("BCEWithLogitsLoss")
        plt.title("Training vs Validation Loss — Breast Cancer NN")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to '{save_path}'")


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> None:
    """
    Final evaluation on held-out test set.

    Reports accuracy, F1, confusion matrix, and full classification report.
    Decision threshold is 0.0 (logit space), equivalent to 0.5 in probability space.

    In a medical context, recall on the malignant class is the most
    clinically important metric — a missed malignant tumor (false negative)
    is more dangerous than a false alarm.
    """
    model.eval()
    with torch.no_grad():
        outputs   = model(X_test)
        predicted = (outputs > 0.0).float()

    y_pred = predicted.cpu().numpy().ravel()
    y_true = y_test.cpu().numpy().ravel()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*55}")
    print("Final Evaluation")
    print(f"{'='*55}")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Weighted F1:    {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["malignant", "benign"]))


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Initialize model and trainer
    model   = BreastCancerNet(input_dim=X_train.shape[1], dropout_rate=DROPOUT_RATE)
    trainer = Trainer(model, lr=LEARNING_RATE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Train
    trainer.fit(X_train, y_train, X_test, y_test, num_epochs=NUM_EPOCHS)

    # 4. Plot loss curves
    trainer.plot_losses("loss_curve.png")

    # 5. Final evaluation
    evaluate(model, X_test, y_test)
