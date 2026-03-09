# food_classifier.py
"""
Food-101 Image Classification — EfficientNetB0 Transfer Learning
=================================================================
101-class food image classification using a pretrained EfficientNetB0
backbone fine-tuned on the Food-101 dataset.

Two-stage training strategy:
    Stage 1 — Feature Extraction:
        Freeze the EfficientNetB0 backbone.
        Train only the classification head for 5 epochs.
        Allows the head to learn meaningful representations before
        the backbone weights are disturbed.

    Stage 2 — Full Fine-Tuning:
        Unfreeze the entire network.
        Train end-to-end at a much lower learning rate for 10 epochs.
        Allows the pretrained features to adapt to food-specific patterns.

Pipeline:
    1. Load Food-101 via torchvision datasets
    2. Data augmentation (train) and normalization (train + val)
    3. Load pretrained EfficientNetB0, replace classifier head
    4. Stage 1: freeze backbone, train head
    5. Stage 2: unfreeze all, fine-tune end-to-end
    6. Evaluate (top-1 and top-5 accuracy)
    7. Visualize predictions on sample images
    8. Plot training curves
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE     = 42
DATA_DIR         = "./food101_data"
NUM_CLASSES      = 101
BATCH_SIZE       = 32
NUM_WORKERS      = 2
IMAGE_SIZE       = 224           # EfficientNetB0 default input size

# Stage 1 — head only
STAGE1_EPOCHS    = 5
STAGE1_LR        = 1e-3          # Higher LR — only training the head

# Stage 2 — full fine-tuning
STAGE2_EPOCHS    = 10
STAGE2_LR        = 1e-4          # Lower LR — fine-tuning pretrained weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization stats — required for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def get_transforms() -> tuple:
    """
    Define image transforms for training and validation.

    Training: augmentation + normalization
    Validation: resize/crop + normalization only (no augmentation)

    Returns:
        train_transform, val_transform
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def load_data(data_dir: str) -> tuple:
    """
    Download and load the Food-101 dataset.

    Food-101:
        - 101 food categories
        - 750 training images per class (75,750 total)
        - 250 test images per class (25,250 total)
        - Downloaded automatically on first run (~5GB)

    Args:
        data_dir: Directory to download/cache the dataset.

    Returns:
        train_loader, val_loader, class_names
    """
    train_transform, val_transform = get_transforms()

    print("Loading Food-101 dataset (downloading if first run — ~5GB)...")
    train_dataset = datasets.Food101(
        root=data_dir, split="train", transform=train_transform, download=True
    )
    val_dataset = datasets.Food101(
        root=data_dir, split="test", transform=val_transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    print(f"Classes:       {len(class_names)}")

    return train_loader, val_loader, class_names


# ------------------------------------------------
# Model
# ------------------------------------------------

class FoodClassifier:
    """
    EfficientNetB0 fine-tuned for 101-class food classification.

    EfficientNet scales model depth, width, and resolution together
    using a compound coefficient — achieving better accuracy per FLOP
    than ResNet architectures. EfficientNetB0 is the smallest variant,
    well-suited for fine-tuning on a single GPU.

    Two-stage training:
        Stage 1: Freeze backbone → train head only (fast convergence)
        Stage 2: Unfreeze all → fine-tune end-to-end (best accuracy)
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.device     = device
        self.num_classes = num_classes
        self.history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Load pretrained EfficientNetB0
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace the classifier head
        # Original: Linear(1280, 1000) for ImageNet
        # Ours:     Dropout(0.4) → Linear(1280, 101)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

        self.model.to(device)
        print(f"Device: {device}")
        print(f"Total parameters:    {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable (stage 1): {sum(p.numel() for p in self.model.classifier.parameters()):,}")

    def _freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head."""
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        print("\nBackbone frozen — training head only")

    def _unfreeze_all(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("\nAll layers unfrozen — full fine-tuning")

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one training epoch. Returns avg loss and top-1 accuracy."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total

    def _val_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one validation epoch. Returns avg loss and top-1 accuracy."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs     = self.model(images)
                loss        = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (outputs.argmax(dim=1) == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, correct / total

    def fit_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Stage 1: Feature extraction.
        Freeze backbone, train classification head only.
        """
        self._freeze_backbone()

        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=STAGE1_LR,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)

        print(f"\n{'='*55}")
        print(f"Stage 1 — Feature Extraction ({STAGE1_EPOCHS} epochs)")
        print(f"{'='*55}")

        for epoch in range(STAGE1_EPOCHS):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc     = self._val_epoch(val_loader, criterion)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch+1:02d}/{STAGE1_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    def fit_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Stage 2: Full fine-tuning.
        Unfreeze all layers, train end-to-end at lower learning rate.
        """
        self._unfreeze_all()

        optimizer = Adam(self.model.parameters(), lr=STAGE2_LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

        print(f"\n{'='*55}")
        print(f"Stage 2 — Full Fine-Tuning ({STAGE2_EPOCHS} epochs)")
        print(f"{'='*55}")

        best_val_acc  = 0.0
        best_state    = None

        for epoch in range(STAGE2_EPOCHS):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc     = self._val_epoch(val_loader, criterion)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  ✓ New best model saved (val acc: {best_val_acc:.4f})")

            print(
                f"Epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best model — Val Acc: {best_val_acc:.4f}")

    def evaluate(self, val_loader: DataLoader) -> None:
        """
        Final evaluation: top-1 and top-5 accuracy.

        Top-5 accuracy is standard for ImageNet-style classification —
        it measures whether the correct class appears in the model's
        top 5 predictions. On a 101-class problem this is a meaningful
        secondary metric.
        """
        self.model.eval()
        top1_correct, top5_correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs        = self.model(images)

                # Top-1
                top1_correct += (outputs.argmax(dim=1) == labels).sum().item()

                # Top-5
                top5_preds    = outputs.topk(5, dim=1).indices
                top5_correct += sum(
                    labels[i].item() in top5_preds[i].tolist()
                    for i in range(labels.size(0))
                )
                total += labels.size(0)

        print(f"\n{'='*55}")
        print("Final Evaluation")
        print(f"{'='*55}")
        print(f"Top-1 Accuracy: {top1_correct / total:.4f}")
        print(f"Top-5 Accuracy: {top5_correct / total:.4f}")

    def plot_training_curves(self, save_path: str = "training_curves.png") -> None:
        """Plot loss and accuracy curves for both training stages."""
        epochs   = range(1, len(self.history["train_loss"]) + 1)
        stage1_end = STAGE1_EPOCHS

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"],   label="Val Loss")
        axes[0].axvline(x=stage1_end + 0.5, color="gray", linestyle="--", label="Stage 2 start")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], label="Train Acc")
        axes[1].plot(epochs, self.history["val_acc"],   label="Val Acc")
        axes[1].axvline(x=stage1_end + 0.5, color="gray", linestyle="--", label="Stage 2 start")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.suptitle("EfficientNetB0 — Food-101 Training Curves", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved to '{save_path}'")

    def visualize_predictions(
        self,
        val_loader: DataLoader,
        class_names: list[str],
        save_path: str = "sample_predictions.png",
        n: int = 12,
    ) -> None:
        """
        Display n sample validation images with predicted and true labels.
        Correct predictions shown in green, incorrect in red.
        """
        self.model.eval()
        images_shown, preds_shown, labels_shown = [], [], []

        _, val_transform = get_transforms()
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
            std=[1 / s for s in IMAGENET_STD],
        )

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = self.model(images.to(self.device))
                preds   = outputs.argmax(dim=1).cpu()
                for i in range(images.size(0)):
                    if len(images_shown) >= n:
                        break
                    images_shown.append(inv_normalize(images[i]))
                    preds_shown.append(preds[i].item())
                    labels_shown.append(labels[i].item())
                if len(images_shown) >= n:
                    break

        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        axes = axes.flatten()

        for i in range(n):
            img = images_shown[i].permute(1, 2, 0).numpy().clip(0, 1)
            axes[i].imshow(img)
            correct = preds_shown[i] == labels_shown[i]
            color   = "green" if correct else "red"
            axes[i].set_title(
                f"Pred: {class_names[preds_shown[i]]}\nTrue: {class_names[labels_shown[i]]}",
                color=color, fontsize=8,
            )
            axes[i].axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Sample Predictions (Green = Correct, Red = Incorrect)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sample predictions saved to '{save_path}'")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    torch.manual_seed(RANDOM_STATE)
    print(f"Device: {DEVICE}")

    # 1. Load data
    train_loader, val_loader, class_names = load_data(DATA_DIR)

    # 2. Initialize model
    classifier = FoodClassifier(NUM_CLASSES, DEVICE)

    # 3. Stage 1 — train head only
    classifier.fit_stage1(train_loader, val_loader)

    # 4. Stage 2 — full fine-tuning
    classifier.fit_stage2(train_loader, val_loader)

    # 5. Final evaluation
    classifier.evaluate(val_loader)

    # 6. Training curves
    classifier.plot_training_curves()

    # 7. Sample predictions
    classifier.visualize_predictions(val_loader, class_names)
