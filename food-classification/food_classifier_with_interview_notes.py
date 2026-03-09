# food_classifier_with_interview_notes.py
"""
NOTES TO INTERVIEWER:

This project fine-tunes EfficientNetB0 on the Food-101 dataset for
101-class food image classification.

The core story this project tells:
- Training a CNN from scratch on 75,750 images would overfit badly
- ImageNet pretraining gives us a backbone that already understands
  edges, textures, shapes, and object parts
- Fine-tuning adapts those general visual features to food-specific patterns
- Two-stage training protects the pretrained weights during early training
  when the classification head is still random and noisy
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

"""
NOTES TO INTERVIEWER:

torchvision provides three things we use here:
1. datasets.Food101 — downloads and loads the Food-101 dataset with
   one line of code, handling caching automatically
2. transforms — composable image preprocessing and augmentation pipeline
3. models.efficientnet_b0 — pretrained EfficientNetB0 with ImageNet weights

EfficientNet_B0_Weights.IMAGENET1K_V1 is the recommended way to load
pretrained weights in torchvision 0.13+. The older weights=True syntax
is deprecated. Using the explicit weights enum also tells you exactly
which checkpoint you're loading, which matters for reproducibility.
"""


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE     = 42
DATA_DIR         = "./food101_data"
NUM_CLASSES      = 101
BATCH_SIZE       = 32
NUM_WORKERS      = 2
IMAGE_SIZE       = 224

STAGE1_EPOCHS    = 5
STAGE1_LR        = 1e-3

STAGE2_EPOCHS    = 10
STAGE2_LR        = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

"""
NOTES TO INTERVIEWER:

The learning rate difference between stages is critical and deliberate:

Stage 1 (lr=1e-3): Only the classification head is being trained.
The head starts with random weights and needs a relatively large learning
rate to converge quickly. The backbone is frozen so there is no risk of
corrupting the pretrained features.

Stage 2 (lr=1e-4): The entire network is now being trained. The backbone
weights are well-calibrated from ImageNet pretraining — aggressive updates
would destroy that prior knowledge. A 10x smaller learning rate makes
small, careful adjustments to adapt the backbone to food-specific features
without overwriting what was learned on 1.2M ImageNet images.

This is directly analogous to the BERT fine-tuning project — both use a
much lower learning rate when fine-tuning pretrained weights (2e-5 for BERT,
1e-4 for EfficientNet) for exactly the same reason.

IMAGENET_MEAN and IMAGENET_STD are the channel-wise mean and standard
deviation computed over the full ImageNet training set. Any model pretrained
on ImageNet expects its inputs normalized with these exact statistics.
Using different normalization would shift the input distribution and
degrade the pretrained features immediately.

BATCH_SIZE = 32 is a safe choice for the T4 GPU at 224x224 resolution.
Larger batches use more GPU memory but provide more stable gradient estimates.
"""


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def get_transforms() -> tuple:
    """
    Define image transforms for training and validation.

    Training uses augmentation to artificially expand the effective
    dataset size and improve generalization.
    Validation uses only deterministic transforms for consistent evaluation.
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

"""
NOTES TO INTERVIEWER:

Data augmentation is one of the most important regularization techniques
in computer vision. With only 750 training images per class, without
augmentation the model would memorize training images rather than learning
generalizable visual features.

Each augmentation is chosen to simulate real-world variation in food photos:

RandomResizedCrop(scale=0.7-1.0): Simulates the photographer being at
different distances from the food. The model learns to recognize a dish
whether it fills the entire frame or appears as part of a larger scene.

RandomHorizontalFlip: A plate of pasta looks the same mirrored left-right.
This effectively doubles the training data for symmetric subjects.

ColorJitter: Restaurant lighting varies enormously — warm incandescent,
cool fluorescent, natural daylight. Training with brightness/contrast/
saturation variation makes the model robust to these differences.

RandomRotation(15°): Food is often photographed at a slight angle.
15° is conservative — enough to add variation without distorting the image
beyond what you'd see in real photos.

Important: augmentation is applied ONLY to training data. Applying it to
validation would introduce randomness into the evaluation, making metrics
non-deterministic across runs. The val transform uses a fixed Resize(256)
followed by CenterCrop(224) — the standard ImageNet evaluation protocol.

Resize(256) then CenterCrop(224) rather than direct Resize(224): Resizing
to exactly 224 squishes the image slightly depending on aspect ratio.
Resizing to 256 first then cropping the center 224x224 preserves the
natural aspect ratio and focuses on the center of the image where the
food typically appears.
"""


def load_data(data_dir: str) -> tuple:
    """
    Download and load the Food-101 dataset via torchvision.

    Food-101 statistics:
        - 101 food categories
        - 750 training images per class (75,750 total)
        - 250 test images per class (25,250 total)
        - ~5GB download, cached after first run
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

"""
NOTES TO INTERVIEWER:

pin_memory=True is a GPU optimization. It allocates the DataLoader's
output tensors in pinned (page-locked) memory, which allows much faster
CPU-to-GPU data transfers. On a T4 GPU this meaningfully reduces the
data loading bottleneck during training.

num_workers=2 uses 2 background processes to load and preprocess images
in parallel while the GPU is running the forward/backward pass. Without
this, the GPU would sit idle waiting for the CPU to load each batch.

shuffle=True on the training loader randomizes batch composition each
epoch. Without shuffling, the model might see all 750 images of apple_pie
before all images of baby_back_ribs — this would cause the loss to spike
and fall as the class distribution shifts, producing unstable gradients.

shuffle=False on the validation loader is correct — evaluation order
doesn't affect accuracy, and deterministic ordering makes debugging easier.
"""


# ------------------------------------------------
# Model
# ------------------------------------------------

class FoodClassifier:
    """
    EfficientNetB0 fine-tuned for 101-class food classification.

    EfficientNet uses compound scaling — depth, width, and resolution
    are scaled together using a fixed ratio rather than independently.
    This produces better accuracy per parameter than scaling only one
    dimension (as earlier architectures like ResNet did).

    EfficientNetB0 is the baseline (B0 = compound coefficient 0).
    Larger variants (B1-B7) achieve higher accuracy at greater cost.
    B0 is the right choice for fine-tuning on a single GPU.
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.device      = device
        self.num_classes = num_classes
        self.history     = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace the classification head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

        self.model.to(device)
        print(f"Device: {device}")
        print(f"Total parameters:    {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable (stage 1): {sum(p.numel() for p in self.model.classifier.parameters()):,}")

    """
    NOTES TO INTERVIEWER:

    The original EfficientNetB0 classifier is:
        Sequential(Dropout(0.2), Linear(1280, 1000))

    We replace it with:
        Sequential(Dropout(0.4), Linear(1280, 101))

    Two changes:
    1. Output size: 1000 → 101 (ImageNet classes → food classes)
    2. Dropout rate: 0.2 → 0.4. With only 750 training images per class,
       the classification head is at higher risk of overfitting than it
       was on 1.2M ImageNet images. Stronger dropout compensates.

    in_features = self.model.classifier[1].in_features extracts the
    input dimension (1280) from the original head rather than hardcoding
    it. This makes the code robust to using a different EfficientNet
    variant — B1 through B7 have different feature dimensions.

    .to(device) moves the entire model to GPU memory. This must be called
    before creating the optimizer — if you move the model after creating
    the optimizer, the optimizer still holds references to the CPU tensors
    and the training loop will fail with device mismatch errors.
    """

    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers, keep classifier head trainable."""
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

    """
    NOTES TO INTERVIEWER:

    param.requires_grad = False tells PyTorch not to compute gradients
    for that parameter during backpropagation. This has two effects:
    1. The parameter is not updated by the optimizer
    2. PyTorch skips gradient computation for that parameter, saving
       memory and speeding up the backward pass

    In EfficientNetB0, self.model.features contains the convolutional
    backbone (all the pretrained layers). self.model.classifier is the
    head we just replaced.

    The two-stage pattern is important:
    - Stage 1: head weights are random → gradients are large and noisy
      → if backbone were unfrozen, these large gradients would corrupt
      the pretrained features in the first few batches
    - Stage 2: head has converged to reasonable weights → gradients are
      smaller and more meaningful → safe to update the backbone
    """

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """One training epoch. Returns avg loss and top-1 accuracy."""
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
        """One validation epoch. Returns avg loss and top-1 accuracy."""
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

    """
    NOTES TO INTERVIEWER:

    loss.item() * images.size(0) accumulates the total loss across all
    samples rather than averaging batch losses. This is important because
    the last batch of an epoch may be smaller than BATCH_SIZE — averaging
    batch losses would underweight earlier batches and overweight the last.
    Dividing by total at the end gives the correct per-sample average.

    CrossEntropyLoss with label_smoothing=0.1 replaces hard one-hot targets
    (e.g. [0, 0, 1, 0, ...]) with soft targets (e.g. [0.001, 0.001, 0.9,
    0.001, ...]). This prevents the model from becoming overconfident —
    a model that assigns 99.9% probability to one class is poorly calibrated
    and tends to overfit. Label smoothing is standard practice in modern
    image classification (used in EfficientNet's original training).
    """

    def fit_stage1(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Stage 1: freeze backbone, train head only."""
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

    def fit_stage2(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Stage 2: unfreeze all layers, full fine-tuning."""
        self._unfreeze_all()

        optimizer = Adam(self.model.parameters(), lr=STAGE2_LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS)

        print(f"\n{'='*55}")
        print(f"Stage 2 — Full Fine-Tuning ({STAGE2_EPOCHS} epochs)")
        print(f"{'='*55}")

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(STAGE2_EPOCHS):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc     = self._val_epoch(val_loader, criterion)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  ✓ New best model saved (val acc: {best_val_acc:.4f})")

            print(
                f"Epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best model — Val Acc: {best_val_acc:.4f}")

    """
    NOTES TO INTERVIEWER:

    filter(lambda p: p.requires_grad, self.model.parameters()) passes
    only the trainable parameters to the optimizer in Stage 1. Without
    this, the optimizer would track all parameters — even frozen ones —
    wasting memory and potentially causing subtle bugs if requires_grad
    changes mid-training.

    CosineAnnealingLR decays the learning rate following a cosine curve
    from the initial LR down to nearly 0 over T_max epochs. Compared to
    step decay (drop LR by 10x at fixed intervals), cosine annealing
    produces smoother convergence and consistently better final accuracy.
    It is now the default LR schedule in most modern CV training recipes.

    Best model checkpointing is important in Stage 2 because validation
    accuracy can occasionally dip slightly in later epochs as the model
    makes larger weight updates. Saving the best checkpoint and restoring
    it at the end ensures we report and use the genuinely best model,
    not just the last epoch.

    {k: v.clone() for k, v in self.model.state_dict().items()} creates
    a deep copy of the model weights. Without .clone(), the dictionary
    would store references to the current tensors — any subsequent weight
    update would overwrite the saved checkpoint in place.

    weight_decay=1e-4 in Stage 2 applies L2 regularization to all weights.
    This penalizes large weights and helps prevent overfitting when the
    full backbone is being updated on a relatively small dataset.
    """

    def evaluate(self, val_loader: DataLoader) -> None:
        """
        Final evaluation reporting top-1 and top-5 accuracy.

        Top-5 accuracy is the standard secondary metric for ImageNet-style
        classification. It measures whether the correct class appears
        anywhere in the model's top 5 predictions.
        """
        self.model.eval()
        top1_correct, top5_correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs        = self.model(images)
                top1_correct  += (outputs.argmax(dim=1) == labels).sum().item()
                top5_preds     = outputs.topk(5, dim=1).indices
                top5_correct  += sum(
                    labels[i].item() in top5_preds[i].tolist()
                    for i in range(labels.size(0))
                )
                total += labels.size(0)

        print(f"\n{'='*55}")
        print("Final Evaluation")
        print(f"{'='*55}")
        print(f"Top-1 Accuracy: {top1_correct / total:.4f}")
        print(f"Top-5 Accuracy: {top5_correct / total:.4f}")

    """
    NOTES TO INTERVIEWER:

    Top-5 accuracy is worth reporting on a 101-class problem because it
    gives a fairer picture of model quality. Some food categories are
    genuinely visually similar — beef carpaccio vs steak, or different
    types of cake — and a model that ranks the correct class 2nd or 3rd
    is still useful in practice (e.g. a food logging app that shows the
    top 3 suggestions).

    outputs.topk(5, dim=1).indices returns the indices of the 5 highest
    logits for each image in the batch. Checking whether the true label
    is among those 5 is O(5) per sample — very efficient.

    The published EfficientNetB0 result on Food-101 with full fine-tuning
    is approximately 85% top-1 accuracy. Our result should be in the
    75-85% range depending on the number of epochs and augmentation.
    """

    def plot_training_curves(self, save_path: str = "training_curves.png") -> None:
        """Plot loss and accuracy curves across both training stages."""
        epochs     = range(1, len(self.history["train_loss"]) + 1)
        stage1_end = STAGE1_EPOCHS

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"],   label="Val Loss")
        axes[0].axvline(x=stage1_end + 0.5, color="gray", linestyle="--", label="Stage 2 start")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

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

    """
    NOTES TO INTERVIEWER:

    The vertical dashed line marking the Stage 1 → Stage 2 transition
    is the most informative part of this plot. It shows:

    - Stage 1: validation accuracy climbs quickly as the head learns
      to use the frozen pretrained features. Train and val curves are
      close together because the frozen backbone prevents overfitting.

    - Stage 2: accuracy continues rising but more gradually as the
      backbone adapts. The gap between train and val may widen slightly
      as the model has more capacity to fit the training data.

    If val loss starts rising while train loss falls in Stage 2, that
    indicates overfitting — the remedy would be stronger augmentation,
    higher dropout, or fewer Stage 2 epochs.
    """

    def visualize_predictions(
        self,
        val_loader: DataLoader,
        class_names: list[str],
        save_path: str = "sample_predictions.png",
        n: int = 12,
    ) -> None:
        """Display n sample predictions — green for correct, red for incorrect."""
        self.model.eval()
        images_shown, preds_shown, labels_shown = [], [], []

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

    """
    NOTES TO INTERVIEWER:

    inv_normalize reverses the ImageNet normalization before displaying
    images. Without this, pixel values would be in the normalized range
    (roughly -2 to +2) and the images would display as noise.

    The inverse of Normalize(mean, std) is Normalize(-mean/std, 1/std) —
    we compute this analytically rather than storing the original images.

    .clip(0, 1) clamps the pixel values to the valid display range after
    inverse normalization. Floating point arithmetic can produce values
    slightly outside [0, 1] which matplotlib would clamp anyway, but
    doing it explicitly avoids a warning.

    image.permute(1, 2, 0) reorders tensor dimensions from PyTorch's
    (C, H, W) format to matplotlib's expected (H, W, C) format.

    This visualization is particularly valuable for food classification
    because incorrect predictions are often visually interesting —
    the model might confuse beef carpaccio with steak, or cannoli with
    churros. These failure modes reveal what visual features the model
    relies on and suggest directions for improvement.
    """


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    """
    NOTES TO INTERVIEWER:

    The pipeline is structured so each stage is completely independent
    and could be run separately with a saved checkpoint. In a production
    workflow you would save the model after Stage 1 and Stage 2 separately
    so you could resume from either point without retraining from scratch.

    torch.manual_seed ensures reproducible weight initialization for the
    classification head. The backbone weights are loaded from a fixed
    checkpoint so they are already deterministic.
    """

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
