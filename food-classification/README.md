# Food-101 Image Classification — EfficientNetB0 Transfer Learning

This project fine-tunes **EfficientNetB0** on the Food-101 dataset for 101-class food image classification. It demonstrates a complete computer vision pipeline including data augmentation, two-stage transfer learning, and top-1/top-5 evaluation — the standard approach for image classification in industry.

---

## 📌 Problem Statement

Given a photo of food, classify it into one of **101 categories** — from apple pie to waffles.

- **Dataset:** Food-101 — 101,000 real-world food images
- **Task:** 101-class image classification
- **Challenge:** High intra-class variation (the same dish can look very different) and inter-class similarity (many dishes share visual features)

---

## 🧠 Key Concepts Demonstrated

- Transfer learning with a pretrained convolutional neural network
- Two-stage fine-tuning: feature extraction → full fine-tuning
- Data augmentation for improved generalization
- ImageNet normalization for pretrained models
- Label smoothing regularization
- Cosine annealing learning rate schedule
- Best model checkpointing
- Top-1 and Top-5 accuracy evaluation
- Sample prediction visualization

---

## 🔧 Pipeline Overview

```
Food-101 Dataset (101,000 images, 101 classes)
        ↓
Data Augmentation (train) / Normalization (train + val)
RandomResizedCrop · RandomHorizontalFlip · ColorJitter · RandomRotation
        ↓
EfficientNetB0 Backbone (pretrained on ImageNet)
        ↓
Custom Classification Head
Dropout(0.4) → Linear(1280 → 101)
        ↓
Stage 1 — Feature Extraction (5 epochs, lr=1e-3)
Backbone frozen → train head only
        ↓
Stage 2 — Full Fine-Tuning (10 epochs, lr=1e-4)
All layers unfrozen → train end-to-end
        ↓
Evaluation (Top-1 Accuracy, Top-5 Accuracy)
        ↓
Training Curves + Sample Predictions
```

---

## 🗂️ Dataset

- **Source:** [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) via torchvision
- **Size:** 101,000 images across 101 food categories
- **Split:** 750 training images per class (75,750 total) / 250 test images per class (25,250 total)
- **Image size:** Resized and cropped to 224×224 (EfficientNetB0 default)
- **Download:** ~5GB, handled automatically by torchvision on first run

---

## 🖼️ Data Augmentation

Augmentation is applied only to the training set. The validation set uses a deterministic resize and center crop to ensure consistent evaluation.

| Transform | Value | Rationale |
|-----------|-------|-----------|
| `RandomResizedCrop` | 224px, scale (0.7–1.0) | Simulates different distances and framings |
| `RandomHorizontalFlip` | p=0.5 | Food images are horizontally symmetric |
| `ColorJitter` | brightness/contrast/saturation ±0.3 | Simulates lighting variation |
| `RandomRotation` | ±15° | Handles dishes photographed at an angle |
| `Normalize` | ImageNet mean/std | Required for pretrained backbone |

---

## 🤖 Model: EfficientNetB0

EfficientNet scales model depth, width, and resolution jointly using a compound coefficient — achieving better accuracy per parameter than ResNet architectures. EfficientNetB0 is the smallest and fastest variant, making it ideal for fine-tuning on a single GPU.

| Property | Value |
|----------|-------|
| Backbone | EfficientNetB0 (ImageNet pretrained) |
| Backbone output | 1,280-dimensional feature vector |
| Classification head | Dropout(0.4) → Linear(1280, 101) |
| Total parameters | 4,136,929 |
| Trainable params (Stage 1) | 129,381 (head only) |
| Pretrained on | ImageNet-1K (1.2M images, 1,000 classes) |

The original ImageNet head (Linear(1280, 1000)) is replaced with a task-specific head for 101 food classes.

---

## ⚙️ Two-Stage Training Strategy

Training in two stages is the industry best practice for transfer learning. It prevents the randomly initialized classification head from corrupting the pretrained backbone weights during early training.

### Stage 1 — Feature Extraction

| Parameter | Value |
|-----------|-------|
| Backbone | Frozen |
| Trainable params | 129,381 (head only) |
| Epochs | 5 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| LR schedule | Cosine annealing |
| Loss | CrossEntropy + label smoothing (0.1) |

### Stage 2 — Full Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Backbone | Unfrozen |
| Trainable params | 4,136,929 (entire network) |
| Epochs | 10 |
| Learning rate | 1e-4 |
| Optimizer | Adam + weight decay (1e-4) |
| LR schedule | Cosine annealing |
| Loss | CrossEntropy + label smoothing (0.1) |
| Checkpointing | Best val accuracy saved and restored |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **86.07%** |
| Top-5 Accuracy | **97.05%** |

> The original Food-101 benchmark paper achieved ~77% Top-1 accuracy. This implementation exceeds that by ~9 percentage points using EfficientNetB0 with a two-stage fine-tuning strategy.

### Training Log

**Stage 1 — Feature Extraction**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01/05 | 3.1950 | 0.3424 | 2.4408 | 0.5390 |
| 02/05 | 2.9074 | 0.4078 | 2.3625 | 0.5558 |
| 03/05 | 2.8559 | 0.4239 | 2.3353 | 0.5652 |
| 04/05 | 2.8022 | 0.4355 | 2.3075 | 0.5745 |
| 05/05 | 2.7695 | 0.4440 | 2.3087 | 0.5777 |

**Stage 2 — Full Fine-Tuning**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01/10 | 2.1886 | 0.6078 | 1.6176 | 0.7771 ✓ |
| 02/10 | 1.8578 | 0.7048 | 1.4959 | 0.8148 ✓ |
| 03/10 | 1.7079 | 0.7503 | 1.4321 | 0.8334 ✓ |
| 04/10 | 1.6006 | 0.7812 | 1.3975 | 0.8404 ✓ |
| 05/10 | 1.5188 | 0.8067 | 1.3631 | 0.8484 ✓ |
| 06/10 | 1.4532 | 0.8253 | 1.3458 | 0.8530 ✓ |
| 07/10 | 1.3997 | 0.8448 | 1.3367 | 0.8568 ✓ |
| 08/10 | 1.3612 | 0.8574 | 1.3216 | 0.8605 ✓ |
| 09/10 | 1.3413 | 0.8643 | 1.3199 | 0.8607 ✓ |
| 10/10 | 1.3273 | 0.8676 | 1.3157 | 0.8607 |

✓ = new best model checkpoint saved

### Training Curves

![Training Curves](training_curves.png)

The dashed vertical line marks the transition from Stage 1 to Stage 2.

---

## 🍽️ Sample Predictions

![Sample Predictions](sample_predictions.png)

Green = correct prediction · Red = incorrect prediction

---

## 📂 Project Structure

```text
food-classification/
│
├── food_classifier.py                      # Main training and evaluation script
├── food_classifier_with_interview_notes.py # Annotated version with design rationale
├── training_curves.png                     # Loss and accuracy curves
├── sample_predictions.png                  # Sample predictions grid
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Splodz/ai-ml-portfolio.git
cd ai-ml-portfolio/food-classification
pip install -r requirements.txt
python food_classifier.py
```

> A CUDA-capable GPU is strongly recommended. The Food-101 dataset (~5GB) will download automatically on first run.

---

## 📦 Requirements

```text
torch
torchvision
numpy
matplotlib
```

---

## 👤 Author

Graduate student in Artificial Intelligence with a focus on machine learning and deep learning systems.
