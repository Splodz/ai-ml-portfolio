# AI / Machine Learning Portfolio

This repository contains a curated collection of **machine learning and artificial intelligence projects** developed using Python, PyTorch, and Hugging Face Transformers.

Each project is self-contained and includes a clean production-style script, an annotated version documenting design decisions, and a dedicated README explaining the problem, methodology, and results.

---

## 🧠 Skills & Technologies

- **Programming:** Python
- **Frameworks:** PyTorch, scikit-learn, Hugging Face Transformers, XGBoost, torchvision
- **Machine Learning:** Supervised learning, classification, class imbalance handling (SMOTE), model evaluation
- **Deep Learning:** Feedforward neural networks, CNNs, transformer fine-tuning (BERT)
- **Computer Vision:** Transfer learning, data augmentation, two-stage fine-tuning, EfficientNet
- **NLP:** TF-IDF, bigram features, WordPiece tokenization, attention visualization
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Data Processing:** Feature engineering, missing value handling, label encoding, stratified splitting
- **Tools:** Git, GitHub, Google Colab

---

## 📊 Results at a Glance

| # | Project | Model | Key Result |
|---|---------|-------|------------|
| 1 | Breast Cancer Classification | Feedforward NN | **94.74% accuracy · 0.95 weighted F1** |
| 2 | IMDb Sentiment — Baseline | TF-IDF + Logistic Regression | **88.07% accuracy · 0.88 F1** |
| 3 | Hospital Readmission Prediction | XGBoost + SHAP | **AUC-ROC 0.58 · F1 0.19** on 101,766 clinical encounters (imbalanced, ~11% positive class) |
| 4 | IMDb Sentiment — BERT | bert-base-uncased fine-tuned | **92.15% accuracy · 0.9219 F1 · 0.9752 AUC-ROC** (+3.2% over TF-IDF baseline) |
| 5 | Food-101 Classification | EfficientNetB0 | **86.07% Top-1 · 97.05% Top-5** (beats original paper ~77%) |

---

## 📂 Projects

### 1️⃣ Breast Cancer Classification — Neural Network
**Folder:** `breast-cancer-classification/`

A feedforward neural network trained to classify breast tumors as **benign** or **malignant** using structured medical data from the UCI Breast Cancer Wisconsin dataset.

**Key concepts:** Binary classification · BCEWithLogitsLoss · Dropout regularization · Confusion matrix · PyTorch Trainer class

---

### 2️⃣ IMDb Sentiment Analysis — TF-IDF + Logistic Regression
**Folder:** `imdb-sentiment-analysis/`

Classical NLP pipeline for binary sentiment classification on 50,000 IMDb movie reviews. Establishes a strong baseline that the BERT project directly builds on.

**Key concepts:** TF-IDF with bigrams · Data leakage prevention · Feature coefficient analysis · Error analysis · SentimentClassifier class

---

### 3️⃣ Hospital Readmission Prediction — XGBoost + SHAP
**Folder:** `hospital-readmission/`

End-to-end ML pipeline on a real-world clinical dataset (101,766 hospital encounters) predicting 30-day readmission for diabetic patients. Demonstrates messy data handling, domain-informed feature engineering, class imbalance handling, and explainable AI in a healthcare context.

**Key concepts:** ICD-9 feature engineering · SMOTE oversampling · XGBoost · F1 and AUC-ROC on imbalanced data · SHAP interpretability · Clinical evaluation metrics

---

### 4️⃣ IMDb Sentiment Analysis — BERT Fine-Tuning
**Folder:** `bert-sentiment-analysis/`

Fine-tuned `bert-base-uncased` on the IMDb dataset for binary sentiment classification. Direct comparison to the classical TF-IDF baseline — same dataset, fundamentally different paradigm. Includes attention weight visualization showing what BERT focuses on per token.

**Key concepts:** Transfer learning · WordPiece tokenization · AdamW with linear warmup · Gradient clipping · Attention visualization · Transformer fine-tuning

---

### 5️⃣ Food-101 Image Classification — EfficientNetB0
**Folder:** `food-classification/`

101-class food image classification using a pretrained EfficientNetB0 backbone fine-tuned on the Food-101 dataset (101,000 real-world food images). Achieves **86.07% Top-1 and 97.05% Top-5 accuracy**, exceeding the original Food-101 benchmark (~77%). Demonstrates two-stage transfer learning, data augmentation, and top-1/top-5 evaluation.

**Key concepts:** Two-stage fine-tuning · Data augmentation · ImageNet normalization · Label smoothing · Cosine annealing · Best model checkpointing · Top-1 and Top-5 accuracy

---

## 📌 Notes

- Each project is organized in its own folder with a dedicated README.
- Annotated scripts (`*_with_interview_notes.py`) document the reasoning behind every design decision.
- This portfolio is intended for academic, professional, and interview review.

---

## 👤 Author

Graduate student in Artificial Intelligence with a focus on machine learning and deep learning systems.
