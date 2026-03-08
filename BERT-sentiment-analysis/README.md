# IMDb Sentiment Analysis — BERT Fine-Tuning

This project fine-tunes **bert-base-uncased** on the IMDb movie reviews dataset for binary sentiment classification. It serves as a direct comparison to the classical [TF-IDF + Logistic Regression approach](../imdb-sentiment-analysis/), demonstrating how transformer-based transfer learning improves on classical NLP baselines.

---

## 📌 Problem Statement

Given a movie review written in natural language, predict whether the sentiment is:

- **Positive (1)**
- **Negative (0)**

Same task as the classical IMDb project — different paradigm entirely.

---

## 🧠 Key Concepts Demonstrated

- Transfer learning with a pretrained transformer (BERT)
- WordPiece subword tokenization
- Fine-tuning with AdamW optimizer and linear learning rate warmup
- Gradient clipping for stable transformer training
- Attention weight visualization — what BERT focuses on per token
- Direct model comparison against classical TF-IDF baseline

---

## 🔧 Pipeline Overview

```
Raw IMDb Reviews (25,000 train / 25,000 test)
        ↓
WordPiece Tokenization (max 256 tokens)
        ↓
bert-base-uncased (pretrained, 110M parameters)
+ Linear Classification Head ([CLS] token → 2 classes)
        ↓
Fine-Tuning (3 epochs, AdamW, lr=2e-5, warmup)
        ↓
Evaluation (Accuracy, F1, AUC-ROC)
        ↓
Attention Visualization
```

---

## 🤖 Model: BERT

BERT (Bidirectional Encoder Representations from Transformers) is pretrained on BookCorpus and English Wikipedia (~3.3 billion words) using:

- **Masked Language Modeling (MLM):** predict randomly masked tokens
- **Next Sentence Prediction (NSP):** predict whether two sentences are consecutive

Fine-tuning adds a linear classification head on top of the `[CLS]` token representation and trains the entire model end-to-end on the target task.

### Why BERT outperforms TF-IDF

| Limitation of TF-IDF | How BERT addresses it |
|---------------------|----------------------|
| "not good" treated as two independent tokens | Bidirectional attention captures negation context |
| Word meaning is fixed regardless of context | Contextual embeddings change per sentence |
| Vocabulary limited to training corpus | WordPiece handles unseen words via subword units |
| No world knowledge | Pretrained on 3.3B words of general text |

---

## ⚙️ Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model` | bert-base-uncased | 12 layers, 110M params — strong balance of speed and accuracy |
| `max_length` | 256 | Covers most reviews; full 512 doubles memory with minimal gain |
| `batch_size` | 16 | Safe for T4 GPU at max_length=256 |
| `epochs` | 3 | Standard for BERT fine-tuning; more risks overfitting |
| `learning_rate` | 2e-5 | Lower end of recommended range (2e-5 to 5e-5) for stability |
| `warmup_ratio` | 0.1 | 10% of steps used for LR warmup — stabilizes early training |
| `weight_decay` | 0.01 | L2 regularization via AdamW |
| `grad_clip` | 1.0 | Prevents exploding gradients during backprop through 12 layers |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | — |
| F1 Score | — |
| AUC-ROC | — |

> Results to be filled in after running `bert_sentiment.py`

### Confusion Matrix

```text
Results to be added after running the script
```

---

## 🔬 Model Comparison

| Model | Accuracy | F1 | AUC-ROC |
|-------|----------|----|---------|
| TF-IDF + Logistic Regression | ~0.89 | ~0.89 | ~0.96 |
| BERT (bert-base-uncased) | — | — | — |

> BERT results to be filled in after running the script

---

## 👁️ Attention Visualization

BERT's attention mechanism shows which tokens the model focuses on when making a prediction. The plots below show `[CLS]` token attention weights from the final transformer layer for a positive and negative review sample.

![Attention Positive](attention_positive.png)
![Attention Negative](attention_negative.png)

Green = high attention weight, Red = low attention weight.

---

## 🗂️ Dataset

- **Source:** [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb) via Hugging Face
- **Size:** 50,000 reviews (25,000 train / 25,000 test)
- **Balance:** 50% positive, 50% negative
- **Split:** 90% train / 10% validation from training set; test set held out

---

## 📂 Project Structure

```text
bert-sentiment-analysis/
│
├── bert_sentiment.py                      # Main fine-tuning and evaluation script
├── bert_sentiment_with_interview_notes.py # Annotated version with design rationale
├── attention_positive.png                 # Attention heatmap — positive review
├── attention_negative.png                 # Attention heatmap — negative review
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Splodz/ai-ml-portfolio.git
cd ai-ml-portfolio/bert-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run on GPU (recommended):

```bash
python bert_sentiment.py
```

> Note: A CUDA-capable GPU is strongly recommended. Fine-tuning on CPU will take several hours.

---

## 📦 Requirements

```text
torch
transformers
datasets
scikit-learn
numpy
matplotlib
```

---

## 👤 Author

Graduate student in Artificial Intelligence with a focus on machine learning and deep learning systems.
