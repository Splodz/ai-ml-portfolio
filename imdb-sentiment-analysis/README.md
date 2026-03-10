# IMDb Sentiment Analysis — TF-IDF + Logistic Regression

Classical NLP pipeline for binary sentiment classification on 50,000 IMDb movie reviews. Establishes a strong baseline that the [BERT fine-tuning project](../BERT-sentiment-analysis/) directly builds on — same dataset, fundamentally different paradigm.

---

## 📌 Problem Statement

Given a movie review written in natural language, predict whether the sentiment expressed is **positive** or **negative**.

- **Dataset:** IMDb Large Movie Review Dataset — 50,000 reviews
- **Task:** Binary sentiment classification
- **Balance:** Perfectly balanced — 50% positive, 50% negative

---

## 🧠 Key Concepts Demonstrated

- Supervised learning and binary text classification
- Text preprocessing (lowercasing, punctuation removal, stopword filtering)
- TF-IDF feature extraction with unigrams and bigrams
- Data leakage prevention — vectorizer fitted on training data only
- Logistic Regression for high-dimensional sparse text features
- Hyperparameter tuning and rationale
- Model evaluation: accuracy, F1-score, confusion matrix
- Error analysis through misclassified example inspection
- Model interpretability through learned feature coefficients

---

## 🔧 Pipeline Overview

```
Raw IMDb Reviews (25,000 train / 25,000 test)
        ↓
Preprocessing
(lowercase → remove non-alpha → collapse whitespace → stopword removal)
        ↓
TF-IDF Vectorization
(unigrams + bigrams, 30,000 features, min_df=5, max_df=0.9)
        ↓
Logistic Regression Classifier (C=2.0)
        ↓
Evaluation (Accuracy, F1, Confusion Matrix)
        ↓
Error Analysis + Feature Coefficient Interpretability
```

---

## 🗂️ Dataset

- **Source:** [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb) via Hugging Face
- **Size:** 50,000 reviews (25,000 train / 25,000 test)
- **Balance:** 50% positive, 50% negative
- **Split:** 80% train / 20% validation carved from training set; test set held out until final evaluation

---

## 🖊️ Text Preprocessing

Each review is normalized before vectorization:

1. Lowercase all text
2. Remove non-alphabetic characters (punctuation, HTML artifacts)
3. Collapse multiple whitespace to single space
4. Remove English stopwords (scikit-learn's built-in list)

Preprocessing is applied to train, validation, and test sets. The TF-IDF vectorizer is **fitted on training data only** — transforming validation and test sets using training vocabulary to prevent data leakage.

---

## 📐 Model Configuration

**TF-IDF Vectorizer**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 30,000 | Wide vocabulary coverage while limiting noise |
| `ngram_range` | (1, 2) | Bigrams capture phrase-level sentiment (e.g. "not good") |
| `min_df` | 5 | Drops rare terms appearing in fewer than 5 documents |
| `max_df` | 0.9 | Drops near-universal terms with low discriminative signal |

**Logistic Regression Classifier**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `C` | 2.0 | Mild L2 regularization — slightly less than default (C=1.0) to allow flexibility in large feature space |
| `max_iter` | 2000 | Sufficient for convergence on high-dimensional data |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training across all available CPU cores |

Logistic Regression is well-suited to high-dimensional sparse TF-IDF features — it is fast, interpretable, and competitive with more complex models on bag-of-words representations.

---

## 📊 Results

### Test Set (held out)

| Metric | Negative | Positive | Overall |
|--------|----------|----------|---------|
| Precision | 0.88 | 0.88 | — |
| Recall | 0.88 | 0.88 | — |
| F1-Score | 0.88 | 0.88 | — |
| **Accuracy** | | | **0.8807** |
| **Weighted F1** | | | **0.88** |
| **Macro F1** | | | **0.88** |

### Confusion Matrix

```text
[[10996  1504]
 [ 1478 11022]]

              precision    recall  f1-score   support
   negative       0.88      0.88      0.88     12500
   positive       0.88      0.88      0.88     12500

   accuracy                           0.88     25000
  macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000
```

---

## 🔍 Model Interpretability

Logistic Regression coefficients directly indicate feature importance — large positive coefficients signal positive sentiment, large negative coefficients signal negative sentiment.

**Top 10 Positive Features:**

| Feature | Coefficient |
|---------|-------------|
| excellent | 7.1844 |
| great | 7.1395 |
| best | 6.1257 |
| perfect | 5.7559 |
| wonderful | 5.5507 |
| amazing | 5.1028 |
| favorite | 4.8354 |
| loved | 4.4536 |
| love | 4.1001 |
| today | 3.9993 |

**Top 10 Negative Features:**

| Feature | Coefficient |
|---------|-------------|
| worst | -9.9449 |
| bad | -7.7363 |
| awful | -7.3620 |
| waste | -6.5473 |
| boring | -6.3903 |
| poor | -6.3089 |
| worse | -5.9780 |
| terrible | -5.3319 |
| poorly | -5.3011 |
| horrible | -5.1377 |

Note that `today` appearing as a top positive feature is a quirk of the corpus — likely correlated with enthusiastic present-tense review language ("the best film I've seen today") rather than a direct sentiment signal.

---

## 🔬 Why This Matters as a Baseline

This project is intentionally designed as the first half of a two-project comparison:

| Consideration | TF-IDF + LR (this project) | BERT (next project) |
|--------------|---------------------------|---------------------|
| Training time | ~2 minutes (CPU) | ~30 minutes (GPU) |
| Inference speed | <1ms per review | ~50ms per review |
| Accuracy | **88.07%** | **92.15%** |
| Handles negation | Partially (bigrams help) | Yes (bidirectional attention) |
| Handles sarcasm | Poorly | Better |
| Production cost | Very low | Higher |
| Interpretability | High (coefficients) | Lower (attention maps) |

For high-volume, cost-sensitive production systems, TF-IDF + Logistic Regression is often the right engineering choice. The BERT project demonstrates when the additional complexity is justified.

---

## 📂 Project Structure

```text
imdb-sentiment-analysis/
│
├── imdb_sentiment_analysis.py       # Main training and evaluation script
├── imdb_sentiment_analysis_notes.py # Annotated version with design rationale
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Splodz/ai-ml-portfolio.git
cd ai-ml-portfolio/imdb-sentiment-analysis
pip install -r requirements.txt
python imdb_sentiment_analysis.py
```

> No GPU required. Runs on CPU in under 2 minutes.

---

## 📦 Requirements

```text
datasets
scikit-learn
numpy
```

---

## 👤 Author

Graduate student in Artificial Intelligence with a focus on machine learning and deep learning systems.
