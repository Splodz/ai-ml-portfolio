# IMDb Sentiment Analysis (NLP with scikit-learn)

This project performs **sentiment analysis on movie reviews** from the IMDb dataset, classifying reviews as **positive** or **negative** using classical Natural Language Processing (NLP) techniques and supervised machine learning.

The goal of this project is to demonstrate a **complete NLP pipeline**, from raw text to model interpretation, using industry-standard tools.

---

## 📌 Problem Statement

Given a movie review written in natural language, predict whether the sentiment expressed is:

- **Positive (1)**
- **Negative (0)**

This is a **supervised binary classification** problem with:
- Text input
- Integer labels
- Pre-labeled training data

---

## 🧠 Key Concepts Demonstrated

- Supervised learning and binary text classification
- Text preprocessing (lowercasing, punctuation removal, stopword filtering)
- Feature extraction with TF-IDF (unigrams and bigrams)
- Logistic Regression for high-dimensional sparse text features
- Hyperparameter tuning and rationale
- Model evaluation: accuracy, F1-score, confusion matrix
- Error analysis through misclassified example inspection
- Model interpretability through learned feature coefficients

---

## 🔧 Pipeline Overview

```
Raw Text Reviews
      ↓
Preprocessing (lowercase → clean → stopword removal)
      ↓
TF-IDF Vectorization (unigrams + bigrams, 30,000 features)
      ↓
Logistic Regression Classifier
      ↓
Evaluation + Error Analysis + Feature Interpretability
```

---

## 📐 Model Architecture

**Vectorizer: TF-IDF**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 30,000 | Wide vocabulary coverage while limiting noise |
| `ngram_range` | (1, 2) | Bigrams capture phrase-level sentiment (e.g. "not good") |
| `min_df` | 5 | Drops rare terms appearing in fewer than 5 documents |
| `max_df` | 0.9 | Drops near-universal terms with low discriminative signal |

**Classifier: Logistic Regression**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `C` | 2.0 | Mild L2 regularization — slightly less than default to allow flexibility in large feature space |
| `max_iter` | 2000 | Sufficient iterations for convergence on high-dimensional data |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Parallel training across all available CPU cores |

Logistic Regression was chosen as the classifier because it is fast, interpretable, and well-suited to high-dimensional sparse text features produced by TF-IDF.

---

## 🗂️ Dataset

- **Source:** [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb) via Hugging Face
- **Size:** 50,000 reviews (25,000 train / 25,000 test)
- **Balance:** 50% positive, 50% negative
- **Split used:** 80% train / 20% validation from the training set; test set held out for final evaluation only

---

## 🏋️ Training Details

- Preprocessing applied to training data before vectorization
- TF-IDF vectorizer fitted on training data only — `transform()` applied to validation and test sets to prevent data leakage
- Stratified train/validation split to preserve class balance

---

## 📊 Results

### Validation Set

| Metric | Negative | Positive | Overall |
|--------|----------|----------|---------|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |
| **Accuracy** | | | **—** |

### Test Set (held out)

| Metric | Score |
|--------|-------|
| Accuracy | — |
| Weighted F1 | — |
| Macro F1 | — |

> Results will be populated after running `imdb_sentiment.py`

### Confusion Matrix

```text
Results to be added after running the script
```

---

## 🔍 Model Interpretability

Logistic Regression coefficients provide direct insight into which words and phrases the model associates with positive or negative sentiment. Large positive coefficients indicate strong positive sentiment signals; large negative coefficients indicate negative sentiment signals.

Example top features (to be updated after running):

**Top Positive:** `brilliant`, `masterpiece`, `wonderful`, `perfectly`, `loved` ...

**Top Negative:** `worst`, `awful`, `waste`, `terrible`, `boring` ...

---

## 🔬 Model Comparison

| Model | Validation Accuracy |
|-------|-------------------|
| Logistic Regression (TF-IDF, baseline) | ~88–89% |
| Logistic Regression (TF-IDF, tuned) | — |

> Baseline results are approximate reference values. Tuned results will be populated after running the script.

---

## 📂 Project Structure

```text
imdb-sentiment-analysis/
│
├── imdb_sentiment.py                      # Main training and evaluation script
├── imdb_sentiment_with_interview_notes.py # Annotated version with interviewer notes
└── README.md                              # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Splodz/ai-ml-portfolio.git
cd ai-ml-portfolio/imdb-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python imdb_sentiment.py
```

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
