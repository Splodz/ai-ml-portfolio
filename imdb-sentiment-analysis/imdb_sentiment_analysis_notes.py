# ------------------------------------------------
# Phase 1
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
Phase 1 focuses on data acquisition and splitting.
I load a labeled dataset suitable for supervised learning
and create train/validation splits before any modeling.
"""

"""
IMDb Sentiment Analysis – Phase 1: Dataset Loading

This script loads the IMDb movie reviews dataset and prepares
train/validation splits suitable for classical NLP pipelines
(Bag-of-Words, TF-IDF) using scikit-learn.
"""

# Import libraries
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ------------------------------------------------
# Load IMDb Dataset
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I use Hugging Face's IMDb dataset, which provides raw text
reviews paired with sentiment labels (0 = negative, 1 = positive).
"""

dataset = load_dataset("imdb")

X = dataset["train"]["text"]      # List of review strings
y = dataset["train"]["label"]     # 0 = negative, 1 = positive


# ------------------------------------------------
# Train / Validation Split
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I split the data into training and validation sets.
Stratification preserves the class distribution so that
both sets contain a similar ratio of positive and negative reviews.
"""

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# Sanity Checks
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
Before preprocessing, I verify dataset size and inspect
a raw sample review to confirm data integrity.
"""

print("IMDb Dataset Loaded Successfully\n")

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\nSample Review (first 500 characters):\n")
print(X_train[0][:500])

print("\nCorresponding Label:")
print("Positive" if y_train[0] == 1 else "Negative")


# ------------------------------------------------
# Phase 2 Preprocessing
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
This phase cleans raw text by normalizing case,
removing punctuation and stopwords, and preparing
the data for feature extraction.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(text):
    """
    Cleans raw data for NLP processing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

    return " ".join(tokens)

X_train_clean = [preprocess(review) for review in X_train]
X_val_clean = [preprocess(review) for review in X_val]


# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I compare original and preprocessed text to ensure
cleaning behaves as expected before vectorization.
"""

print("\nOriginal Review:")
print(X_train[0][:300])

print("\nPreprocessed Review:")
print(X_train_clean[0][:300])


# ------------------------------------------------
# Phase 3 Transform
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I convert text into numerical features using TF-IDF.
This allows classical ML models to operate on text data.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)

X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_val_tfidf = tfidf.transform(X_val_clean)


# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I verify feature matrix dimensions and inspect
sample feature names to confirm successful transformation.
"""

print("\nTF-IDF Feature Matrix Shape:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)

print("\nSample feature names:")
print(tfidf.get_feature_names_out()[:20])


# ------------------------------------------------
# Phase 4: Modeling
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I use logistic regression as a strong baseline model
for binary sentiment classification with sparse TF-IDF features.
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

model.fit(X_train_tfidf, y_train)


# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
After training, I verify that the model has learned
weights for each feature.
"""

print("\nModel Training Complete.")
print(f"Number of features learned: {model.coef_.shape[1]}")


# ------------------------------------------------
# Phase 5: Evaluation
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
This phase evaluates generalization performance using
unseen validation data.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_val_pred = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

print("\nConfusion Matrix:")


# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
I confirm prediction lengths and label diversity to
ensure valid evaluation output.
"""

print(f"y_val length: {len(y_val)}")
print(f"y_val_pred length: {len(y_val_pred)}")
print("Unique predicted labels:", set(y_val_pred))


# ------------------------------------------------
# Phase 6: Analysis and Improvement
# ------------------------------------------------
"""
NOTES TO INTERVIEWER:
Phase 6 focuses on understanding model errors,
improving features, and interpreting learned weights.
"""

#******************* Phase 6A ***********************

import numpy as np

incorrect_indices = np.where(y_val != y_val_pred)[0]
print(f"\nNumber of incorrect predictions: {len(incorrect_indices)}")

"""
NOTES TO INTERVIEWER:
Reviewing misclassified samples helps identify ambiguity,
sarcasm, or preprocessing limitations.
"""

for idx in incorrect_indices[:5]:
    print("\nReview:")
    print(X_val[idx][:300])
    print("True label:", y_val[idx])
    print("Predicted label:", y_val_pred[idx])


#******************* Phase 6B ***********************

"""
NOTES TO INTERVIEWER:
I improve TF-IDF coverage and regularization strength
to capture richer phrase-level sentiment signals.
"""

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_val_tfidf = vectorizer.transform(X_val_clean)

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    n_jobs=-1,
)

model.fit(X_train_tfidf, y_train)


#******************* Phase 6C ***********************

"""
NOTES TO INTERVIEWER:
Logistic regression is interpretable.
Coefficient magnitude indicates feature importance,
revealing which words influence sentiment decisions.
"""

if not hasattr(model, "coef_"):
    raise RuntimeError("Model must be trained before extracting coefficients.")

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

print("\nTop Positive Words:")
for i in reversed(top_positive):
    print(feature_names[i])

print("\nTop Negative Words:")
for i in top_negative:
    print(feature_names[i])
