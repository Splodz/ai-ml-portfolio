# ------------------------------------------------
# Phase 1: Data Acqusition, Inspection, Splitting
# ------------------------------------------------
"""
IMDb Sentiment Analysis – Phase 1: Dataset Loading

This script loads the IMDb movie reviews dataset and prepares
train/validation splits suitable for classical NLP pipelines
(Bag-of-Words, TF-IDF) using scikit-learn.
"""

# Imports libraries
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ------------------------------------------------
# Load IMDb Dataset
# ------------------------------------------------

# Load the IMDb dataset from Hugging Face
dataset = load_dataset("imdb")

# Extract text reviews and labels from the training split
X = dataset["train"]["text"]      # List of review strings
y = dataset["train"]["label"]     # 0 = negative, 1 = positive


# ------------------------------------------------
# Train / Validation Split
# ------------------------------------------------

# Create a validation split from the training data
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

print("IMDb Dataset Loaded Successfully\n")

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\nSample Review (first 500 characters):\n")
print(X_train[0][:500])

print("\nCorresponding Label:")
print("Positive" if y_train[0] == 1 else "Negative")

# ------------------------------------------------
# Phase 2: Preprocessing
# ------------------------------------------------

# Import library
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Preprocess raw data
def preprocess(text):
    """
    Cleans raw text for NLP processing
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]

    return " ".join(tokens)

# Apply preprocessing without overwriting X_train
X_train_clean = [preprocess(review) for review in X_train]
X_val_clean = [preprocess(review) for review in X_val]

# ------------------------------------------------
# Sanity check
# ------------------------------------------------
print("\nOriginal Review:")
print(X_train[0][:300])

print("\nPreprocessed Review:")
print(X_train_clean[0][:300])

# ------------------------------------------------
# Phase 3: Transform
# ------------------------------------------------

# Import library
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
)

# Fit on training data only
X_train_tfidf = tfidf.fit_transform(X_train_clean)

# Transform valadation data only
X_val_tfidf = tfidf.transform(X_val_clean)

# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
print("\nTF-IDF Feature Matrix Shape:")
print("Training:", X_train_tfidf.shape)
print("Validation:", X_val_tfidf.shape)

print("\nSample feature names:")
print(tfidf.get_feature_names_out()[:20])

# ------------------------------------------------
# Phase 4: Modeling
# ------------------------------------------------

# Import libraries
from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

# Train model
model.fit(X_train_tfidf, y_train)

# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
print("\nModel Training Complete.")
print(f"Number of features learned: {model.coef_.shape[1]}")

# ------------------------------------------------
# Phase 5: Evaluation
# ------------------------------------------------
# Import libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate predicitons
y_val_pred = model.predict(X_val_tfidf)

# Determine accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")

# ------------------------------------------------
# Sanity Check
# ------------------------------------------------
print(f"y_val length: {len(y_val)}")
print(f"y_val_pred length: {len(y_val_pred)}")
print("Unique predicted labels:", set(y_val_pred))

# ------------------------------------------------
# Phase 6: Analysis and Improvement
# ------------------------------------------------
#*******************Phase 6A***********************

# Import library
import numpy as np

# Find incorrect predictions
incorrect_indices = np.where(y_val != y_val_pred)[0]
print(f"\nNumber of incorrect predictions: {len(incorrect_indices)}")

# Inspect a few mistakes
for idx in incorrect_indices[:5]:
    print("\nReview:")
    print(X_val[idx][:300])
    print("True label:", y_val[idx])
    print("Predicted label:", y_val_pred[idx])

#*******************Phase 6B***********************

# Improve TF-IDF
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

# Fit on training data
X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_val_tfidf = vectorizer.transform(X_val_clean)

# Improve logistic regression
model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    n_jobs=-1
)

# Train Model
model.fit(X_train_tfidf, y_train)

#*******************Phase 6C***********************

# Safety check: ensure model is trained
if not hasattr(model, "coef_"):
    raise RuntimeError("Model must be trained before extracting coefficients.")

# Get feature names and learned coefficients 
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top positive and negative features
top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

print("\nTop Positive Words:")
for i in reversed(top_positive):
    print(feature_names[i])

print("\nTop Negative Words:")
for i in top_negative:
    print(feature_names[i])
    




