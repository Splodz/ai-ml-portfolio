# ------------------------------------------------
# Phase 1
# ------------------------------------------------
"""
IMDb Sentiment Analysis – Phase 1: Dataset Loading

This script loads the IMDb movie reviews dataset and prepares
train/validation splits suitable for classical NLP pipelines
(Bag-of-Words, TF-IDF) using scikit-learn.
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

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
# Phase 2
# ------------------------------------------------

# ------------------------------------------------
# Imports
# ------------------------------------------------

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ------------------------------------------------
# Preprocess raw data
# ------------------------------------------------

def preprocess_text(text):
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

# ------------------------------------------------
# Apply preprocessing to tran/validation sets
# ------------------------------------------------

# Apply preprocessing without overwriting X_train
X_train_clean = [preprocess_text(review) for review in X_train]
X_val_clean = [preprocess_text(review) for review in X_val]

# ------------------------------------------------
# Sanity check
# ------------------------------------------------

print("\nOriginal Review:")
print(X_train[0][:300])

