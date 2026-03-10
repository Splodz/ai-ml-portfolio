# imdb_sentiment_analysis.py
"""
IMDb Sentiment Analysis — Classical NLP Pipeline
=================================================
Binary sentiment classification on the IMDb movie reviews dataset
using a TF-IDF feature representation and Logistic Regression.

Pipeline:
    1. Load and split data (Hugging Face datasets)
    2. Text preprocessing (lowercasing, cleaning, stopword removal)
    3. TF-IDF vectorization with bigrams
    4. Logistic Regression classifier
    5. Evaluation (accuracy, F1, confusion matrix)
    6. Model interpretability (top positive/negative features)
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import re
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
MAX_FEATURES    = 30000
NGRAM_RANGE     = (1, 2)
MIN_DF          = 5
MAX_DF          = 0.9
LR_C            = 2.0
LR_MAX_ITER     = 2000


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data() -> tuple[list[str], list[str], list[int], list[int], list[str], list[int]]:
    """
    Load IMDb dataset from Hugging Face and create a train/validation split.

    The test split is held out and only used for final evaluation.
    Stratification preserves the 50/50 positive-negative class balance.

    Note: Hugging Face dataset columns are converted to plain Python lists
    before passing to scikit-learn to avoid Arrow/numpy int64 type errors
    with newer versions of the datasets library.

    Returns:
        X_train, X_val, y_train, y_val, X_test, y_test
    """
    dataset = load_dataset("imdb")

    # Convert to plain Python lists — required for compatibility with
    # newer datasets library versions (Arrow format causes TypeError with
    # scikit-learn's train_test_split stratify parameter)
    X_train_full = list(dataset["train"]["text"])
    y_train_full = list(dataset["train"]["label"])
    X_test       = list(dataset["test"]["text"])
    y_test       = list(dataset["test"]["label"])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print(f"Train samples:      {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    return X_train, X_val, y_train, y_val, X_test, y_test


# ------------------------------------------------
# Preprocessing
# ------------------------------------------------

def preprocess(text: str) -> str:
    """
    Normalize raw review text for TF-IDF vectorization.

    Steps:
        1. Lowercase
        2. Remove non-alphabetic characters
        3. Collapse whitespace
        4. Remove scikit-learn English stopwords

    Args:
        text: Raw review string.

    Returns:
        Cleaned, stopword-filtered string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Apply preprocess() to a list of review strings."""
    return [preprocess(review) for review in corpus]


# ------------------------------------------------
# Vectorizer
# ------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer with tuned hyperparameters.

    Config rationale:
        max_features=30000 : Captures a wide vocabulary while limiting noise
        ngram_range=(1, 2) : Unigrams + bigrams capture phrase-level sentiment
                             (e.g. "not good" vs "good")
        min_df=5           : Drops rare terms that appear in fewer than 5 docs,
                             reducing overfitting to low-frequency noise
        max_df=0.9         : Drops near-universal terms that carry little
                             discriminative signal
    """
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )


# ------------------------------------------------
# Classifier
# ------------------------------------------------

class SentimentClassifier:
    """
    Logistic Regression sentiment classifier with TF-IDF features.

    Logistic Regression is a strong baseline for high-dimensional sparse
    text features. It is fast, interpretable, and often competitive with
    more complex models on bag-of-words representations.

    C=2.0 applies mild L2 regularization — less regularization than the
    default (C=1.0) to allow the model slightly more flexibility given
    the large feature space.
    """

    def __init__(self):
        self.vectorizer = build_vectorizer()
        self.model = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def fit(self, X_train: list[str], y_train: list[int]) -> None:
        """Fit vectorizer and classifier on training data."""
        print("\nFitting TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Feature matrix shape: {X_train_tfidf.shape}")

        print("Training Logistic Regression classifier...")
        self.model.fit(X_train_tfidf, y_train)
        print("Training complete.")

    def predict(self, X: list[str]) -> np.ndarray:
        """Transform and predict on new text data."""
        return self.model.predict(self.vectorizer.transform(X))

    def evaluate(self, X: list[str], y_true: list[int], split_name: str = "Validation") -> None:
        """
        Print accuracy, classification report, and confusion matrix.

        Args:
            X:          Raw (unvectorized) text samples.
            y_true:     Ground truth labels.
            split_name: Label for the printed output (e.g. 'Validation', 'Test').
        """
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)

        print(f"\n{'='*50}")
        print(f"{split_name} Results")
        print(f"{'='*50}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    def top_features(self, n: int = 10) -> None:
        """
        Print the most predictive positive and negative unigrams/bigrams.

        Logistic Regression coefficients directly indicate feature importance
        in a linear model — large positive coefficients signal positive
        sentiment, large negative coefficients signal negative sentiment.

        Args:
            n: Number of top features to display per class.
        """
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("Model must be trained before extracting features.")

        feature_names  = self.vectorizer.get_feature_names_out()
        coefficients   = self.model.coef_[0]

        top_positive = np.argsort(coefficients)[-n:][::-1]
        top_negative = np.argsort(coefficients)[:n]

        print(f"\nTop {n} Positive Features:")
        for i in top_positive:
            print(f"  {feature_names[i]:<25} coef: {coefficients[i]:.4f}")

        print(f"\nTop {n} Negative Features:")
        for i in top_negative:
            print(f"  {feature_names[i]:<25} coef: {coefficients[i]:.4f}")

    def inspect_errors(self, X_raw: list[str], y_true: list[int], n: int = 5) -> None:
        """
        Display misclassified examples for qualitative error analysis.

        Reviewing errors helps identify systematic failure modes — for
        example, sarcastic reviews or domain-specific vocabulary that
        confuses the model.

        Args:
            X_raw:  Raw (unvectorized) text samples.
            y_true: Ground truth labels.
            n:      Number of errors to display.
        """
        y_pred = self.predict(X_raw)
        incorrect = np.where(np.array(y_true) != y_pred)[0]
        print(f"\nTotal misclassified: {len(incorrect)} / {len(y_true)}")

        label_map = {0: "negative", 1: "positive"}
        for idx in incorrect[:n]:
            print(f"\n{'─'*50}")
            print(f"True:      {label_map[y_true[idx]]}")
            print(f"Predicted: {label_map[y_pred[idx]]}")
            print(f"Review:    {X_raw[idx][:300]}")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    # 1. Load data
    X_train, X_val, y_train, y_val, X_test, y_test = load_data()

    # 2. Preprocess
    print("\nPreprocessing text...")
    X_train_clean = preprocess_corpus(X_train)
    X_val_clean   = preprocess_corpus(X_val)
    X_test_clean  = preprocess_corpus(X_test)
    print("Preprocessing complete.")

    # 3. Train
    classifier = SentimentClassifier()
    classifier.fit(X_train_clean, y_train)

    # 4. Validation evaluation
    classifier.evaluate(X_val_clean, y_val, split_name="Validation")

    # 5. Error analysis
    classifier.inspect_errors(X_val, y_val, n=5)

    # 6. Feature interpretability
    classifier.top_features(n=10)

    # 7. Final test set evaluation (held-out — run once only)
    classifier.evaluate(X_test_clean, y_test, split_name="Test")
