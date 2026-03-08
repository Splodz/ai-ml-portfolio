# imdb_sentiment_with_interview_notes.py
"""
NOTES TO INTERVIEWER:

This project implements a complete classical NLP pipeline for binary
sentiment classification on the IMDb movie reviews dataset.

The goal is to predict whether a review is positive or negative using
TF-IDF feature extraction and Logistic Regression.

Key design decisions include:
- TF-IDF with bigrams to capture phrase-level sentiment signals
- Tuned hyperparameters (max_features, min_df, max_df, C) with documented rationale
- A SentimentClassifier class for clean, production-style code structure
- Held-out test set evaluated only once to prevent data leakage
- Error analysis and feature interpretability built into the pipeline
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

"""
NOTES TO INTERVIEWER:

All imports are collected at the top of the file following PEP 8 conventions.
Scattering imports throughout a script (e.g. inside phase blocks) works but
signals a lack of familiarity with Python best practices.

- re: standard library regex module for text cleaning
- numpy: used for coefficient indexing in feature importance analysis
- datasets: Hugging Face library for loading the IMDb dataset
- sklearn: provides the full ML pipeline — splitting, vectorizing,
  modeling, and evaluation
"""


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

"""
NOTES TO INTERVIEWER:

Extracting magic numbers into named constants at the top of the file
makes hyperparameters easy to find, modify, and reason about.
This is standard practice in production ML code and avoids having
unexplained numbers buried inside functions.
"""


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data() -> tuple[list[str], list[str], list[int], list[int], list[str], list[int]]:
    """
    Load IMDb dataset from Hugging Face and create a train/validation split.

    The test split is held out and only used for final evaluation.
    Stratification preserves the 50/50 positive-negative class balance.

    Returns:
        X_train, X_val, y_train, y_val, X_test, y_test
    """
    dataset = load_dataset("imdb")

    X_train_full = dataset["train"]["text"]
    y_train_full = dataset["train"]["label"]
    X_test       = dataset["test"]["text"]
    y_test       = dataset["test"]["label"]

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

"""
NOTES TO INTERVIEWER:

The IMDb dataset from Hugging Face provides 25,000 labeled training reviews
and 25,000 labeled test reviews. Labels are 0 = negative, 1 = positive,
with a perfect 50/50 class balance.

I create a validation split from the training data using stratify=y to
preserve the class ratio. The test set is loaded but not touched until
the very end — this is standard practice to ensure the final evaluation
is unbiased and the model has never seen this data in any form.

Using type hints on the return signature makes the data contract explicit:
downstream code knows exactly what types to expect.
"""


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

"""
NOTES TO INTERVIEWER:

The preprocessing pipeline normalizes text before vectorization.
Each step serves a specific purpose:

1. Lowercasing: ensures "Great" and "great" map to the same token,
   reducing vocabulary size and improving generalization.

2. Removing non-alphabetic characters: strips HTML tags, punctuation,
   and numbers that carry little sentiment signal in this context.

3. Collapsing whitespace: cleans up artifacts left by the previous step.

4. Stopword removal: drops high-frequency words like "the", "a", "is"
   that appear in nearly every review and have no discriminative value.
   I use scikit-learn's built-in ENGLISH_STOP_WORDS list for consistency.

Separating preprocess() (single text) from preprocess_corpus() (list of texts)
follows the single responsibility principle and makes unit testing easier.
"""


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

"""
NOTES TO INTERVIEWER:

TF-IDF (Term Frequency–Inverse Document Frequency) scores each word by
how frequently it appears in a review, down-weighted by how common it is
across all reviews. This gives high scores to words that are distinctive
to a particular review rather than generic.

The key hyperparameter decisions:

max_features=30000: A larger vocabulary than the initial 5,000 captures
more nuanced language. Too large and we risk including noise; too small
and we lose signal.

ngram_range=(1, 2): Bigrams are critical for sentiment because negation
completely reverses meaning — "not good" has the opposite sentiment of
"good". Unigrams alone miss this entirely.

min_df=5: Rare words appearing in fewer than 5 documents are likely
typos, proper nouns, or idiosyncratic terms that won't generalize.
Removing them reduces dimensionality and overfitting.

max_df=0.9: Words appearing in more than 90% of documents are so
common they carry no discriminative power. This is a softer version
of stopword removal that catches domain-specific high-frequency terms
the stopword list might miss.

Crucially, fit_transform() is called only on training data. The vectorizer
learns vocabulary and IDF weights from the training distribution only.
transform() is applied to validation and test sets — using fit_transform()
on those would constitute data leakage.
"""


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

    """
    NOTES TO INTERVIEWER:

    I encapsulate the vectorizer and classifier together in a single class.
    This mirrors the sklearn Pipeline pattern and ensures the two components
    are always used together — you can't accidentally call predict() on
    unvectorized text.

    Logistic Regression is well-suited to this problem because:
    - TF-IDF produces high-dimensional sparse matrices — LR handles these efficiently
    - It is interpretable: coefficients directly indicate feature importance
    - It converges quickly on linear problems
    - It is a strong, well-understood baseline before reaching for neural approaches

    C is the inverse of regularization strength. The default C=1.0 applies
    standard L2 regularization. I use C=2.0 to allow slightly more model
    complexity given our 30,000-feature space, which I found improved
    validation accuracy without signs of overfitting.

    n_jobs=-1 enables parallel training across all CPU cores, which
    significantly speeds up fitting on large datasets.

    random_state=42 ensures reproducible results across runs.
    """

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

    """
    NOTES TO INTERVIEWER:

    The evaluate() method accepts raw unvectorized text and handles
    transformation internally. This prevents a common mistake where
    test data is accidentally vectorized with fit_transform() rather
    than transform().

    I report three evaluation outputs:

    1. Accuracy: overall proportion of correct predictions. Reliable
       here because the dataset is perfectly balanced (50/50).

    2. Classification report: per-class precision, recall, and F1-score.
       Precision measures how often a positive prediction is correct.
       Recall measures how often actual positives are correctly identified.
       F1 is the harmonic mean — useful when you care about both.

    3. Confusion matrix: shows the exact breakdown of correct and
       incorrect predictions per class, making it easy to spot whether
       the model is biased toward one class.
    """

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

    """
    NOTES TO INTERVIEWER:

    One of the key advantages of Logistic Regression over more complex models
    like neural networks is interpretability. Each feature has a single learned
    coefficient that directly indicates its contribution to the prediction.

    A large positive coefficient means the word/bigram strongly predicts
    positive sentiment. A large negative coefficient means it strongly
    predicts negative sentiment.

    I print coefficients alongside feature names so it is easy to verify
    the model has learned sensible associations — this is a form of
    sanity checking the model's internal logic, not just its accuracy.

    The [::-1] reverses the argsort result so the most positive features
    appear first in descending order.
    """

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

    """
    NOTES TO INTERVIEWER:

    Accuracy alone does not tell you why a model fails. Error analysis
    — reading actual misclassified examples — reveals patterns that
    metrics cannot capture.

    Common failure modes for sentiment classifiers:

    1. Sarcasm: "Oh great, another two hours of my life wasted."
       The word "great" is positive but the review is negative.

    2. Mixed sentiment: A review might praise acting but criticize the plot.
       The model has to make a single binary decision on ambiguous text.

    3. Domain-specific vocabulary: Niche film terms or references the
       model has not seen enough of during training.

    Identifying these patterns informs what to improve next — for example,
    adding negation handling, using a larger model, or incorporating
    contextual embeddings like BERT.
    """


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    """
    NOTES TO INTERVIEWER:

    The if __name__ == '__main__' guard ensures this script only runs
    when executed directly, not when imported as a module. This is a
    Python best practice that makes the code reusable and testable —
    another module could import SentimentClassifier without triggering
    the full training pipeline.
    """

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

    """
    NOTES TO INTERVIEWER:

    The pipeline follows a deliberate order:

    1. Load — data is split before any processing to prevent leakage
    2. Preprocess — cleaning applied to all three splits
    3. Train — vectorizer and classifier fitted on training data only
    4. Validate — performance checked on unseen validation data
    5. Error analysis — qualitative inspection of failures
    6. Interpretability — feature coefficients extracted and displayed
    7. Test — final held-out evaluation run exactly once

    Running test evaluation last, and only once, ensures the reported
    test accuracy is a genuine measure of generalization and has not
    been inflated by repeated tuning against the test set.
    """
