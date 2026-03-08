pip install xgboost shap imbalanced-learn

# hospital_readmission.py
"""
Hospital Readmission Prediction
================================
Predicts whether a diabetic patient will be readmitted to hospital
within 30 days using the UCI Diabetes 130-US Hospitals dataset.

Pipeline:
    1. Data loading and inspection
    2. Data cleaning (missing values, inconsistent entries, uninformative columns)
    3. Feature engineering (diagnosis codes, medication flags, visit aggregates)
    4. Preprocessing (encoding, scaling, SMOTE class balancing)
    5. XGBoost classification with tuned hyperparameters
    6. Evaluation (F1, AUC-ROC, confusion matrix)
    7. SHAP interpretability (global + local feature importance)
"""

# ------------------------------------------------
# Imports
# ------------------------------------------------

import io
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)


# ------------------------------------------------
# Constants
# ------------------------------------------------

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
DATA_URL      = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
TARGET_COLUMN = "readmitted"

COLUMNS_TO_DROP = [
    "encounter_id",
    "patient_nbr",
    "examide",
    "citoglipton",
    "weight",
    "payer_code",
    "medical_specialty",
]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


# ------------------------------------------------
# Data Loading
# ------------------------------------------------

def load_data(url: str) -> pd.DataFrame:
    """
    Load the UCI Diabetes 130-US Hospitals dataset.

    The ZIP contains multiple files — we extract the main data CSV directly.

    Args:
        url: URL to the zipped dataset on UCI repository.

    Returns:
        Raw DataFrame with original column names and values.
    """
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("dataset_diabetes/diabetic_data.csv") as f:
            df = pd.read_csv(f, na_values="?")

    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ------------------------------------------------
# Data Cleaning
# ------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw hospital data.

    Steps:
        1. Drop administratively uninformative and high-missingness columns
        2. Drop rows with remaining critical missing values
        3. Binarize target: '<30' days = 1 (readmitted), else 0

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with binary target column.
    """
    df = df.copy()

    # Drop uninformative columns
    df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], inplace=True)

    # Drop rows with missing race
    df.dropna(subset=["race"], inplace=True)

    # Binarize target: readmitted within 30 days = 1, otherwise = 0
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "<30").astype(int)

    print(f"After cleaning: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Class balance — Readmitted (<30 days): {df[TARGET_COLUMN].mean():.2%}")

    return df


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinically meaningful features from raw columns.

    Engineered features:
        - num_medications_changed: count of medications with dosage adjustments
        - is_insulin_changed:      flag for insulin dosage change specifically
        - total_diagnoses:         count of non-null diagnosis codes
        - age_numeric:             ordinal encoding of age brackets
        - has_diabetes_diagnosis:  flag if any diagnosis is diabetes-related

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Count how many medications had their dosage changed
    med_cols_present = [c for c in MEDICATION_COLS if c in df.columns]
    df["num_medications_changed"] = (
        df[med_cols_present].apply(lambda col: col != "Steady").sum(axis=1)
    )

    # Insulin change is clinically significant — flag it separately
    if "insulin" in df.columns:
        df["is_insulin_changed"] = (df["insulin"] != "Steady").astype(int)

    # Count non-null diagnosis codes as a proxy for clinical complexity
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    df["total_diagnoses"] = df[diag_cols].notna().sum(axis=1)

    # Flag if any diagnosis code falls in the diabetes range (ICD-9: 250.xx)
    def is_diabetes_code(code):
        try:
            return str(code).startswith("250")
        except Exception:
            return False

    df["has_diabetes_diagnosis"] = df[diag_cols].apply(
        lambda col: col.apply(is_diabetes_code)
    ).any(axis=1).astype(int)

    # Convert age brackets to ordinal numeric values
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
        "[80-90)": 8, "[90-100)": 9,
    }
    if "age" in df.columns:
        df["age_numeric"] = df["age"].map(age_map)
        df.drop(columns=["age"], inplace=True)

    # Drop raw columns after feature extraction
    df.drop(columns=diag_cols, inplace=True)
    df.drop(columns=med_cols_present, inplace=True)

    print(f"After feature engineering: {df.shape[1]} columns")

    return df


# ------------------------------------------------
# Preprocessing
# ------------------------------------------------

def preprocess(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Encode categorical features, split data, and apply SMOTE to training set.

    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
    samples of the minority class to address class imbalance.
    Applied ONLY to training data to prevent leakage.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        X_train_res, X_test, y_train_res, y_test, feature_names
    """
    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Label-encode remaining categorical columns
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill any remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    feature_names = list(X.columns)

    # Stratified split preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Apply SMOTE only to training data
    print(f"\nBefore SMOTE — Train class balance: {y_train.mean():.2%} positive")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE  — Train class balance: {y_train_res.mean():.2%} positive")
    print(f"Training samples after SMOTE: {len(X_train_res):,}")

    return X_train_res, X_test, y_train_res.values, y_test.values, feature_names


# ------------------------------------------------
# Model
# ------------------------------------------------

class ReadmissionClassifier:
    """
    XGBoost classifier for hospital readmission prediction.

    XGBoost is well-suited to this problem because:
    - Handles mixed feature types natively
    - Robust to remaining missing values via built-in sparse awareness
    - Strong performance on tabular data with minimal preprocessing
    - Native SHAP support for interpretability
    """

    def __init__(self, scale_pos_weight: float = 1.0):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Train XGBoost with validation loss logging."""
        self.feature_names = feature_names
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        print(f"\nTraining complete. Total estimators: {self.model.n_estimators}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, split_name: str = "Test") -> None:
        """
        Print F1, AUC-ROC, accuracy, classification report, and confusion matrix.

        For imbalanced medical classification, F1 and AUC-ROC are the
        primary metrics. Accuracy alone is misleading on skewed classes.
        """
        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)

        print(f"\n{'='*55}")
        print(f"{split_name} Results")
        print(f"{'='*55}")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
        print(f"AUC-ROC:   {roc_auc_score(y_true, y_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=["Not Readmitted", "Readmitted <30d"]
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    def plot_roc_curve(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        save_path: str = "roc_curve.png",
    ) -> None:
        """Plot and save the ROC curve."""
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(
            y_true, self.predict_proba(X), ax=ax, name="XGBoost"
        )
        ax.set_title("ROC Curve — Hospital Readmission")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"ROC curve saved to '{save_path}'")

    def plot_shap(
        self,
        X: np.ndarray,
        save_path: str = "shap_summary.png",
    ) -> None:
        """
        Generate SHAP summary plot showing global feature importance.

        SHAP (SHapley Additive exPlanations) assigns each feature a
        contribution value grounded in game theory. Unlike simple feature
        importance, SHAP shows both magnitude and direction of each
        feature's effect on the prediction.
        """
        print("\nComputing SHAP values (this may take a moment)...")
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Feature Importance — Hospital Readmission")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to '{save_path}'")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    # 1. Load
    df_raw = load_data(DATA_URL)

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Engineer features
    df_engineered = engineer_features(df_clean)

    # 4. Preprocess + SMOTE
    X_train, X_test, y_train, y_test, feature_names = preprocess(df_engineered)

    # 5. Train — split a validation slice for loss logging
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    classifier = ReadmissionClassifier()
    classifier.fit(X_tr, y_tr, X_val, y_val, feature_names)

    # 6. Evaluate on held-out test set
    classifier.evaluate(X_test, y_test, split_name="Test")

    # 7. ROC curve
    classifier.plot_roc_curve(X_test, y_test)

    # 8. SHAP interpretability (subset for speed)
    classifier.plot_shap(X_test[:500])
