"""Shared utility functions for splitting, metrics, and filesystem setup.

These helpers are used by multiple scripts so the project uses the same split,
feature list, leakage checks, and metric calculations everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    HYPERPARAMETER_TRIALS_PATH,
    LEAKAGE_COLUMNS,
    METRICS_TABLE_PATH,
    MISSING_VALUES_PATH,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_METRICS_PATH,
)


# Filesystem helpers
def ensure_directories() -> None:
    """Create output directories used by the project."""
    # All scripts write into these folders. Creating them up front keeps the
    # later save steps simple.
    output_directories = [
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        FINAL_MODEL_PATH.parent,
        METRICS_TABLE_PATH.parent,
        HYPERPARAMETER_TRIALS_PATH.parent,
        TEST_METRICS_PATH.parent,
        MISSING_VALUES_PATH.parent,
    ]

    for path in output_directories:
        Path(path).mkdir(parents=True, exist_ok=True)


# Column validation
def validate_model_columns(df: pd.DataFrame) -> None:
    """Validate required columns and make leakage violations explicit."""
    # The model should only use approved feature columns plus the target.
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing required processed columns: {missing_columns}")

    # Leakage columns contain diagnostic information about the failure itself.
    # They are allowed in the processed CSV for EDA, but they must never appear
    # in FEATURE_COLUMNS.
    leaked_features = sorted(set(FEATURE_COLUMNS).intersection(LEAKAGE_COLUMNS))
    if leaked_features:
        raise ValueError(f"Leakage columns are configured as model features: {leaked_features}")


# Feature/target splitting
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the approved feature matrix and binary target vector."""
    validate_model_columns(df)

    # X contains only the approved model inputs. y is the binary class label.
    feature_table = df[FEATURE_COLUMNS].copy()
    target_values = df[TARGET_COLUMN].astype(int).copy()
    return feature_table, target_values


# Train/validation/test split
def make_train_validation_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a stratified 70/15/15 train/validation/test split."""
    features, target = split_features_target(df)

    # First split off 30%, then split that evenly into validation and test sets.
    # stratify keeps the rare failure class in each split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # The validation split is used for model selection. The test split stays
    # separate until src/evaluate.py performs the final evaluation.
    return X_train, X_validation, X_test, y_train, y_validation, y_test


# Probability extraction
def positive_class_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability for class 1, meaning machine failure."""
    # scikit-learn returns one probability column per class. Column index 1 is
    # the probability of the positive class: machine_failure = 1.
    class_probabilities = model.predict_proba(X)
    positive_probabilities = class_probabilities[:, 1]
    return positive_probabilities


# Metrics
def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Compute the project metrics for binary classification."""
    # Keep metric names stable because result CSVs and report text refer to
    # these exact names.
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # zero_division=0 avoids warnings if a model predicts no positive cases.
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2_score"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # ROC-AUC only makes sense when both classes are present in y_true.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


# Confusion matrix
def confusion_matrix_values(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TN, FP, FN, TP values for the binary confusion matrix."""
    # labels=[0, 1] fixes the order so ravel() always returns TN, FP, FN, TP.
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    values = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return values


# Metric text formatting
def metrics_to_text(metrics: dict[str, float]) -> str:
    """Format metric values for terminal output."""
    lines = []

    # Format each value to four decimals to match the report and console output.
    for name, value in metrics.items():
        lines.append(f"{name}: {value:.4f}")

    return "\n".join(lines)
