"""Shared utility functions for splitting, metrics, and filesystem setup."""

from __future__ import annotations

import math
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


def ensure_directories() -> None:
    """Create output directories used by the project."""
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


def validate_model_columns(df: pd.DataFrame) -> None:
    """Validate required columns and make leakage violations explicit."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(f"Missing required processed columns: {missing_columns}")

    leaked_features = sorted(set(FEATURE_COLUMNS).intersection(LEAKAGE_COLUMNS))
    if leaked_features:
        raise ValueError(f"Leakage columns are configured as model features: {leaked_features}")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the approved feature matrix and binary target vector."""
    validate_model_columns(df)
    feature_table = df[FEATURE_COLUMNS].copy()
    target_values = df[TARGET_COLUMN].astype(int).copy()
    return feature_table, target_values


def make_train_validation_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a stratified 70/15/15 train/validation/test split."""
    features, target = split_features_target(df)

    # First split off 30%, then split that evenly into validation and test sets.
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
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def positive_class_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities for class 1, using a safe fallback if needed."""
    if hasattr(model, "predict_proba"):
        class_probabilities = model.predict_proba(X)
        positive_probabilities = class_probabilities[:, 1]
        return positive_probabilities

    scores = model.decision_function(X)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if math.isclose(min_score, max_score):
        return np.full(shape=len(scores), fill_value=0.5)

    normalized_scores = (scores - min_score) / (max_score - min_score)
    return normalized_scores


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Compute the project metrics for binary classification."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2_score"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def confusion_matrix_values(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TN, FP, FN, TP values for the binary confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    values = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return values


def metrics_to_text(metrics: dict[str, float]) -> str:
    """Format metric values for terminal output."""
    lines = []

    for name, value in metrics.items():
        lines.append(f"{name}: {value:.4f}")

    return "\n".join(lines)
