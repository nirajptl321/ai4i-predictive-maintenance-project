"""Shared helpers for the MLP and XGBoost scripts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TYPE_COLUMN,
)
from src.data_loading import load_processed_data  # noqa: E402
from src.utils import make_train_validation_test_split  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_PLOTS_DIR = OUTPUT_RESULTS_DIR / "plots"

MLP_MODEL_PATH = OUTPUT_MODELS_DIR / "mlp_classifier.joblib"
XGBOOST_MODEL_PATH = OUTPUT_MODELS_DIR / "xgboost_classifier.joblib"
BEST_MODEL_PATH = OUTPUT_MODELS_DIR / "best_mlp_xgboost_model.joblib"

VALIDATION_SUMMARY_PATH = OUTPUT_RESULTS_DIR / "validation_metrics.csv"
HYPERPARAMETER_TRIALS_PATH = OUTPUT_RESULTS_DIR / "hyperparameter_trials.csv"
TEST_METRICS_PATH = OUTPUT_RESULTS_DIR / "test_metrics.csv"
TEST_PREDICTIONS_PATH = OUTPUT_RESULTS_DIR / "test_predictions.csv"
ERROR_ANALYSIS_PATH = OUTPUT_RESULTS_DIR / "error_analysis.md"
TRAINING_NOTES_PATH = OUTPUT_RESULTS_DIR / "training_notes.md"

SELECTION_METRIC = "f1_score"
THRESHOLD_METRIC = "f1_score"
SAVEFIG_KWARGS = {"dpi": 180, "bbox_inches": "tight", "pad_inches": 0.2}

MODEL_PATHS = {
    "MLPClassifier": MLP_MODEL_PATH,
    "XGBoost": XGBOOST_MODEL_PATH,
}


def ensure_output_directories() -> None:
    """Create output directories used by these scripts."""
    for path in [OUTPUT_MODELS_DIR, OUTPUT_RESULTS_DIR, OUTPUT_PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Load data and create the same train/validation/test split."""
    df = load_processed_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = make_train_validation_test_split(df)
    return X_train, X_validation, X_test, y_train, y_validation, y_test, df


def make_tabular_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step for tabular models."""
    return ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [TYPE_COLUMN]),
            ("numeric", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities for class 1."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if math.isclose(min_score, max_score):
            return np.full(shape=len(scores), fill_value=0.5)
        return (scores - min_score) / (max_score - min_score)
    raise AttributeError("Model must expose predict_proba or decision_function.")


def predictions_from_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities into binary predictions."""
    return (np.asarray(probability) >= float(threshold)).astype(int)


def confusion_counts(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TN, FP, FN, and TP counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    y_true_array = np.asarray(y_true)
    metrics = {
        "accuracy": accuracy_score(y_true_array, y_pred),
        "precision": precision_score(y_true_array, y_pred, zero_division=0),
        "recall": recall_score(y_true_array, y_pred, zero_division=0),
        "f1_score": f1_score(y_true_array, y_pred, zero_division=0),
        "f2_score": fbeta_score(y_true_array, y_pred, beta=2, zero_division=0),
        "average_precision": average_precision_score(y_true_array, y_probability),
    }
    if len(np.unique(y_true_array)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true_array, y_probability)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute metrics at one threshold."""
    y_pred = predictions_from_threshold(y_probability, threshold)
    return binary_classification_metrics(y_true, y_pred, y_probability)


def choose_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    metric_name: str = THRESHOLD_METRIC,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the best validation threshold."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 91), 2)

    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for threshold in thresholds:
        current_metrics = metrics_at_threshold(y_true, y_probability, float(threshold))
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = current_metrics
            continue

        current_key = (
            current_metrics[metric_name],
            current_metrics["recall"],
            current_metrics["f2_score"],
            current_metrics["roc_auc"],
        )
        best_key = (
            best_metrics[metric_name],
            best_metrics["recall"],
            best_metrics["f2_score"],
            best_metrics["roc_auc"],
        )
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = current_metrics

    if best_metrics is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best_threshold, best_metrics


def is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    """Compare models using validation metrics."""
    if incumbent is None:
        return True
    candidate_key = (
        candidate[SELECTION_METRIC],
        candidate["recall"],
        candidate["f2_score"],
        candidate["roc_auc"],
    )
    incumbent_key = (
        incumbent[SELECTION_METRIC],
        incumbent["recall"],
        incumbent["f2_score"],
        incumbent["roc_auc"],
    )
    return candidate_key > incumbent_key


def json_dumps(value: Any) -> str:
    """Dump values as JSON text."""
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(value, sort_keys=True, default=default)


def save_package(package: dict[str, Any], path: Path) -> None:
    """Save a joblib package."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)


def load_package(path: Path) -> dict[str, Any]:
    """Load a saved joblib package."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model package at {path}. Run training first.")
    package = joblib.load(path)
    required_keys = {"pipeline", "model_name", "feature_columns", "target_column", "best_threshold"}
    missing = required_keys.difference(package)
    if missing:
        raise ValueError(f"Model package at {path} is missing keys: {sorted(missing)}")
    return package


def transformed_feature_names(pipeline: Any) -> list[str]:
    """Return names after preprocessing."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    return list(FEATURE_COLUMNS)


def make_error_type(y_true: int, y_pred: int) -> str:
    """Label a row by prediction outcome."""
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    if y_true == 1 and y_pred == 0:
        return "false_negative"
    return "true_positive"


def ordered_metric_columns() -> list[str]:
    """Return the metric table column order."""
    return [
        "model",
        "split",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "f2_score",
        "roc_auc",
        "average_precision",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]


def write_markdown(path: Path, text: str) -> None:
    """Write a Markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
