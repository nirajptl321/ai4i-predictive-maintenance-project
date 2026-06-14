"""Evaluate the saved final model once on the held-out test split."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai4i")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.config import FINAL_MODEL_PATH, PLOTS_DIR, RANDOM_STATE, TEST_METRICS_PATH
from src.data_loading import load_processed_data
from src.utils import (
    classification_metrics,
    confusion_matrix_values,
    ensure_directories,
    make_train_validation_test_split,
    metrics_to_text,
    positive_class_probability,
)

sns.set_theme(style="whitegrid")


def save_confusion_matrix_plot(y_true: pd.Series, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No failure", "Failure"],
    )
    display.plot(values_format="d", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_feature_importance_plot(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], xerr=importance_df["std"], color="#4C78A8")
    ax.set_xlabel("Mean permutation importance using F1-score")
    ax.set_title("Final Model Feature Importance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError("Final model not found. Run python -m src.train first.")

    model_package = joblib.load(FINAL_MODEL_PATH)
    model = model_package["pipeline"]
    model_name = model_package["model_name"]

    df = load_processed_data()
    _, _, X_test, _, _, y_test = make_train_validation_test_split(df)

    y_pred = model.predict(X_test)
    y_probability = positive_class_probability(model, X_test)
    metrics = classification_metrics(y_test, y_pred, y_probability)
    metrics_row = {
        "model": model_name,
        "split": "test",
        **metrics,
        **confusion_matrix_values(y_test, y_pred),
    }
    pd.DataFrame([metrics_row]).to_csv(TEST_METRICS_PATH, index=False)

    save_confusion_matrix_plot(y_test, y_pred)
    save_feature_importance_plot(model, X_test, y_test)

    print(f"Final model: {model_name}")
    print(f"Test metrics saved to: {TEST_METRICS_PATH}")
    print(metrics_to_text(metrics))
    print(f"Confusion matrix plot saved to: {PLOTS_DIR / 'confusion_matrix.png'}")
    print(f"Feature importance plot saved to: {PLOTS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
