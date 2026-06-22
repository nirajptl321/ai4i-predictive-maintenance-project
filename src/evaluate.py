"""Evaluate the saved final model once on the held-out test split."""

from __future__ import annotations

# Matplotlib setup
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

# Plot style constants
SAVEFIG_KWARGS = {"dpi": 180, "bbox_inches": "tight", "pad_inches": 0.2}


# Confusion matrix plot
def save_confusion_matrix_plot(y_true: pd.Series, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No failure", "Failure"],
    )
    display.plot(values_format="d", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Test Confusion Matrix", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", **SAVEFIG_KWARGS)
    plt.close(fig)


# Feature importance plot
def save_feature_importance_plot(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    importance_result = permutation_importance(
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
            "importance": importance_result.importances_mean,
            "std": importance_result.importances_std,
        }
    ).sort_values("importance", ascending=True)

    # Wider canvas and explicit x-limit keep long labels and error bars visible.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(importance_df["feature"], importance_df["importance"], xerr=importance_df["std"], color="#4C78A8")
    x_max = (importance_df["importance"] + importance_df["std"]).max()
    ax.set_xlim(left=0, right=x_max * 1.08)
    ax.set_xlabel("Mean permutation importance using F1-score")
    ax.set_title("Final Model Feature Importance", pad=12)
    fig.subplots_adjust(left=0.28, bottom=0.18, right=0.97, top=0.9)
    fig.savefig(PLOTS_DIR / "feature_importance.png", **SAVEFIG_KWARGS)
    plt.close(fig)


def main() -> None:
    # Create output folders before saving metrics or plots.
    ensure_directories()

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError("Final model not found. Run python -m src.train first.")

    # Load the saved final model package.
    model_package = joblib.load(FINAL_MODEL_PATH)
    model = model_package["pipeline"]
    model_name = model_package["model_name"]

    # Load the processed data and recreate the same split.
    processed_data = load_processed_data()
    _, _, X_test, _, _, y_test = make_train_validation_test_split(processed_data)

    # Evaluate only the saved final model on the held-out test set.
    y_pred = model.predict(X_test)
    y_probability = positive_class_probability(model, X_test)
    metrics = classification_metrics(y_test, y_pred, y_probability)

    # Save the test metrics with confusion-matrix counts.
    metrics_row = {
        "model": model_name,
        "split": "test",
        **metrics,
        **confusion_matrix_values(y_test, y_pred),
    }
    pd.DataFrame([metrics_row]).to_csv(TEST_METRICS_PATH, index=False)

    # Save the final evaluation plots.
    save_confusion_matrix_plot(y_test, y_pred)
    save_feature_importance_plot(model, X_test, y_test)

    # Print the same metrics shown in the report.
    print(f"Final model: {model_name}")
    print(f"Test metrics saved to: {TEST_METRICS_PATH}")
    print(metrics_to_text(metrics))
    print(f"Confusion matrix plot saved to: {PLOTS_DIR / 'confusion_matrix.png'}")
    print(f"Feature importance plot saved to: {PLOTS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
