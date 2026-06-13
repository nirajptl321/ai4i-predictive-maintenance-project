"""Train, tune, compare, select, and save the final AI4I model."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai4i")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_PATH,
    LEAKAGE_COLUMNS,
    METRICS_TABLE_PATH,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loading import load_processed_data
from src.modeling import fit_final_model, train_and_validate_models
from src.utils import ensure_directories, make_train_validation_test_split

sns.set_theme(style="whitegrid")


def save_model_comparison_plot(metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df.sort_values("f1_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["model"], plot_df["f1_score"], color="#4C78A8", label="F1-score")
    ax.scatter(plot_df["recall"], plot_df["model"], color="#F58518", label="Recall", zorder=3)
    ax.scatter(plot_df["f2_score"], plot_df["model"], color="#54A24B", label="F2-score", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation score")
    ax.set_title("Model Comparison on Validation Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    df = load_processed_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = make_train_validation_test_split(df)
    del X_test, y_test

    metrics_df, selected_models, best_model_name = train_and_validate_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    metrics_df.to_csv(METRICS_TABLE_PATH, index=False)
    save_model_comparison_plot(metrics_df)

    best_params = selected_models[best_model_name]["best_params"]
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    final_pipeline = fit_final_model(best_model_name, best_params, X_train_validation, y_train_validation)

    package = {
        "pipeline": final_pipeline,
        "model_name": best_model_name,
        "best_params": best_params,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_f1_score",
        "validation_metrics": selected_models[best_model_name]["validation_metrics"],
        "all_validation_results": json.loads(metrics_df.to_json(orient="records")),
    }
    joblib.dump(package, FINAL_MODEL_PATH)

    print(f"Validation metrics saved to: {METRICS_TABLE_PATH}")
    print(f"Model comparison plot saved to: {PLOTS_DIR / 'model_comparison.png'}")
    print(f"Selected model: {best_model_name}")
    print(f"Best parameters: {best_params}")
    print(f"Final model saved to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
