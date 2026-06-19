"""Evaluate saved MLP and XGBoost models on the test split."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from mlp_xgboost.common import (
    ERROR_ANALYSIS_PATH,
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_PATHS,
    OUTPUT_PLOTS_DIR,
    OUTPUT_RESULTS_DIR,
    RANDOM_STATE,
    SAVEFIG_KWARGS,
    TEST_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    binary_classification_metrics,
    confusion_counts,
    ensure_output_directories,
    load_package,
    load_split_data,
    make_error_type,
    ordered_metric_columns,
    positive_class_probability,
    predictions_from_threshold,
    transformed_feature_names,
    write_markdown,
)


def save_confusion_matrix_plot(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Save a confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(matrix)
    ax.set_title(f"{model_name} Test Confusion Matrix", pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["No failure", "Failure"])
    ax.set_yticks([0, 1], labels=["No failure", "Failure"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    file_name = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(OUTPUT_PLOTS_DIR / file_name, **SAVEFIG_KWARGS)
    plt.close(fig)


def save_roc_and_pr_plots(evaluation_frames: list[dict[str, object]], y_test: pd.Series) -> None:
    """Save ROC and precision-recall curves."""
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for item in evaluation_frames:
        RocCurveDisplay.from_predictions(
            y_test,
            item["probability"],
            name=str(item["model"]),
            ax=ax,
        )
    ax.set_title("MLP vs XGBoost ROC Curve")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "mlp_xgboost_roc_curve.png", **SAVEFIG_KWARGS)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for item in evaluation_frames:
        PrecisionRecallDisplay.from_predictions(
            y_test,
            item["probability"],
            name=str(item["model"]),
            ax=ax,
        )
    ax.set_title("MLP vs XGBoost Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "mlp_xgboost_precision_recall_curve.png", **SAVEFIG_KWARGS)
    plt.close(fig)


def save_xgboost_importance_plot(package: dict) -> None:
    """Save XGBoost feature importance."""
    pipeline = package["pipeline"]
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return

    names = transformed_feature_names(pipeline)
    importance_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "xgboost_feature_importance.png", **SAVEFIG_KWARGS)
    plt.close(fig)


def save_mlp_permutation_importance_plot(package: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save MLP permutation importance."""
    pipeline = package["pipeline"]
    result = permutation_importance(
        pipeline,
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
    ).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(importance_df["feature"], importance_df["importance"], xerr=importance_df["std"])
    ax.set_xlabel("Mean permutation importance using F1-score")
    ax.set_title("MLP Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "mlp_permutation_importance.png", **SAVEFIG_KWARGS)
    plt.close(fig)


def prediction_frame(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probability: np.ndarray,
    prediction: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return row-level test predictions."""
    columns = FEATURE_COLUMNS + [column for column in LEAKAGE_COLUMNS if column in df.columns]
    base = df.loc[X_test.index, columns].copy()
    base.insert(0, "row_index", X_test.index)
    base.insert(1, "model", model_name)
    base["true_label"] = y_test.to_numpy(dtype=int)
    base["predicted_label"] = prediction.astype(int)
    base["predicted_probability_failure"] = probability
    base["error_type"] = [make_error_type(int(t), int(p)) for t, p in zip(base["true_label"], base["predicted_label"])]
    return base


def numeric_profile_table(frame: pd.DataFrame, numeric_columns: list[str]) -> str:
    """Return error-group numeric means."""
    if frame.empty:
        return "No prediction rows were available."
    return frame.groupby("error_type")[numeric_columns].mean().round(3).to_markdown()


def failure_mode_summary(frame: pd.DataFrame) -> str:
    """Return diagnostic failure-mode counts."""
    mode_columns = [column for column in LEAKAGE_COLUMNS if column in frame.columns]
    if not mode_columns:
        return "No diagnostic failure-mode columns were available."

    pieces = []
    for error_type in ["false_negative", "false_positive"]:
        subset = frame[frame["error_type"] == error_type]
        if subset.empty:
            pieces.append(f"- `{error_type}`: none")
        else:
            counts = subset[mode_columns].sum().astype(int).to_dict()
            pieces.append(f"- `{error_type}` count = {len(subset)}; diagnostic mode sums = `{counts}`")
    return "\n".join(pieces)


def save_error_analysis(all_predictions: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Write error analysis files."""
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "type"]
    lines = [
        "# MLP and XGBoost Error Analysis",
        "",
        "Test metrics were computed once after model and threshold selection.",
        "Diagnostic failure-mode columns are used here only for interpretation.",
        "",
        "## Test metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
    ]

    for model_name in metrics_df["model"].tolist():
        frame = all_predictions[all_predictions["model"] == model_name].copy()
        false_negatives = frame[frame["error_type"] == "false_negative"].sort_values(
            "predicted_probability_failure",
            ascending=False,
        )
        false_positives = frame[frame["error_type"] == "false_positive"].sort_values(
            "predicted_probability_failure",
            ascending=False,
        )

        safe_name = model_name.lower().replace(" ", "_")
        false_negatives.to_csv(OUTPUT_RESULTS_DIR / f"{safe_name}_false_negatives.csv", index=False)
        false_positives.to_csv(OUTPUT_RESULTS_DIR / f"{safe_name}_false_positives.csv", index=False)

        lines.extend(
            [
                f"## {model_name}",
                "",
                f"- False negatives: {len(false_negatives)}",
                f"- False positives: {len(false_positives)}",
                "",
                "### Mean numeric feature values by prediction group",
                "",
                numeric_profile_table(frame, numeric_columns),
                "",
                "### Diagnostic failure-mode summary for errors",
                "",
                failure_mode_summary(frame),
                "",
            ]
        )

    write_markdown(ERROR_ANALYSIS_PATH, "\n".join(lines))


def main() -> None:
    ensure_output_directories()
    _, _, X_test, _, _, y_test, df = load_split_data()

    packages = {model_name: load_package(path) for model_name, path in MODEL_PATHS.items()}

    metric_rows = []
    prediction_frames = []
    evaluation_frames = []

    for model_name, package in packages.items():
        pipeline = package["pipeline"]
        threshold = float(package.get("best_threshold", 0.50))
        probability = positive_class_probability(pipeline, X_test)
        prediction = predictions_from_threshold(probability, threshold)
        metrics = binary_classification_metrics(y_test, prediction, probability)
        counts = confusion_counts(y_test, prediction)
        metric_rows.append(
            {
                "model": model_name,
                "split": "test",
                "threshold": threshold,
                **metrics,
                **counts,
            }
        )
        prediction_frames.append(prediction_frame(model_name, X_test, y_test, probability, prediction, df))
        evaluation_frames.append({"model": model_name, "probability": probability, "prediction": prediction})
        save_confusion_matrix_plot(model_name, y_test, prediction)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df[[column for column in ordered_metric_columns() if column in metrics_df.columns]]
    metrics_df.to_csv(TEST_METRICS_PATH, index=False)

    all_predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)
    all_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    save_roc_and_pr_plots(evaluation_frames, y_test)
    save_xgboost_importance_plot(packages["XGBoost"])
    save_mlp_permutation_importance_plot(packages["MLPClassifier"], X_test, y_test)
    save_error_analysis(all_predictions, metrics_df)

    print(f"Test metrics saved to: {TEST_METRICS_PATH}")
    print(f"Test predictions saved to: {TEST_PREDICTIONS_PATH}")
    print(f"Error analysis saved to: {ERROR_ANALYSIS_PATH}")
    print(f"Plots saved to: {OUTPUT_PLOTS_DIR}")
    print("\nTest summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
