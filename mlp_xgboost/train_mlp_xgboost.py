"""Train, tune, compare, and save MLP and XGBoost models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required. Install it with: pip install -r mlp_xgboost/requirements-xgboost.txt"
    ) from exc

from mlp_xgboost.common import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    HYPERPARAMETER_TRIALS_PATH,
    LEAKAGE_COLUMNS,
    MLP_MODEL_PATH,
    OUTPUT_PLOTS_DIR,
    RANDOM_STATE,
    SAVEFIG_KWARGS,
    SELECTION_METRIC,
    TARGET_COLUMN,
    THRESHOLD_METRIC,
    TRAINING_NOTES_PATH,
    VALIDATION_SUMMARY_PATH,
    XGBOOST_MODEL_PATH,
    binary_classification_metrics,
    choose_best_threshold,
    confusion_counts,
    ensure_output_directories,
    is_better,
    json_dumps,
    load_split_data,
    make_tabular_preprocessor,
    predictions_from_threshold,
    positive_class_probability,
    save_package,
    write_markdown,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class ModelSearchSpec:
    name: str
    param_grid: dict[str, list[Any]]


def positive_class_weight_ratio(y: pd.Series) -> float:
    """Return the negative-to-positive class ratio."""
    counts = y.value_counts()
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def model_search_specs(y_train: pd.Series) -> list[ModelSearchSpec]:
    """Return the model search grids."""
    imbalance_ratio = positive_class_weight_ratio(y_train)
    return [
        ModelSearchSpec(
            name="MLPClassifier",
            param_grid={
                "hidden_layer_sizes": [(16,), (32,)],
                "activation": ["relu"],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.003],
            },
        ),
        ModelSearchSpec(
            name="XGBoost",
            param_grid={
                "n_estimators": [80, 150],
                "max_depth": [2, 3],
                "learning_rate": [0.05, 0.10],
                "subsample": [0.80],
                "colsample_bytree": [0.80],
                "min_child_weight": [1],
                "scale_pos_weight": [imbalance_ratio],
            },
        ),
    ]


def make_estimator(model_name: str, params: dict[str, Any]):
    """Create a classifier for one trial."""
    if model_name == "MLPClassifier":
        return MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=10,
            batch_size=512,
            **params,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
            **params,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    """Attach preprocessing to a classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", make_tabular_preprocessor()),
            ("model", make_estimator(model_name, params)),
        ]
    )


def fit_trial_pipeline(pipeline: Pipeline, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit one hyperparameter trial."""
    if model_name == "MLPClassifier":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        try:
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        except TypeError:
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_validation_trial(
    pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Score one validation trial."""
    probability = positive_class_probability(pipeline, X_validation)
    default_predictions = predictions_from_threshold(probability, 0.50)
    default_metrics = binary_classification_metrics(y_validation, default_predictions, probability)
    best_threshold, threshold_metrics = choose_best_threshold(
        y_validation,
        probability,
        metric_name=THRESHOLD_METRIC,
    )
    return best_threshold, threshold_metrics, default_metrics


def save_validation_comparison_plot(metrics_df: pd.DataFrame) -> None:
    """Save the validation comparison plot."""
    plot_df = metrics_df.sort_values(SELECTION_METRIC, ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(plot_df["model"], plot_df["f1_score"], label="F1-score")
    ax.scatter(plot_df["recall"], plot_df["model"], label="Recall", zorder=3)
    ax.scatter(plot_df["f2_score"], plot_df["model"], label="F2-score", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation score")
    ax.set_title("MLP vs XGBoost Validation Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "mlp_xgboost_validation_comparison.png", **SAVEFIG_KWARGS)
    plt.close(fig)


def save_training_notes(metrics_df: pd.DataFrame, trial_df: pd.DataFrame, best_model_name: str) -> None:
    """Write the training summary file."""
    top = metrics_df.iloc[0]
    notes = f"""
# MLP and XGBoost Training Notes

Two supervised tabular classifiers were trained on the processed AI4I feature set.

## Inputs

- Features: `{', '.join(FEATURE_COLUMNS)}`
- Target: `{TARGET_COLUMN}`
- Excluded columns: `{', '.join(LEAKAGE_COLUMNS)}`
- Split: stratified 70/15/15 train/validation/test

## Search setup

- `MLPClassifier`: hidden layer size, alpha, and learning rate
- `XGBoost`: tree count, depth, learning rate, and class weight
- Trial selection metric: `{SELECTION_METRIC}`
- Threshold selection metric: `{THRESHOLD_METRIC}`

## Saved files

- Validation summary: `{VALIDATION_SUMMARY_PATH.relative_to(VALIDATION_SUMMARY_PATH.parents[2])}`
- Hyperparameter trials: `{HYPERPARAMETER_TRIALS_PATH.relative_to(HYPERPARAMETER_TRIALS_PATH.parents[2])}`
- MLP model: `{MLP_MODEL_PATH.relative_to(MLP_MODEL_PATH.parents[2])}`
- XGBoost model: `{XGBOOST_MODEL_PATH.relative_to(XGBOOST_MODEL_PATH.parents[2])}`
- Best model: `{BEST_MODEL_PATH.relative_to(BEST_MODEL_PATH.parents[2])}`

## Best validation model

`{best_model_name}` had validation F1 = {top['f1_score']:.4f}, recall = {top['recall']:.4f}, F2 = {top['f2_score']:.4f}, and ROC-AUC = {top['roc_auc']:.4f}.

Total validation trials: {len(trial_df)}.
"""
    write_markdown(TRAINING_NOTES_PATH, notes)


def main() -> None:
    ensure_output_directories()
    X_train, X_validation, _, y_train, y_validation, _, _ = load_split_data()

    all_trial_records: list[dict[str, Any]] = []
    selected_packages: dict[str, dict[str, Any]] = {}
    summary_records: list[dict[str, Any]] = []

    for spec in model_search_specs(y_train):
        best_metrics: dict[str, float] | None = None
        best_threshold = 0.50
        best_params: dict[str, Any] | None = None
        best_pipeline: Pipeline | None = None

        grid = list(ParameterGrid(spec.param_grid))
        print(f"Training {spec.name}: {len(grid)} validation trials")

        for trial_number, params in enumerate(grid, start=1):
            pipeline = make_pipeline(spec.name, dict(params))
            pipeline = fit_trial_pipeline(pipeline, spec.name, X_train, y_train)
            threshold, metrics, default_metrics = evaluate_validation_trial(pipeline, X_validation, y_validation)

            trial_record = {
                "model": spec.name,
                "trial_number": trial_number,
                "split": "validation",
                "params": json_dumps(params),
                "threshold_metric": THRESHOLD_METRIC,
                "selected_threshold": threshold,
                **metrics,
                **{f"default_threshold_{key}": value for key, value in default_metrics.items()},
            }
            all_trial_records.append(trial_record)

            if is_better(metrics, best_metrics):
                best_metrics = metrics
                best_threshold = threshold
                best_params = dict(params)
                best_pipeline = pipeline

        if best_metrics is None or best_params is None or best_pipeline is None:
            raise RuntimeError(f"No successful trials for {spec.name}")

        validation_probability = positive_class_probability(best_pipeline, X_validation)
        validation_prediction = predictions_from_threshold(validation_probability, best_threshold)
        package = {
            "pipeline": best_pipeline,
            "model_name": spec.name,
            "best_params": best_params,
            "best_threshold": best_threshold,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "excluded_leakage_columns": LEAKAGE_COLUMNS,
            "random_state": RANDOM_STATE,
            "selection_metric": SELECTION_METRIC,
            "threshold_metric": THRESHOLD_METRIC,
            "validation_metrics": best_metrics,
            "validation_confusion_counts": confusion_counts(y_validation, validation_prediction),
        }
        selected_packages[spec.name] = package
        save_package(package, MLP_MODEL_PATH if spec.name == "MLPClassifier" else XGBOOST_MODEL_PATH)

        summary_records.append(
            {
                "model": spec.name,
                "split": "validation",
                "threshold": best_threshold,
                "best_params": json_dumps(best_params),
                **best_metrics,
                **confusion_counts(y_validation, validation_prediction),
            }
        )

    metrics_df = pd.DataFrame(summary_records).sort_values(
        by=[SELECTION_METRIC, "recall", "f2_score", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    metrics_df.insert(0, "rank", range(1, len(metrics_df) + 1))

    trial_df = pd.DataFrame(all_trial_records)
    metrics_df.to_csv(VALIDATION_SUMMARY_PATH, index=False)
    trial_df.to_csv(HYPERPARAMETER_TRIALS_PATH, index=False)
    save_validation_comparison_plot(metrics_df)

    best_model_name = str(metrics_df.loc[0, "model"])
    save_package(selected_packages[best_model_name], BEST_MODEL_PATH)
    save_training_notes(metrics_df, trial_df, best_model_name)

    print(f"Validation metrics saved to: {VALIDATION_SUMMARY_PATH}")
    print(f"Hyperparameter trials saved to: {HYPERPARAMETER_TRIALS_PATH}")
    print(f"Training notes saved to: {TRAINING_NOTES_PATH}")
    print(f"Best model: {best_model_name}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
