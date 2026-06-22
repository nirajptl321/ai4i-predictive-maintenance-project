"""Model definitions, tuning, and validation comparison helpers."""

from __future__ import annotations

# Imports
import json
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import NUMERIC_FEATURE_COLUMNS, RANDOM_STATE, TYPE_COLUMN
from src.utils import classification_metrics, positive_class_probability


# Model specification container
@dataclass(frozen=True)
class ModelSpec:
    """Keep one model, its display name, and its optional tuning grid together."""

    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list] | None = None


# Preprocessing pipeline
def make_preprocessor() -> ColumnTransformer:
    """Build the common preprocessing step for all models."""
    # Machine type is categorical, so it is converted into one-hot columns.
    type_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Numeric sensor readings are scaled before they reach the model.
    numeric_transformer = StandardScaler()

    transformers = [
        ("type", type_transformer, [TYPE_COLUMN]),
        ("numeric", numeric_transformer, NUMERIC_FEATURE_COLUMNS),
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Attach preprocessing to a classifier."""
    # Every candidate model gets the same preprocessing for a fair comparison.
    pipeline_steps = [
        ("preprocessor", make_preprocessor()),
        ("model", estimator),
    ]
    return Pipeline(steps=pipeline_steps)


# Candidate model definitions
def model_specs() -> OrderedDict[str, ModelSpec]:
    """Return exactly the five models required for the project."""
    specs = OrderedDict()
    specs["Logistic Regression"] = ModelSpec(
        name="Logistic Regression",
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
    )
    specs["Decision Tree"] = ModelSpec(
        name="Decision Tree",
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 5, 10],
        },
    )
    specs["Random Forest"] = ModelSpec(
        name="Random Forest",
        estimator=RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [8, None],
            "min_samples_leaf": [1, 5],
        },
    )
    specs["Extra Trees"] = ModelSpec(
        name="Extra Trees",
        estimator=ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    specs["HistGradientBoostingClassifier"] = ModelSpec(
        name="HistGradientBoostingClassifier",
        estimator=HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "learning_rate": [0.05, 0.10],
            "max_iter": [100, 200],
            "max_leaf_nodes": [15, 31],
        },
    )
    return specs


# Model comparison helper
def _is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    """Return True when a trial beats the current best validation result."""
    if incumbent is None:
        return True

    candidate_key = (
        candidate["f1_score"],
        candidate["recall"],
        candidate["f2_score"],
        candidate["roc_auc"],
    )
    incumbent_key = (
        incumbent["f1_score"],
        incumbent["recall"],
        incumbent["f2_score"],
        incumbent["roc_auc"],
    )
    return candidate_key > incumbent_key


# Training and validation loop
def train_and_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict], str]:
    """Tune required models on train/validation data and select by validation F1."""
    summary_records = []
    selected_models: dict[str, dict] = {}

    for spec in model_specs().values():
        # ParameterGrid expands each model's grid into one trial per parameter combination.
        parameter_grid = list(ParameterGrid(spec.param_grid or {}))
        trial_records = []
        best_metrics: dict[str, float] | None = None
        best_params: dict | None = None
        best_pipeline: Pipeline | None = None

        for trial_number, params in enumerate(parameter_grid, start=1):
            # Clone the estimator so each trial starts from a fresh, unfitted model.
            estimator = clone(spec.estimator).set_params(**params)
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_train, y_train)

            # Each validation trial is saved so the tuning process can be checked later.
            y_pred = pipeline.predict(X_validation)
            y_probability = positive_class_probability(pipeline, X_validation)
            metrics = classification_metrics(y_validation, y_pred, y_probability)

            trial_record = {
                "model": spec.name,
                "trial_number": trial_number,
                "split": "validation",
                "tuned": bool(spec.param_grid),
                "params": json.dumps(params, sort_keys=True),
                **metrics,
            }
            trial_records.append(trial_record)

            if _is_better(metrics, best_metrics):
                # Keep the best trial for this model family.
                best_metrics = metrics
                best_params = params
                best_pipeline = pipeline

        if best_metrics is None or best_params is None or best_pipeline is None:
            raise RuntimeError(f"No model candidate was trained for {spec.name}")

        summary_record = {
            "model": spec.name,
            "split": "validation",
            "tuned": bool(spec.param_grid),
            "best_params": json.dumps(best_params, sort_keys=True),
            **best_metrics,
        }
        summary_records.append(summary_record)

        selected_models[spec.name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "validation_metrics": best_metrics,
            "validation_trials": trial_records,
            "tuned": bool(spec.param_grid),
        }

    metrics_df = pd.DataFrame(summary_records)

    # Rank models by validation F1 first, then by recall, F2, and ROC-AUC as tie breakers.
    metrics_df = metrics_df.sort_values(
        by=["f1_score", "recall", "f2_score", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    metrics_df.insert(0, "rank", range(1, len(metrics_df) + 1))
    best_model_name = metrics_df.loc[0, "model"]
    return metrics_df, selected_models, best_model_name


# Final refit
def fit_final_model(
    model_name: str,
    best_params: dict,
    X_train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
) -> Pipeline:
    """Refit the selected model on train + validation before final testing."""
    specs = model_specs()
    if model_name not in specs:
        raise ValueError(f"Unknown model name: {model_name}")

    estimator = clone(specs[model_name].estimator).set_params(**best_params)
    pipeline = make_pipeline(estimator)
    pipeline.fit(X_train_validation, y_train_validation)
    return pipeline
