"""Simple local demo for the saved MLP and XGBoost models."""

from __future__ import annotations

import argparse

import pandas as pd

from mlp_xgboost.common import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    MLP_MODEL_PATH,
    TARGET_COLUMN,
    XGBOOST_MODEL_PATH,
    load_package,
    load_split_data,
    positive_class_probability,
    predictions_from_threshold,
)

MODEL_CHOICES = {
    "best": BEST_MODEL_PATH,
    "mlp": MLP_MODEL_PATH,
    "xgboost": XGBOOST_MODEL_PATH,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run sample predictions for the saved MLP/XGBoost models.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="best",
        help="Saved model to load. Default: best",
    )
    parser.add_argument(
        "--rows-per-class",
        type=int,
        default=3,
        help="Number of failure and non-failure rows to print. Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = load_package(MODEL_CHOICES[args.model])
    pipeline = package["pipeline"]
    threshold = float(package["best_threshold"])

    _, _, X_test, _, _, y_test, df = load_split_data()
    del y_test
    test_rows = df.loc[X_test.index].copy()
    non_failure = test_rows[test_rows[TARGET_COLUMN] == 0].head(args.rows_per_class)
    failure = test_rows[test_rows[TARGET_COLUMN] == 1].head(args.rows_per_class)
    demo_rows = pd.concat([non_failure, failure], axis=0)

    probabilities = positive_class_probability(pipeline, demo_rows[FEATURE_COLUMNS])
    predictions = predictions_from_threshold(probabilities, threshold)

    print("MLP/XGBoost Prediction Demo")
    print("This demo prints sample predictions from the held-out test split.")
    print(f"Loaded model: {package['model_name']}")
    print(f"Decision threshold: {threshold:.2f}")
    print()

    for display_index, (row_index, row) in enumerate(demo_rows.iterrows(), start=1):
        print(f"Sample {display_index}")
        print(f"  Original processed row index: {row_index}")
        print(f"  True class: {int(row[TARGET_COLUMN])}")
        print(f"  Predicted class: {int(predictions[display_index - 1])}")
        print(f"  Predicted probability of machine failure: {probabilities[display_index - 1]:.4f}")
        print("  Input feature values:")
        for column in FEATURE_COLUMNS:
            print(f"    {column}: {row[column]}")
        print()


if __name__ == "__main__":
    main()
