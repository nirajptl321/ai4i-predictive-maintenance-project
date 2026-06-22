"""Simple local demo for the saved AI4I machine failure model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "final_model.joblib"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "ai4i_processed.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing models/final_model.joblib. Run python -m src.train first.")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/processed/ai4i_processed.csv. Run python -m src.preprocessing first."
        )
    # Load the saved model package and the columns it expects.
    package = joblib.load(MODEL_PATH)
    model = package["pipeline"]
    model_name = package["model_name"]
    feature_columns = package["feature_columns"]
    target_column = package["target_column"]

    # Load the processed dataset used by the final model.
    data = pd.read_csv(PROCESSED_DATA_PATH)

    # Show three examples without failure and three examples with failure.
    no_failure_rows = data[data[target_column] == 0].head(3)
    failure_rows = data[data[target_column] == 1].head(3)
    demo_rows = pd.concat([no_failure_rows, failure_rows], axis=0)

    # Predict classes and failure probabilities for the selected demo rows.
    input_features = demo_rows[feature_columns]
    predicted_classes = model.predict(input_features)
    all_probabilities = model.predict_proba(input_features)
    failure_probabilities = all_probabilities[:, 1]

    # Print a small readable table of results.
    print("AI4I Machine Failure Prediction Demo")
    print("This demo shows a few sample predictions only. Full evaluation is done by python -m src.evaluate.")
    print("This is a simple local demo, not deployment software.")
    print(f"Loaded model: {model_name}")
    print()

    for sample_number in range(len(demo_rows)):
        row = demo_rows.iloc[sample_number]
        original_row_index = demo_rows.index[sample_number]
        display_number = sample_number + 1

        print(f"Sample {display_number}")
        print(f"  Original processed row index: {original_row_index}")
        print(f"  True class: {int(row[target_column])}")
        print(f"  Predicted class: {int(predicted_classes[sample_number])}")
        print(f"  Predicted probability of machine failure: {failure_probabilities[sample_number]:.4f}")
        print("  Input feature values:")
        print(f"  Type: {row['type']}")
        print(f"  Air temperature [K]: {row['air_temperature_k']}")
        print(f"  Process temperature [K]: {row['process_temperature_k']}")
        print(f"  Rotational speed [rpm]: {row['rotational_speed_rpm']}")
        print(f"  Torque [Nm]: {row['torque_nm']}")
        print(f"  Tool wear [min]: {row['tool_wear_min']}")
        print()


if __name__ == "__main__":
    main()
