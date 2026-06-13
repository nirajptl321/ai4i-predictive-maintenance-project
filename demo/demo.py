"""Simple local demo for the saved AI4I machine failure model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

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

    package = joblib.load(MODEL_PATH)
    model = package["pipeline"]
    feature_columns = package["feature_columns"]
    target_column = package["target_column"]

    df = pd.read_csv(PROCESSED_DATA_PATH)
    positive_examples = df[df[target_column] == 1].head(1)
    negative_examples = df[df[target_column] == 0].head(1)
    demo_rows = pd.concat([negative_examples, positive_examples], axis=0)

    probabilities = model.predict_proba(demo_rows[feature_columns])[:, 1]
    predictions = model.predict(demo_rows[feature_columns])

    print("AI4I Machine Failure Prediction Demo")
    print("This is a simple local demo, not deployment software.")
    print(f"Loaded model: {package['model_name']}")
    print()

    for display_index, (row_index, row) in enumerate(demo_rows.iterrows(), start=1):
        print(f"Sample {display_index} from processed row index {row_index}")
        print(f"  Type: {row['type']}")
        print(f"  Air temperature [K]: {row['air_temperature_k']}")
        print(f"  Process temperature [K]: {row['process_temperature_k']}")
        print(f"  Rotational speed [rpm]: {row['rotational_speed_rpm']}")
        print(f"  Torque [Nm]: {row['torque_nm']}")
        print(f"  Tool wear [min]: {row['tool_wear_min']}")
        print(f"  True class: {int(row[target_column])}")
        print(f"  Predicted class: {int(predictions[display_index - 1])}")
        print(f"  Predicted probability of machine failure: {probabilities[display_index - 1]:.4f}")
        print()


if __name__ == "__main__":
    main()

