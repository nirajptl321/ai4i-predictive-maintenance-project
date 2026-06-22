"""Preprocess the raw AI4I dataset into a clean project CSV.

Running this file creates data/processed/ai4i_processed.csv. That processed
file is the single dataset used by EDA, training, evaluation, and the demo.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    ALL_REQUIRED_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loading import clean_column_names, load_raw_data
from src.utils import ensure_directories


def create_processed_dataset() -> pd.DataFrame:
    """Clean columns, remove IDs, and save the processed AI4I dataset."""
    # Create output folders before writing the processed CSV.
    ensure_directories()

    # Load the raw CSV and clean its column names.
    raw_data = load_raw_data()
    processed_data = clean_column_names(raw_data)

    # Make sure the model inputs and target are present after cleaning.
    missing_columns = []
    for column in ALL_REQUIRED_COLUMNS:
        if column not in processed_data.columns:
            missing_columns.append(column)

    # Stop immediately if an expected column is missing. Without this check the
    # model could train on the wrong data or fail later with a less clear error.
    if missing_columns:
        raise ValueError(f"Required columns are missing after cleaning: {missing_columns}")

    # Remove ID columns because they do not describe machine behavior.
    processed_data = processed_data.drop(columns=ID_COLUMNS, errors="ignore")

    # Put the model columns first, then keep any extra dataset columns after them.
    important_columns = []
    for column in ALL_REQUIRED_COLUMNS + LEAKAGE_COLUMNS:
        if column in processed_data.columns and column not in important_columns:
            important_columns.append(column)

    # Extra columns are preserved after the important columns so the processed
    # CSV still contains the full cleaned dataset order where possible.
    extra_columns = []
    for column in processed_data.columns:
        if column not in important_columns:
            extra_columns.append(column)

    ordered_columns = important_columns + extra_columns
    processed_data = processed_data[ordered_columns]

    # Store the target as integers so classification metrics are consistent.
    processed_data[TARGET_COLUMN] = processed_data[TARGET_COLUMN].astype(int)

    # Save the processed CSV used by training, evaluation, EDA, and the demo.
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    return processed_data


def main() -> None:
    # main() is the command-line entry point used by:
    # python -m src.preprocessing
    processed_data = create_processed_dataset()
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(processed_data):,}")
    print(f"Columns: {list(processed_data.columns)}")
    print("Model feature columns exclude the diagnostic failure mode columns.")


if __name__ == "__main__":
    main()
