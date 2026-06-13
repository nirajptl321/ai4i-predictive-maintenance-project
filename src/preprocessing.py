"""Preprocess the raw AI4I dataset into a clean project CSV."""

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
    ensure_directories()
    raw_df = load_raw_data()
    df = clean_column_names(raw_df)

    missing = [column for column in ALL_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing after cleaning: {missing}")

    df = df.drop(columns=ID_COLUMNS, errors="ignore")

    ordered_columns = []
    for column in ALL_REQUIRED_COLUMNS + LEAKAGE_COLUMNS:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)

    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    df = df[ordered_columns + remaining_columns]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def main() -> None:
    df = create_processed_dataset()
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("Model feature columns exclude the diagnostic failure mode columns.")


if __name__ == "__main__":
    main()

