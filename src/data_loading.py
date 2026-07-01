"""Data loading and column-name cleaning helpers.

The raw AI4I CSV uses mixed-case names, spaces, units, and brackets. These
helpers convert those headers into stable snake_case names so the rest of the
project can refer to columns consistently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def clean_column_name(column_name: str) -> str:
    """Convert original AI4I column names into Python-friendly snake_case."""
    cleaned_name = str(column_name)
    cleaned_name = cleaned_name.replace("\ufeff", "")
    cleaned_name = cleaned_name.strip().lower()
    cleaned_name = cleaned_name.replace("[", "").replace("]", "")
    cleaned_name = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned_name)
    cleaned_name = re.sub(r"_+", "_", cleaned_name)
    return cleaned_name.strip("_")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with cleaned column names."""
    cleaned_data = df.copy()
    cleaned_columns = []

    for column in cleaned_data.columns:
        cleaned_columns.append(clean_column_name(column))

    cleaned_data.columns = cleaned_columns
    return cleaned_data


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw AI4I CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place ai4i2020.csv in data/raw/."
        )

    # utf-8-sig strips a possible byte-order mark from the first header.
    raw_data = pd.read_csv(path, encoding="utf-8-sig")
    return raw_data


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed AI4I CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run python -m src.preprocessing first."
        )

    processed_data = pd.read_csv(path)
    return processed_data
