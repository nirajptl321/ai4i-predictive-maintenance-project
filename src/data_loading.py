"""Data loading and column-name cleaning helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def clean_column_name(column_name: str) -> str:
    """Convert original AI4I column names into Python-friendly snake_case."""
    cleaned = str(column_name).replace("\ufeff", "").strip().lower()
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with cleaned column names."""
    cleaned = df.copy()
    cleaned.columns = [clean_column_name(column) for column in cleaned.columns]
    return cleaned


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw AI4I CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place ai4i2020.csv in data/raw/."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed AI4I CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run python -m src.preprocessing first."
        )
    return pd.read_csv(path)

