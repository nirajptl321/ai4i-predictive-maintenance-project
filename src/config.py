"""Shared configuration for the AI4I predictive maintenance project.

This file keeps paths, column names, split settings, and model names in one
place. The other scripts import these constants so the project uses the same
filenames and columns everywhere.
"""

from pathlib import Path

# Project root
# Path(__file__) points to this config file. parents[1] moves up from
# src/config.py to the project folder, so every path below works no matter
# where the command is run from.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
# Raw data is the original downloaded CSV. Processed data is the cleaned CSV
# created by src/preprocessing.py and reused by training, EDA, evaluation, and
# the demo.
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "ai4i_processed.csv"

# Model and result paths
# The final trained pipeline is saved with joblib. Results and plots are saved
# separately so they can be inspected without rerunning the full workflow.
MODELS_DIR = ROOT_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"

RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_TABLE_PATH = RESULTS_DIR / "metrics_table.csv"
HYPERPARAMETER_TRIALS_PATH = RESULTS_DIR / "hyperparameter_trials.csv"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.csv"
FULL_RUN_LOG_PATH = RESULTS_DIR / "full_reproducibility_run.txt"
DEMO_OUTPUT_PATH = RESULTS_DIR / "demo_output.txt"
MISSING_VALUES_PATH = RESULTS_DIR / "missing_values_summary.csv"

# Split settings
# RANDOM_STATE keeps the train/validation/test split reproducible. The split is
# 70% training, 15% validation for model selection, and 15% final test data.
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Column groups
# TARGET_COLUMN is what the model predicts. TYPE_COLUMN is categorical. ID
# columns identify rows/products but do not describe machine condition, so they
# are dropped during preprocessing.
TARGET_COLUMN = "machine_failure"
TYPE_COLUMN = "type"
ID_COLUMNS = ["udi", "product_id"]

# These diagnostic failure-mode columns reveal the reason for failure. They are
# useful for EDA but are excluded from model inputs because they leak target
# information.
LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]

# Feature columns
# These are the only input columns used by the trained models. Leakage columns
# are intentionally not included here.
NUMERIC_FEATURE_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

FEATURE_COLUMNS = [
    TYPE_COLUMN,
    *NUMERIC_FEATURE_COLUMNS,
]

ALL_REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Model names
# These names define the required model families compared during validation.
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoostingClassifier",
]
