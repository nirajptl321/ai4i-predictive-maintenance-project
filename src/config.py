"""Shared configuration for the AI4I predictive maintenance project.

This file keeps paths, column names, split settings, and model names in one
place. The other scripts import these constants so the project uses the same
filenames and columns everywhere.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "ai4i_processed.csv"

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

# Shared split settings keep preprocessing, training, and evaluation reproducible.
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

TARGET_COLUMN = "machine_failure"
TYPE_COLUMN = "type"
ID_COLUMNS = ["udi", "product_id"]

# Diagnostic failure-mode columns are kept for EDA but excluded from model inputs
# because they leak information about the target.
LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]

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

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoostingClassifier",
]
