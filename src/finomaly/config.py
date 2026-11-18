from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset configuration
DATASET_NAME = "PS_20174392719_1491204439457_log.csv"

DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

# Required dataset columns
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Machine learning configuration
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_CONTAMINATION = 0.01
RANDOM_STATE = 42

# Model configuration
MODEL_ESTIMATORS = 150