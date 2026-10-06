from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "bank.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURE_MANIFEST_PATH = ARTIFACTS_DIR / "features.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Target configuration
TARGET = "y"
TARGET_MAPPING = {"no": 0, "yes": 1}
TARGET_NAMES = ["No (Will Not Subscribe)", "Yes (Will Subscribe)"]

# Operational Pre-Call Features (Excludes 'duration' to prevent look-ahead data leakage)
OPERATIONAL_NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

# Benchmark Features (Includes 'duration' for historical/analytical comparisons)
BENCHMARK_NUMERICAL_FEATURES = OPERATIONAL_NUMERICAL_FEATURES + ["duration"]

BINARY_FEATURES = [
    "default",
    "housing",
    "loan",
]

NOMINAL_FEATURES = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
]

CATEGORICAL_FEATURES = BINARY_FEATURES + NOMINAL_FEATURES

ALL_RAW_FEATURES = OPERATIONAL_NUMERICAL_FEATURES + ["duration"] + CATEGORICAL_FEATURES

# Split Ratios & Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5

# Lead Quality Tier Thresholds
LEAD_TIER_HIGH_THRESHOLD = 0.60
LEAD_TIER_MEDIUM_THRESHOLD = 0.25
