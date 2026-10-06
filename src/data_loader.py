from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

try:
    from src.config import RAW_DATA_PATH, ALL_RAW_FEATURES, TARGET
except ModuleNotFoundError:
    from config import RAW_DATA_PATH, ALL_RAW_FEATURES, TARGET


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw bank marketing data from CSV.
    The raw dataset uses semicolon (';') delimiters.
    """
    path = Path(file_path) if file_path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path, sep=";")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate raw dataset schema, check missing values, duplicates, and target distribution.
    Does NOT modify the dataframe or apply preprocessing.
    """
    expected_cols = set(ALL_RAW_FEATURES + [TARGET])
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        raise ValueError(f"Missing expected columns in raw data: {missing_cols}")

    # Check for empty dataframe
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    # Target integrity check
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")

    unique_targets = set(df[TARGET].dropna().unique())
    expected_targets = {"no", "yes"}
    if not unique_targets.issubset(expected_targets):
        raise ValueError(f"Unexpected target values: {unique_targets - expected_targets}")

    # Compute validation and profile summary
    validation_report = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET].value_counts(normalize=True).to_dict(),
        "target_counts": df[TARGET].value_counts().to_dict(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    return validation_report


def inspect_dataset() -> Dict[str, Any]:
    """Convenience function to load and validate raw data, printing summary."""
    df = load_raw_data()
    report = validate_raw_data(df)

    print("=" * 60)
    print("BANK MARKETING DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Shape: {report['num_rows']} rows x {report['num_columns']} columns")
    print(f"Total Null Values: {report['total_nulls']}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print("\nTarget Distribution:")
    for label, count in report["target_counts"].items():
        pct = report["target_distribution"][label] * 100
        print(f"  - '{label}': {count:,} ({pct:.2f}%)")
    print("=" * 60)
    return report


if __name__ == "__main__":
    inspect_dataset()
