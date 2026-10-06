import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.config import (
        PROCESSED_DATA_DIR,
        PREPROCESSOR_PATH,
        FEATURE_MANIFEST_PATH,
        TARGET,
        TARGET_MAPPING,
        OPERATIONAL_NUMERICAL_FEATURES,
        BENCHMARK_NUMERICAL_FEATURES,
        NOMINAL_FEATURES,
        RANDOM_STATE,
        TEST_SIZE,
        VAL_SIZE,
    )
    from src.data_loader import load_raw_data
    from src.feature_engineering import engineer_features
except ModuleNotFoundError:
    from config import (
        PROCESSED_DATA_DIR,
        PREPROCESSOR_PATH,
        FEATURE_MANIFEST_PATH,
        TARGET,
        TARGET_MAPPING,
        OPERATIONAL_NUMERICAL_FEATURES,
        BENCHMARK_NUMERICAL_FEATURES,
        NOMINAL_FEATURES,
        RANDOM_STATE,
        TEST_SIZE,
        VAL_SIZE,
    )
    from data_loader import load_raw_data
    from feature_engineering import engineer_features



def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split into Train (70%), Validation (15%), and Test (15%).
    Saves split dataframes to data/processed for full reproducibility.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=random_state,
    )

    # 2. Split remaining into train and validation
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df[TARGET],
        random_state=random_state,
    )

    # Save to disk
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    return train_df, val_df, test_df


def build_preprocessor(include_duration: bool = False) -> Tuple[ColumnTransformer, List[str]]:
    """
    Construct Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.
    Excludes 'duration' by default to avoid look-ahead data leakage for pre-call lead scoring.
    """
    base_numeric = (
        BENCHMARK_NUMERICAL_FEATURES if include_duration else OPERATIONAL_NUMERICAL_FEATURES
    )
    # Replace pdays with pdays_clean in numeric list
    numeric_cols = [col if col != "pdays" else "pdays_clean" for col in base_numeric]

    binary_engineered_cols = [
        "was_previously_contacted",
        "balance_is_negative",
        "has_any_loan",
        "default_num",
        "housing_num",
        "loan_num",
    ]

    nominal_cols = NOMINAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal_cols,
            ),
            ("bin", "passthrough", binary_engineered_cols),
        ],
        remainder="drop",
    )

    expected_input_cols = list(set(base_numeric + nominal_cols + ["default", "housing", "loan"]))
    return preprocessor, expected_input_cols


def fit_and_save_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_duration: bool = False,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    feature_manifest_path: Path = FEATURE_MANIFEST_PATH,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Fit preprocessor ONLY on training data, transform splits, and serialize artifacts.
    Prevents train/test contamination and leakage.
    """
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Feature Engineering
    train_feat = engineer_features(train_df)
    val_feat = engineer_features(val_df)
    test_feat = engineer_features(test_df)

    y_train = train_feat[TARGET].map(TARGET_MAPPING).values
    y_val = val_feat[TARGET].map(TARGET_MAPPING).values
    y_test = test_feat[TARGET].map(TARGET_MAPPING).values

    # 2. Build and fit preprocessor on train only
    preprocessor, expected_input_cols = build_preprocessor(include_duration=include_duration)
    X_train = preprocessor.fit_transform(train_feat)
    X_val = preprocessor.transform(val_feat)
    X_test = preprocessor.transform(test_feat)

    # 3. Retrieve output feature names
    feature_names = list(preprocessor.get_feature_names_out())

    # 4. Serialize preprocessor and feature manifest
    joblib.dump(preprocessor, preprocessor_path)

    manifest = {
        "include_duration": include_duration,
        "input_features": expected_input_cols,
        "output_feature_count": len(feature_names),
        "output_features": feature_names,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }

    with open(feature_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return preprocessor, X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def prepare_data(include_duration: bool = False):
    """Convenience pipeline runner from raw data to transformed train/val/test matrices."""
    df = load_raw_data()
    train_df, val_df, test_df = split_dataset(df)
    return fit_and_save_pipeline(train_df, val_df, test_df, include_duration=include_duration)


if __name__ == "__main__":
    print("Preparing and transforming dataset...")
    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_data()
    print(f"X_train shape: {X_train.shape}, y_train mean: {y_train.mean():.4f}")
    print(f"X_val shape:   {X_val.shape}, y_val mean:   {y_val.mean():.4f}")
    print(f"X_test shape:  {X_test.shape}, y_test mean:  {y_test.mean():.4f}")
    print(f"Total features created: {len(feat_names)}")
    print(f"Artifacts saved to {PREPROCESSOR_PATH} and {FEATURE_MANIFEST_PATH}")
