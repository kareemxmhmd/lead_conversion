import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, TARGET
from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.preprocessing import split_dataset, build_preprocessor, fit_and_save_pipeline


def test_engineer_features():
    df = load_raw_data(RAW_DATA_PATH).head(50)
    df_feat = engineer_features(df)

    assert "was_previously_contacted" in df_feat.columns
    assert "pdays_clean" in df_feat.columns
    assert "balance_is_negative" in df_feat.columns
    assert "has_any_loan" in df_feat.columns
    assert "default_num" in df_feat.columns
    assert "housing_num" in df_feat.columns
    assert "loan_num" in df_feat.columns

    # Verify input df was not mutated
    assert "was_previously_contacted" not in df.columns


def test_split_dataset_stratification():
    df = load_raw_data(RAW_DATA_PATH)
    train_df, val_df, test_df = split_dataset(df)

    total_len = len(df)
    assert abs(len(train_df) / total_len - 0.70) < 0.02
    assert abs(len(val_df) / total_len - 0.15) < 0.02
    assert abs(len(test_df) / total_len - 0.15) < 0.02

    # Verify stratification (positive class ratio roughly 11.5% across all)
    overall_pos_rate = (df[TARGET] == "yes").mean()
    train_pos_rate = (train_df[TARGET] == "yes").mean()
    val_pos_rate = (val_df[TARGET] == "yes").mean()
    test_pos_rate = (test_df[TARGET] == "yes").mean()

    assert abs(train_pos_rate - overall_pos_rate) < 0.01
    assert abs(val_pos_rate - overall_pos_rate) < 0.02
    assert abs(test_pos_rate - overall_pos_rate) < 0.02


def test_preprocessor_fit_transform_and_reload(tmp_path):
    df = load_raw_data(RAW_DATA_PATH)
    train_df, val_df, test_df = split_dataset(df)

    temp_prep_path = tmp_path / "preprocessor.pkl"
    temp_manifest_path = tmp_path / "features.json"

    preprocessor, X_train, y_train, X_val, y_val, X_test, y_test, feat_names = fit_and_save_pipeline(
        train_df,
        val_df,
        test_df,
        include_duration=False,
        preprocessor_path=temp_prep_path,
        feature_manifest_path=temp_manifest_path,
    )

    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_test.shape[0] == len(test_df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Verify reloaded preprocessor reproduces transformations
    loaded_prep = joblib.load(temp_prep_path)
    test_feat = engineer_features(test_df)
    X_test_reloaded = loaded_prep.transform(test_feat)

    np.testing.assert_array_almost_equal(X_test, X_test_reloaded)
