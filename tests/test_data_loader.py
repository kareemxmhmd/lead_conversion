import pytest
import pandas as pd
from src.data_loader import load_raw_data, validate_raw_data
from src.config import RAW_DATA_PATH, ALL_RAW_FEATURES, TARGET


def test_load_raw_data_success():
    df = load_raw_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (4521, 17)
    assert TARGET in df.columns


def test_validate_raw_data_valid():
    df = load_raw_data(RAW_DATA_PATH)
    report = validate_raw_data(df)
    assert report["num_rows"] == 4521
    assert report["num_columns"] == 17
    assert report["total_nulls"] == 0
    assert report["duplicate_rows"] == 0
    assert "no" in report["target_counts"]
    assert "yes" in report["target_counts"]
    assert report["target_counts"]["no"] == 4000
    assert report["target_counts"]["yes"] == 521


def test_validate_raw_data_missing_column():
    df = load_raw_data(RAW_DATA_PATH).drop(columns=[TARGET])
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_raw_data(df)


def test_validate_raw_data_invalid_target_values():
    df = load_raw_data(RAW_DATA_PATH)
    df_corrupted = df.copy()
    df_corrupted.loc[0, TARGET] = "invalid_class"
    with pytest.raises(ValueError, match="Unexpected target values"):
        validate_raw_data(df_corrupted)
