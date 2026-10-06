import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain features for the Bank Marketing dataset.
    Preserves input dataframe by making a deep copy.
    Calculations do not introduce data leakage (row-independent).
    """
    df_out = df.copy()

    # 1. Contact History Indicators
    # pdays == -1 means client was never contacted before in a previous campaign
    if "pdays" in df_out.columns:
        df_out["was_previously_contacted"] = (df_out["pdays"] != -1).astype(int)
        # 999 is the standard benchmark convention for clients not previously contacted
        df_out["pdays_clean"] = df_out["pdays"].replace(-1, 999)

    # 2. Financial Strain Indicators
    if "balance" in df_out.columns:
        df_out["balance_is_negative"] = (df_out["balance"] < 0).astype(int)

    # 3. Overall Loan Burden
    if "housing" in df_out.columns and "loan" in df_out.columns:
        df_out["has_any_loan"] = (
            (df_out["housing"] == "yes") | (df_out["loan"] == "yes")
        ).astype(int)

    # 4. Standardize Binary Categoricals to Integers (0/1)
    for col in ["default", "housing", "loan"]:
        if col in df_out.columns:
            df_out[f"{col}_num"] = (df_out[col] == "yes").astype(int)

    return df_out
