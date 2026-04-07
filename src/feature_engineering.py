"""Feature engineering module for E-Commerce Churn dataset."""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from the cleaned dataset.

    New features:
        1. tenure_bucket — categorical binning of Tenure
        2. engagement_score — HourSpendOnApp * OrderCount
        3. cashback_per_order — CashbackAmount / OrderCount
        4. is_recent_buyer — 1 if DaySinceLastOrder <= 3
        5. has_multi_device — 1 if NumberOfDeviceRegistered >= 4
        6. is_high_spender — 1 if OrderAmountHikeFromlastYear > 20
    """
    df = df.copy()

    # 1. Tenure bucket
    bins = [0, 6, 12, 24, np.inf]
    labels = ["new", "settling", "established", "loyal"]
    df["tenure_bucket"] = pd.cut(df["Tenure"], bins=bins, labels=labels, include_lowest=True)

    # 2. Engagement score
    df["engagement_score"] = df["HourSpendOnApp"] * df["OrderCount"]

    # 3. Cashback per order (handle division by zero)
    df["cashback_per_order"] = np.where(
        df["OrderCount"] > 0,
        df["CashbackAmount"] / df["OrderCount"],
        0,
    )

    # 4. Is recent buyer
    df["is_recent_buyer"] = (df["DaySinceLastOrder"] <= 3).astype(int)

    # 5. Has multi device
    df["has_multi_device"] = (df["NumberOfDeviceRegistered"] >= 4).astype(int)

    # 6. Is high spender
    df["is_high_spender"] = (df["OrderAmountHikeFromlastYear"] > 20).astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables.

    Label encoding: Gender, MaritalStatus, tenure_bucket
    One-hot encoding: PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat
    """
    df = df.copy()

    # Label encoding
    label_maps = {
        "Gender": {"Male": 0, "Female": 1},
        "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
        "tenure_bucket": {"new": 0, "settling": 1, "established": 2, "loyal": 3},
    }
    for col, mapping in label_maps.items():
        df[col] = df[col].map(mapping)

    # One-hot encoding
    ohe_cols = ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
        1. Create 6 new features
        2. Encode categorical variables

    Returns:
        Feature-engineered DataFrame ready for modeling.
    """
    df = create_features(df)
    df = encode_features(df)
    return df


if __name__ == "__main__":
    from data_cleaning import load_raw_data, clean

    raw = load_raw_data()
    cleaned = clean(raw)
    print(f"Cleaned shape: {cleaned.shape}")

    engineered = engineer_features(cleaned)
    print(f"Engineered shape: {engineered.shape}")
    print(f"\nColumns:\n{engineered.columns.tolist()}")
    print(f"\nDtypes:\n{engineered.dtypes}")

    engineered.to_csv("data/processed/engineered.csv", index=False)
    print("\nSaved to data/processed/engineered.csv")
