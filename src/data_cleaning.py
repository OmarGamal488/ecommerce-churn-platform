"""Data cleaning module for E-Commerce Churn dataset."""

import pandas as pd
import numpy as np


def load_raw_data(path: str = "data/raw/E Commerce Dataset.xlsx") -> pd.DataFrame:
    """Load the raw dataset from the Excel file."""
    return pd.read_excel(path, sheet_name="E Comm")


def merge_duplicate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate category names in categorical columns."""
    df = df.copy()

    # PreferredLoginDevice: "Phone" → "Mobile Phone"
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")

    # PreferredPaymentMode: "CC" → "Credit Card", "COD" → "Cash on Delivery"
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    })

    # PreferedOrderCat: "Mobile" → "Mobile Phone"
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

    return df


def fix_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data entry errors in WarehouseToHome."""
    df = df.copy()
    # Values 126 and 127 are data entry errors (prepended "1")
    df["WarehouseToHome"] = df["WarehouseToHome"].replace({126: 26, 127: 27})
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with median for numeric columns."""
    df = df.copy()
    cols_to_impute = [
        "Tenure", "WarehouseToHome", "HourSpendOnApp",
        "OrderAmountHikeFromlastYear", "CouponUsed",
        "OrderCount", "DaySinceLastOrder",
    ]
    for col in cols_to_impute:
        df[col] = df[col].fillna(df[col].median())
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Steps:
        1. Drop CustomerID
        2. Merge duplicate categories
        3. Fix WarehouseToHome outliers
        4. Impute missing values with median

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()
    df = df.drop(columns=["CustomerID"])
    df = merge_duplicate_categories(df)
    df = fix_outliers(df)
    df = impute_missing(df)
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    print(f"Raw shape: {raw.shape}")
    print(f"Missing before: {raw.isnull().sum().sum()}")

    cleaned = clean(raw)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing after: {cleaned.isnull().sum().sum()}")

    cleaned.to_csv("data/processed/cleaned.csv", index=False)
    print("Saved to data/processed/cleaned.csv")
