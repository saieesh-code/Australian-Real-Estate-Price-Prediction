import pandas as pd
import numpy as np


def load_data(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def handle_missing_values(df):

    # =========================================
    # NUMERICAL COLUMNS
    # =========================================

    num_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in num_cols:

        if df[col].isnull().sum() > 0:

            median_value = df[col].median()

            df[col] = df[col].fillna(
                median_value
            )

    # =========================================
    # SPECIAL DOMAIN COLUMNS
    # =========================================

    special_cols = [
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "Alley"
    ]

    for col in special_cols:

        if col in df.columns:

            df[col] = df[col].astype(object)

            df[col] = df[col].fillna(
                "None"
            )

    # =========================================
    # CATEGORICAL COLUMNS
    # =========================================

    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in cat_cols:

        if df[col].isnull().sum() > 0:

            mode_value = df[col].mode()[0]

            df[col] = df[col].fillna(
                mode_value
            )

    return df


def remove_outliers(df):

    if "SalePrice" in df.columns:

        df = df[
            ~(
                (df["GrLivArea"] > 4000) &
                (df["SalePrice"] < 300000)
            )
        ]

    return df