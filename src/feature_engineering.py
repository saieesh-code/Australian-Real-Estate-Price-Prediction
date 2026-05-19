def create_features(df):

    # Total Square Feet
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = (
            df["TotalBsmtSF"] +
            df["1stFlrSF"] +
            df["2ndFlrSF"]
        )

    # House Age
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Total Bathrooms
    if {"FullBath", "HalfBath"}.issubset(df.columns):
        df["TotalBath"] = (
            df["FullBath"] +
            (0.5 * df["HalfBath"])
        )

    # Has Garage
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    return df