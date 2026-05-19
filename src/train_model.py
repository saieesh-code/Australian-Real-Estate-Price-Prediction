import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
import joblib


def prepare_data(df):

    X = df.drop("SalePrice", axis=1)
    y = np.log1p(df["SalePrice"])

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 30)
    print(f"MSE : {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2  : {r2}")

    return rmse


def train_models(df):

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():

        if name == "Random Forest":
            model.fit(X_train, y_train)
            rmse = evaluate_model(
                name,
                model,
                X_test,
                y_test
            )
        else:
            model.fit(X_train_scaled, y_train)

            rmse = evaluate_model(
                name,
                model,
                X_test_scaled,
                y_test
            )

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    joblib.dump(best_model, "outputs/models/best_model.pkl")

    print("\nBest model saved successfully.")