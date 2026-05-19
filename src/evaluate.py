import joblib
import pandas as pd
import numpy as np


def predict_test_data(test_df):

    model = joblib.load("outputs/models/best_model.pkl")

    ids = test_df["Id"]

    test_df = pd.get_dummies(test_df, drop_first=True)

    predictions = model.predict(test_df)

    predictions = np.expm1(predictions)

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions
    })

    submission.to_csv(
        "outputs/predictions/submission.csv",
        index=False
    )

    print("Predictions saved successfully.")