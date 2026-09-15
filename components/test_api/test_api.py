"""
Test the deployed Customer Churn Prediction FastAPI service.

This module performs integration tests against a running FastAPI server
using HTTP requests. It verifies that the root endpoint is accessible and
that the prediction endpoint correctly predicts both non-churn and churn
customer examples from the training/validation dataset.

The FastAPI server must be running before executing these tests.
"""

import requests
import pandas as pd


# Categorical columns that are not required by the deployed model
cat_columns = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category"
]

# URL of the running FastAPI server
url_path = "http://127.0.0.1:8000"


def test_root():
    """
    Test the root endpoint of the FastAPI service.

    Sends a GET request to the root endpoint and verifies that the API
    returns HTTP status code 200 and the expected welcome message.
    """
    response = requests.get(
        f"{url_path}/",
        timeout=10
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": (
            "Welcome to the Customer Churn Predictor API. "
            "Use the /predict endpoint to get churn predictions."
        )
    }


def test_predict_churn_0():
    """
    Test prediction for a non-churn customer.

    Loads the training/validation dataset, selects the first customer whose
    actual churn label is 0, removes columns that are not required by the
    deployed model, and sends the customer's features to the prediction
    endpoint.

    The returned prediction is expected to be 0.
    """
    df = pd.read_csv("../../data/trainval_data.csv")

    df.drop(columns=cat_columns, inplace=True)

    input_data = df[df["Churn"] == 0].iloc[0]

    # Remove target variable before sending features to the API
    input_data = input_data.drop("Churn")

    response = requests.post(
        f"{url_path}/predict",
        json=input_data.to_dict(),
        timeout=10
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "0"


def test_predict_churn_1():
    """
    Test prediction for a churn customer.

    Loads the training/validation dataset, selects the first customer whose
    actual churn label is 1, removes columns that are not required by the
    deployed model, and sends the customer's features to the prediction
    endpoint.

    The returned prediction is expected to be 1.
    """
    df = pd.read_csv("../../data/trainval_data.csv")

    df.drop(columns=cat_columns, inplace=True)

    input_data = df[df["Churn"] == 1].iloc[0]

    # Remove target variable before sending features to the API
    input_data = input_data.drop("Churn")

    response = requests.post(
        f"{url_path}/predict",
        json=input_data.to_dict(),
        timeout=10
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "1"
