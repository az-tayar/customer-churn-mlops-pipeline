"""
Deploy a trained Customer Churn prediction model as a FastAPI service.
This module downloads an exported machine learning model artifact from
Weights & Biases (W&B), loads the model using MLflow, and exposes prediction
functionality through a FastAPI REST API.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn
import wandb
import argparse

# Define Pydantic models for request and response


class PredictionRequest(BaseModel):
    """
    Define the response schema for a churn prediction.
    """
    prediction: str


class InputData(BaseModel):
    """
    Define the input features required by the customer churn model.
    """
    Customer_Age: int
    Dependent_count: int
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: int
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Gender_Churn: float
    Education_Level_Churn: float
    Marital_Status_Churn: float
    Income_Category_Churn: float
    Card_Category_Churn: float


def create_app(args):
    """
    Create and configure the FastAPI application for model inference.
    Args:
        args: Command-line arguments containing the exported W&B model
            artifact identifier.
    Returns:
        fastapi.FastAPI: Configured FastAPI application.
    """
    # Initialize a W&B run for deployment
    run = wandb.init(job_type="deploy", entity='az-tayar-university-of-ottawa', project='CustomerChurnAI')
    run.config.update(args)

    # Download the exported model artifact from W&B
    model_path = run.use_artifact(args.export_model).download()
    model = mlflow.sklearn.load_model(model_path)

    # Create FastAPI app
    app = FastAPI(title="Customer Churn Predictor API",
                  description="Starter API for ML model deployment",
                  version="1.0.0")

    # Define API endpoints
    @app.get("/")
    async def root():
        """
        Return a welcome message and confirm that the API is running.
        Returns:
            dict: Message describing how to access the prediction endpoint.
        """
        return {"message": "Welcome to the Customer Churn Predictor API. Use the /predict endpoint to get churn predictions."}

    @app.post("/predict", response_model=PredictionRequest)
    async def predict_churn(input_data: InputData):
        """
        Predict whether a customer will churn.
        Args:
            input_data: Validated customer features required by the model.
        Returns:
            PredictionRequest: Predicted churn class for the customer.
        """
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])

        pred = model.predict(input_df)

        return {"prediction": str(pred[0])}

    return app


if __name__ == "__main__":
    # Extract the arguments
    parser = argparse.ArgumentParser(
        description="Run the FastAPI server for the Customer Churn Predictor.")

    parser.add_argument("--ip_address",
                        type=str,
                        help="the IP address for deploying the prod model",
                        required=True,
                        )

    parser.add_argument("--port",
                        type=int,
                        help="The port number for listening to the FastAPI server",
                        required=True,
                        )

    parser.add_argument("--export_model",
                        type=str,
                        help="exported model Artifact for deployment",
                        required=True,
                        )
    
    args = parser.parse_args()

    # Create the app
    app = create_app(args)
    uvicorn.run(app, host=args.ip_address, port=args.port)
