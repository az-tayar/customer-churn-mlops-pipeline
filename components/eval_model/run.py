"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import wandb
import mlflow
import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
               'Income_Category', 'Card_Category']

METRICS_DIR = '../../results/metrics'
TEST_FILE = 'test_data_metrics.json'

def go(args):
    """
    Evaluate the trained MLflow model against the test dataset and log classification metrics.

    Args:
        args: Command-line arguments containing the MLflow model artifact
            and test dataset artifact.
    """

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    # Download input artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    df = pd.read_csv(test_dataset_path)

    # Drop the original categorical columns from the DataFrame
    df.drop(columns=cat_columns, inplace=True)

    X_test = df.drop(columns=["Churn"])
    y_test = df["Churn"]

    model = mlflow.sklearn.load_model(model_local_path)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Log metrics
    run.summary['accuracy'] = accuracy
    run.summary['precision'] = precision
    run.summary['recall'] = recall
    run.summary['f1'] = f1

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # save the metrics locally
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(f'{METRICS_DIR}/{TEST_FILE}', 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
