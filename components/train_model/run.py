"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import yaml
import pandas as pd
import wandb
import json

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
               'Income_Category', 'Card_Category']


def go(args):
    """
    Train, evaluate, and export a Random Forest classification pipeline.

    Args:
        args: Command-line arguments containing the training artifact,
            validation size, random seed, stratification setting, and
            output artifact name.
    """

    run = wandb.init(job_type="train")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open('../../config.yaml') as f:
        rf_config = yaml.safe_load(f)['modeling']
    run.config.update(rf_config)

    # Get the train and validation artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(trainval_local_path)

    # Drop the original categorical columns from the DataFrame
    df.drop(columns=cat_columns, inplace=True)

    # Split the dataset into features (X) and target (y)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=rf_config['val_size'], stratify=y, random_state=rf_config['random_seed'])

    logger.info("Preparing sklearn pipeline")

    sk_pipe = get_inference_pipeline(rf_config['random_seed'])

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    param_grid = {
        "model__n_estimators": rf_config['random_forest']['n_estimators'],
        "model__max_depth": rf_config['random_forest']['max_depth'],
        "model__min_samples_split": rf_config['random_forest']['min_samples_split'],
        "model__min_samples_leaf": rf_config['random_forest']['min_samples_leaf'],
        "model__max_features": rf_config['random_forest']['max_features'],
        "model__criterion": rf_config['random_forest']['criterion']}

    grid_search = GridSearchCV(
        estimator=sk_pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # Fit the pipeline sk_pipe by calling the .fit method on X_train and y_train
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    train_metrics = {
        'best_params': grid_search.best_params_,
        'best_cv_f1': grid_search.best_score_
    }

    # save the best model, parameters, and score
    os.makedirs('../../metrics', exist_ok=True)
    with open('../../metrics/train_data_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=4)

    # Compute the metrics on the validation set
    logger.info("Metrics Computation")

    y_pred = best_model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, pos_label=1)
    recall = recall_score(y_val, y_pred, pos_label=1)
    f1 = f1_score(y_val, y_pred, pos_label=1)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    if os.path.exists("../../model"):
        shutil.rmtree("../../model")

    # Save the sk_pipe pipeline as a mlflow.sklearn model
    mlflow.sklearn.save_model(best_model, "../../model")

    # Upload the model we just exported to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export",
        description="Random Forest model exported in MLFlow format",
        metadata=train_metrics
    )

    artifact.add_dir("../../model")
    run.log_artifact(artifact)

    # Plot feature importance
    fig_conf_mtrx = plot_confusion_matrix(y_val, y_pred)

    # Saving the confusion matrix figure locally
    with open("../../visualizations/confusion_matrix.png", "wb") as f:
        fig_conf_mtrx.savefig(f, format="png")

    # Save metrics results to W&B summary
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
    os.makedirs('../../metrics', exist_ok=True)
    with open('../../metrics/val_data_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Upload to W&B the confusion matrix figure
    run.log({"confusion_matrix": wandb.Image(fig_conf_mtrx)})


def plot_confusion_matrix(y_val, y_pred):
    """
    Create a confusion matrix figure from validation labels and predictions.

    Args:
        y_val: True target values from the validation dataset.
        y_pred: Predicted target values from the trained model.

    Returns:
        matplotlib.figure.Figure: Figure containing the confusion matrix.
    """

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            0,
            1]).plot(
        ax=ax)
    return fig


def get_inference_pipeline(random_seed):
    """
    Create the preprocessing and Random Forest inference pipeline.

    Args:
        random_seed: Random seed used for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The sklearn pipeline.
    """

    # Create a median imputer for missing values
    median_imputer = SimpleImputer(strategy="median")

    # Create the sklearn pipeline with the preprocessor and the Random Forest
    # model
    sk_pipe = Pipeline([
        ('imputer', median_imputer),
        ('model', RandomForestClassifier(random_state=random_seed, n_jobs=-1))
    ])

    return sk_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()
    go(args)
