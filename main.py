import mlflow
import os
import hydra
import logging

steps = [
    "data_ingestion",
    "preprocessing",
    "eda",
    "test_data",
    "data_split",
    "train_model",
    "test_model",
    ]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config):
    """
    Execute the configured ML pipeline steps using Hydra, MLflow, and W&B.

    Args:
        config: Hydra configuration object containing project settings,
            pipeline steps, ETL parameters, data checks, and modeling parameters.
    """
    # Set up logging
    LOGS_DIR = "./logs"
    LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")
    os.makedirs(LOGS_DIR, exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )

    logging.info("Pipeline started.")

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    if steps_par == "all":
        active_steps = steps
    elif isinstance(steps_par, str):
        active_steps = steps_par.split(",")
    else:
        active_steps = list(steps_par)


    if "data_ingestion" in active_steps:
        try:
            # Download file and load in W&B
            _ = mlflow.run(
                "components/data_ingestion",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "dataset.csv",
                    "artifact_type": "raw_dataset",
                    "artifact_description": "Raw file as downloaded"
                },
            )
            logging.info("Data ingestion step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running data_ingestion: {e}")

    if "preprocessing" in active_steps:
        try:
            # Run the preprocessing step using MLflow
            _ = mlflow.run(
                "components/preprocessing",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "input_artifact": "dataset.csv:latest",
                    "output_artifact": "clean_dataset.csv",
                    "output_type": "clean_dataset",
                    "output_description": "Data with outliers and null values removed",
                },
            )
            logging.info("Preprocessing step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running preprocessing: {e}")

    if "eda" in active_steps:
        try:
            # Run the EDA step using MLflow
            _ = mlflow.run(
                "components/eda",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "input_artifact": "clean_dataset.csv:latest",
                    "output_artifact": "eda_report.html",
                    "output_type": "eda_report",
                    "output_description": "Exploratory data analysis report"
                },
            )
            logging.info("EDA step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running eda: {e}")

    if "test_data" in active_steps:
        try:
            # Run the data testing step using MLflow
            _ = mlflow.run(
                "components/test_data",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "csv": "clean_dataset.csv:latest",
                    "ref": "clean_dataset.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                },
            )
            logging.info("Data testing step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running data testing: {e}")

    if "data_split" in active_steps:
        try:
            # Run the data splitting step using MLflow
            _ = mlflow.run(
                "components/data_split",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "input": "clean_dataset.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )
            logging.info("Data splitting step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running data splitting: {e}")

    if "train_model" in active_steps:
        try:
            # Run the model training step using MLflow
            _ = mlflow.run(
                "components/train_model",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "output_artifact": 'random_forest_export'
                },
            )
            logging.info("Model training step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running model training: {e}")

    if "test_model" in active_steps:
        try:
            # Run the model testing step using MLflow
            _ = mlflow.run(
                "components/test_model",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:latest",
                    "test_dataset": "test_data.csv:latest"
                },
            )
            logging.info("Model testing step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running model testing: {e}")

    if "deploy_model" in active_steps:
        try:
            # Run the model deployment step using MLflow
            _ = mlflow.run(
                "components/deploy_model",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "ip_address": config["main"]["ip_address"],
                    "port": config["main"]["port"],
                    "export_model": "random_forest_export:prod",
                },
            )
            logging.info("Model deployment step completed successfully.")

        except Exception as e:
            logging.error(f"Error occurred while running model deployment: {e}")

    if "test_api" in active_steps:
        try:
            # Run some tests for fastapi running server
            _ = mlflow.run(
                "components/test_api",
                entry_point="main",
                env_manager="conda",
            )
            logging.info("API testing step completed successfully.") 

        except Exception as e:
            logging.error(f"Error occurred while testing API running server: {e}")


if __name__ == "__main__":
    go()
