import mlflow
import os
import hydra
import logging
import time

steps = [
    "data_ingestion",
    "preprocessing",
    "eda",
    "test_data",
    "data_split",
    "train_model",
    "eval_model",
    "test_model"
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
            ingestion_start_time = time.time()

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
            ingestion_end_time = time.time()
            ingestion_exc_time = ingestion_end_time - ingestion_start_time
            logging.info(f"Data ingestion step completed successfully - execution time: {ingestion_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running data_ingestion: {e}")
            raise

    if "preprocessing" in active_steps:
        try:
            preprocessing_start_time = time.time()

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
            preprocessing_end_time = time.time()
            preprocessing_exc_time = preprocessing_end_time - preprocessing_start_time
            logging.info(f"Preprocessing step completed successfully - execution time: {preprocessing_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running preprocessing: {e}")
            raise

    if "eda" in active_steps:
        try:
            eda_start_time = time.time()

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
            eda_end_time = time.time()
            eda_exc_time = eda_end_time - eda_start_time
            logging.info(f"EDA step completed successfully - execution time: {eda_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running eda: {e}")

    if "test_data" in active_steps:
        try:
            test_data_start_time = time.time()

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
            test_data_end_time = time.time()
            test_data_exc_time = test_data_end_time - test_data_start_time
            logging.info(f"Data testing step completed successfully - execution time: {test_data_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running data testing: {e}")
            raise

    if "data_split" in active_steps:
        try:
            data_split_start_time = time.time()

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
            data_split_end_time = time.time()
            data_split_exc_time = data_split_end_time - data_split_start_time
            logging.info(f"Data splitting step completed successfully - execution time: {data_split_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running data splitting: {e}")
            raise

    if "train_model" in active_steps:
        try:
            train_start_time = time.time()

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
            train_end_time = time.time()
            train_exc_time = train_end_time - train_start_time
            logging.info(f"Model training step completed successfully - execution time: {train_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running model training: {e}")
            raise

    if "eval_model" in active_steps:
        try:
            eval_model_start_time = time.time()

            # Run the model evaluating step using MLflow
            _ = mlflow.run(
                "components/eval_model",
                entry_point="main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                },
            )
            eval_model_end_time = time.time()
            eval_model_exc_time = eval_model_end_time - eval_model_start_time
            logging.info(f"Model evaluating step completed successfully - execution time: {eval_model_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running model testing: {e}")
            raise

    if "test_model" in active_steps:
        try:
            test_model_start_time = time.time()

            # Run the model testing step using MLflow
            _ = mlflow.run(
                "components/test_model",
                entry_point="main",
                env_manager="conda",
            )
            test_model_end_time = time.time()
            test_model_exc_time = test_model_end_time - test_model_start_time
            logging.info(f"Model testing step completed successfully - execution time: {test_model_exc_time:.2f} sec")

        except Exception as e:
            logging.error(f"Error occurred while running model testing: {e}")    
            raise

    if "deploy_model" in active_steps:
        try:
            logging.info("Starting model deployment service...")

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
            
        except Exception as e:
            logging.error(f"Error occurred while running model deployment: {e}")
            raise

    if "test_api" in active_steps:
        try:
            test_api_start_time = time.time()

            # Run some tests for fastapi running server
            _ = mlflow.run(
                "components/test_api",
                entry_point="main",
                env_manager="conda",
            )
            test_api_end_time = time.time()
            test_api_exc_time = test_api_end_time - test_api_start_time
            logging.info(f"API testing step completed successfully - execution time: {test_api_exc_time:.2f} sec") 

        except Exception as e:
            logging.error(f"Error occurred while testing API running server: {e}")
            raise


if __name__ == "__main__":
    go()
