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

LOGS_DIR = "./results/logs"
LOGS_FILE = "pipeline.log"

def run_component(name, path, parameters=None, critical=True):
    """Run an MLflow component with timing and error logging."""
    start_time = time.perf_counter()
    logging.info(f"Starting {name} step.")

    try:
        result = mlflow.run(
            path,
            entry_point="main",
            env_manager="conda",
            parameters=parameters
        )
        execution_time = (time.perf_counter() - start_time) / 60
        logging.info(
            f"{name} completed successfully - "
            f"execution time: {execution_time:.2f} mins"
        )
        return result

    except Exception:
        execution_time = (time.perf_counter() - start_time) / 60
        logging.exception(
            f"{name} failed after {execution_time:.2f} mins"
        )
        if critical:
            raise

        return None


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
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=f'{LOGS_DIR}/{LOGS_FILE}',
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


    # --------------------------------------------------
    # 1. DATA INGESTION
    # --------------------------------------------------
    if "data_ingestion" in active_steps:
        run_component(
            name="Data ingestion",
            path="components/data_ingestion",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "dataset.csv",
                "artifact_type": "raw_dataset",
                "artifact_description": "Raw file as downloaded"
            }
        )

    # -------------------------------------------------
    # 2. PREPROCESSING
    # --------------------------------------------------
    if "preprocessing" in active_steps:
        run_component(
            name="Preprocessing",
            path="components/preprocessing",
            parameters={
                "input_artifact": "dataset.csv:latest",
                "output_artifact": "clean_dataset.csv",
                "output_type": "clean_dataset",
                "output_description":
                    "Data with outliers and null values removed"
            }
        )

    # --------------------------------------------------
    # 3. EDA
    # --------------------------------------------------
    if "eda" in active_steps:
        run_component(
            name="EDA",
            path="components/eda",
            parameters={
                "input_artifact": "clean_dataset.csv:latest",
                "output_artifact": "eda_report.html",
                "output_type": "eda_report",
                "output_description":
                    "Exploratory data analysis report"
            },
            critical=False
        )

    # --------------------------------------------------
    # 4. DATA TESTING
    # --------------------------------------------------
    if "test_data" in active_steps:
        run_component(
            name="Data testing",
            path="components/test_data",
            parameters={
                "csv": "clean_dataset.csv:latest",
                "ref": "clean_dataset.csv:reference",
                "kl_threshold":
                    config["data_check"]["kl_threshold"]
            }
        )

    # --------------------------------------------------
    # 5. DATA SPLITTING
    # --------------------------------------------------
    if "data_split" in active_steps:
        run_component(
            name="Data splitting",
            path="components/data_split",
            parameters={
                "input": "clean_dataset.csv:latest",
                "test_size":
                    config["modeling"]["test_size"],
                "random_seed":
                    config["modeling"]["random_seed"],
                "stratify_by":
                    config["modeling"]["stratify_by"]
            }
        )

    # --------------------------------------------------
    # 6. MODEL TRAINING
    # --------------------------------------------------
    if "train_model" in active_steps:
        run_component(
            name="Model training",
            path="components/train_model",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "output_artifact": "random_forest_export"
            }
        )

    # --------------------------------------------------
    # 7. MODEL EVALUATION
    # --------------------------------------------------
    if "eval_model" in active_steps:
        run_component(
            name="Model evaluation",
            path="components/eval_model",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_dataset": "test_data.csv:latest"
            }
        )

    # --------------------------------------------------
    # 8. MODEL TESTING
    # --------------------------------------------------
    if "test_model" in active_steps:
        run_component(
            name="Model testing",
            path="components/test_model"
        )

    # --------------------------------------------------
    # 9. API TESTING
    # --------------------------------------------------
    if "test_api" in active_steps:
        run_component(
            name="API testing",
            path="components/test_api"
        )

    # --------------------------------------------------
    # 10. MODEL DEPLOYMENT
    # --------------------------------------------------
    if "deploy_model" in active_steps:
        run_component(
            name="Model deployment",
            path="components/deploy_model",
            parameters={
                "ip_address": config["main"]["ip_address"],
                "port": config["main"]["port"],
                "export_model": "random_forest_export:prod"
            }
        )


if __name__ == "__main__":
    go()
