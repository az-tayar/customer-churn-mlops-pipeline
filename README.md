# Customer Churn MLOps Pipeline

This project is a full machine learning operations workflow for predicting customer churn using a tabular customer dataset. It follows a reproducible pipeline pattern with MLflow, Weights & Biases (W&B), Hydra-based configuration, modular components, automated validation, and a deployable FastAPI inference service.

The repository is designed to:
- ingest a customer dataset from W&B artifacts
- clean and prepare data for modeling
- run exploratory data analysis
- validate data quality and schema expectations
- split data into train/validation/test sets
- train and tune a Random Forest classifier
- evaluate model quality with standard metrics
- test the model against acceptance thresholds
- deploy the trained model as an API for real-time predictions
- provide a Streamlit web application for interactive customer churn predictions

## Project overview

This project focuses on customer churn prediction, where the target variable is `Churn` and the model predicts whether a customer is likely to churn (1) or stay (0). The pipeline uses a Random Forest model because it performs well on structured tabular data and is easy to deploy with MLflow.

The project is organized around reusable ML pipeline components under the `components/` directory, each with its own `MLproject`, `conda.yml`, and Python entry script. The orchestration is handled by the root `main.py` and `config.yaml` files.

## Core files

### `main.py`
This is the orchestrator for the end-to-end pipeline. It defines a list of pipeline stages and executes each one via `mlflow.run(...)` using Hydra configuration values. It also sets up logging and environment variables used by W&B.

### `config.yaml`
This file stores the project configuration, including:
- project name and experiment name
- pipeline step selection
- API host, URL, and port
- Streamlit frontend port
- data validation threshold
- train/test split settings
- Random Forest hyperparameter grid

### `MLproject`
This file defines the project entry point for MLflow, enabling commands like:

```bash
mlflow run .
```

The root `MLproject` can run the whole pipeline or a subset of steps with the `steps` parameter.

## Pipeline stages

The pipeline is structured into modular steps, as defined in `main.py`.

### 1) Data ingestion
Location: `components/data_ingestion/`

- uploads the raw dataset into a W&B artifact
- stores it as a reusable artifact for downstream steps

### 2) Preprocessing
Location: `components/preprocessing/`

- loads the dataset artifact
- removes duplicate records
- drops rows with missing values
- creates a binary target field `Churn`
- derives churn-encoded categorical features
- saves the cleaned dataset as a new artifact

### 3) Exploratory data analysis (EDA)
Location: `components/eda/`

- generates the EDA report and visual summaries
- marks this step as non-critical so the pipeline can keep going even if analysis fails

### 4) Data testing
Location: `components/test_data/`

- validates the expected dataset schema
- ensures row counts are reasonable
- checks that the target distribution matches a reference distribution within a KL divergence threshold

### 5) Data splitting
Location: `components/data_split/`

- splits the cleaned dataset into train/validation and test sets
- creates W&B artifacts for both split datasets

### 6) Model training
Location: `components/train_model/`

- loads the train/validation dataset
- drops the original categorical columns used for engineered churn features
- trains a `RandomForestClassifier` with a grid search over hyperparameters
- uses F1 score as the selection metric
- saves the best model in MLflow format
- logs confusion matrix and validation metrics to W&B

### 7) Model evaluation
Location: `components/eval_model/`

- loads the saved model
- evaluates it using the test dataset
- computes accuracy, precision, recall, and F1 score
- writes metrics to `results/metrics/`

### 8) Model testing
Location: `components/test_model/`

- verifies the generated metrics file exists
- checks that the model meets minimum quality thresholds

### 9) API testing
Location: `components/test_api/`

- tests the FastAPI service using `TestClient`
- validates the root endpoint and churn prediction responses

### 10) Model deployment
Location: `components/deploy_model/`

- loads the exported model from W&B
- deploys it behind a FastAPI server
- exposes a prediction endpoint for inference

### 11) Streamlit web application
Location: `components/streamlit_ui/`

- loads the test-data and exported-model artifacts from W&B
- provides an interactive web interface for selecting a customer
- sends the selected customer's features to the deployed FastAPI service
- displays the churn prediction, churn probability, and actual test value

This stage runs after model deployment when the pipeline is executed with `steps=all`. It can also be run independently with `steps=streamlit_ui`, provided the API service is already available.

## Project configuration

The project uses Hydra configuration via `config.yaml`.

Key settings include:

```yaml
main:
  project_name: CustomerChurnAI
  experiment_name: development
  steps: all
  host: 0.0.0.0
  api_url: http://backend:8000
  api_port: 8000
  streamlit_port: 8501

etl:
  sample: "dataset.csv"

data_check:
  kl_threshold: 0.2

modeling:
  test_size: 0.2
  val_size: 0.2
  random_seed: 42
  stratify_by: Churn
  n_jobs: -1
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [5, 10, 15, 20]
    min_samples_split: [2, 4, 8]
    min_samples_leaf: [1, 2, 3]
    max_features: [0.33, 0.5]
    criterion: ["gini", "entropy", "log_loss"]
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.13 (recommended in the project environment setup)
- Conda or Miniconda installed
- Access to W&B and a valid API key
- MLflow available in the environment
- Internet access for downloading artifacts and packages

## Setup

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd customer-churn-mlops-pipeline
```

### 2) Create the environment

```bash
conda env create -f conda.yml
conda activate components
```

If you prefer to use the requirements file in a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Log in to W&B

```bash
wandb login
```

You can also set your entity/project environment variables if needed.

## Running with Docker Compose

Install Docker with Docker Compose and create a `.env` file in the project root with your W&B API key (`WANDB_API_KEY`). Run the following commands from the project root.

### Run the pipeline service

```bash
docker compose run --rm pipeline
```

This runs the pipeline and removes the service container when it finishes. Output files are saved in the local `results/` directory.

### Start the API service

```bash
docker compose up -d backend
```

This starts the API service in the background, available at `http://localhost:8000` once startup completes. The configured model artifact must be accessible from W&B.

### Start the frontend web application

The Streamlit frontend uses the test-data artifact to let you select a customer and sends that customer's features to the FastAPI backend for prediction. Start the backend and frontend together:

```bash
docker compose up --build backend frontend
```

Open `http://localhost:8501` in a browser after the services start. The frontend expects the W&B artifacts configured by the pipeline:

- `test_data.csv:latest` for the customer records
- `random_forest_export:prod` for the exported model

The frontend reads the backend URL from `config.yaml`. When running with Docker Compose, the configured URL is `http://backend:8000`, which allows the frontend container to reach the backend service by its Compose service name.

## Running the pipeline

### Run the entire pipeline

```bash
mlflow run .
```

This executes all pipeline stages defined in `main.py` and the configuration file.

### Run only selected steps

```bash
mlflow run . -P steps=data_ingestion,preprocessing
```

```bash
mlflow run . -P steps=train_model,eval_model,test_model
```

## Model deployment and API usage

The project includes a FastAPI service for model inference.

### Start the API locally

```bash
mlflow run . -P steps=deploy_model
```

Once running, the app exposes:

- `GET /` — welcome endpoint
- `POST /predict` — churn prediction endpoint

### Example request payload

```json
{
  "Customer_Age": 45,
  "Dependent_count": 2,
  "Months_on_book": 120,
  "Total_Relationship_Count": 4,
  "Months_Inactive_12_mon": 2,
  "Contacts_Count_12_mon": 1,
  "Credit_Limit": 12000.0,
  "Total_Revolving_Bal": 300,
  "Avg_Open_To_Buy": 11700.0,
  "Total_Amt_Chng_Q4_Q1": 0.75,
  "Total_Trans_Amt": 4500,
  "Total_Trans_Ct": 60,
  "Total_Ct_Chng_Q4_Q1": 0.8,
  "Avg_Utilization_Ratio": 0.25,
  "Gender_Churn": 0.4,
  "Education_Level_Churn": 0.3,
  "Marital_Status_Churn": 0.5,
  "Income_Category_Churn": 0.2,
  "Card_Category_Churn": 0.7
}
```

### Example response

```json
{
  "prediction": 0,
  "churn_probability": 0.12
}
```

The API returns a numeric prediction and the estimated probability of churn where:
- `0` = customer is not likely to churn
- `1` = customer is likely to churn
- `churn_probability` is a value between `0` and `1`

## Streamlit web application

The interactive frontend is implemented in [`components/streamlit_ui/web_app.py`](components/streamlit_ui/web_app.py). It displays customer details from the test-data artifact and, when **Analyze Customer** is selected, shows:

- the predicted churn class
- the churn probability
- the actual churn value from the selected test record

### Run the frontend through MLflow

Start the API in one terminal:

```bash
mlflow run . -P steps=deploy_model
```

In a second terminal, start the Streamlit component:

```bash
mlflow run . -P steps=streamlit_ui
```

The default Streamlit port is `8501`. If running outside Docker Compose, update `main.api_url` in `config.yaml` to the address where the FastAPI service is reachable (for example, `http://localhost:8000`).

## Validation and testing

The repository includes validation tests for dataset quality and API behavior.

### Data validation tests

```bash
mlflow run . -P steps=test_data
```

These tests verify:
- correct column names and ordering
- acceptable row counts
- acceptable class distribution divergence using KL divergence

### API tests

```bash
mlflow run . -P steps=test_api
```

These tests validate:
- the root endpoint is healthy
- the prediction endpoint accepts valid features
- predictions match expected churn outputs for known examples

### Model threshold checks

The model tests verify that the model achieves minimum acceptable thresholds, including:
- accuracy >= 0.90
- precision >= 0.80
- recall >= 0.80
- F1 score >= 0.80

## Output artifacts

This project stores model and pipeline artifacts under the `results/` directory.

```text
results/
├── data/
├── logs/
├── metrics/
├── model/
└── visualizations/
```

## W&B and MLflow integration

This project relies on both W&B and MLflow for tracking and artifact management.

### W&B
W&B is used for:
- dataset artifacts
- model artifacts
- experiment tracking
- metric summaries
- confusion matrix visualization

### MLflow
MLflow is used for:
- model packaging and tracking
- reproducible model export
- scalable experiment management
- loading and serving deployed models

## Summary

This repository is a complete example of an MLOps-style customer churn prediction pipeline. It covers:
- reproducible data processing
- model training with hyperparameter tuning
- model validation and testing
- experiment tracking with W&B and MLflow
- deployment as a FastAPI service
- interactive customer churn dashboard using Streamlit
- automated checks for model quality and API health

It is suitable for learning, experimentation, and real-world project demonstrations in ML engineering and MLOps workflows.
