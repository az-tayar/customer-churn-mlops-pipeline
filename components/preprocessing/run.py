"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

def go(args):
    """
    Download the raw dataset from W&B, apply basic data cleaning, and log the cleaned dataset as a new artifact.

    Args:
        args: Command-line arguments containing the input artifact and
            output artifact configuration.
    """

    run = wandb.init(
        job_type="preprocessing",
        settings=wandb.Settings(disable_job_creation=False),
    )
    run.config.update(args)

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Map the 'Attrition_Flag' column to a binary 'Churn' column
    df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    df.drop(columns=["Attrition_Flag"], inplace=True)

    # Create new features based on categorical columns and the 'Churn' column
    for cat_col in cat_columns:
        df[cat_col + '_' + 'Churn'] = df[cat_col].map(
            df.groupby(cat_col)['Churn'].mean()
        )
  
    logger.info("Cleaned data has %s rows and %s columns", *df.shape)

    # Save cleaned data
    df.to_csv(os.path.join("../../data", args.output_artifact), index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(os.path.join("../../data", args.output_artifact))
    run.log_artifact(artifact)

    artifact.wait()
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True
    )

    args = parser.parse_args()

    go(args)
