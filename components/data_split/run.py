"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    Split the input dataset into training/validation and test sets and log them as W&B artifacts.

    Args:
        args: Command-line arguments containing the input artifact, test size,
            random seed, and optional stratification column.
    """

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for split_df, key in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {key}_data.csv dataset")

        split_df.to_csv(
            os.path.join(
                "../../data",
                f'{key}_data.csv'),
            index=False)

        # Log to W&B
        artifact = wandb.Artifact(
            f"{key}_data.csv",
            type=f"{key}_data",
            description=f"{key} split of dataset",
        )
        artifact.add_file(os.path.join("../../data", f'{key}_data.csv'))
        run.log_artifact(artifact)

        # Wait for the artifact to be logged before proceeding
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False)

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False)

    args = parser.parse_args()

    go(args)
