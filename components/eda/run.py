import wandb
import pandas as pd
import argparse

import ydata_profiling


def go(args):
    """
    Perform exploratory data analysis on a W&B dataset artifact and generate a profiling report.

    Args:
        args: Command-line arguments containing the input artifact and
            output artifact configuration.
    """

    # Initialize a new W&B run
    run = wandb.init(project='CustomerChurnAI', group='eda')
    run.config.update(args)

    # load the input artifact
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    # Generate EDA report
    print(f'The 1st 5 rows of the dataset: \n{df.head()}')
    print(f'\nInfo for the dataset: \n{df.info()}')
    print(f'\nDescriptive statistics for the dataset: \n{df.describe()}')
    print(f'\nMissing values for the dataset: \n{df.isnull().sum()}')
    print(f'\nData types for the dataset: \n{df.dtypes}')

    profile = ydata_profiling.ProfileReport(df)
    profile.to_file("../../visualizations/eda_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis report")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact to do EDA on",
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
