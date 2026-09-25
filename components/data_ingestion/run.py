"""
This script download a URL to a local destination
"""
import argparse
import wandb

DATA_DIR = '../../results/data'

def go(args):
    """
    Upload a local data sample to Weights & Biases as an artifact.

    Args:
        args: Command-line arguments containing the sample file name,
            artifact name, artifact type, and artifact description.
    """

    run = wandb.init(job_type="data_ingestion")
    run.config.update(args)

    # Log to W&B
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(f'{DATA_DIR}/{args.sample}')
    run.log_artifact(artifact)

    # Wait for the artifact to be logged before proceeding
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download URL to a local destination")

    parser.add_argument(
        "sample",
        type=str,
        help="Name of the sample to download")

    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name for the output artifact")

    parser.add_argument(
        "artifact_type",
        type=str,
        help="Output artifact type.")

    parser.add_argument(
        "artifact_description",
        type=str,
        help="A brief description of this artifact")

    args = parser.parse_args()

    go(args)
