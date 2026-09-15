import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    """
    Add custom command-line options for the pytest data tests.

    Args:
        parser: Pytest command-line option parser.
    """
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")


@pytest.fixture(scope='session')
def data(request):
    """
    Download and load the input dataset used for testing.

    Args:
        request: Pytest request object containing the command-line options.

    Returns:
        pandas.DataFrame: The input dataset.
    """
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact.
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    """
    Download and load the reference dataset used for comparison.

    Args:
        request: Pytest request object containing the command-line options.

    Returns:
        pandas.DataFrame: The reference dataset.
    """
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact.
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    """
    Retrieve the KL divergence threshold provided through the command line.

    Args:
        request: Pytest request object containing the command-line options.

    Returns:
        float: The KL divergence threshold.
    """
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)
