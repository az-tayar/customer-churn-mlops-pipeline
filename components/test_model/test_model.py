"""
Test model performance metrics against minimum acceptable thresholds.
"""

import json
from pathlib import Path

METRICS_PATH = (
    Path(__file__).resolve().parents[2]
    / "metrics"
    / "test_data_metrics.json"
)

def test_metrics_file_exists():
    """Verify that model evaluation generated the metrics file."""
    assert METRICS_PATH.exists()


def test_model_metrics():
    """Verify that the model meets minimum performance requirements."""
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert metrics["accuracy"] >= 0.90
    assert metrics["precision"] >= 0.80
    assert metrics["recall"] >= 0.80
    assert metrics["f1"] >= 0.80
