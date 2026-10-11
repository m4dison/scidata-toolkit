from pathlib import Path

from scidata.cli import analyze
from scidata.io import load_columns

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cooling_experiment.csv"


def test_example_dataset_has_known_outliers():
    x, y = load_columns(EXAMPLE, "time_s", "temperature_c")
    report, smoothed, _, _ = analyze(x, y, window=5, threshold=2.5)
    assert report["summary"]["count"] == 120
    assert report["linear_fit"]["slope"] < 0
    assert report["outliers"]["count"] >= 2
    assert smoothed.size == y.size
