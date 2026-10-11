import pytest


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    rows = ["time,temp"] + [f"{t},{2.0 * t + 1.0}" for t in range(10)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
