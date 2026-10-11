import numpy as np
import pytest

from scidata.io import DataError, load_columns


def test_load_columns(csv_file):
    x, y = load_columns(csv_file, "time", "temp")
    assert x.size == 10
    assert np.allclose(y, 2.0 * x + 1.0)


def test_load_sorts_by_x(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n3,30\n1,10\n2,20\n", encoding="utf-8")
    x, y = load_columns(path, "a", "b")
    assert list(x) == [1.0, 2.0, 3.0]
    assert list(y) == [10.0, 20.0, 30.0]


def test_load_skips_empty_cells(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a;b\n1;10\n2;\n3;30\n", encoding="utf-8")
    x, _ = load_columns(path, "a", "b", delimiter=";")
    assert list(x) == [1.0, 3.0]


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_columns(tmp_path / "none.csv", "a", "b")


def test_missing_column(csv_file):
    with pytest.raises(DataError, match="column 'pressure'"):
        load_columns(csv_file, "time", "pressure")


def test_non_numeric_value(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\nx,3\n", encoding="utf-8")
    with pytest.raises(DataError, match="line 3"):
        load_columns(path, "a", "b")


def test_too_few_points(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DataError, match="two data points"):
        load_columns(path, "a", "b")
