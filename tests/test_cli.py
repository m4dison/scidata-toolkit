import json

import pytest

from scidata.cli import main


def test_text_report(csv_file, capsys):
    assert main([str(csv_file), "--x", "time", "--y", "temp"]) == 0
    out = capsys.readouterr().out
    assert "samples:    10" in out
    assert "y = 2.0000 * x + 1.0000" in out


def test_json_report_and_plot(csv_file, tmp_path, capsys):
    image = tmp_path / "chart.png"
    code = main([str(csv_file), "--x", "time", "--y", "temp", "--json", "--plot", str(image)])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["count"] == 10
    assert report["linear_fit"]["slope"] == pytest.approx(2.0)
    assert report["outliers"]["count"] == 0
    assert image.exists() and image.stat().st_size > 0


def test_large_window_is_clamped(csv_file):
    assert main([str(csv_file), "--x", "time", "--y", "temp", "--window", "50"]) == 0


def test_error_exit_code(csv_file, capsys):
    assert main([str(csv_file), "--x", "time", "--y", "missing"]) == 1
    assert "error:" in capsys.readouterr().err
