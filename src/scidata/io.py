import csv
from pathlib import Path

import numpy as np


class DataError(ValueError):
    pass


def load_columns(path, x_column, y_column, delimiter=","):
    path = Path(path)
    if not path.exists():
        raise DataError(f"file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = reader.fieldnames or []
        for name in (x_column, y_column):
            if name not in fields:
                raise DataError(f"column '{name}' not found, available: {', '.join(fields)}")
        xs, ys = [], []
        for line_number, row in enumerate(reader, start=2):
            raw_x, raw_y = row[x_column].strip(), row[y_column].strip()
            if raw_x == "" or raw_y == "":
                continue
            try:
                xs.append(float(raw_x))
                ys.append(float(raw_y))
            except ValueError as exc:
                raise DataError(f"non numeric value at line {line_number}") from exc
    if len(xs) < 2:
        raise DataError("at least two data points are required")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]
