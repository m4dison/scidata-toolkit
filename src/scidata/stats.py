from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float
    std: float
    min: float
    median: float
    max: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r_squared: float

    def predict(self, x):
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def as_dict(self):
        return asdict(self)


def describe(values):
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("cannot describe an empty array")
    std = float(data.std(ddof=1)) if data.size > 1 else 0.0
    return Summary(
        count=int(data.size),
        mean=float(data.mean()),
        std=std,
        min=float(data.min()),
        median=float(np.median(data)),
        max=float(data.max()),
    )


def moving_average(values, window):
    data = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be a positive integer")
    if window > data.size:
        raise ValueError("window is larger than the number of samples")
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def zscore_outliers(values, threshold=3.0):
    data = np.asarray(values, dtype=float)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    std = data.std()
    if std == 0:
        return np.zeros(data.shape, dtype=bool)
    return np.abs((data - data.mean()) / std) > threshold


def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        raise ValueError("at least two points are required")
    if np.all(x == x[0]):
        raise ValueError("x values must not be constant")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 if total == 0 else 1.0 - float(np.sum(residual**2) / total)
    return LinearFit(float(slope), float(intercept), r_squared)
