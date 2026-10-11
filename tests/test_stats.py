import numpy as np
import pytest

from scidata.stats import describe, linear_fit, moving_average, zscore_outliers


def test_describe_basic():
    s = describe([1, 2, 3, 4, 5])
    assert s.count == 5
    assert s.mean == pytest.approx(3.0)
    assert s.median == pytest.approx(3.0)
    assert s.std == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert (s.min, s.max) == (1.0, 5.0)


def test_describe_single_value_has_zero_std():
    assert describe([7]).std == 0.0


def test_describe_empty_raises():
    with pytest.raises(ValueError):
        describe([])


def test_moving_average_keeps_length_and_constant_signal():
    result = moving_average(np.full(10, 4.0), 3)
    assert result.shape == (10,)
    assert np.allclose(result, 4.0)


def test_moving_average_window_one_is_identity():
    data = np.array([1.0, 5.0, 2.0])
    assert np.allclose(moving_average(data, 1), data)


@pytest.mark.parametrize("window", [0, 11])
def test_moving_average_invalid_window(window):
    with pytest.raises(ValueError):
        moving_average(np.arange(10), window)


def test_zscore_detects_spike():
    data = np.zeros(50)
    data[10] = 100.0
    mask = zscore_outliers(data, 3.0)
    assert mask.sum() == 1
    assert mask[10]


def test_zscore_constant_signal_has_no_outliers():
    assert not zscore_outliers(np.ones(5)).any()


def test_zscore_invalid_threshold():
    with pytest.raises(ValueError):
        zscore_outliers([1, 2, 3], 0)


def test_linear_fit_exact_line():
    x = np.arange(20)
    fit = linear_fit(x, 3.0 * x - 2.0)
    assert fit.slope == pytest.approx(3.0)
    assert fit.intercept == pytest.approx(-2.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert np.allclose(fit.predict([0, 1]), [-2.0, 1.0])


def test_linear_fit_noisy_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 0.5 * x + 4 + rng.normal(0, 0.1, x.size)
    fit = linear_fit(x, y)
    assert fit.slope == pytest.approx(0.5, abs=0.02)
    assert 0.95 < fit.r_squared <= 1.0


def test_linear_fit_constant_y():
    assert linear_fit([0, 1, 2], [5, 5, 5]).r_squared == 1.0


@pytest.mark.parametrize(
    "x, y",
    [([1, 2, 3], [1, 2]), ([1], [1]), ([2, 2, 2], [1, 2, 3])],
)
def test_linear_fit_invalid_input(x, y):
    with pytest.raises(ValueError):
        linear_fit(x, y)
