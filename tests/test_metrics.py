"""Unit tests for core regression metrics."""

from util.metrics import kge, mae, mse, nse, regression_metrics, rmse


def test_basic_metrics_values() -> None:
    observed = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 4.0]

    assert mse(observed, predicted) == 1.0 / 3.0
    assert mae(observed, predicted) == 1.0 / 3.0
    assert rmse(observed, predicted) > 0.0
    assert nse(observed, predicted) < 1.0
    assert kge(observed, predicted) < 1.0


def test_regression_metrics_contains_expected_keys() -> None:
    metrics = regression_metrics([1.0, 2.0], [1.0, 2.5])

    assert set(metrics) == {"mse", "mae", "rmse", "nse", "kge"}
