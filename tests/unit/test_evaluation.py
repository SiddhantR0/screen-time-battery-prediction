import numpy as np
import pytest

from screen_time_battery.evaluation import (
    calculate_mse,
    calculate_r2,
    calculate_rmse,
    evaluate_predictions,
)


def test_calculate_mse() -> None:
    """Test mean squared error calculation."""
    actual = np.array([2.0, 4.0, 6.0])
    predicted = np.array([1.0, 5.0, 7.0])

    assert calculate_mse(actual, predicted) == pytest.approx(1.0)


def test_calculate_rmse() -> None:
    """Test root mean squared error calculation."""
    actual = np.array([2.0, 4.0, 6.0])
    predicted = np.array([1.0, 5.0, 7.0])

    assert calculate_rmse(actual, predicted) == pytest.approx(1.0)


def test_calculate_r2() -> None:
    """Test R-squared calculation."""
    actual = np.array([2.0, 4.0, 6.0])
    predicted = np.array([2.0, 4.0, 6.0])

    assert calculate_r2(actual, predicted) == pytest.approx(1.0)


def test_evaluate_predictions_returns_all_metrics() -> None:
    """Test the combined evaluation function."""
    actual = np.array([2.0, 4.0, 6.0])
    predicted = np.array([1.0, 5.0, 7.0])

    metrics = evaluate_predictions(actual, predicted)

    assert set(metrics) == {"mse", "rmse", "r2"}
    assert metrics["mse"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)