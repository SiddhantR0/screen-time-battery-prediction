from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_mse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate mean squared error."""
    return float(mean_squared_error(actual, predicted))


def calculate_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate root mean squared error."""
    mse = calculate_mse(actual, predicted)
    return float(np.sqrt(mse))


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate the coefficient of determination."""
    return float(r2_score(actual, predicted))


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate all baseline evaluation metrics."""
    return {
        "mse": calculate_mse(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "r2": calculate_r2(actual, predicted),
    }