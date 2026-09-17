from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class ScreenTimeBatteryModel:
    """Simple linear regression model for battery-use prediction."""

    def __init__(self) -> None:
        """Initialize an unfitted linear regression model."""
        self._model = LinearRegression()
        self._is_fitted = False

    def fit(
        self,
        screen_time: np.ndarray,
        battery_used: np.ndarray,
    ) -> None:
        """Fit the regression model to training data."""
        if screen_time.ndim != 2:
            raise ValueError("screen_time must be a two-dimensional array.")

        self._model.fit(screen_time, battery_used)
        self._is_fitted = True

    def predict(self, screen_time: np.ndarray) -> np.ndarray:
        """Predict battery use for supplied screen-time values."""
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")

        if screen_time.ndim != 2:
            raise ValueError("screen_time must be a two-dimensional array.")

        return self._model.predict(screen_time)

    @property
    def coefficient(self) -> float:
        """Return the fitted screen-time coefficient."""
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before inspection.")

        return float(self._model.coef_[0])

    @property
    def intercept(self) -> float:
        """Return the fitted model intercept."""
        if not self._is_fitted:
            raise RuntimeError("The model must be fitted before inspection.")

        return float(self._model.intercept_)