from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from screen_time_battery.model import ScreenTimeBatteryModel


def split_dataset(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split the dataset into training and testing subsets."""
    return train_test_split(
        screen_time,
        battery_used,
        test_size=test_size,
        random_state=random_state,
    )


def train_model(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    ScreenTimeBatteryModel,
    np.ndarray,
    np.ndarray,
]:
    """Split data, fit the model, and return held-out predictions."""
    (
        screen_time_train,
        screen_time_test,
        battery_used_train,
        battery_used_test,
    ) = split_dataset(
        screen_time,
        battery_used,
        test_size=test_size,
        random_state=random_state,
    )

    model = ScreenTimeBatteryModel()
    model.fit(screen_time_train, battery_used_train)

    predictions = model.predict(screen_time_test)

    return model, battery_used_test, predictions