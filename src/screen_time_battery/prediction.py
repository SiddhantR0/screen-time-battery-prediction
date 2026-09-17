from __future__ import annotations

import math

import numpy as np

from screen_time_battery.data import load_dataset
from screen_time_battery.training import train_model


def predict_battery_used(screen_time: float) -> float:
    """Predict battery percentage points consumed for screen time."""
    if not isinstance(screen_time, (int, float)):
        raise TypeError("screen_time must be a number.")

    screen_time_value = float(screen_time)

    if not math.isfinite(screen_time_value):
        raise ValueError("screen_time must be finite.")

    if screen_time_value < 0:
        raise ValueError("screen_time cannot be negative.")

    features, target = load_dataset()

    model, _, _ = train_model(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    prediction = model.predict(
        np.array([[screen_time_value]], dtype=np.float64),
    )

    return float(prediction[0])