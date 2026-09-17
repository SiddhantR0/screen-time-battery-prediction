from typing import Final

import numpy as np
from numpy.typing import NDArray


SCREEN_TIME_VALUES: Final[tuple[float, ...]] = (
    10.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
    135.0,
    150.0,
)

BATTERY_USED_VALUES: Final[tuple[float, ...]] = (
    2.0,
    5.0,
    6.5,
    10.0,
    14.0,
    18.0,
    20.0,
    23.0,
    26.0,
    30.0,
)


def load_dataset() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    screen_time = np.asarray(SCREEN_TIME_VALUES, dtype=np.float64)
    battery_used = np.asarray(BATTERY_USED_VALUES, dtype=np.float64)

    validate_dataset(screen_time, battery_used)

    return screen_time.reshape(-1, 1), battery_used


def validate_dataset(
    screen_time: NDArray[np.float64],
    battery_used: NDArray[np.float64],
) -> None:
    if screen_time.size == 0 or battery_used.size == 0:
        raise ValueError("Dataset must not be empty.")

    if screen_time.size != battery_used.size:
        raise ValueError(
            "Screen-time and battery-used values must have matching lengths."
        )

    if not np.issubdtype(screen_time.dtype, np.number):
        raise ValueError("Screen-time values must be numeric.")

    if not np.issubdtype(battery_used.dtype, np.number):
        raise ValueError("Battery-used values must be numeric.")

    if not np.all(np.isfinite(screen_time)):
        raise ValueError("Screen-time values must be finite.")

    if not np.all(np.isfinite(battery_used)):
        raise ValueError("Battery-used values must be finite.")