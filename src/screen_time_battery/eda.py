from __future__ import annotations

from typing import Any

import numpy as np


def calculate_summary_statistics(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Calculate descriptive statistics for both dataset variables."""
    return {
        "screen_time": _summary(screen_time),
        "battery_used": _summary(battery_used),
    }


def check_missing_values(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
) -> dict[str, int]:
    """Count missing values in each dataset variable."""
    return {
        "screen_time": int(np.isnan(screen_time).sum()),
        "battery_used": int(np.isnan(battery_used).sum()),
    }


def check_duplicate_observations(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
) -> int:
    """Count duplicate screen-time/battery-use observations."""
    observations = np.column_stack((screen_time, battery_used))
    return int(observations.shape[0] - np.unique(observations, axis=0).shape[0])


def calculate_correlation(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
) -> float:
    """Calculate Pearson correlation between the two variables."""
    return float(np.corrcoef(screen_time, battery_used)[0, 1])


def _summary(values: np.ndarray) -> dict[str, float]:
    """Return descriptive statistics for a one-dimensional array."""
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
    }