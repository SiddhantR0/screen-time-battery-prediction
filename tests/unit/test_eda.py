import numpy as np
import pytest

from screen_time_battery.data import load_dataset
from screen_time_battery.eda import (
    calculate_correlation,
    calculate_summary_statistics,
    check_duplicate_observations,
    check_missing_values,
)


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return the supplied project dataset."""
    return load_dataset()


def test_summary_statistics(dataset: tuple[np.ndarray, np.ndarray]) -> None:
    """Test descriptive statistics for the supplied dataset."""
    screen_time, battery_used = dataset

    statistics = calculate_summary_statistics(
        screen_time.ravel(),
        battery_used,
    )

    assert statistics["screen_time"]["count"] == 10.0
    assert statistics["screen_time"]["minimum"] == 10.0
    assert statistics["screen_time"]["maximum"] == 150.0
    assert statistics["battery_used"]["minimum"] == 2.0
    assert statistics["battery_used"]["maximum"] == 30.0


def test_missing_values_are_absent(
    dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that the supplied dataset contains no missing values."""
    screen_time, battery_used = dataset

    missing = check_missing_values(
        screen_time.ravel(),
        battery_used,
    )

    assert missing == {
        "screen_time": 0,
        "battery_used": 0,
    }


def test_duplicate_observations_are_absent(
    dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that the supplied dataset contains no duplicate observations."""
    screen_time, battery_used = dataset

    duplicates = check_duplicate_observations(
        screen_time.ravel(),
        battery_used,
    )

    assert duplicates == 0


def test_correlation_is_strongly_positive(
    dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test the direction and strength of the observed relationship."""
    screen_time, battery_used = dataset

    correlation = calculate_correlation(
        screen_time.ravel(),
        battery_used,
    )

    assert correlation > 0.95