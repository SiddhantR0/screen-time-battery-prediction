import numpy as np
import pytest

from screen_time_battery.data import (
    BATTERY_USED_VALUES,
    SCREEN_TIME_VALUES,
    load_dataset,
    validate_dataset,
)


def test_load_dataset_returns_expected_shape() -> None:
    """Verify that features and targets have the expected dimensions."""
    screen_time, battery_used = load_dataset()

    assert screen_time.shape == (10, 1)
    assert battery_used.shape == (10,)


def test_load_dataset_preserves_source_values() -> None:
    """Verify that the supplied dataset values are preserved."""
    screen_time, battery_used = load_dataset()

    np.testing.assert_array_equal(
        screen_time.ravel(),
        np.asarray(SCREEN_TIME_VALUES),
    )
    np.testing.assert_array_equal(
        battery_used,
        np.asarray(BATTERY_USED_VALUES),
    )


def test_validate_dataset_accepts_valid_data() -> None:
    """Verify that valid numeric arrays pass validation."""
    screen_time = np.array([10.0, 20.0])
    battery_used = np.array([2.0, 4.0])

    validate_dataset(screen_time, battery_used)


def test_validate_dataset_rejects_empty_data() -> None:
    """Verify that empty datasets are rejected."""
    screen_time = np.array([])
    battery_used = np.array([])

    with pytest.raises(ValueError, match="must not be empty"):
        validate_dataset(screen_time, battery_used)


def test_validate_dataset_rejects_mismatched_lengths() -> None:
    """Verify that feature and target lengths must match."""
    screen_time = np.array([10.0, 20.0])
    battery_used = np.array([2.0])

    with pytest.raises(ValueError, match="matching lengths"):
        validate_dataset(screen_time, battery_used)


def test_validate_dataset_rejects_non_finite_values() -> None:
    """Verify that NaN and infinite values are rejected."""
    screen_time = np.array([10.0, np.nan])
    battery_used = np.array([2.0, 4.0])

    with pytest.raises(ValueError, match="finite"):
        validate_dataset(screen_time, battery_used)