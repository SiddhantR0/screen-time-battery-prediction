import math

import pytest

from screen_time_battery.prediction import predict_battery_used


def test_predict_battery_used_returns_float() -> None:
    """Test that a valid prediction returns a float."""
    prediction = predict_battery_used(60)

    assert isinstance(prediction, float)


def test_predict_battery_used_returns_expected_prediction() -> None:
    """Test prediction using the established baseline model."""
    prediction = predict_battery_used(60)

    assert prediction == pytest.approx(
        10.9327,
        abs=0.01,
    )


def test_predict_battery_used_accepts_integer_input() -> None:
    """Test integer screen-time input."""
    prediction = predict_battery_used(30)

    assert prediction > 0.0


def test_predict_battery_used_rejects_negative_input() -> None:
    """Test that negative screen time is rejected."""
    with pytest.raises(ValueError, match="cannot be negative"):
        predict_battery_used(-1)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_predict_battery_used_rejects_non_finite_input(
    invalid_value: float,
) -> None:
    """Test that non-finite screen-time values are rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        predict_battery_used(invalid_value)


def test_predict_battery_used_rejects_non_numeric_input() -> None:
    """Test that non-numeric screen time is rejected."""
    with pytest.raises(TypeError, match="must be a number"):
        predict_battery_used("60")  # type: ignore[arg-type]