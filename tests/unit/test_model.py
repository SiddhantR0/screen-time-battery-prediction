import numpy as np
import pytest

from screen_time_battery.model import ScreenTimeBatteryModel


def test_model_rejects_prediction_before_fitting() -> None:
    """Test that prediction requires a fitted model."""
    model = ScreenTimeBatteryModel()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[30.0]]))


def test_model_rejects_one_dimensional_training_data() -> None:
    """Test that training features must be two-dimensional."""
    model = ScreenTimeBatteryModel()

    with pytest.raises(ValueError):
        model.fit(
            np.array([10.0, 20.0, 30.0]),
            np.array([2.0, 4.0, 6.0]),
        )


def test_model_fits_and_exposes_parameters() -> None:
    """Test model fitting and learned parameters."""
    screen_time = np.array(
        [[10.0], [20.0], [30.0]],
    )
    battery_used = np.array([2.0, 4.0, 6.0])

    model = ScreenTimeBatteryModel()
    model.fit(screen_time, battery_used)

    assert model.coefficient == pytest.approx(0.2)
    assert model.intercept == pytest.approx(0.0)


def test_model_predicts_expected_values() -> None:
    """Test predictions from a fitted model."""
    screen_time = np.array(
        [[10.0], [20.0], [30.0]],
    )
    battery_used = np.array([2.0, 4.0, 6.0])

    model = ScreenTimeBatteryModel()
    model.fit(screen_time, battery_used)

    predictions = model.predict(
        np.array([[40.0], [50.0]]),
    )

    np.testing.assert_allclose(
        predictions,
        np.array([8.0, 10.0]),
    )