import numpy as np

from screen_time_battery.data import load_dataset
from screen_time_battery.training import split_dataset, train_model
from screen_time_battery.training import train_and_evaluate

def test_split_dataset_is_reproducible() -> None:
    """Test that identical random seeds produce identical splits."""
    screen_time, battery_used = load_dataset()

    first_split = split_dataset(
        screen_time,
        battery_used,
        random_state=42,
    )
    second_split = split_dataset(
        screen_time,
        battery_used,
        random_state=42,
    )

    for first, second in zip(first_split, second_split):
        np.testing.assert_array_equal(first, second)


def test_train_model_returns_predictions() -> None:
    """Test that the training workflow produces held-out predictions."""
    screen_time, battery_used = load_dataset()

    model, actual, predictions = train_model(
        screen_time,
        battery_used,
        random_state=42,
    )

    assert actual.shape == predictions.shape
    assert actual.size == 2
    assert model.coefficient > 0.0

def test_train_and_evaluate_is_reproducible() -> None:
    """Test that evaluation produces deterministic results."""
    screen_time, battery_used = load_dataset()

    _, first_metrics = train_and_evaluate(
        screen_time,
        battery_used,
        test_size=0.2,
        random_state=42,
    )

    _, second_metrics = train_and_evaluate(
        screen_time,
        battery_used,
        test_size=0.2,
        random_state=42,
    )

    assert first_metrics == second_metrics