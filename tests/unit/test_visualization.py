import matplotlib

matplotlib.use("Agg")

import numpy as np

from screen_time_battery.visualization import plot_regression_line


def test_plot_regression_line_creates_plot() -> None:
    """Test that the regression-line visualization executes successfully."""
    screen_time = np.array([10.0, 20.0, 30.0])
    battery_used = np.array([2.0, 4.0, 6.0])

    plot_regression_line(
        screen_time=screen_time,
        battery_used=battery_used,
        coefficient=0.2,
        intercept=0.0,
    )