from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_screen_time_vs_battery(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
) -> None:
    """Display a scatter plot of screen time against battery use."""
    plt.figure(figsize=(8, 5))
    plt.scatter(screen_time, battery_used)
    plt.xlabel("Screen Time (minutes)")
    plt.ylabel("Battery Used (percentage points)")
    plt.title("Screen Time vs. Battery Use")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_regression_line(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
    coefficient: float,
    intercept: float,
) -> None:
    """Display observations together with the fitted regression line."""
    x_values = np.linspace(
        float(np.min(screen_time)),
        float(np.max(screen_time)),
        100,
    )
    y_values = intercept + coefficient * x_values

    plt.figure(figsize=(8, 5))
    plt.scatter(screen_time, battery_used, label="Observed data")
    plt.plot(x_values, y_values, label="Regression line")
    plt.xlabel("Screen Time (minutes)")
    plt.ylabel("Battery Used (percentage points)")
    plt.title("Screen Time vs. Battery Use")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()