from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT_DIRECTORY = Path("assets")

def plot_screen_time_vs_battery(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Display and optionally save a scatter plot of screen time against battery use."""
    figure = plt.figure(figsize=(8, 5))
    plt.scatter(screen_time, battery_used)
    plt.xlabel("Screen Time (minutes)")
    plt.ylabel("Battery Used (percentage points)")
    plt.title("Screen Time vs. Battery Use")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(figure)


def plot_regression_line(
    screen_time: np.ndarray,
    battery_used: np.ndarray,
    coefficient: float,
    intercept: float,
    output_path: Path | None = None,
) -> None:
    """Display and optionally save observations with the fitted regression line."""
    x_values = np.linspace(
        float(np.min(screen_time)),
        float(np.max(screen_time)),
        100,
    )
    y_values = intercept + coefficient * x_values

    figure = plt.figure(figsize=(8, 5))
    plt.scatter(screen_time, battery_used, label="Observed data")
    plt.plot(x_values, y_values, label="Regression line")
    plt.xlabel("Screen Time (minutes)")
    plt.ylabel("Battery Used (percentage points)")
    plt.title("Screen Time vs. Battery Use")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(figure)