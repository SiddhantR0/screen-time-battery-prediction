"""Visualization utilities for exploratory data analysis."""

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