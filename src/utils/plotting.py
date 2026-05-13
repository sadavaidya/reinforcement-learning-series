"""Plotting helpers for Week 1 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def plot_average_reward(results: dict[float, dict[str, object]], save_path: str | Path | None = None) -> None:
    """Plot average reward over time for each epsilon."""
    plt.figure(figsize=(10, 6))

    for epsilon, data in results.items():
        plt.plot(data["average_rewards"], label=f"epsilon = {epsilon}")

    plt.title("Average Reward over Time")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


def plot_optimal_action_percentage(
    results: dict[float, dict[str, object]], save_path: str | Path | None = None
) -> None:
    """Plot optimal action selection percentage for each epsilon."""
    plt.figure(figsize=(10, 6))

    for epsilon, data in results.items():
        plt.plot(data["optimal_action_percentage"], label=f"epsilon = {epsilon}")

    plt.title("Optimal Action Selection over Time")
    plt.xlabel("Steps")
    plt.ylabel("Optimal Action (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()
