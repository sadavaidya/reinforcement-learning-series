"""Plotting helpers for bandit experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def _format_result_label(label: object) -> str:
    """Format result labels while preserving Week 1 epsilon labels."""
    if isinstance(label, (int, float)):
        return f"epsilon = {label}"
    return str(label)


def plot_average_reward(
    results: dict[object, dict[str, object]],
    save_path: str | Path | None = None,
    title: str = "Average Reward over Time",
) -> None:
    """Plot average reward over time for one or more agent results."""
    plt.figure(figsize=(10, 6))

    for label, data in results.items():
        plt.plot(data["average_rewards"], label=_format_result_label(label))

    plt.title(title)
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
    results: dict[object, dict[str, object]],
    save_path: str | Path | None = None,
    title: str = "Optimal Action Selection over Time",
) -> None:
    """Plot optimal action selection percentage for one or more agent results."""
    plt.figure(figsize=(10, 6))

    for label, data in results.items():
        plt.plot(
            data["optimal_action_percentage"],
            label=_format_result_label(label),
        )

    plt.title(title)
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
