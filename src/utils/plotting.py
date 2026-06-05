"""Plotting helpers for bandit experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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


def _plot_policy_summary_bars(
    summary: dict[str, dict[str, float]],
    metric_key: str,
    ylabel: str,
    title: str,
    multiplier: float = 1.0,
    value_suffix: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Plot a bar chart for a single summary metric."""
    policy_names = list(summary.keys())
    values = [summary[name][metric_key] * multiplier for name in policy_names]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[index % len(colors)] for index in range(len(policy_names))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policy_names, values, color=bar_colors)

    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    value_span = max_value - min_value
    padding = max(1.0, value_span * 0.12 if value_span > 0 else 1.0)

    ax.set_ylim(min_value - padding, max_value + padding)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)

    for bar, value in zip(bars, values):
        x_position = bar.get_x() + bar.get_width() / 2
        label = f"{value:.2f}{value_suffix}"
        if value >= 0:
            y_position = value + padding * 0.08
            vertical_alignment = "bottom"
        else:
            y_position = value - padding * 0.08
            vertical_alignment = "top"

        ax.text(
            x_position,
            y_position,
            label,
            va=vertical_alignment,
            ha="center",
        )

    ax.set_title(title)
    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_average_return_by_policy(
    summary: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot average return by policy."""
    _plot_policy_summary_bars(
        summary=summary,
        metric_key="average_return",
        ylabel="Average Return",
        title="Average Return by Policy",
        save_path=save_path,
    )


def plot_average_episode_length_by_policy(
    summary: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot average episode length by policy."""
    _plot_policy_summary_bars(
        summary=summary,
        metric_key="average_episode_length",
        ylabel="Average Episode Length",
        title="Average Episode Length by Policy",
        save_path=save_path,
    )


def plot_success_rate_by_policy(
    summary: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot success rate by policy."""
    _plot_policy_summary_bars(
        summary=summary,
        metric_key="success_rate",
        ylabel="Success Rate (%)",
        title="Success Rate by Policy",
        multiplier=100.0,
        value_suffix="%",
        save_path=save_path,
    )


def plot_state_visitation_heatmap(
    visitation_counts: np.ndarray,
    env: object,
    title: str = "State Visitation Heatmap",
    save_path: str | Path | None = None,
) -> None:
    """Plot a Gridworld visitation heatmap with obstacle, start, and goal labels."""
    heatmap_data = np.array(visitation_counts, dtype=float, copy=True)

    for obstacle in getattr(env, "obstacles", set()):
        heatmap_data[obstacle] = np.nan

    plt.figure(figsize=(7, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    image = plt.imshow(heatmap_data, cmap=cmap, origin="upper")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.colorbar(image, label="Visit Count")

    rows, cols = heatmap_data.shape
    plt.xticks(range(cols))
    plt.yticks(range(rows))

    for obstacle in getattr(env, "obstacles", set()):
        plt.text(obstacle[1], obstacle[0], "X", ha="center", va="center", color="black")

    start_row, start_col = env.start_state
    goal_row, goal_col = env.goal_state
    plt.text(start_col, start_row, "S", ha="center", va="center", color="white")
    plt.text(goal_col, goal_row, "G", ha="center", va="center", color="white")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()
