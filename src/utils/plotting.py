"""Plotting helpers for bandit and Gridworld experiments."""

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


# ---------------------------------------------------------------------------
# Week 5 - value function plotting helpers
# ---------------------------------------------------------------------------


def plot_value_function_grid(
    value_grid: np.ndarray,
    env: object,
    title: str = "State-Value Function",
    save_path: str | Path | None = None,
) -> None:
    """Heatmap of V(s) over the Gridworld with per-cell value labels.

    Obstacles appear in light-gray (NaN cells).  The start cell is annotated
    with 'S' and the goal cell with 'G'.
    """
    rows, cols = value_grid.shape
    obstacles = getattr(env, "obstacles", set())
    start_state = getattr(env, "start_state", None)
    goal_state = getattr(env, "goal_state", None)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="lightgray")

    image = ax.imshow(value_grid, cmap=cmap, origin="upper")
    fig.colorbar(image, ax=ax, label="V(s)")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in obstacles:
                ax.text(col, row, "X", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")
            else:
                val = value_grid[row, col]
                val_text = f"{val:.2f}" if not np.isnan(val) else ""
                marker = ""
                if state == start_state:
                    marker = "\nS"
                elif state == goal_state:
                    marker = "\nG"
                ax.text(col, row, val_text + marker, ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_value_convergence(
    convergence_histories: dict[str, list[float]],
    title: str = "Policy Evaluation Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Max Value Change (delta)",
    save_path: str | Path | None = None,
) -> None:
    """Plot max-delta vs iteration for one or more policy evaluation runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, history in convergence_histories.items():
        ax.plot(range(1, len(history) + 1), history, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_iteration_comparison(
    comparison: dict[str, float],
    ylabel: str = "Count",
    title: str = "Dynamic Programming Iteration Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing one scalar metric across algorithms."""
    labels = list(comparison.keys())
    values = [comparison[label] for label in labels]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=bar_colors)

    ymax = max(values) if values else 0.0
    padding = max(0.5, ymax * 0.1 if ymax > 0 else 0.5)
    ax.set_ylim(0.0, ymax + padding)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + padding * 0.05,
            f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_policy_value_comparison(
    summary: dict[str, float],
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing average state value across policies."""
    policy_names = list(summary.keys())
    values = [summary[name] for name in policy_names]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[i % len(colors)] for i in range(len(policy_names))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policy_names, values, color=bar_colors)

    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    value_span = max_value - min_value
    padding = max(0.5, value_span * 0.12 if value_span > 0 else 0.5)

    ax.set_ylim(min_value - padding, max_value + padding)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)

    for bar, value in zip(bars, values):
        x_pos = bar.get_x() + bar.get_width() / 2
        label = f"{value:.4f}"
        if value >= 0:
            y_pos = value + padding * 0.08
            va = "bottom"
        else:
            y_pos = value - padding * 0.08
            va = "top"
        ax.text(x_pos, y_pos, label, va=va, ha="center")

    ax.set_title("Average State Value by Policy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average V(s)")
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


# ---------------------------------------------------------------------------
# Week 6 - policy iteration plotting helpers
# ---------------------------------------------------------------------------


def _policy_to_mapping(policy: object) -> dict[tuple[int, int], str]:
    if hasattr(policy, "policy_dict"):
        return dict(policy.policy_dict)
    return dict(policy)


def plot_policy_arrows(
    policy: object,
    env: object,
    title: str = "Policy",
    save_path: str | Path | None = None,
) -> None:
    """Visualize a deterministic Gridworld policy using arrow annotations."""
    arrow_map = {
        "up": "\u2191",
        "down": "\u2193",
        "left": "\u2190",
        "right": "\u2192",
    }
    rows, cols = env.grid_size
    policy_mapping = _policy_to_mapping(policy)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.zeros((rows, cols)), cmap="Greys", alpha=0.08, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in getattr(env, "obstacles", set()):
                label = "X"
            elif state == getattr(env, "start_state", None):
                label = "S"
            elif state == getattr(env, "goal_state", None):
                label = "G"
            elif state in policy_mapping:
                label = arrow_map.get(policy_mapping[state], "?")
            else:
                label = ""

            if label:
                ax.text(col, row, label, ha="center", va="center", fontsize=16, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_policy_changes(
    policy_changes: list[int],
    save_path: str | Path | None = None,
) -> None:
    """Plot the number of changed states after each policy-improvement step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = np.arange(1, len(policy_changes) + 1)
    ax.plot(iterations, policy_changes, marker="o")
    ax.set_title("Policy Changes per Iteration")
    ax.set_xlabel("Policy Iteration")
    ax.set_ylabel("Number of Changed States")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def plot_trajectory_on_grid(
    trajectory: dict[str, object],
    env: object,
    title: str = "Policy Trajectory",
    save_path: str | Path | None = None,
) -> None:
    """Plot a visited state trajectory on the Gridworld."""
    rows, cols = env.grid_size
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.zeros((rows, cols)), cmap="Blues", alpha=0.08, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for obstacle in getattr(env, "obstacles", set()):
        ax.text(obstacle[1], obstacle[0], "X", ha="center", va="center", fontsize=16, fontweight="bold")

    start_row, start_col = env.start_state
    goal_row, goal_col = env.goal_state
    ax.text(start_col, start_row, "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(goal_col, goal_row, "G", ha="center", va="center", fontsize=14, fontweight="bold")

    states = trajectory["states"]
    x_values = [state[1] for state in states]
    y_values = [state[0] for state in states]
    ax.plot(x_values, y_values, marker="o", linewidth=2, color="tab:red")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def _draw_policy_grid(
    ax: plt.Axes,
    policy: object,
    env: object,
    title: str,
    changed_states: set[tuple[int, int]] | None = None,
) -> None:
    """Render one deterministic policy on a provided axis."""
    arrow_map = {
        "up": "\u2191",
        "down": "\u2193",
        "left": "\u2190",
        "right": "\u2192",
    }
    rows, cols = env.grid_size
    policy_mapping = _policy_to_mapping(policy)
    changed_states = changed_states or set()

    ax.imshow(np.zeros((rows, cols)), cmap="Greys", alpha=0.08, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in getattr(env, "obstacles", set()):
                ax.text(col, row, "X", ha="center", va="center", fontsize=16, fontweight="bold")
                continue
            if state == getattr(env, "start_state", None):
                ax.text(col, row, "S", ha="center", va="center", fontsize=14, fontweight="bold")
                continue
            if state == getattr(env, "goal_state", None):
                ax.text(col, row, "G", ha="center", va="center", fontsize=14, fontweight="bold")
                continue
            if state not in policy_mapping:
                continue

            color = "tab:red" if state in changed_states else "black"
            bbox = (
                {
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "#ffe6e6",
                    "edgecolor": "tab:red",
                    "linewidth": 1.0,
                }
                if state in changed_states
                else None
            )
            ax.text(
                col,
                row,
                arrow_map.get(policy_mapping[state], "?"),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold" if state in changed_states else None,
                color=color,
                bbox=bbox,
            )


def plot_policy_comparison(
    initial_policy: object,
    final_policy: object,
    env: object,
    title: str = "Initial vs Final Policy",
    save_path: str | Path | None = None,
) -> None:
    """Plot initial and final deterministic policies side by side.

    States whose actions changed are highlighted in the final policy panel.
    """
    initial_mapping = _policy_to_mapping(initial_policy)
    final_mapping = _policy_to_mapping(final_policy)
    changed_states = {
        state
        for state, action in final_mapping.items()
        if initial_mapping.get(state) != action
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _draw_policy_grid(axes[0], initial_mapping, env, "Initial Random Policy")
    _draw_policy_grid(
        axes[1],
        final_mapping,
        env,
        "Final Improved Policy",
        changed_states=changed_states,
    )
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def _draw_value_grid(
    ax: plt.Axes,
    value_grid: np.ndarray,
    env: object,
    title: str,
) -> None:
    """Render one value grid on a provided axis."""
    rows, cols = value_grid.shape
    obstacles = getattr(env, "obstacles", set())
    start_state = getattr(env, "start_state", None)
    goal_state = getattr(env, "goal_state", None)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="lightgray")

    image = ax.imshow(value_grid, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in obstacles:
                ax.text(col, row, "X", ha="center", va="center", fontsize=11, fontweight="bold")
            else:
                value = value_grid[row, col]
                value_text = f"{value:.2f}" if not np.isnan(value) else ""
                marker = ""
                if state == start_state:
                    marker = "\nS"
                elif state == goal_state:
                    marker = "\nG"
                ax.text(col, row, value_text + marker, ha="center", va="center", fontsize=7)

    return image


def plot_value_function_comparison(
    initial_value_grid: np.ndarray,
    final_value_grid: np.ndarray,
    env: object,
    title: str = "Initial vs Final State-Value Function",
    save_path: str | Path | None = None,
) -> None:
    """Plot initial and final value functions side by side.

    Each panel uses its own color normalization so the comparison preserves the
    same visual meaning as the standalone value-function plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, grid, panel_title in zip(
        axes,
        [initial_value_grid, final_value_grid],
        ["Initial Policy Values", "Final Policy Values"],
        strict=False,
    ):
        image = _draw_value_grid(ax, grid, env, panel_title)
        fig.colorbar(image, ax=ax, label="V(s)", shrink=0.82, pad=0.03)

    fig.suptitle(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
