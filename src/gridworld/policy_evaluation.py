"""Iterative policy evaluation for Week 5 value function experiments."""

from __future__ import annotations

import numpy as np

from .environment import GridworldMDP


def get_action_probabilities(
    policy: object,
    state: tuple[int, int],
    env: GridworldMDP,
) -> dict[str, float]:
    """Dispatch to the policy's get_action_probabilities method.

    Returns a dict mapping each action to its probability under the policy.
    """
    del env
    return policy.get_action_probabilities(state)


def compute_bellman_update(
    env: GridworldMDP,
    policy: object,
    values: dict[tuple[int, int], float],
    state: tuple[int, int],
    gamma: float = 1.0,
) -> float:
    """Compute one Bellman expectation backup for a single state.

    Terminal states always return 0.0.
    For non-terminal states applies:
        v(s) = sum_a pi(a|s) [r(s,a) + gamma * V(s')]
    """
    if env.is_terminal(state):
        return 0.0

    action_probs = policy.get_action_probabilities(state)
    total = 0.0
    for action, prob in action_probs.items():
        next_state, reward, _done, _info = env.get_transition(state, action)
        next_value = values.get(next_state, 0.0)
        total += prob * (reward + gamma * next_value)
    return total


def iterative_policy_evaluation(
    env: GridworldMDP,
    policy: object,
    gamma: float = 1.0,
    theta: float = 1e-4,
    max_iterations: int = 1000,
) -> tuple[dict[tuple[int, int], float], list[float]]:
    """Compute V(s) for all valid states under a fixed policy.

    Uses synchronous Bellman backups: all states are updated with the same
    old value table each sweep, then the table is replaced.

    Returns
    -------
    values
        Dict mapping each valid non-obstacle state to its estimated value.
    convergence_history
        List of max-delta values, one per sweep, for plotting convergence.
    """
    states = env.get_all_states()
    values: dict[tuple[int, int], float] = {s: 0.0 for s in states}
    convergence_history: list[float] = []

    for _ in range(max_iterations):
        delta = 0.0
        new_values: dict[tuple[int, int], float] = {}

        for state in states:
            if env.is_terminal(state):
                new_values[state] = 0.0
            else:
                new_val = compute_bellman_update(env, policy, values, state, gamma)
                delta = max(delta, abs(new_val - values[state]))
                new_values[state] = new_val

        values = new_values
        convergence_history.append(delta)

        if delta < theta:
            break

    return values, convergence_history


def values_to_grid(
    values: dict[tuple[int, int], float],
    env: GridworldMDP,
    obstacle_value: float = np.nan,
) -> np.ndarray:
    """Convert a value dict to a 2-D numpy array matching the grid shape.

    Obstacle cells are filled with obstacle_value (default np.nan).
    """
    rows, cols = env.grid_size
    grid = np.full((rows, cols), obstacle_value, dtype=float)
    for (row, col), value in values.items():
        grid[row, col] = value
    return grid


def manual_bellman_update_details(
    env: GridworldMDP,
    policy: object,
    values: dict[tuple[int, int], float],
    state: tuple[int, int],
    gamma: float = 1.0,
) -> dict[str, object]:
    """Return a breakdown of one Bellman backup for educational purposes.

    Returns a dict with keys:
        state          - the input state
        action_details - list of per-action dicts (action, probability,
                         next_state, reward, next_state_value, contribution)
        updated_value  - sum of all contributions
    """
    if env.is_terminal(state):
        return {"state": state, "action_details": [], "updated_value": 0.0}

    action_probs = policy.get_action_probabilities(state)
    action_details: list[dict[str, object]] = []
    total = 0.0

    for action, prob in action_probs.items():
        next_state, reward, _done, _info = env.get_transition(state, action)
        next_value = values.get(next_state, 0.0)
        contribution = prob * (reward + gamma * next_value)
        total += contribution
        action_details.append(
            {
                "action": action,
                "probability": prob,
                "next_state": next_state,
                "reward": reward,
                "next_state_value": next_value,
                "contribution": contribution,
            }
        )

    return {"state": state, "action_details": action_details, "updated_value": total}
