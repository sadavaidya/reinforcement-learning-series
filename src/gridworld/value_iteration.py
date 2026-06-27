"""Value iteration helpers for Gridworld dynamic programming experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .environment import GridworldMDP
from .policy_iteration import one_step_lookahead, policy_iteration


def value_iteration(
    env: GridworldMDP,
    gamma: float = 1.0,
    theta: float = 1e-4,
    max_iterations: int = 1000,
) -> tuple[dict[tuple[int, int], float], list[float], int]:
    """Compute the optimal state-value function with synchronous value iteration."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    states = env.get_all_states()
    values: dict[tuple[int, int], float] = {state: 0.0 for state in states}
    convergence_history: list[float] = []

    for _ in range(max_iterations):
        delta = 0.0
        new_values: dict[tuple[int, int], float] = {}

        for state in states:
            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            action_scores = one_step_lookahead(
                env=env,
                values=values,
                state=state,
                gamma=gamma,
            )
            # Bellman optimality backup: V(s) <- max_a sum_{s', r} p(s', r | s, a)
            # [r + gamma * V(s')].
            updated_value = max(action_scores.values())
            delta = max(delta, abs(updated_value - values[state]))
            new_values[state] = updated_value

        values = new_values
        convergence_history.append(delta)

        if delta < theta:
            break

    return values, convergence_history, len(convergence_history)


def extract_greedy_policy(
    env: GridworldMDP,
    values: Mapping[tuple[int, int], float],
    gamma: float = 1.0,
    tie_breaking: str = "first",
    seed: int | None = None,
) -> dict[tuple[int, int], str]:
    """Extract a deterministic greedy policy from a value function."""
    if tie_breaking not in {"first", "random"}:
        raise ValueError("tie_breaking must be 'first' or 'random'.")

    rng = np.random.default_rng(seed)
    actions = env.get_valid_actions()
    greedy_policy: dict[tuple[int, int], str] = {}

    for state in env.get_all_states():
        if env.is_terminal(state):
            continue

        action_scores = one_step_lookahead(
            env=env,
            values=values,
            state=state,
            gamma=gamma,
        )
        best_score = max(action_scores.values())
        best_actions = [action for action in actions if action_scores[action] == best_score]

        if tie_breaking == "random":
            greedy_policy[state] = str(rng.choice(best_actions))
        else:
            greedy_policy[state] = best_actions[0]

    return greedy_policy


def compare_value_and_policy_iteration(
    env: GridworldMDP,
    gamma: float = 1.0,
    theta: float = 1e-4,
    max_value_iterations: int = 1000,
    max_policy_iterations: int = 100,
    max_eval_iterations: int = 1000,
    seed: int | None = None,
) -> dict[str, object]:
    """Run value iteration and policy iteration on the same Gridworld."""
    vi_values, vi_history, vi_sweeps = value_iteration(
        env=env,
        gamma=gamma,
        theta=theta,
        max_iterations=max_value_iterations,
    )
    vi_policy = extract_greedy_policy(env=env, values=vi_values, gamma=gamma)

    pi_results = policy_iteration(
        env=env,
        gamma=gamma,
        theta=theta,
        max_policy_iterations=max_policy_iterations,
        max_eval_iterations=max_eval_iterations,
        seed=seed,
    )

    pi_policy = dict(pi_results["final_policy"])
    pi_values = dict(pi_results["final_values"])
    compared_states = sorted(vi_policy.keys())
    matching_actions = sum(vi_policy[state] == pi_policy[state] for state in compared_states)
    max_value_difference = max(
        abs(vi_values[state] - pi_values[state]) for state in env.get_all_states()
    )
    policy_evaluation_sweeps = sum(len(history) for history in pi_results["value_histories"])

    return {
        "value_iteration": {
            "final_values": vi_values,
            "final_policy": vi_policy,
            "convergence_history": vi_history,
            "num_sweeps": vi_sweeps,
        },
        "policy_iteration": {
            "final_values": pi_values,
            "final_policy": pi_policy,
            "policy_changes": list(pi_results["policy_changes"]),
            "value_histories": [list(history) for history in pi_results["value_histories"]],
            "num_iterations": int(pi_results["num_iterations"]),
            "policy_stable": bool(pi_results["policy_stable"]),
            "initial_policy": dict(pi_results["initial_policy"]),
            "num_evaluation_sweeps": policy_evaluation_sweeps,
        },
        "comparison": {
            "max_value_difference": float(max_value_difference),
            "matching_actions": int(matching_actions),
            "num_compared_states": len(compared_states),
            "policy_match_fraction": matching_actions / len(compared_states),
            "policies_match_exactly": matching_actions == len(compared_states),
        },
    }
