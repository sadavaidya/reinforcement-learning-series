"""Policy improvement and policy iteration helpers for Gridworld."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .environment import GridworldMDP
from .policy_evaluation import iterative_policy_evaluation


class DeterministicPolicy:
    """Deterministic state-action mapping with a policy-evaluation interface."""

    def __init__(
        self,
        policy_dict: Mapping[tuple[int, int], str],
        env: GridworldMDP,
    ) -> None:
        self.policy_dict = dict(policy_dict)
        self.actions = env.get_valid_actions()

    def select_action(self, state: tuple[int, int]) -> str:
        """Return the selected action for the given state."""
        return self.policy_dict[state]

    def get_action_probabilities(self, state: tuple[int, int]) -> dict[str, float]:
        """Return a one-hot action distribution."""
        selected_action = self.select_action(state)
        return {
            action: 1.0 if action == selected_action else 0.0
            for action in self.actions
        }


def _iter_policy_states(env: GridworldMDP) -> list[tuple[int, int]]:
    return [state for state in env.get_all_states() if not env.is_terminal(state)]


def initialize_random_deterministic_policy(
    env: GridworldMDP,
    seed: int | None = None,
) -> dict[tuple[int, int], str]:
    """Create a reproducible random deterministic policy over valid non-terminal states."""
    rng = np.random.default_rng(seed)
    actions = env.get_valid_actions()
    return {
        state: str(rng.choice(actions))
        for state in _iter_policy_states(env)
    }


def get_policy_action_probabilities(
    policy_dict: Mapping[tuple[int, int], str],
    state: tuple[int, int],
    env: GridworldMDP,
) -> dict[str, float]:
    """Return the one-hot action distribution for a deterministic policy dict."""
    selected_action = policy_dict[state]
    return {
        action: 1.0 if action == selected_action else 0.0
        for action in env.get_valid_actions()
    }


def one_step_lookahead(
    env: GridworldMDP,
    values: Mapping[tuple[int, int], float],
    state: tuple[int, int],
    gamma: float = 1.0,
) -> dict[str, float]:
    """Estimate action values for all actions from a state using the current value function."""
    if env.is_terminal(state):
        return {}

    action_scores: dict[str, float] = {}
    for action in env.get_valid_actions():
        next_state, reward, done, _ = env.get_transition(state, action)
        next_value = 0.0 if done else float(values.get(next_state, 0.0))
        action_scores[action] = float(reward + gamma * next_value)

    return action_scores


def improve_policy(
    env: GridworldMDP,
    policy: Mapping[tuple[int, int], str] | DeterministicPolicy,
    values: Mapping[tuple[int, int], float],
    gamma: float = 1.0,
    tie_breaking: str = "first",
    seed: int | None = None,
) -> tuple[dict[tuple[int, int], str], bool, int]:
    """Greedily improve a deterministic policy using one-step lookahead."""
    if tie_breaking not in {"first", "random"}:
        raise ValueError("tie_breaking must be 'first' or 'random'.")

    rng = np.random.default_rng(seed)
    current_policy = dict(policy.policy_dict) if isinstance(policy, DeterministicPolicy) else dict(policy)
    actions = env.get_valid_actions()
    new_policy: dict[tuple[int, int], str] = {}
    num_policy_changes = 0

    for state in _iter_policy_states(env):
        old_action = current_policy[state]
        action_scores = one_step_lookahead(env=env, values=values, state=state, gamma=gamma)
        best_score = max(action_scores.values())
        best_actions = [action for action in actions if action_scores[action] == best_score]

        if tie_breaking == "random":
            selected_action = str(rng.choice(best_actions))
        else:
            selected_action = best_actions[0]

        new_policy[state] = selected_action
        if selected_action != old_action:
            num_policy_changes += 1

    policy_stable = num_policy_changes == 0
    return new_policy, policy_stable, num_policy_changes


def policy_iteration(
    env: GridworldMDP,
    gamma: float = 1.0,
    theta: float = 1e-4,
    max_policy_iterations: int = 100,
    max_eval_iterations: int = 1000,
    seed: int | None = None,
) -> dict[str, object]:
    """Run policy iteration until the policy becomes stable or the iteration cap is reached."""
    if max_policy_iterations <= 0:
        raise ValueError("max_policy_iterations must be positive.")

    policy = initialize_random_deterministic_policy(env=env, seed=seed)
    initial_policy = dict(policy)
    policy_changes: list[int] = []
    value_histories: list[list[float]] = []
    final_values: dict[tuple[int, int], float] = {
        state: 0.0 for state in env.get_all_states()
    }
    policy_stable = False

    for iteration_index in range(max_policy_iterations):
        evaluation_results = iterative_policy_evaluation(
            env=env,
            policy=DeterministicPolicy(policy, env),
            gamma=gamma,
            theta=theta,
            max_iterations=max_eval_iterations,
        )
        final_values, convergence_history = evaluation_results
        final_values = dict(final_values)
        value_histories.append(list(convergence_history))

        improved_policy, policy_stable, num_policy_changes = improve_policy(
            env=env,
            policy=policy,
            values=final_values,
            gamma=gamma,
            tie_breaking="first",
            seed=None if seed is None else seed + iteration_index,
        )
        policy_changes.append(num_policy_changes)
        policy = improved_policy

        if policy_stable:
            return {
                "final_policy": policy,
                "final_values": final_values,
                "policy_changes": policy_changes,
                "value_histories": value_histories,
                "num_iterations": iteration_index + 1,
                "policy_stable": True,
                "initial_policy": initial_policy,
            }

    return {
        "final_policy": policy,
        "final_values": final_values,
        "policy_changes": policy_changes,
        "value_histories": value_histories,
        "num_iterations": max_policy_iterations,
        "policy_stable": False,
        "initial_policy": initial_policy,
    }


def extract_greedy_trajectory(
    env: GridworldMDP,
    policy: Mapping[tuple[int, int], str] | DeterministicPolicy,
    max_steps: int = 100,
) -> dict[str, object]:
    """Follow a deterministic policy from the start state until termination or cutoff."""
    policy_dict = dict(policy.policy_dict) if isinstance(policy, DeterministicPolicy) else dict(policy)
    state = env.reset()
    states = [state]
    actions: list[str] = []
    success = env.is_terminal(state)

    for _ in range(max_steps):
        if env.is_terminal(state):
            success = True
            break

        action = policy_dict[state]
        next_state, _, done, _ = env.step(action)
        actions.append(action)
        states.append(next_state)
        state = next_state

        if done:
            success = state == env.goal_state
            break

    return {
        "states": states,
        "actions": actions,
        "success": success,
        "episode_length": len(actions),
    }
