"""Experiment runners for Week 1 bandit comparisons."""

from __future__ import annotations

import numpy as np

from .agents import EpsilonGreedyAgent
from .environment import TenArmedBandit


def run_single_experiment(
    epsilon: float,
    n_steps: int = 1000,
    n_actions: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one epsilon-greedy bandit experiment."""
    bandit = TenArmedBandit(n_actions=n_actions, seed=seed)
    agent = EpsilonGreedyAgent(n_actions=n_actions, epsilon=epsilon, seed=seed)

    rewards = np.zeros(n_steps, dtype=float)
    optimal_action_selected = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)

        rewards[step] = reward
        optimal_action_selected[step] = float(action == bandit.optimal_action)

    return rewards, optimal_action_selected


def run_multiple_experiments(
    epsilons: list[float],
    n_runs: int = 2000,
    n_steps: int = 1000,
    n_actions: int = 10,
    seed: int | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Average many independent runs for each epsilon value."""
    master_rng = np.random.default_rng(seed)
    results: dict[float, dict[str, np.ndarray]] = {}

    for epsilon in epsilons:
        rewards_per_run = np.zeros((n_runs, n_steps), dtype=float)
        optimal_per_run = np.zeros((n_runs, n_steps), dtype=float)

        for run_index in range(n_runs):
            run_seed = int(master_rng.integers(0, 2**32 - 1))
            rewards, optimal = run_single_experiment(
                epsilon=epsilon,
                n_steps=n_steps,
                n_actions=n_actions,
                seed=run_seed,
            )
            rewards_per_run[run_index] = rewards
            optimal_per_run[run_index] = optimal

        results[epsilon] = {
            "average_rewards": rewards_per_run.mean(axis=0),
            "optimal_action_percentage": optimal_per_run.mean(axis=0) * 100.0,
        }

    return results
