"""Experiment runners for Week 1 and Week 2 bandit comparisons."""

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

    return run_single_agent_experiment(agent=agent, bandit=bandit, n_steps=n_steps)


def run_single_agent_experiment(
    agent: object,
    bandit: TenArmedBandit,
    n_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one bandit experiment for a pre-built agent and environment."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rewards = np.zeros(n_steps, dtype=float)
    optimal_action_selected = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        action = agent.select_action()
        reward = bandit.step(action)
        agent.update(action, reward)

        rewards[step] = reward
        optimal_action_selected[step] = float(action == bandit.optimal_action)

    return rewards, optimal_action_selected


def run_multiple_agent_experiments(
    agent_configs: dict[str, dict[str, object]],
    n_runs: int = 2000,
    n_steps: int = 1000,
    n_actions: int = 10,
    bandit_kwargs: dict[str, object] | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Average many independent runs for a set of agent configurations."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    master_rng = np.random.default_rng(seed)
    bandit_kwargs = dict(bandit_kwargs or {})
    results: dict[str, dict[str, np.ndarray]] = {}

    for label, config in agent_configs.items():
        agent_class = config["agent_class"]
        agent_kwargs = dict(config.get("agent_kwargs", {}))

        rewards_per_run = np.zeros((n_runs, n_steps), dtype=float)
        optimal_per_run = np.zeros((n_runs, n_steps), dtype=float)

        for run_index in range(n_runs):
            run_seed = int(master_rng.integers(0, 2**32 - 1))
            bandit = TenArmedBandit(
                n_actions=n_actions,
                seed=run_seed,
                **bandit_kwargs,
            )
            agent = agent_class(
                n_actions=n_actions,
                seed=run_seed,
                **agent_kwargs,
            )
            rewards, optimal = run_single_agent_experiment(
                agent=agent,
                bandit=bandit,
                n_steps=n_steps,
            )
            rewards_per_run[run_index] = rewards
            optimal_per_run[run_index] = optimal

        results[label] = {
            "average_rewards": rewards_per_run.mean(axis=0),
            "optimal_action_percentage": optimal_per_run.mean(axis=0) * 100.0,
        }

    return results


def run_multiple_experiments(
    epsilons: list[float],
    n_runs: int = 2000,
    n_steps: int = 1000,
    n_actions: int = 10,
    seed: int | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Average many independent runs for each epsilon value."""
    agent_configs = {
        float(epsilon): {
            "agent_class": EpsilonGreedyAgent,
            "agent_kwargs": {"epsilon": epsilon, "initial_value": 0.0},
        }
        for epsilon in epsilons
    }
    generic_results = run_multiple_agent_experiments(
        agent_configs=agent_configs,
        n_runs=n_runs,
        n_steps=n_steps,
        n_actions=n_actions,
        seed=seed,
    )
    return {float(key): value for key, value in generic_results.items()}
