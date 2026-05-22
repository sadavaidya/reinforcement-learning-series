"""Bandit agents for bandit experiments."""

from __future__ import annotations

import numpy as np


class EpsilonGreedyAgent:
    """Epsilon-greedy agent with sample-average action-value updates."""

    def __init__(
        self,
        n_actions: int = 10,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be between 0 and 1.")

        self.n_actions = n_actions
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)

        self.q_estimates = np.full(self.n_actions, self.initial_value, dtype=float)
        self.action_counts = np.zeros(self.n_actions, dtype=int)

    def select_action(self) -> int:
        """Select an action using epsilon-greedy exploration."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        best_value = np.max(self.q_estimates)
        best_actions = np.flatnonzero(self.q_estimates == best_value)
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        """Update the selected action estimate with the sample-average rule."""
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Action must be between 0 and {self.n_actions - 1}, got {action}."
            )

        self.action_counts[action] += 1
        count = self.action_counts[action]
        estimate = self.q_estimates[action]
        self.q_estimates[action] = estimate + (reward - estimate) / count

    def reset(self) -> None:
        """Reset estimates and counts for a fresh run."""
        self.q_estimates.fill(self.initial_value)
        self.action_counts.fill(0)


class UCBAgent:
    """Upper-confidence-bound action selection with sample-average updates."""

    def __init__(
        self,
        n_actions: int = 10,
        c: float = 2.0,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if c < 0:
            raise ValueError("c must be non-negative.")

        self.n_actions = n_actions
        self.c = c
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)

        self.q_estimates = np.full(self.n_actions, self.initial_value, dtype=float)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.time_step = 0

    def select_action(self) -> int:
        """Select an action using the UCB rule with random tie-breaking."""
        self.time_step += 1

        untried_actions = np.flatnonzero(self.action_counts == 0)
        if untried_actions.size > 0:
            return int(self.rng.choice(untried_actions))

        exploration_bonus = self.c * np.sqrt(
            np.log(self.time_step) / self.action_counts
        )
        ucb_scores = self.q_estimates + exploration_bonus
        best_score = np.max(ucb_scores)
        best_actions = np.flatnonzero(ucb_scores == best_score)
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        """Update the selected action estimate with the sample-average rule."""
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Action must be between 0 and {self.n_actions - 1}, got {action}."
            )

        self.action_counts[action] += 1
        count = self.action_counts[action]
        estimate = self.q_estimates[action]
        self.q_estimates[action] = estimate + (reward - estimate) / count

    def reset(self) -> None:
        """Reset estimates, counts, and time step for a fresh run."""
        self.q_estimates.fill(self.initial_value)
        self.action_counts.fill(0)
        self.time_step = 0
