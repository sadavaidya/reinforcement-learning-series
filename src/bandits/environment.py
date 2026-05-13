"""Bandit environments for Week 1."""

from __future__ import annotations

import numpy as np


class TenArmedBandit:
    """Stationary n-armed bandit with normally distributed rewards."""

    def __init__(
        self,
        n_actions: int = 10,
        true_reward_mean: float = 0.0,
        true_reward_std: float = 1.0,
        reward_std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")

        self.n_actions = n_actions
        self.true_reward_mean = true_reward_mean
        self.true_reward_std = true_reward_std
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed)

        self.q_true = self.rng.normal(
            loc=self.true_reward_mean,
            scale=self.true_reward_std,
            size=self.n_actions,
        )
        self.optimal_action = int(np.argmax(self.q_true))

    def step(self, action: int) -> float:
        """Return a sampled reward for the selected action."""
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Action must be between 0 and {self.n_actions - 1}, got {action}."
            )

        reward = self.rng.normal(loc=self.q_true[action], scale=self.reward_std)
        return float(reward)
