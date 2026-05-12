import numpy as np


class TenArmedBandit:
    """
    A stationary 10-armed bandit environment.

    Each action has a true value q*(a), sampled from a normal distribution.
    When an action is selected, the reward is sampled from a normal distribution
    centered around that action's true value.
    """

    def __init__(
        self,
        n_actions: int = 10,
        true_reward_mean: float = 0.0,
        true_reward_std: float = 1.0,
        reward_std: float = 1.0,
        seed: int | None = None,
    ):
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
        """
        Take an action and return a sampled reward.
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Action must be between 0 and {self.n_actions - 1}")

        reward = self.rng.normal(
            loc=self.q_true[action],
            scale=self.reward_std,
        )

        return float(reward)