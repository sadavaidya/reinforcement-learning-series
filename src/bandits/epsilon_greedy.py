import numpy as np


class EpsilonGreedyAgent:
    """
    Epsilon-greedy agent using sample-average action-value estimation.
    """

    def __init__(
        self,
        n_actions: int = 10,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.rng = np.random.default_rng(seed)

        self.q_estimates = np.full(self.n_actions, self.initial_value, dtype=float)
        self.action_counts = np.zeros(self.n_actions, dtype=int)

    def select_action(self) -> int:
        """
        Select an action using epsilon-greedy action selection.
        """
        explore = self.rng.random() < self.epsilon

        if explore:
            return int(self.rng.integers(self.n_actions))

        max_value = np.max(self.q_estimates)
        greedy_actions = np.flatnonzero(self.q_estimates == max_value)

        return int(self.rng.choice(greedy_actions))

    def update(self, action: int, reward: float) -> None:
        """
        Update action-value estimate using the sample-average update rule.
        """
        self.action_counts[action] += 1

        n = self.action_counts[action]
        old_estimate = self.q_estimates[action]

        self.q_estimates[action] = old_estimate + (reward - old_estimate) / n

    def reset(self) -> None:
        """
        Reset agent estimates and action counts.
        """
        self.q_estimates = np.full(self.n_actions, self.initial_value, dtype=float)
        self.action_counts = np.zeros(self.n_actions, dtype=int)