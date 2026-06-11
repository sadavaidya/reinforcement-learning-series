"""Fixed policies for Week 4 Gridworld comparisons."""

from __future__ import annotations

import numpy as np

from .environment import GridworldMDP


def _manhattan_distance(
    state: tuple[int, int],
    goal_state: tuple[int, int],
) -> int:
    return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])


class RandomPolicy:
    """Uniformly random action selection."""

    def __init__(
        self,
        actions: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.actions = list(GridworldMDP.ACTION_DELTAS.keys()) if actions is None else list(actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: tuple[int, int]) -> str:
        """Return a uniformly random action."""
        del state
        return str(self.rng.choice(self.actions))

    def get_action_probabilities(self, state: tuple[int, int]) -> dict[str, float]:
        """Return uniform probability over all actions."""
        del state
        prob = 1.0 / len(self.actions)
        return {action: prob for action in self.actions}


class GoalDirectedPolicy:
    """Heuristic policy that tends to reduce distance to the goal."""

    def __init__(
        self,
        goal_state: tuple[int, int],
        grid_size: tuple[int, int] = (5, 5),
        obstacles: set[tuple[int, int]] | None = None,
        exploration_prob: float = 0.1,
        actions: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if exploration_prob < 0.0 or exploration_prob > 1.0:
            raise ValueError("exploration_prob must be between 0 and 1.")

        self.goal_state = goal_state
        self.grid_size = grid_size
        self.obstacles = set(obstacles or set())
        self.exploration_prob = exploration_prob
        self.actions = list(GridworldMDP.ACTION_DELTAS.keys()) if actions is None else list(actions)
        self.rng = np.random.default_rng(seed)

    def _candidate_state(
        self,
        state: tuple[int, int],
        action: str,
    ) -> tuple[int, int]:
        delta_row, delta_col = GridworldMDP.ACTION_DELTAS[action]
        return state[0] + delta_row, state[1] + delta_col

    def _is_valid_state(self, state: tuple[int, int]) -> bool:
        rows, cols = self.grid_size
        row, col = state
        return 0 <= row < rows and 0 <= col < cols and state not in self.obstacles

    def select_action(self, state: tuple[int, int]) -> str:
        """Prefer valid actions that improve distance to the goal."""
        if self.rng.random() < self.exploration_prob:
            return str(self.rng.choice(self.actions))

        current_distance = _manhattan_distance(state, self.goal_state)
        scored_actions: list[tuple[str, int, bool]] = []

        for action in self.actions:
            next_state = self._candidate_state(state, action)
            is_valid = self._is_valid_state(next_state)
            distance = (
                _manhattan_distance(next_state, self.goal_state)
                if is_valid
                else current_distance + 1000
            )
            scored_actions.append((action, distance, is_valid))

        valid_actions = [item for item in scored_actions if item[2]]
        if valid_actions:
            best_distance = min(item[1] for item in valid_actions)
            best_actions = [
                item[0]
                for item in valid_actions
                if item[1] == best_distance and item[1] <= current_distance
            ]
            if not best_actions:
                best_actions = [
                    item[0] for item in valid_actions if item[1] == best_distance
                ]
            return str(self.rng.choice(best_actions))

        return str(self.rng.choice(self.actions))

    def get_action_probabilities(self, state: tuple[int, int]) -> dict[str, float]:
        """Return expected probability for each action under this policy."""
        # Random component: exploration_prob spread uniformly over all actions.
        probs = {action: self.exploration_prob / len(self.actions) for action in self.actions}

        # Goal-directed component: (1 - exploration_prob) assigned to best actions.
        current_distance = _manhattan_distance(state, self.goal_state)
        scored: list[tuple[str, int, bool]] = []
        for action in self.actions:
            next_state = self._candidate_state(state, action)
            is_valid = self._is_valid_state(next_state)
            distance = (
                _manhattan_distance(next_state, self.goal_state)
                if is_valid
                else current_distance + 1000
            )
            scored.append((action, distance, is_valid))

        valid = [item for item in scored if item[2]]
        if valid:
            best_dist = min(item[1] for item in valid)
            best_actions = [
                item[0] for item in valid
                if item[1] == best_dist and item[1] <= current_distance
            ]
            if not best_actions:
                best_actions = [item[0] for item in valid if item[1] == best_dist]
        else:
            best_actions = list(self.actions)

        goal_prob_each = (1.0 - self.exploration_prob) / len(best_actions)
        for action in best_actions:
            probs[action] += goal_prob_each

        return probs


class BadPolicy:
    """Biased policy that often picks poor directions for this grid."""

    def __init__(
        self,
        bad_action_prob: float = 0.7,
        preferred_bad_actions: tuple[str, str] = ("up", "left"),
        actions: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if bad_action_prob < 0.0 or bad_action_prob > 1.0:
            raise ValueError("bad_action_prob must be between 0 and 1.")

        self.bad_action_prob = bad_action_prob
        self.preferred_bad_actions = tuple(preferred_bad_actions)
        self.actions = list(GridworldMDP.ACTION_DELTAS.keys()) if actions is None else list(actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: tuple[int, int]) -> str:
        """Usually pick poor directions, with some randomness."""
        del state
        if self.rng.random() < self.bad_action_prob:
            return str(self.rng.choice(self.preferred_bad_actions))
        return str(self.rng.choice(self.actions))

    def get_action_probabilities(self, state: tuple[int, int]) -> dict[str, float]:
        """Return expected probability for each action under this policy."""
        del state
        # Random component: (1 - bad_action_prob) spread uniformly over all actions.
        probs = {action: (1.0 - self.bad_action_prob) / len(self.actions) for action in self.actions}
        # Bad-action component: bad_action_prob split equally among preferred bad actions.
        bad_prob_each = self.bad_action_prob / len(self.preferred_bad_actions)
        for action in self.preferred_bad_actions:
            probs[action] += bad_prob_each
        return probs
