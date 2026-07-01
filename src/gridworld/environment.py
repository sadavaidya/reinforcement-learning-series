"""Gridworld environment for Week 4 finite MDP experiments."""

from __future__ import annotations

import numpy as np


class GridworldMDP:
    """A simple episodic Gridworld Markov Decision Process."""

    ACTION_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(
        self,
        grid_size: tuple[int, int] = (5, 5),
        start_state: tuple[int, int] = (0, 0),
        goal_state: tuple[int, int] = (4, 4),
        obstacles: set[tuple[int, int]] | None = None,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
        invalid_move_reward: float = -5.0,
    ) -> None:
        rows, cols = grid_size
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_size dimensions must be positive.")

        default_obstacles = {(1, 1), (1, 3), (2, 3), (3, 0), (4, 2)}
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = set(default_obstacles if obstacles is None else obstacles)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.invalid_move_reward = invalid_move_reward
        self.actions = list(self.ACTION_DELTAS.keys())

        if start_state in self.obstacles:
            raise ValueError("start_state cannot be an obstacle.")
        if goal_state in self.obstacles:
            raise ValueError("goal_state cannot be an obstacle.")
        if not self._is_within_bounds(start_state):
            raise ValueError("start_state must lie within the grid.")
        if not self._is_within_bounds(goal_state):
            raise ValueError("goal_state must lie within the grid.")

        self.current_state = self.start_state

    def _is_within_bounds(self, state: tuple[int, int]) -> bool:
        row, col = state
        rows, cols = self.grid_size
        return 0 <= row < rows and 0 <= col < cols

    def reset(self) -> tuple[int, int]:
        """Reset the environment to the start state."""
        self.current_state = self.start_state
        return self.current_state

    def reset_to_state(self, state: tuple[int, int]) -> tuple[int, int]:
        """Reset the environment to a specific valid non-obstacle state."""
        if not self.is_valid_state(state):
            raise ValueError("state must be a valid non-obstacle state.")

        self.current_state = state
        return self.current_state

    def is_terminal(self, state: tuple[int, int]) -> bool:
        """Return True when the state is terminal."""
        return state == self.goal_state

    def is_valid_state(self, state: tuple[int, int]) -> bool:
        """Return True when the state is inside the grid and not an obstacle."""
        return self._is_within_bounds(state) and state not in self.obstacles

    def get_all_states(self) -> list[tuple[int, int]]:
        """Return all valid, non-obstacle states in the grid."""
        rows, cols = self.grid_size
        states: list[tuple[int, int]] = []

        for row in range(rows):
            for col in range(cols):
                state = (row, col)
                if self.is_valid_state(state):
                    states.append(state)

        return states

    def get_non_terminal_states(self) -> list[tuple[int, int]]:
        """Return all valid states except the terminal goal state."""
        return [state for state in self.get_all_states() if not self.is_terminal(state)]

    def sample_non_terminal_state(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        """Sample one valid non-terminal state uniformly at random."""
        non_terminal_states = self.get_non_terminal_states()
        if not non_terminal_states:
            raise ValueError("environment must contain at least one non-terminal state.")

        rng = np.random.default_rng() if rng is None else rng
        state_index = int(rng.integers(0, len(non_terminal_states)))
        return non_terminal_states[state_index]

    def get_valid_actions(self) -> list[str]:
        """Return the available action names."""
        return self.actions.copy()

    def get_transition(
        self,
        state: tuple[int, int],
        action: str,
    ) -> tuple[tuple[int, int], float, bool, dict[str, bool]]:
        """Compute a hypothetical transition without modifying current_state."""
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action}")

        if self.is_terminal(state):
            return state, 0.0, True, {"invalid_move": False}

        delta_row, delta_col = self.ACTION_DELTAS[action]
        row, col = state
        proposed_state = (row + delta_row, col + delta_col)

        if not self.is_valid_state(proposed_state):
            return (
                state,
                float(self.invalid_move_reward),
                False,
                {"invalid_move": True},
            )

        next_state = proposed_state
        if self.is_terminal(next_state):
            return (
                next_state,
                float(self.goal_reward),
                True,
                {"invalid_move": False},
            )

        return (
            next_state,
            float(self.step_reward),
            False,
            {"invalid_move": False},
        )

    def step(
        self,
        action: str,
    ) -> tuple[tuple[int, int], float, bool, dict[str, bool]]:
        """Apply one environment transition from the current state."""
        next_state, reward, done, info = self.get_transition(self.current_state, action)
        self.current_state = next_state
        return next_state, reward, done, info

    def render(self) -> str:
        """Return a simple text rendering of the grid."""
        rows, cols = self.grid_size
        symbols: list[str] = []

        for row in range(rows):
            cells: list[str] = []
            for col in range(cols):
                state = (row, col)
                if state == self.current_state:
                    cells.append("A")
                elif state == self.start_state:
                    cells.append("S")
                elif state == self.goal_state:
                    cells.append("G")
                elif state in self.obstacles:
                    cells.append("X")
                else:
                    cells.append(".")
            symbols.append(" ".join(cells))

        return "\n".join(symbols)
