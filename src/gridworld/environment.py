"""Gridworld environment for Week 4 finite MDP experiments."""

from __future__ import annotations


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

    def get_valid_actions(self) -> list[str]:
        """Return the available action names."""
        return self.actions.copy()

    def step(
        self,
        action: str,
    ) -> tuple[tuple[int, int], float, bool, dict[str, bool]]:
        """Apply one environment transition from the current state."""
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"Unknown action: {action}")

        if self.is_terminal(self.current_state):
            return self.current_state, 0.0, True, {"invalid_move": False}

        delta_row, delta_col = self.ACTION_DELTAS[action]
        row, col = self.current_state
        proposed_state = (row + delta_row, col + delta_col)

        if not self.is_valid_state(proposed_state):
            return (
                self.current_state,
                float(self.invalid_move_reward),
                False,
                {"invalid_move": True},
            )

        self.current_state = proposed_state
        if self.is_terminal(self.current_state):
            return (
                self.current_state,
                float(self.goal_reward),
                True,
                {"invalid_move": False},
            )

        return (
            self.current_state,
            float(self.step_reward),
            False,
            {"invalid_move": False},
        )

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
