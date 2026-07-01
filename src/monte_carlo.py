"""Monte Carlo prediction utilities for Week 8."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


State = tuple[int, int]
Transition = tuple[State, str, float]


def generate_episode(
    env: object,
    policy: object,
    max_steps: int = 100,
    start_state: State | None = None,
) -> list[Transition]:
    """Generate one episode by interacting with the environment."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if start_state is None:
        state = env.reset()
    else:
        state = env.reset_to_state(start_state)

    episode: list[Transition] = []

    for _ in range(max_steps):
        if env.is_terminal(state):
            break

        action = policy.select_action(state)
        next_state, reward, done, _info = env.step(action)
        episode.append((state, action, float(reward)))
        state = next_state

        if done:
            break

    return episode


def compute_returns(
    episode: list[Transition],
    gamma: float = 1.0,
) -> list[float]:
    """Compute backward returns G_t aligned with the episode time steps."""
    returns = [0.0] * len(episode)
    running_return = 0.0

    for time_index in range(len(episode) - 1, -1, -1):
        reward = episode[time_index][2]
        running_return = reward + gamma * running_return
        returns[time_index] = float(running_return)

    return returns


def first_visit_mc_prediction(
    env: object,
    policy: object,
    num_episodes: int = 5000,
    gamma: float = 1.0,
    max_steps: int = 100,
    seed: int | None = None,
) -> tuple[dict[State, float], dict[str, object]]:
    """Estimate V_pi(s) with First-Visit Monte Carlo prediction."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    states = env.get_all_states()
    values: dict[State, float] = {state: 0.0 for state in states}
    returns_sum: defaultdict[State, float] = defaultdict(float)
    returns_count: defaultdict[State, int] = defaultdict(int)
    rng = np.random.default_rng(seed)

    episode_returns = np.zeros(num_episodes, dtype=float)
    episode_lengths = np.zeros(num_episodes, dtype=int)
    state_visit_counts: dict[State, int] = {state: 0 for state in states}
    first_visit_counts: dict[State, int] = {state: 0 for state in states}
    value_snapshots: list[dict[str, object]] = []
    checkpoint_interval = max(1, num_episodes // 20)

    for episode_index in range(num_episodes):
        start_state = env.sample_non_terminal_state(rng=rng)
        episode = generate_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            start_state=start_state,
        )
        returns = compute_returns(episode, gamma=gamma)

        visited_states: set[State] = set()
        for time_index, ((state, _action, _reward), return_t) in enumerate(
            zip(episode, returns, strict=False)
        ):
            state_visit_counts[state] += 1
            if state in visited_states:
                continue

            visited_states.add(state)
            first_visit_counts[state] += 1
            returns_sum[state] += return_t
            returns_count[state] += 1
            values[state] = returns_sum[state] / returns_count[state]

        if returns:
            episode_returns[episode_index] = returns[0]
        episode_lengths[episode_index] = len(episode)

        if (
            episode_index == 0
            or (episode_index + 1) % checkpoint_interval == 0
            or episode_index == num_episodes - 1
        ):
            value_snapshots.append(
                {
                    "episode": episode_index + 1,
                    "values": dict(values),
                }
            )

    history = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "state_visit_counts": state_visit_counts,
        "first_visit_counts": first_visit_counts,
        "value_snapshots": value_snapshots,
        "returns_count": dict(returns_count),
    }

    return values, history
