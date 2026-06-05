"""Experiment helpers for Week 4 Gridworld policy comparisons."""

from __future__ import annotations

import numpy as np

from .environment import GridworldMDP


def run_episode(
    env: GridworldMDP,
    policy: object,
    max_steps: int = 100,
    gamma: float = 1.0,
) -> dict[str, object]:
    """Run one episode and collect trajectories and summary statistics."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    state = env.reset()
    states = [state]
    actions: list[str] = []
    rewards: list[float] = []
    success = False

    for _ in range(max_steps):
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        states.append(next_state)
        state = next_state

        if done:
            success = state == env.goal_state
            break

    discounted_return = 0.0
    for step_index, reward in enumerate(rewards):
        discounted_return += (gamma**step_index) * reward

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "return": float(discounted_return),
        "episode_length": len(actions),
        "success": success,
    }


def _build_policy(
    policy_config: object,
    env: GridworldMDP,
    seed: int | None = None,
) -> object:
    if isinstance(policy_config, dict):
        policy_class = policy_config["policy_class"]
        policy_kwargs = dict(policy_config.get("policy_kwargs", {}))
    else:
        policy_class = policy_config
        policy_kwargs = {}

    policy_kwargs.setdefault("actions", env.get_valid_actions())
    if "goal_state" in policy_class.__init__.__code__.co_varnames:
        policy_kwargs.setdefault("goal_state", env.goal_state)
    if "grid_size" in policy_class.__init__.__code__.co_varnames:
        policy_kwargs.setdefault("grid_size", env.grid_size)
    if "obstacles" in policy_class.__init__.__code__.co_varnames:
        policy_kwargs.setdefault("obstacles", env.obstacles)
    if "seed" in policy_class.__init__.__code__.co_varnames:
        policy_kwargs.setdefault("seed", seed)

    return policy_class(**policy_kwargs)


def run_policy_evaluation_experiment(
    policy_configs: dict[str, object],
    n_episodes: int = 500,
    max_steps: int = 100,
    gamma: float = 1.0,
    env_kwargs: dict[str, object] | None = None,
    seed: int | None = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate fixed policies over many Gridworld episodes."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")

    env_kwargs = dict(env_kwargs or {})
    master_rng = np.random.default_rng(seed)
    results: dict[str, dict[str, np.ndarray]] = {}

    for label, policy_config in policy_configs.items():
        env = GridworldMDP(**env_kwargs)
        policy_seed = int(master_rng.integers(0, 2**32 - 1))
        policy = _build_policy(policy_config=policy_config, env=env, seed=policy_seed)

        returns = np.zeros(n_episodes, dtype=float)
        episode_lengths = np.zeros(n_episodes, dtype=int)
        successes = np.zeros(n_episodes, dtype=float)
        visitation_counts = np.zeros(env.grid_size, dtype=float)

        for obstacle in env.obstacles:
            visitation_counts[obstacle] = np.nan

        for episode_index in range(n_episodes):
            episode = run_episode(
                env=env,
                policy=policy,
                max_steps=max_steps,
                gamma=gamma,
            )
            returns[episode_index] = float(episode["return"])
            episode_lengths[episode_index] = int(episode["episode_length"])
            successes[episode_index] = float(episode["success"])

            for state in episode["states"]:
                row, col = state
                if not np.isnan(visitation_counts[row, col]):
                    visitation_counts[row, col] += 1

        results[label] = {
            "returns": returns,
            "episode_lengths": episode_lengths,
            "successes": successes,
            "state_visitation_counts": visitation_counts,
        }

    return results


def summarize_policy_results(
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Summarize average return, episode length, and success rate by policy."""
    summary: dict[str, dict[str, float]] = {}

    for label, data in results.items():
        summary[label] = {
            "average_return": float(np.mean(data["returns"])),
            "average_episode_length": float(np.mean(data["episode_lengths"])),
            "success_rate": float(np.mean(data["successes"])),
        }

    return summary
