from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gridworld.environment import GridworldMDP
from src.gridworld.experiments import (
    run_episode,
    run_policy_evaluation_experiment,
    summarize_policy_results,
)
from src.gridworld.policies import BadPolicy, GoalDirectedPolicy, RandomPolicy


def test_gridworld_initialization_and_reset():
    env = GridworldMDP()

    assert env.grid_size == (5, 5)
    assert env.start_state == (0, 0)
    assert env.goal_state == (4, 4)
    assert env.obstacles == {(1, 1), (1, 3), (2, 3), (3, 0), (4, 2)}

    state = env.reset()
    assert state == env.start_state
    assert env.current_state == env.start_state


def test_gridworld_valid_actions():
    env = GridworldMDP()
    assert env.get_valid_actions() == ["up", "down", "left", "right"]


def test_gridworld_valid_movement():
    env = GridworldMDP()
    env.reset()

    next_state, reward, done, info = env.step("right")

    assert next_state == (0, 1)
    assert reward == env.step_reward
    assert done is False
    assert info["invalid_move"] is False


def test_gridworld_invalid_movement():
    env = GridworldMDP()
    env.reset()

    next_state, reward, done, info = env.step("up")

    assert next_state == (0, 0)
    assert reward == env.invalid_move_reward
    assert done is False
    assert info["invalid_move"] is True


def test_gridworld_goal_transition():
    env = GridworldMDP()
    env.current_state = (4, 3)

    next_state, reward, done, info = env.step("right")

    assert next_state == (4, 4)
    assert reward == env.goal_reward
    assert done is True
    assert info["invalid_move"] is False


def test_gridworld_obstacle_collision():
    env = GridworldMDP()
    env.current_state = (1, 0)

    next_state, reward, done, info = env.step("right")

    assert next_state == (1, 0)
    assert reward == env.invalid_move_reward
    assert done is False
    assert info["invalid_move"] is True


def test_random_policy_selects_valid_action():
    policy = RandomPolicy(seed=0)
    action = policy.select_action((0, 0))
    assert action in ["up", "down", "left", "right"]


def test_goal_directed_policy_selects_valid_distance_reducing_action():
    policy = GoalDirectedPolicy(
        goal_state=(4, 4),
        exploration_prob=0.0,
        seed=0,
    )
    action = policy.select_action((0, 0))
    assert action in ["up", "down", "left", "right"]
    assert action in ["down", "right"]


def test_bad_policy_selects_valid_action():
    policy = BadPolicy(seed=0)
    action = policy.select_action((2, 2))
    assert action in ["up", "down", "left", "right"]


def test_run_episode_returns_expected_keys_and_length():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    episode = run_episode(env=env, policy=policy, max_steps=10, gamma=1.0)

    assert set(episode.keys()) == {
        "states",
        "actions",
        "rewards",
        "return",
        "episode_length",
        "success",
    }
    assert episode["episode_length"] <= 10
    assert len(episode["actions"]) == episode["episode_length"]
    assert len(episode["rewards"]) == episode["episode_length"]
    assert len(episode["states"]) == episode["episode_length"] + 1


def test_run_policy_evaluation_experiment_returns_expected_shapes():
    policy_configs = {
        "Random Policy": {
            "policy_class": RandomPolicy,
            "policy_kwargs": {},
        },
        "Goal-Directed Policy": {
            "policy_class": GoalDirectedPolicy,
            "policy_kwargs": {"goal_state": (4, 4), "exploration_prob": 0.0},
        },
        "Bad Policy": {
            "policy_class": BadPolicy,
            "policy_kwargs": {},
        },
    }

    results = run_policy_evaluation_experiment(
        policy_configs=policy_configs,
        n_episodes=3,
        max_steps=10,
        gamma=1.0,
        env_kwargs={},
        seed=0,
    )

    assert set(results.keys()) == set(policy_configs.keys())
    for policy_result in results.values():
        assert policy_result["returns"].shape == (3,)
        assert policy_result["episode_lengths"].shape == (3,)
        assert policy_result["successes"].shape == (3,)
        assert policy_result["state_visitation_counts"].shape == (5, 5)


def test_summarize_policy_results_returns_expected_metrics():
    results = {
        "Random Policy": {
            "returns": np.array([1.0, 2.0, 3.0]),
            "episode_lengths": np.array([4, 5, 6]),
            "successes": np.array([0.0, 1.0, 1.0]),
            "state_visitation_counts": np.zeros((5, 5), dtype=float),
        }
    }

    summary = summarize_policy_results(results)

    assert "Random Policy" in summary
    assert "average_return" in summary["Random Policy"]
    assert "average_episode_length" in summary["Random Policy"]
    assert "success_rate" in summary["Random Policy"]
    assert np.isclose(summary["Random Policy"]["average_return"], 2.0)
    assert np.isclose(summary["Random Policy"]["average_episode_length"], 5.0)
    assert np.isclose(summary["Random Policy"]["success_rate"], 2.0 / 3.0)
