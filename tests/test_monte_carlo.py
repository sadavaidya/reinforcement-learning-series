from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gridworld.environment import GridworldMDP
from src.gridworld.policies import GoalDirectedPolicy, RandomPolicy
from src.gridworld.policy_evaluation import iterative_policy_evaluation
from src.monte_carlo import compute_returns, first_visit_mc_prediction, generate_episode


def test_reset_to_state_sets_current_state():
    env = GridworldMDP()

    state = env.reset_to_state((2, 2))

    assert state == (2, 2)
    assert env.current_state == (2, 2)


def test_sample_non_terminal_state_never_returns_goal():
    env = GridworldMDP()
    rng = np.random.default_rng(0)

    for _ in range(50):
        state = env.sample_non_terminal_state(rng=rng)
        assert env.is_valid_state(state)
        assert not env.is_terminal(state)


def test_generate_episode_returns_state_action_reward_transitions():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    episode = generate_episode(
        env=env,
        policy=policy,
        max_steps=10,
        start_state=(0, 0),
    )

    assert len(episode) > 0
    assert len(episode) <= 10
    for state, action, reward in episode:
        assert env.is_valid_state(state)
        assert action in env.get_valid_actions()
        assert isinstance(reward, float)


def test_compute_returns_matches_backward_discounted_sum():
    episode = [
        ((0, 0), "right", -1.0),
        ((0, 1), "down", -1.0),
        ((1, 1), "right", 10.0),
    ]

    returns = compute_returns(episode, gamma=0.9)

    assert np.allclose(returns, [6.2, 8.0, 10.0])


def test_first_visit_mc_prediction_returns_expected_structure():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    values, history = first_visit_mc_prediction(
        env=env,
        policy=policy,
        num_episodes=20,
        gamma=1.0,
        max_steps=20,
        seed=0,
    )

    assert set(values.keys()) == set(env.get_all_states())
    assert set(history.keys()) == {
        "episode_returns",
        "episode_lengths",
        "state_visit_counts",
        "first_visit_counts",
        "value_snapshots",
        "returns_count",
    }
    assert history["episode_returns"].shape == (20,)
    assert history["episode_lengths"].shape == (20,)
    assert len(history["value_snapshots"]) >= 1
    assert values[env.goal_state] == 0.0


def test_first_visit_mc_prediction_moves_toward_dp_values():
    env = GridworldMDP()
    policy = GoalDirectedPolicy(
        goal_state=env.goal_state,
        grid_size=env.grid_size,
        obstacles=env.obstacles,
        exploration_prob=0.1,
        seed=123,
    )

    dp_values, _ = iterative_policy_evaluation(
        env=env,
        policy=policy,
        gamma=1.0,
        theta=1e-4,
        max_iterations=1000,
    )
    mc_values, history = first_visit_mc_prediction(
        env=env,
        policy=policy,
        num_episodes=3000,
        gamma=1.0,
        max_steps=100,
        seed=123,
    )

    visited_states = [
        state for state, count in history["returns_count"].items() if count > 0 and not env.is_terminal(state)
    ]
    mae = np.mean([abs(mc_values[state] - dp_values[state]) for state in visited_states])

    assert len(visited_states) > 0
    assert mae < 4.0
