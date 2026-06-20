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
from src.gridworld.policy_evaluation import (
    compute_bellman_update,
    iterative_policy_evaluation,
    manual_bellman_update_details,
    values_to_grid,
)
from src.gridworld.policy_iteration import (
    DeterministicPolicy,
    extract_greedy_trajectory,
    get_policy_action_probabilities,
    improve_policy,
    initialize_random_deterministic_policy,
    one_step_lookahead,
    policy_iteration,
)


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


# ---------------------------------------------------------------------------
# Week 5 — get_transition tests
# ---------------------------------------------------------------------------


def test_get_transition_does_not_mutate_current_state():
    env = GridworldMDP()
    env.reset()
    state_before = env.current_state

    env.get_transition((0, 0), "right")

    assert env.current_state == state_before


def test_get_transition_valid_move_from_start():
    env = GridworldMDP()
    next_state, reward, done, info = env.get_transition((0, 0), "right")

    assert next_state == (0, 1)
    assert reward == env.step_reward
    assert done is False
    assert info["invalid_move"] is False
    assert env.current_state == env.start_state  # unchanged


def test_get_transition_wall_returns_same_state():
    env = GridworldMDP()
    next_state, reward, done, info = env.get_transition((0, 0), "up")

    assert next_state == (0, 0)
    assert reward == env.invalid_move_reward
    assert done is False
    assert info["invalid_move"] is True


def test_get_transition_obstacle_returns_same_state():
    env = GridworldMDP()
    # (1,0) moving right hits obstacle (1,1)
    next_state, reward, done, info = env.get_transition((1, 0), "right")

    assert next_state == (1, 0)
    assert reward == env.invalid_move_reward
    assert info["invalid_move"] is True


def test_get_transition_goal_transition():
    env = GridworldMDP()
    next_state, reward, done, info = env.get_transition((4, 3), "right")

    assert next_state == (4, 4)
    assert reward == env.goal_reward
    assert done is True


def test_step_still_mutates_current_state():
    env = GridworldMDP()
    env.reset()

    next_state, reward, done, info = env.step("right")

    assert env.current_state == (0, 1)
    assert next_state == (0, 1)
    assert reward == env.step_reward


# ---------------------------------------------------------------------------
# Week 5 — policy probability tests
# ---------------------------------------------------------------------------


def test_random_policy_action_probabilities_sum_to_one():
    policy = RandomPolicy()
    probs = policy.get_action_probabilities((0, 0))

    assert set(probs.keys()) == {"up", "down", "left", "right"}
    assert np.isclose(sum(probs.values()), 1.0)
    for p in probs.values():
        assert np.isclose(p, 0.25)


def test_goal_directed_policy_action_probabilities_sum_to_one():
    policy = GoalDirectedPolicy(goal_state=(4, 4), exploration_prob=0.1)
    probs = policy.get_action_probabilities((0, 0))

    assert set(probs.keys()) == {"up", "down", "left", "right"}
    assert np.isclose(sum(probs.values()), 1.0)
    for p in probs.values():
        assert p >= 0.0


def test_goal_directed_policy_probabilities_all_states():
    env = GridworldMDP()
    policy = GoalDirectedPolicy(goal_state=env.goal_state, obstacles=env.obstacles)
    for state in env.get_all_states():
        if not env.is_terminal(state):
            probs = policy.get_action_probabilities(state)
            assert np.isclose(sum(probs.values()), 1.0), f"probs don't sum to 1 at {state}"


def test_bad_policy_action_probabilities_sum_to_one():
    policy = BadPolicy(bad_action_prob=0.7)
    probs = policy.get_action_probabilities((2, 2))

    assert set(probs.keys()) == {"up", "down", "left", "right"}
    assert np.isclose(sum(probs.values()), 1.0)
    # bad actions should have higher probability
    assert probs["up"] > probs["down"]
    assert probs["left"] > probs["right"]


# ---------------------------------------------------------------------------
# Week 5 — Bellman update tests
# ---------------------------------------------------------------------------


def test_compute_bellman_update_returns_float():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    result = compute_bellman_update(env, policy, values, (0, 0), gamma=0.9)

    assert isinstance(result, float)


def test_compute_bellman_update_terminal_state_returns_zero():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    result = compute_bellman_update(env, policy, values, env.goal_state, gamma=0.9)

    assert result == 0.0


def test_compute_bellman_update_nonterminal_is_finite():
    env = GridworldMDP()
    policy = GoalDirectedPolicy(goal_state=env.goal_state, obstacles=env.obstacles)
    values = {s: 0.0 for s in env.get_all_states()}

    result = compute_bellman_update(env, policy, values, (0, 0), gamma=0.9)

    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Week 5 — iterative policy evaluation tests
# ---------------------------------------------------------------------------

GAMMA = 0.9
THETA = 1e-3
MAX_ITER = 100


def test_iterative_policy_evaluation_returns_values_and_history():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    values, history = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    assert isinstance(values, dict)
    assert isinstance(history, list)
    assert len(history) > 0


def test_iterative_policy_evaluation_terminal_value_is_zero():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    assert values[env.goal_state] == 0.0


def test_iterative_policy_evaluation_all_valid_states_present():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    for state in env.get_all_states():
        assert state in values


def test_iterative_policy_evaluation_obstacles_not_in_values():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    for obstacle in env.obstacles:
        assert obstacle not in values


def test_iterative_policy_evaluation_convergence():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)

    _, history = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    # Either converged below theta or ran to max_iterations
    assert history[-1] < THETA or len(history) == MAX_ITER


def test_iterative_policy_evaluation_goal_directed_higher_than_random():
    env = GridworldMDP()
    random_policy = RandomPolicy(seed=0)
    goal_policy = GoalDirectedPolicy(goal_state=env.goal_state, obstacles=env.obstacles, seed=0)

    random_values, _ = iterative_policy_evaluation(
        env, random_policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )
    goal_values, _ = iterative_policy_evaluation(
        env, goal_policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    random_avg = np.mean([v for s, v in random_values.items() if not env.is_terminal(s)])
    goal_avg = np.mean([v for s, v in goal_values.items() if not env.is_terminal(s)])

    assert goal_avg > random_avg


# ---------------------------------------------------------------------------
# Week 5 — values_to_grid tests
# ---------------------------------------------------------------------------


def test_values_to_grid_returns_correct_shape():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    grid = values_to_grid(values, env)

    assert grid.shape == env.grid_size


def test_values_to_grid_obstacles_are_nan():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    grid = values_to_grid(values, env)

    for obstacle in env.obstacles:
        assert np.isnan(grid[obstacle])


def test_values_to_grid_start_and_goal_are_finite():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values, _ = iterative_policy_evaluation(
        env, policy, gamma=GAMMA, theta=THETA, max_iterations=MAX_ITER
    )

    grid = values_to_grid(values, env)

    start_row, start_col = env.start_state
    goal_row, goal_col = env.goal_state
    assert np.isfinite(grid[start_row, start_col])
    assert np.isfinite(grid[goal_row, goal_col])


# ---------------------------------------------------------------------------
# Week 5 — manual_bellman_update_details tests
# ---------------------------------------------------------------------------


def test_manual_bellman_update_details_structure():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    details = manual_bellman_update_details(env, policy, values, (0, 0), gamma=0.9)

    assert "state" in details
    assert "action_details" in details
    assert "updated_value" in details
    assert details["state"] == (0, 0)
    assert len(details["action_details"]) == len(env.get_valid_actions())


def test_manual_bellman_update_details_action_fields():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    details = manual_bellman_update_details(env, policy, values, (0, 0), gamma=0.9)

    for d in details["action_details"]:
        assert "action" in d
        assert "probability" in d
        assert "next_state" in d
        assert "reward" in d
        assert "next_state_value" in d
        assert "contribution" in d


def test_manual_bellman_update_details_contributions_sum_to_updated_value():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    details = manual_bellman_update_details(env, policy, values, (0, 0), gamma=0.9)

    total_from_contributions = sum(d["contribution"] for d in details["action_details"])
    assert np.isclose(total_from_contributions, details["updated_value"])


def test_manual_bellman_update_details_terminal_state():
    env = GridworldMDP()
    policy = RandomPolicy(seed=0)
    values = {s: 0.0 for s in env.get_all_states()}

    details = manual_bellman_update_details(env, policy, values, env.goal_state, gamma=0.9)

    assert details["action_details"] == []
    assert details["updated_value"] == 0.0


# ---------------------------------------------------------------------------
# Week 6 - policy iteration tests
# ---------------------------------------------------------------------------


def test_initialize_random_deterministic_policy_covers_valid_nonterminal_states():
    env = GridworldMDP()
    policy = initialize_random_deterministic_policy(env=env, seed=42)

    expected_states = {state for state in env.get_all_states() if not env.is_terminal(state)}

    assert isinstance(policy, dict)
    assert set(policy.keys()) == expected_states
    assert env.goal_state not in policy
    for obstacle in env.obstacles:
        assert obstacle not in policy
    for action in policy.values():
        assert action in env.get_valid_actions()


def test_deterministic_policy_probabilities_are_one_hot():
    env = GridworldMDP()
    policy_dict = initialize_random_deterministic_policy(env=env, seed=42)
    state = env.start_state

    probabilities = get_policy_action_probabilities(policy_dict, state, env)
    wrapped_policy = DeterministicPolicy(policy_dict, env)
    wrapped_probabilities = wrapped_policy.get_action_probabilities(state)

    assert np.isclose(sum(probabilities.values()), 1.0)
    assert probabilities[policy_dict[state]] == 1.0
    assert probabilities == wrapped_probabilities
    for action, probability in probabilities.items():
        if action != policy_dict[state]:
            assert probability == 0.0


def test_one_step_lookahead_returns_finite_scores_without_mutating_state():
    env = GridworldMDP()
    values = {state: 0.0 for state in env.get_all_states()}
    env.current_state = (0, 1)

    scores = one_step_lookahead(env=env, values=values, state=(0, 1), gamma=1.0)

    assert set(scores.keys()) == set(env.get_valid_actions())
    assert all(np.isfinite(score) for score in scores.values())
    assert env.current_state == (0, 1)


def test_improve_policy_returns_expected_metadata():
    env = GridworldMDP()
    policy = initialize_random_deterministic_policy(env=env, seed=42)
    values, _ = iterative_policy_evaluation(
        env=env,
        policy=DeterministicPolicy(policy, env),
        gamma=1.0,
        theta=1e-3,
        max_iterations=100,
    )

    new_policy, policy_stable, num_policy_changes = improve_policy(
        env=env,
        policy=policy,
        values=values,
        gamma=1.0,
    )

    assert isinstance(new_policy, dict)
    assert isinstance(policy_stable, bool)
    assert isinstance(num_policy_changes, int)
    assert num_policy_changes >= 0


def test_policy_iteration_returns_expected_structure_and_valid_entries():
    env = GridworldMDP()
    results = policy_iteration(
        env=env,
        gamma=1.0,
        theta=1e-3,
        max_policy_iterations=20,
        max_eval_iterations=100,
        seed=42,
    )

    assert set(results.keys()) == {
        "final_policy",
        "final_values",
        "policy_changes",
        "value_histories",
        "num_iterations",
        "policy_stable",
        "initial_policy",
    }
    assert results["num_iterations"] > 0
    assert isinstance(results["policy_stable"], bool)
    assert len(results["policy_changes"]) == results["num_iterations"]
    assert len(results["value_histories"]) == results["num_iterations"]

    for action in results["final_policy"].values():
        assert action in env.get_valid_actions()
    assert set(results["final_values"].keys()) == set(env.get_all_states())


def test_extract_greedy_trajectory_returns_expected_structure():
    env = GridworldMDP()
    results = policy_iteration(
        env=env,
        gamma=1.0,
        theta=1e-3,
        max_policy_iterations=20,
        max_eval_iterations=100,
        seed=42,
    )

    trajectory = extract_greedy_trajectory(
        env=env,
        policy=results["final_policy"],
        max_steps=100,
    )

    assert set(trajectory.keys()) == {"states", "actions", "success", "episode_length"}
    assert trajectory["episode_length"] <= 100
    assert len(trajectory["states"]) >= 1
    assert len(trajectory["actions"]) == trajectory["episode_length"]
