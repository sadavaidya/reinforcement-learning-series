from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bandits.agents import EpsilonGreedyAgent, GradientBanditAgent, UCBAgent
from src.bandits.environment import TenArmedBandit
from src.bandits.experiments import (
    run_multiple_agent_experiments,
    run_multiple_experiments,
    run_single_experiment,
)


def test_bandit_creates_correct_number_of_actions():
    bandit = TenArmedBandit(n_actions=10, seed=0)
    assert bandit.n_actions == 10
    assert bandit.q_true.shape == (10,)


def test_bandit_step_returns_float():
    bandit = TenArmedBandit(seed=0)
    reward = bandit.step(0)
    assert isinstance(reward, float)


def test_bandit_optimal_action_is_valid():
    bandit = TenArmedBandit(seed=0)
    assert 0 <= bandit.optimal_action < bandit.n_actions


def test_agent_selects_valid_action():
    agent = EpsilonGreedyAgent(n_actions=10, epsilon=0.1, seed=0)
    action = agent.select_action()
    assert 0 <= action < agent.n_actions


def test_agent_update_changes_estimate_and_count():
    agent = EpsilonGreedyAgent(n_actions=10, epsilon=0.1, seed=0)
    action = 3
    reward = 2.5

    old_estimate = agent.q_estimates[action]
    old_count = agent.action_counts[action]
    agent.update(action, reward)

    assert agent.action_counts[action] == old_count + 1
    assert agent.q_estimates[action] != old_estimate
    assert np.isclose(agent.q_estimates[action], reward)


def test_optimistic_agent_initializes_all_estimates_to_initial_value():
    agent = EpsilonGreedyAgent(
        n_actions=10,
        epsilon=0.0,
        initial_value=5.0,
        seed=0,
    )
    assert np.all(agent.q_estimates == 5.0)


def test_optimistic_agent_update_changes_only_selected_action():
    agent = EpsilonGreedyAgent(
        n_actions=5,
        epsilon=0.0,
        initial_value=5.0,
        seed=0,
    )

    agent.update(action=2, reward=1.0)

    assert agent.action_counts[2] == 1
    assert np.isclose(agent.q_estimates[2], 1.0)
    assert np.all(agent.q_estimates[[0, 1, 3, 4]] == 5.0)


def test_ucb_agent_selects_valid_action():
    agent = UCBAgent(n_actions=10, c=2.0, seed=0)
    action = agent.select_action()
    assert 0 <= action < agent.n_actions


def test_ucb_agent_initially_selects_untried_actions():
    agent = UCBAgent(n_actions=4, c=2.0, seed=0)
    selected_actions: list[int] = []

    for _ in range(agent.n_actions):
        action = agent.select_action()
        selected_actions.append(action)
        agent.update(action, reward=1.0)

    assert len(set(selected_actions)) == agent.n_actions
    assert np.all(agent.action_counts == 1)


def test_ucb_agent_update_changes_estimate_and_count():
    agent = UCBAgent(n_actions=5, c=2.0, seed=0)

    action = agent.select_action()
    agent.update(action, reward=2.5)

    assert agent.action_counts[action] == 1
    assert np.isclose(agent.q_estimates[action], 2.5)


def test_ucb_agent_returns_valid_action_after_all_actions_tried():
    agent = UCBAgent(n_actions=3, c=2.0, seed=0)

    for _ in range(agent.n_actions):
        action = agent.select_action()
        agent.update(action, reward=1.0)

    action = agent.select_action()
    assert 0 <= action < agent.n_actions


def test_ucb_agent_avoids_division_by_zero_after_warmup():
    agent = UCBAgent(n_actions=3, c=2.0, seed=0)

    for _ in range(agent.n_actions):
        action = agent.select_action()
        agent.update(action, reward=0.5)

    action = agent.select_action()
    assert 0 <= action < agent.n_actions


def test_single_experiment_returns_correct_shapes():
    rewards, optimal_action_selected = run_single_experiment(
        epsilon=0.1,
        n_steps=25,
        n_actions=10,
        seed=0,
    )
    assert rewards.shape == (25,)
    assert optimal_action_selected.shape == (25,)


def test_multiple_experiment_runner_returns_expected_keys():
    epsilons = [0.0, 0.01, 0.1]
    results = run_multiple_experiments(
        epsilons=epsilons,
        n_runs=3,
        n_steps=10,
        n_actions=10,
        seed=0,
    )

    assert set(results.keys()) == set(epsilons)
    for epsilon in epsilons:
        assert "average_rewards" in results[epsilon]
        assert "optimal_action_percentage" in results[epsilon]


def test_multiple_agent_experiments_returns_expected_keys_and_shapes():
    agent_configs = {
        "epsilon_greedy_0.1": {
            "agent_class": EpsilonGreedyAgent,
            "agent_kwargs": {"epsilon": 0.1, "initial_value": 0.0},
        },
        "optimistic_greedy": {
            "agent_class": EpsilonGreedyAgent,
            "agent_kwargs": {"epsilon": 0.0, "initial_value": 5.0},
        },
        "ucb_c_2": {
            "agent_class": UCBAgent,
            "agent_kwargs": {"c": 2.0, "initial_value": 0.0},
        },
    }

    results = run_multiple_agent_experiments(
        agent_configs=agent_configs,
        n_runs=3,
        n_steps=10,
        n_actions=10,
        seed=0,
    )

    assert set(results.keys()) == set(agent_configs.keys())
    for result in results.values():
        assert result["average_rewards"].shape == (10,)
        assert result["optimal_action_percentage"].shape == (10,)


def test_gradient_bandit_initialization():
    agent = GradientBanditAgent(n_actions=10, alpha=0.1, use_baseline=True, seed=0)

    assert agent.preferences.shape == (10,)
    assert np.all(agent.preferences == 0.0)
    assert agent.action_probabilities.shape == (10,)
    assert np.allclose(agent.action_probabilities, np.full(10, 0.1))
    assert agent.average_reward == 0.0
    assert agent.time_step == 0


def test_gradient_bandit_softmax_returns_valid_uniform_distribution():
    agent = GradientBanditAgent(n_actions=4, alpha=0.1, seed=0)

    probabilities = agent._softmax()

    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities, np.full(4, 0.25))


def test_gradient_bandit_select_action_returns_valid_action_and_updates_policy():
    agent = GradientBanditAgent(n_actions=5, alpha=0.1, seed=0)

    action = agent.select_action()

    assert 0 <= action < agent.n_actions
    assert np.isclose(agent.action_probabilities.sum(), 1.0)
    assert np.all(agent.action_probabilities >= 0.0)


def test_gradient_bandit_update_with_baseline_changes_preferences_and_tracking():
    agent = GradientBanditAgent(n_actions=4, alpha=0.1, use_baseline=True, seed=0)
    action = agent.select_action()
    old_preferences = agent.preferences.copy()

    agent.update(action=action, reward=2.0)

    assert not np.allclose(agent.preferences, old_preferences)
    assert np.isclose(agent.average_reward, 2.0)
    assert agent.time_step == 1


def test_gradient_bandit_update_without_baseline_changes_preferences_and_tracking():
    agent = GradientBanditAgent(n_actions=4, alpha=0.1, use_baseline=False, seed=0)
    action = agent.select_action()
    old_preferences = agent.preferences.copy()

    agent.update(action=action, reward=1.5)

    assert not np.allclose(agent.preferences, old_preferences)
    assert np.isclose(agent.average_reward, 1.5)
    assert agent.time_step == 1


def test_multiple_agent_experiments_supports_gradient_bandits():
    agent_configs = {
        "Gradient bandit α=0.1 with baseline": {
            "agent_class": GradientBanditAgent,
            "agent_kwargs": {"alpha": 0.1, "use_baseline": True},
        },
        "Gradient bandit α=0.1 without baseline": {
            "agent_class": GradientBanditAgent,
            "agent_kwargs": {"alpha": 0.1, "use_baseline": False},
        },
    }

    results = run_multiple_agent_experiments(
        agent_configs=agent_configs,
        n_runs=3,
        n_steps=5,
        n_actions=10,
        bandit_kwargs={
            "true_reward_mean": 4.0,
            "true_reward_std": 1.0,
            "reward_std": 1.0,
        },
        seed=0,
    )

    assert set(results.keys()) == set(agent_configs.keys())
    for result in results.values():
        assert result["average_rewards"].shape == (5,)
        assert result["optimal_action_percentage"].shape == (5,)
