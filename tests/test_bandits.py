from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bandits.agents import EpsilonGreedyAgent
from src.bandits.environment import TenArmedBandit
from src.bandits.experiments import run_multiple_experiments, run_single_experiment


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
