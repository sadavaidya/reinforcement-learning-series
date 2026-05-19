"""Bandit components for the reinforcement learning series."""

from .agents import EpsilonGreedyAgent, UCBAgent
from .environment import TenArmedBandit
from .experiments import (
    run_multiple_agent_experiments,
    run_multiple_experiments,
    run_single_agent_experiment,
    run_single_experiment,
)

__all__ = [
    "TenArmedBandit",
    "EpsilonGreedyAgent",
    "UCBAgent",
    "run_single_agent_experiment",
    "run_single_experiment",
    "run_multiple_agent_experiments",
    "run_multiple_experiments",
]
