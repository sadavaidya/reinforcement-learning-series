# Week 1: Epsilon-Greedy Action Selection in the 10-Armed Bandit

## Context

The multi-armed bandit problem is one of the simplest reinforcement learning settings. An agent repeatedly chooses from a fixed set of actions, receives an immediate reward, and tries to learn which action has the highest expected payoff over time.

For this week, I used the standard 10-armed testbed from Sutton and Barto:

- each arm has an unknown true action value
- true action values are sampled from a normal distribution
- observed rewards are noisy samples around those true values
- the environment is stationary, so the true values do not change during a run

That makes the focus very clear: the main question is how the agent should balance exploration and exploitation.

## Core Idea

The key tension is:

- exploitation: choose the action that currently looks best
- exploration: try other actions to gather more information

A purely greedy agent only exploits. An epsilon-greedy agent explores with probability `epsilon` and acts greedily the rest of the time.

## Action-Value Update

The agent estimates action values using the sample-average update rule. If `Q_t(a)` is the current estimate for action `a`, and the action has been selected `N` times, then after observing reward `R` the update is:

`Q_new = Q_old + (R - Q_old) / N`

This works well in the stationary setting because:

- the estimate moves toward the new reward
- the step size becomes smaller as more evidence is collected
- early rewards matter more, but later updates still refine the estimate

## Epsilon-Greedy Policy

Greedy action selection always picks the action with the largest estimated value. Epsilon-greedy changes that by introducing a small amount of random exploration:

- with probability `epsilon`, choose a random action
- otherwise, choose the action with the highest current estimate

This helps prevent the agent from committing too early to an action that only looked good because of a lucky initial reward.

## Evaluation

I used two metrics to compare performance across many independent runs:

- average reward: how much reward the agent receives over time on average
- optimal action percentage: how often the agent selects the action with the highest true value

These two views are useful together. Average reward shows overall performance, while optimal action percentage shows how quickly the agent identifies the best arm.

## Week 1 Takeaway

The main result from Week 1 is simple:

- pure greedy action selection can get stuck because it may stop exploring too early
- epsilon-greedy gives up a small amount of short-term certainty
- that extra exploration usually leads to better long-term learning
