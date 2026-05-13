# Week 1: Epsilon-Greedy Action Selection in the 10-Armed Bandit

The multi-armed bandit problem is one of the simplest reinforcement learning settings. An agent repeatedly chooses from a fixed set of actions, receives an immediate reward, and tries to learn which action gives the highest expected payoff over time.

For this week, the implementation uses the standard 10-armed testbed from Sutton and Barto. Each arm has an unknown true action value sampled from a normal distribution. When an arm is pulled, the observed reward is a noisy sample centered on that arm's true value. Because the environment is stationary, those underlying action values do not change during a run.

The main learning tension is exploration versus exploitation. Exploitation means choosing the action that currently looks best according to the agent's estimates. Exploration means trying other actions to gather information that might improve future decisions. A purely greedy strategy exploits only, while epsilon-greedy explores with a small probability `epsilon`.

The agent estimates action values using sample averages. If `Q_t(a)` is the current estimate for action `a`, and the action has been selected `N` times, then the estimate is updated after receiving reward `R` as:

`Q_new = Q_old + (R - Q_old) / N`

This update rule is simple and intuitive: the estimate moves toward the new reward, but the step size gets smaller as more evidence is collected.

Greedy action selection chooses the action with the largest estimated value. Epsilon-greedy modifies that rule by choosing a random action with probability `epsilon`, and otherwise acting greedily. This helps prevent the agent from getting stuck on an early but misleading estimate.

Two metrics are used to evaluate performance. Average reward tracks how much reward the agent receives over time on average across many independent runs. Percentage optimal action tracks how often the agent selects the arm that truly has the highest expected reward in each bandit instance.

The key learning from Week 1 is that pure greed can fail because it may stop exploring too soon. Epsilon-greedy gives up a small amount of short-term certainty in exchange for better long-term learning.
