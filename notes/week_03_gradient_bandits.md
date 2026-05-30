# Week 3 - Gradient Bandit Algorithms

## 1. Context

Previous bandit methods estimated action values `Q(a)` and then used those estimates to decide which action to take. Gradient bandits take a different route: they learn action preferences `H(a)` directly. Instead of asking, "What reward do I expect from this action?", the method asks, "How strongly should I prefer this action over the others?"

That makes Week 3 an important transition point. It still uses the familiar multi-armed bandit setting, but the update is now policy-oriented rather than value-oriented. This is one of the first clear steps toward later policy-gradient methods in reinforcement learning.

## 2. Action Preferences

`H(a)` is a preference score, not a reward estimate. A higher preference means the corresponding action becomes more likely to be selected, but it does not mean the action has a reward equal to that number.

This distinction matters:
- In epsilon-greedy and UCB, the agent tries to estimate how good each action is.
- In gradient bandits, the agent directly adjusts how likely each action should be.

So the learned quantity is no longer a value function. It is a policy parameter.

## 3. Softmax Action Selection

Preferences are converted into probabilities with the softmax function:

```math
\pi(a) = \frac{\exp(H(a))}{\sum_b \exp(H(b))}
```

Here:
- `H(a)` is the preference for action `a`
- `π(a)` is the probability of selecting action `a`
- the denominator sums over all possible actions

Softmax turns arbitrary preference scores into a valid probability distribution:
- every probability is non-negative
- all probabilities sum to 1
- actions with larger preferences get larger probabilities

If all preferences are equal, then every action has the same probability. If one action's preference becomes larger than the others, its probability rises smoothly rather than switching abruptly to always-greedy behavior.

## 4. Gradient Bandit Update

After selecting an action and receiving a reward, the agent compares that reward against a baseline. The selected action is then pushed up or down depending on whether the outcome was better or worse than expected.

The intuition is:
- if the reward is better than expected, increase the preference for the chosen action
- if the reward is worse than expected, decrease the preference for the chosen action
- adjust the non-selected actions in the opposite direction so the policy remains balanced

This is why the algorithm is called a gradient bandit method. It changes the policy parameters in the direction that should improve expected reward.

## 5. Reward Baseline

The baseline is usually the average reward observed so far. It is updated every iteration and helps decide whether the latest reward was better or worse than expected.

This gives the update a useful reference point:
- a reward above the average should reinforce the chosen action
- a reward below the average should discourage it

In practice, this usually improves stability because the update depends on relative performance, not just the raw reward magnitude.

## 6. With Baseline vs Without Baseline

With a baseline, the update uses:

`R - average_reward`

Without a baseline, the update uses:

`R`

The baseline becomes especially useful when rewards are shifted away from zero. In the Sutton and Barto style setup for this experiment, the true action values are centered around `4.0` instead of `0.0`. That means rewards are often positive even when an action is not especially good relative to the others.

Without a baseline, those raw positive rewards can make the update less informative. With a baseline, the agent asks whether the reward was good relative to what it has been seeing on average, which usually leads to cleaner learning.

## 7. Why Gradient Bandits Matter

Gradient bandits directly learn the policy, meaning they adjust how likely each action is to be selected rather than estimating how valuable each action is.

That idea becomes important again later in reinforcement learning:
- policy-gradient methods also optimize policy parameters directly
- stochastic policies remain useful when exploration should be built into the policy itself
- the distinction between value learning and policy learning starts becoming concrete here

## 8. Week 3 Takeaway

Gradient bandits learn how likely each action should be selected, instead of directly estimating how valuable each action is. The softmax policy converts preferences into probabilities, and the reward baseline makes those preference updates more meaningful and stable.
