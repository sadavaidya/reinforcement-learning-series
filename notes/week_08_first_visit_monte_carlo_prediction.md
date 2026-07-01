# Week 8 - Monte Carlo Methods: First-Visit Prediction

## Book Reference

Main reference: Sutton & Barto, Chapter 5, mainly Section 5.1 on Monte Carlo prediction.

## From Dynamic Programming to Monte Carlo

Weeks 5 to 7 used Dynamic Programming (DP) methods in the Gridworld because the model was known. In particular, policy evaluation used the transition dynamics and rewards directly through Bellman expectation backups.

Monte Carlo (MC) methods shift the perspective:

- DP uses the known model `p(s', r | s, a)`.
- MC does not require the transition model.
- MC learns from complete sampled episodes.

So MC is **model-free** in the sense that it estimates values from experience rather than from explicit knowledge of the environment dynamics.

## Prediction Under a Fixed Policy

Week 8 is still a **prediction** problem, not a control problem.

The question remains:

How good is each state under a fixed policy `pi`?

The target quantity is the state-value function:

v<sub>&pi;</sub>(s) = E<sub>&pi;</sub>[ G<sub>t</sub> | S<sub>t</sub> = s ]

Here:

- S<sub>t</sub> is the state at time `t`
- A<sub>t</sub> is the action at time `t`
- R<sub>t+1</sub> is the reward after taking action A<sub>t</sub>
- G<sub>t</sub> is the return from time `t`

## Episodes and Returns

A Monte Carlo method works with **episodes**, meaning finite trajectories that start in some state and continue until termination or truncation.

For an episode

S<sub>t</sub>, A<sub>t</sub>, R<sub>t+1</sub>, S<sub>t+1</sub>, A<sub>t+1</sub>, R<sub>t+2</sub>, ...

the return is

G<sub>t</sub> = R<sub>t+1</sub> + &gamma; R<sub>t+2</sub> + &gamma;<sup>2</sup> R<sub>t+3</sub> + ...

Monte Carlo prediction waits until the episode ends, then computes returns from the sampled rewards.

## First-Visit MC Algorithm

First-Visit MC estimates `V(s)` by averaging returns observed after the **first** occurrence of state `s` within each episode.

Algorithm sketch:

1. Generate an episode using the fixed policy `pi`.
2. Compute returns G<sub>t</sub> backward through the episode.
3. For each state S<sub>t</sub>:
   - if this is the first visit to S<sub>t</sub> in the episode, record G<sub>t</sub>
   - update `V(S_t)` as the average of all recorded returns for that state
4. Repeat for many episodes.

This gives the estimator:

`V(s) = average of observed returns following first visits to s`

## First-Visit vs Every-Visit MC

- **First-Visit MC** updates a state only on its first occurrence within an episode.
- **Every-Visit MC** updates a state on every occurrence within an episode.

Both are valid Monte Carlo prediction methods. This week implements **First-Visit MC only**.

## Why MC Is Model-Free

Monte Carlo prediction does not need Bellman expectation backups or the explicit model `p(s', r | s, a)`.

Instead, it:

- samples full episodes by interacting with the environment
- observes rewards directly
- estimates expected returns by averaging

This is what makes MC a model-free method.

## Why MC Does Not Bootstrap

DP methods bootstrap because they update estimates using other current estimates, for example through terms like `r + gamma V(s')`.

Monte Carlo does **not** bootstrap.

It updates from complete sampled returns G<sub>t</sub>, which are built from actual rewards observed later in the same episode.

## Gridworld Experiment Summary

The Week 8 experiment reuses the existing Gridworld from the earlier DP weeks, but now treats it as an interactive environment for sampling episodes.

The implementation:

- samples episodes under a fixed policy
- computes returns backward
- updates `V(s)` from first visits only
- compares the final MC estimate with Week 5 iterative policy evaluation

To ensure broad state coverage, episodes are started from randomly sampled valid non-terminal states while still following the same fixed policy.

## Comparison with DP Policy Evaluation

Both methods estimate v<sub>&pi;</sub>(s), but they use different information:

- **DP policy evaluation** uses the known model and Bellman expectation updates.
- **First-Visit MC prediction** uses sampled episodes and averages observed returns.

With enough episodes, the MC estimate should move closer to the DP value function for the same policy.

## Key Takeaways

- DP uses a known model.
- MC learns from complete sampled episodes.
- First-Visit MC updates each state once per episode.
- MC estimates `V(s)` by averaging observed returns.
- MC is model-free.
- MC does not bootstrap.
- This sets up the transition toward Temporal-Difference learning in later weeks.
