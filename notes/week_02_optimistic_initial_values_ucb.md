# Week 2: Optimistic Initial Values and UCB Action Selection

## Context

After implementing epsilon-greedy action selection in Week 1, the next step was to look at two other ways to handle exploration in the same 10-armed bandit setting:

- optimistic initial values
- Upper-Confidence-Bound (UCB) action selection

The environment is still the standard stationary 10-armed testbed from Sutton and Barto:

- each action has a true value that stays fixed during a run
- the observed reward is a noisy sample around that true value
- the environment does not change while the agent is learning

That keeps the comparison focused on the action-selection strategy rather than on changes in the environment.

## Nonstationary Background

Section 2.5 of the book discusses what changes in a nonstationary problem. In a stationary problem, sample-average updates work well because all past rewards are sampled from the same underlying value distribution. In a nonstationary problem, the true action values can drift over time, so older rewards become less useful.

The main issue is that the sample-average step size becomes `1 / N`, which means the estimate adapts more slowly as more data is collected. A constant step-size update avoids that by giving more weight to recent rewards.

I did not implement the nonstationary case this week, but it matters as background:

- sample-average updates are a good fit for stationary bandits
- constant step-size updates are usually better when action values change over time

## Optimistic Initial Values

Optimistic initial values encourage exploration without adding random actions. Instead of starting from `Q(a) = 0`, the agent starts from a deliberately high estimate such as Q<sub>0</sub> = 5.

I used Q<sub>0</sub> = 5 because it is clearly above the typical reward scale in this testbed, so the optimistic effect is easy to see.

The idea is straightforward:

- at the start, every action looks promising
- once an action is selected, its estimate drops toward a more realistic value
- untried actions still look better for a while
- even a greedy agent with `epsilon = 0` ends up exploring early on

### What shows up in the curves

- the optimistic greedy agent usually explores aggressively at the beginning
- optimal-action percentage often rises quickly after the first few updates
- average reward can look noisy early because the initial estimates are being corrected

### Limitation

The main limitation is that the exploration is mostly front-loaded. Once the optimistic estimates have been corrected, the method behaves much more like an ordinary greedy strategy.

That means:

- it does not keep encouraging exploration later in the run
- if the early estimates move in the wrong direction, there is no explicit mechanism to keep checking alternatives

## UCB Action Selection

UCB takes a different approach. Instead of exploring randomly, it adds an uncertainty bonus to each action estimate. The selected action is:

`A_t = argmax_a [ Q_t(a) + c sqrt(ln(t) / N_t(a)) ]`

This rule combines:

- estimated value: actions that currently look good
- uncertainty bonus: actions that have not been tried often

I used `c = 2` as a simple baseline. It is large enough to make directed exploration visible, but not so large that the uncertainty term dominates the value estimates for too long.

### Why it behaves differently from epsilon-greedy

Epsilon-greedy explores by occasionally choosing a random action, even if that action already looks poor. UCB still explores, but it does so in a more targeted way by preferring actions that are uncertain.

### What shows up in the curves

- after the initial warm-up, UCB usually looks more stable than epsilon-greedy
- exploration is spent on under-sampled actions rather than random retries
- that often leads to stronger average reward and a steadier rise in optimal-action selection

## Comparison

The three approaches can be summarized like this:

- epsilon-greedy explores with randomness
- optimistic initial values explore through high starting estimates
- UCB explores through an explicit uncertainty bonus

They are all trying to solve the same exploration problem, but they do it in different ways.

## Week 2 Takeaway

The main result from Week 2 is that exploration does not need to come from randomness alone.

- optimistic initial values give a simple push toward early exploration
- UCB makes exploration depend on both current estimates and uncertainty
- UCB is generally more deliberate than epsilon-greedy

That makes Week 2 a useful extension of Week 1: the core problem is still exploration versus exploitation, but the mechanism for exploration becomes more informed.
