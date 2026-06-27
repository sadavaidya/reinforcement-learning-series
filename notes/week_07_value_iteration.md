# Week 7 - Dynamic Programming: Value Iteration

## 1. Context

Week 6 used policy iteration to alternate between evaluating a policy and improving it greedily.

Week 7 keeps the same Gridworld and the same Dynamic Programming setting, but changes the update rule. Instead of fully evaluating the current policy before improving it, Value Iteration applies the Bellman optimality backup directly.

The goal now is not to estimate `v_pi(s)` for one fixed policy, but to move straight toward the optimal value function `v_*(s)`.

## 2. Short Recap of Policy Iteration

Policy iteration repeats two steps:
- policy evaluation
- policy improvement

Policy evaluation computes the value of the current policy.
Policy improvement then chooses the greedy action in each state using those values.

This continues until the policy is stable.

## 3. Bellman Optimality Equation

The optimal state-value function satisfies:

v<sub>*</sub>(s) = max<sub>a</sub> sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma v<sub>*</sub>(s') ]

This is the Bellman optimality equation for state values.

In the deterministic Gridworld used in this repo, each action leads to one next state and one reward, so the update becomes:

v<sub>*</sub>(s) = max<sub>a</sub> [ r(s, a) + gamma V(s') ]

where `s'` is the next state reached by taking action `a` in state `s`.

## 4. Policy Evaluation vs Value Iteration

Policy evaluation uses the Bellman **expectation** equation:

v<sub>pi</sub>(s) = sum<sub>a</sub> pi(a | s) sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma v<sub>pi</sub>(s') ]

So the update is an average under the current policy.

Value iteration uses the Bellman **optimality** equation:

v<sub>*</sub>(s) = max<sub>a</sub> sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma v<sub>*</sub>(s') ]

So the update always keeps the best action value available from that state.

Policy evaluation asks:
"How good is this policy?"

Value iteration asks:
"If I act optimally from this state onward, how good can this state become?"

## 5. Value Iteration Algorithm

For a finite MDP, the basic synchronous algorithm is:

1. Initialize `V(s) = 0` for all valid states.
2. Repeat for each sweep:
   - for each non-terminal state `s`
   - compute all one-step action returns
   - update `V(s)` to the maximum action return
3. Track
   - `delta = max_s |V_new(s) - V(s)|`
4. Stop when `delta < theta`.

In this repo, the update is:

`V_new(s) = max_a [ reward + gamma * V(next_state) ]`

Terminal states stay at value `0`.

## 6. Greedy Policy Extraction

After the value function has converged, a greedy policy can be extracted by computing:

q(s, a) = sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma V(s') ]

for every action, then choosing the maximizing action:

pi(s) = argmax<sub>a</sub> q(s, a)

This converts the converged optimal value information into a control policy.

## 7. Why `max` Gives the Value and `argmax` Gives the Action

The quantity inside the backup is the return estimate for taking a specific action and then continuing with the best future behavior.

- `max` returns the **largest numerical return**, so it gives the optimal state value.
- `argmax` returns the **action index or label** that achieved that largest return, so it gives the greedy action.

So:
- `max` answers "how good is the best choice?"
- `argmax` answers "which choice is the best?"

## 8. Gridworld Intuition

In Gridworld, states near the goal usually improve first because they can reach the goal reward in fewer steps.

Then that information propagates backward:
- states one step away become valuable
- then states two steps away
- and so on

Invalid moves and obstacles reduce value because they either waste steps or produce penalties.

This backward propagation is exactly what repeated Bellman optimality backups produce.

## 9. Comparison with Policy Iteration

Both policy iteration and value iteration are Dynamic Programming control methods and both assume access to the environment model.

Main difference:
- Policy iteration separates evaluation and improvement.
- Value iteration merges them into one repeated optimality update.

Typical behavior:
- Policy iteration often uses fewer outer improvement steps.
- But each outer step can be expensive because it includes policy evaluation sweeps.
- Value iteration usually performs more simple sweeps.
- Each sweep is cheaper because it applies only the Bellman optimality backup.

In this Week 7 Gridworld, the greedy policy extracted from value iteration should match the optimal policy found by policy iteration, up to deterministic tie-breaking.

## 10. Main Takeaways

- Value iteration targets the optimal value function `v_*(s)` directly.
- The Bellman optimality equation replaces the policy-weighted average with a max over actions.
- After convergence, a greedy policy can be extracted with `argmax`.
- `max` gives the optimal value; `argmax` gives the action that achieves it.
- Value iteration and policy iteration should agree on the optimal solution in this finite Gridworld.
