# Week 6 - Dynamic Programming: Policy Iteration

## 1. Context

Week 5 focused on evaluating a fixed policy. Week 6 takes the next step: use the value function to improve the policy itself.

This is the first control method in the series. Instead of only asking how good a policy is, the main question becomes how to make it better.

## 2. Policy Evaluation Recap

Given a policy `pi`, policy evaluation computes:

v<sub>pi</sub>(s)

This is the expected return from state `s` when the agent follows `pi`.

During evaluation, the policy is fixed. The value estimates change, but the action choices do not.

## 3. Policy Improvement

Once a value function is available, it can be used to compare actions.

In deterministic Gridworld, a one-step lookahead score is:

`action_score = reward + gamma V(next_state)`

This asks: if the agent takes an action now and then follows the current value estimate afterward, how good does that action look?

## 4. Greedy Policy

A greedy improvement step chooses the action with the highest one-step lookahead score.

This can be written as:

`pi_new(s) = argmax_a [r + gamma V(s')]`

So the updated policy acts greedily with respect to the current value function.

## 5. Policy Improvement Theorem

The policy improvement theorem says that if the new policy is chosen greedily with respect to the current policy's value function, then the new policy is at least as good as the old one.

This is the key reason policy iteration works: evaluation estimates the current policy, and improvement uses those estimates to move toward a better policy.

## 6. Policy Iteration

Policy iteration alternates between:
- policy evaluation
- policy improvement

The loop is:
1. evaluate the current policy
2. improve the policy greedily
3. repeat until the policy stops changing

This combines prediction and control into one Dynamic Programming procedure.

## 7. Policy Stability

A policy is stable when the improvement step does not change any state's action.

If every state keeps the same greedy action after improvement, the algorithm has converged for that finite MDP under the chosen dynamics and rewards.

## 8. Dynamic Programming Assumption

Dynamic Programming assumes the environment model is known.

That means the transition and reward structure can be queried directly. In this Gridworld, the next state and reward for each `(state, action)` pair are known, so policy evaluation and policy improvement can be computed exactly from the model.

## 9. Week 6 Experiment

This week's experiment:
- starts with a random deterministic policy
- evaluates that policy
- improves it greedily with one-step lookahead
- repeats until the policy is stable
- visualizes the initial policy, final policy, final value function, policy changes, and a final trajectory

The number of changed states per iteration is a useful convergence signal because it shows how quickly the policy structure settles.

## 10. Main Takeaways

- Policy evaluation tells how good the current policy is.
- Policy improvement uses those values to choose better actions.
- Policy iteration repeatedly evaluates and improves until convergence.
- Policy iteration is a model-based control algorithm.
