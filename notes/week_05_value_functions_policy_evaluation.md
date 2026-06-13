# Week 5 - Value Functions and Policy Evaluation

## 1. Context

Week 4 introduced Markov Decision Processes using a Gridworld environment and compared three fixed policies by their average return, episode length, and success rate.

Week 5 builds on the same Gridworld and asks a different question: given that we are following a fixed policy, how good is each state?

This is what value functions answer.

## 2. State-Value Function

The state-value function v<sub>pi</sub>(s) is the expected return when starting in state s and following policy pi thereafter.

v<sub>pi</sub>(s) = E<sub>pi</sub>[ G<sub>t</sub> | S<sub>t</sub> = s ]

Key points:
- The value depends entirely on the policy. A good policy makes states more valuable.
- Terminal states have value 0 by convention. There is no future return once the goal is reached.
- States closer to the goal under a good policy generally have higher values.
- v<sub>pi</sub>(s) does not tell us how to improve the policy. It only evaluates the current policy.

## 3. Action-Value Function

The action-value function q<sub>pi</sub>(s, a) is the expected return when starting in state s, taking action a, and then following policy pi.

q<sub>pi</sub>(s, a) = E<sub>pi</sub>[ G<sub>t</sub> | S<sub>t</sub> = s, A<sub>t</sub> = a ]

Key points:
- q<sub>pi</sub>(s, a) distinguishes the quality of different actions in the same state.
- Later weeks use q<sub>pi</sub> directly for SARSA and Q-learning.
- In Week 5, q<sub>pi</sub> is not computed explicitly, but the Bellman update performs it implicitly for each action.

## 4. Relationship Between v<sub>pi</sub>(s) and q<sub>pi</sub>(s, a)

v<sub>pi</sub>(s) = sum<sub>a</sub> pi(a | s) q<sub>pi</sub>(s, a)

The state value is the policy-weighted average of the action values.

Expanding q<sub>pi</sub>(s, a):

q<sub>pi</sub>(s, a) = sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma v<sub>pi</sub>(s') ]

Combining these gives the Bellman expectation equation for v<sub>pi</sub>.

## 5. Bellman Expectation Equation

The return satisfies this recursive relationship:

G<sub>t</sub> = R<sub>t+1</sub> + gamma G<sub>t+1</sub>

Substituting into the definition of v<sub>pi</sub>(s) gives the Bellman expectation equation:

v<sub>pi</sub>(s) = sum<sub>a</sub> pi(a | s) sum<sub>s', r</sub> p(s', r | s, a) [ r + gamma v<sub>pi</sub>(s') ]

For the deterministic Gridworld from Week 4, the transition is fully determined by (s, a), so `p(s', r | s, a) = 1` for a single (s', r) pair. The equation simplifies to:

v<sub>pi</sub>(s) = sum<sub>a</sub> pi(a | s) [ r(s, a) + gamma v<sub>pi</sub>(s') ]

where s' is the unique next state after taking action a in state s.

## 6. Bellman Backup

A Bellman backup is one update of a state value using the Bellman expectation equation:

V<sub>new</sub>(s) = sum<sub>a</sub> pi(a | s) [ r(s, a) + gamma V(s') ]

It expresses the new value of state s as a weighted sum over all actions, where each term combines:
- the immediate reward r(s, a)
- the discounted value of the resulting next state gamma V(s')

## 7. Policy Evaluation

Policy evaluation (also called the prediction problem) is the task of computing v<sub>pi</sub>(s) for all states given a fixed policy pi.

The policy is not changed during this step. The only goal is to estimate how good each state is under the given policy.

This is the first step toward policy improvement, which comes in Week 6.

## 8. Iterative Policy Evaluation

Because the Bellman equation is a system of linear equations (one per state), it can be solved by repeatedly applying Bellman backups until convergence.

**Algorithm:**

1. Initialize V(s) = 0 for all valid non-obstacle states.
2. For each sweep through all states:
   - For each non-terminal state s:
     - V<sub>new</sub>(s) = sum<sub>a</sub> pi(a | s) [ r(s, a) + gamma V(s') ]
   - delta = max<sub>s</sub> | V<sub>new</sub>(s) - V(s) |
   - Replace V with V<sub>new</sub> (synchronous update)
3. Stop when delta < theta (convergence threshold).

The use of V<sub>new</sub> computed from the old V is called a synchronous update. All states are updated using the same old value table in each sweep.

**Convergence:** For episodic tasks with finite state spaces and gamma <= 1, iterative policy evaluation converges to the true v<sub>pi</sub>.

## 9. Terminal States

Terminal state values are always 0 by definition:

v<sub>pi</sub>(terminal) = 0

There is no future return after reaching the terminal state.
The goal reward is received when transitioning *into* the terminal state, not while being in it.

So the value of a state adjacent to the goal reflects the expected future return including that goal reward.

## 10. Obstacles

Obstacle states are not valid states and are not included in the policy evaluation sweep.
When a policy tries to move into an obstacle, the environment returns the current state unchanged along with the invalid move reward.

The invalid move penalty contributes negatively to the value of states that often attempt invalid moves.

## 11. Week 5 Experiment

Using the 5 x 5 Gridworld from Week 4:
- gamma = 0.9
- theta = 1e-4
- max_iterations = 1000

Three fixed policies were evaluated:
- `RandomPolicy`: uniform over all four actions
- `GoalDirectedPolicy`: prefers actions that reduce Manhattan distance to the goal
- `BadPolicy`: 70% probability of choosing "up" or "left", away from the goal

Results:
- GoalDirectedPolicy produces the highest state values because it reaches the goal efficiently.
- RandomPolicy produces lower values because random actions often result in invalid move penalties and many unnecessary steps.
- BadPolicy produces the lowest values because it systematically moves away from the goal.
- In this experiment, the convergence plot shows delta shrinking toward zero across sweeps.
- States closer to the goal under a good policy have values closer to the goal reward.

## 12. Main Takeaways

- v<sub>pi</sub>(s) evaluates each state under a fixed policy.
- q<sub>pi</sub>(s, a) evaluates each action in each state under a fixed policy.
- The Bellman expectation equation connects current state value to immediate reward and discounted next-state value.
- Iterative policy evaluation computes v<sub>pi</sub> by repeatedly applying Bellman backups until convergence.
- Policy evaluation does not change the policy. It only answers: given this policy, how good is each state?
- This is the foundation for policy improvement and policy iteration in Week 6.
