# Week 4 - Finite Markov Decision Processes

## 1. Context

Weeks 1-3 focused on bandit problems, where the structure is simple: choose an action and observe a reward. That setting is useful for learning exploration ideas, but it leaves out a core part of reinforcement learning: state.

In a Markov Decision Process (MDP), the agent is no longer acting in a stateless world. The action affects both the immediate reward and the next state, which then changes the decisions that come after it.

This makes Week 4 the transition from:
- bandits: action -> reward
- MDPs: state -> action -> reward -> next state

## 2. Agent-Environment Interface

The standard reinforcement learning interaction looks like this:

S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, S<sub>1</sub>, A<sub>1</sub>, R<sub>2</sub>, S<sub>2</sub>, ...

At time `t`:
- the agent observes state S<sub>t</sub>
- selects action A<sub>t</sub>
- receives reward R<sub>t+1</sub>
- transitions to the next state S<sub>t+1</sub>

The reward is written as R<sub>t+1</sub> because it is received after the action is taken.

## 3. Finite MDP

A finite MDP has:
- a finite set of states
- a finite set of actions
- rewards
- transition dynamics
- a discount factor

The transition dynamics are written as:

`p(s', r | s, a)`

This means: the probability of moving to next state `s'` and receiving reward `r`, given that the agent was in state `s` and took action `a`.

## 4. Markov Property

The defining idea of an MDP is the Markov property:

the future depends only on the current state and action, not on the full history.

So if the current state is well-defined, it already contains the information needed for predicting what comes next.

## 5. Rewards and Goals

The reward signal defines what the agent should achieve. It should describe the goal of behavior, not hard-code the exact method for reaching it.

In this week's Gridworld:
- each normal step gives reward `-1`
- reaching the goal gives reward `+10`
- invalid moves give reward `-5`

That reward design pushes the agent toward shorter successful paths and discourages wasting steps or repeatedly walking into walls.

## 6. Returns

The return is the cumulative reward collected from a time step onward.

For an episodic task, the undiscounted return is:

G<sub>t</sub> = R<sub>t+1</sub> + R<sub>t+2</sub> + ... + R<sub>T</sub>

This is the quantity that reinforcement learning ultimately tries to maximize, not just the immediate next reward.

## 7. Discounted Return

When future rewards should matter less than immediate rewards, discounting is introduced:

G<sub>t</sub> = R<sub>t+1</sub> + gamma R<sub>t+2</sub> + gamma<sup>2</sup> R<sub>t+3</sub> + ...

Here:
- gamma = 0 means only immediate reward matters
- gamma close to 1 means future rewards remain important

Discounting is useful both conceptually and computationally. It expresses how much the agent values delayed outcomes.

## 8. Episodic vs Continuing Tasks

An episodic task has a natural ending point. Gridworld is episodic because the episode ends when the agent reaches the goal or when the run is truncated by a maximum step limit.

A continuing task has no natural terminal state and keeps going indefinitely.

This distinction matters because the meaning of return changes across the two settings.

## 9. Policies

A policy tells the agent how to act in each state.

It is written as:

`pi(a | s)`

This means the probability of choosing action `a` in state `s`.

In a bandit problem, one action can be globally best because there is no state. In an MDP, the quality of an action depends on where the agent currently is.

## 10. Basic Value Function Preview

Later chapters will formalize how good states and actions are under a policy.

Two standard quantities are:
- v<sub>pi</sub>(s): expected return from state `s` under policy `pi`
- q<sub>pi</sub>(s, a): expected return from taking action `a` in state `s`, then following `pi`

These are only a preview for now. Week 4 focuses on states, transitions, returns, and policy behavior rather than full value-function machinery.

## 11. Week 4 Experiment

This week's implementation uses a `5 x 5` Gridworld MDP with:
- a start state
- a goal state
- obstacles
- deterministic transitions
- fixed rewards for normal steps, invalid moves, and goal reaching

Three fixed policies are compared:
- `RandomPolicy`
- `GoalDirectedPolicy`
- `BadPolicy`

The measured metrics are:
- average return
- average episode length
- success rate
- state visitation frequency

This is enough to show a central MDP idea: different policies can produce very different long-term outcomes even when they share the same environment.

## 12. Main Takeaways

- Bandits only model action and immediate reward.
- MDPs introduce states and transitions.
- Actions matter because they affect both reward and future states.
- Return measures long-term performance, not just the next reward.
- Policies in MDPs must respond to state.
- Week 4 sets up the project for value functions and Bellman equations in later weeks.
