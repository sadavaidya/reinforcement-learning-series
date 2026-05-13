# reinforcement-learning-from-scratch

This repository is a week-by-week reinforcement learning implementation series based mainly on Sutton and Barto's *Reinforcement Learning: An Introduction*. The goal is to build core RL ideas from scratch in clean, readable Python while documenting the reasoning and results in a portfolio-friendly way.

The project motivation is straightforward: reinforcement learning concepts become much clearer when the underlying environments, agents, updates, and evaluation loops are implemented directly rather than treated as black boxes.

Current roadmap at a high level:
- Week 1: epsilon-greedy action selection in the 10-armed bandit problem
- Future weeks: additional chapters and algorithms will be added incrementally

Current repository structure:

```text
reinforcement-learning-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── notes/
│   └── week_01_epsilon_greedy_bandits.md
├── notebooks/
│   └── week_01_epsilon_greedy_bandits.ipynb
├── src/
│   ├── bandits/
│   │   ├── agents.py
│   │   ├── environment.py
│   │   └── experiments.py
│   └── utils/
│       └── plotting.py
├── results/
│   └── week_01/
│       ├── average_reward.png
│       └── optimal_action_percentage.png
└── tests/
    └── test_bandits.py
```

## Week 1

Week 1 implements the 10-armed bandit testbed and compares `epsilon = 0`, `epsilon = 0.01`, and `epsilon = 0.1` using sample-average action-value estimation. Performance is evaluated with average reward and percentage optimal action.

Key insight: a pure greedy strategy can get stuck after early lucky rewards, while epsilon-greedy improves long-term learning by continuing to explore.

### Week 1 Results

Average reward:

![Average Reward](results/week_01/average_reward.png)

Optimal action percentage:

![Optimal Action Percentage](results/week_01/optimal_action_percentage.png)

Reference:
- Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*.
