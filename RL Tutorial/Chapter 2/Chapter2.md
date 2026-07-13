<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" defer></script>

## Chapter 2 - Mathematical Foundations of Reinforcement Learning

### Markov Decision Processes

In RL problems, there is usually plenty of *uncertainty*, which refers to the fact that you do not know the inner workings of the environment, or how the agent's actions affect it. Such problems can be represented with Markov Decision Processes (MDPs). Additionally, the environment may be *stochastic*, rather than *deterministic*, meaning the intended action or transition may not always take place, which is befitting of the real world.

In the case of MDPs, the state is fully observable, meaning the observation and the state at a time step are the same. Partially Observable Markov Decision Processes (POMDPs), on the other hand, uphold the fact that the agent does not have access to the full state. MDPs have a property known as the *Markov property*, which refers to the requirement that states must contain all the variables necessary to make them independent of all other states. In other words, you only need the current state and action to know what happens next, and you do not need the history of states that were visited.

<p align="center">
  <img src="Figures/MarkovProperty.png" width="75%"/>
  <figcaption align="center">The Markov property (Morales, 2020).</figcaption>
</p>

The set of all states in the MDP is denoted $S^+$. There is a subset of $S^+$ called the set of starting or initial states, denoted $S^i$. To begin interacting with an MDP, we draw a state from $S^i$ from a probability distribution. There is a unique state called the absorbing or terminal state, and the set of all non-terminal states is denoted $S$. When the agent reaches a terminal state, the next state is guaranteed to be the same as the terminal state. The set of available actions, $A$, is determined by the current state, $s$, denoted as $A(s)$. The way the environment changes as a response to actions is referred to as the state-transition probabilities, or more simply, the transition function, and is denoted by $T(s, a, s')$. Notice that $T$ also describes a probability distribution $p( · | s, a)$ determining how the system will evolve in an interaction cycle from selecting action $a$ in state $s$.

<p align="center">
  <img src="Figures/TransitionFunction.png" width="75%"/>
  <figcaption align="center">The Markov property (Morales, 2020).</figcaption>
</p>

Note that either one, or both, of the environment and actions can be stochastic. A stochastic environment means even if you take a specific action given a specific state, the next state is not guaranteed to be the same every time. A stochastic action means a particular action you intend to take may not take place, but instead a different action may be performed unintentionally.

### Rewards

Reinforcement learning is evaluative, so evaluations of the agent's actions are defined by a reward function. The reward function provides a measure of how rewarding an action is, given a specific state. This helps in determining how valuable it is to take a certain action or even just being in a certain state. A *return* is the sum of rewards collected in a single episode.

<p align="center">
  <img src="Figures/RewardFunction.png" width="75%"/>
  <figcaption align="center">The reward function (Morales, 2020).</figcaption>
</p>

Tasks undertaken by the agent can be time-dependent, meaning the time or order in which actions are taken is important. This is defined as the *planning horizon*.

**Finite Horizon:** The agent knows the task will terminate in a finite number of time steps.

**Infinite Horizon:** The task may be continuing or episodic and terminate, however, the agent acts as if the task will go on forever.

In most cases, it is better for the agent to make crucial decisions early on, rather than later. To encourage the agent to act this way, we can introduce a *discount* on the reward that scales the output of the reward function down as the number of elapsed time steps increases.

<p align="center">
  <img src="Figures/DiscountEffect.png" width="75%"/>
  <figcaption align="center">The effect of discount factor and time on the value of rewards (Morales, 2020).</figcaption>
</p>

Discounts on the reward can be applied by using a *discount factor*, $0<\gamma<1$, which is exponentiated by the number of elapsed time steps and multiplied with the reward at the specific time step.

<p align="center">
  <img src="Figures/Return.png" width="75%"/>
  <figcaption align="center">Return and discount (Morales, 2020).</figcaption>
</p>

Return is a crucial concept in training agents, because it can be used to associate states and actions with long-term rewards, rather than short-term rewards.

## Sources

Morales, M. (2020). *Grokking deep reinforcement learning*. Manning Publications.