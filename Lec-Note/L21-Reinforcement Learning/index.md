# Reinforcement Learning

So far, we have discussed *supervised learning*, and a little bit *unsupervised learning* (in UCB-data100 :wink:)

## What is Reinforcement Learning
> at time $t$, env $\rightarrow^{state}$ agent $\rightarrow^{action}$ env $\rightarrow^{reward}$ agent, then env changed, agent learned, then repeated.

- state can be partial -> noisy
- reward can be delayed, implicit and sparse -> noisy
- AND Nondifferentiable :astonished:
- Nonstationary environment, change over time :sunglasses:

Generative Adversarial Networks (GANs) somehow is a part of Reinforcement Learning. 

## MDP (Markov Decision Process)

> formalization of reinforcement learning problem: *tuple* $(S, A, P, R, \gamma)$
> Agent wanna find a *policy* $\pi$ that maximizes the expected reward over time

- $S$ : set of states
- $A$ : set of actions
- $P(s'|s,a)$ : state transition probability
- $R(s,a,s')$ : reward function
- $\gamma$ : discount factor(tradeoff between immediate and future reward)

How good is a policy $\pi$?

- Value function $V_{\pi}(s)$ : expected reward starting from state $s$ under policy $\pi$

$$
V^{\pi}(s) = \mathbb{E}[\sum_{t \geq 0}\gamma^tr_t | s_0=s, \pi]
$$

- Q function $Q_{\pi}(s,a)$ : expected reward starting from state $s$ and taking action $a$ under policy $\pi$

$$
Q^{\pi}(s,a) = \mathbb{E}[\sum_{t \geq 0}\gamma^tr_t | s_0=s, a_0=a, \pi]
$$

$$
Q^*(s,a) = \max_{\pi} \mathbb{E}[\sum_{t \geq 0}\gamma^tr_t | s_0=s, a_0=a, \pi]
$$

易推出

$$
\pi^*(s) = argmax_{a'}Q(s,a')
$$


### bellman equation
Q-star satisfies the Bellman equation:

$$
Q^*(s,a) = \mathbb{E}[R(s,a) + \gamma \max_{a'}Q^*(s',a')]
$$
- $R(s,a)$ : immediate reward
- $\gamma \max_{a'}Q^*(s',a')$ : expected future reward

> if a function $V$ satisfies the Bellman equation, then it must be a Q-star
> if using Bellman equation to update from random to infinite, we will converge to $Q^*$
> We can use *DL* to learn to approximate Q-star: *DQL(Deep Q-learning)*

## Q-learning

Q-learning is a simple and effective algorithm for learning Q-functions.

- Train a NN $Q_\theta(s,a)$ to estimate future rewards for every (state, action) pair.
- move forward to Bellman equation (which is the loss function for training)
- Some problems are easier to learn a mapping from states to actions, others are harder.

## Policy Gradient
> train a NN $\pi_\theta(a|s)$ that takes in a state and outputs a probability distribution over actions......

> Obj fn, expected reward under policy $\pi_\theta$: $J(\theta) = \mathbb{E}_{r \sim p_\theta}[\sum_{t\geq 0}\gamma^tr_t ]$
> then we can find the optimal policy by maximizing $\theta^* = argmax_{\theta} J(\theta)$, using *gradient ascent*.

can not find $\partial J / \partial \theta$ directly 

> General formulation: $J(\theta) = \mathbb{E}_{x \sim p_\theta}[f(x)]$, where $f(x)$ is a rewarding fn, and $x$ is trajectory of states, rewards, actions under policy $p_\theta$.



$$
\begin{align}
\frac{\partial J}{\partial \theta} = \frac{\partial}{\partial \theta} \mathbb{E}_{x \sim p_\theta} [f(x)] &= \frac{\partial}{\partial \theta} \int_{X} p_\theta(x) f(x) dx = \int_{X} f(x) \frac{\partial}{\partial \theta} p_\theta(x) dx \\
\because \frac{\partial}{\partial \theta} logp_\theta(x) &= \frac{1}{p_\theta(x)} \frac{\partial}{\partial \theta} p_\theta(x) \\
\therefore \frac{\partial J}{\partial \theta} &= \int_{X} f(x) p_\theta(x) \frac{\partial}{\partial \theta} logp_\theta(x) dx\\
&= \mathbb{E}_{x \sim p_\theta}[f(x) \frac{\partial}{\partial \theta} logp_\theta(x)]\\ 
\end{align}
$$

then we can sampling to estimate the expectation (sth like Monte Carlo :thinking:)

let $x = (s_0, a_0, s_1, a_1, ..., s_t, a_t, r_t)$, random: $x \sim p_\theta(x)$

then write out the probability of $x$ under policy $\pi_\theta$ (with Markov property):

$$
p_\theta(x) = \Pi_{t\geq 0} P(s_{t+1}|s_t)\pi_\theta(a_t|s_t) \Rightarrow logp_\theta(x) = \sum_{t\geq 0} logP(s_{t+1}|s_t) + log\pi_\theta(a_t|s_t)
$$

where $P(s_{t+1}|s_t)$ is the ENV's transition probability, can not compute!
$\pi_\theta(a_t|s_t)$ is the NN's output (which is action probability), can compute!

note that $P$ is not a function of $\theta$, so we can write the gradient as:

$$
\frac{\partial}{\partial \theta} logp_\theta(x) = \sum_{t\geq 0} \frac{\partial}{\partial \theta} log\pi_\theta(a_t|s_t)
$$

put it back into (4)

$$
\begin{align}
\frac{\partial J}{\partial \theta} 
&= \mathbb{E}_{x \sim p_\theta}[f(x) \frac{\partial}{\partial \theta} logp_\theta(x)]\\ 
&= \mathbb{E}_{x \sim p_\theta}[f(x) \sum_{t\geq 0} \frac{\partial}{\partial \theta} log\pi_\theta(a_t|s_t)]\\ 
\end{align}
$$
- $x\sim p_\theta$: can be sampled from the policy, because it is a sequence of states, actions, rewards
- $f(x)$: rewarding function, can be observed and computed
- $\frac{\partial}{\partial \theta}log\pi_\theta(a_t|s_t)$: NN's output gradient w.r.t. model weights $\theta$


so we come up to a algorithm:
> 1. Initialize $\theta$
> 2. Collect trajectories $x_t$ and get rewards $f(x)$ by running policy $\pi_\theta$ in the environment
> 3. Compute the gradient of $J(\theta)$ w.r.t. $\theta$ 
> 4. Update $\theta$ using *gradient ascent*
> 5. Repeat 2-4 until convergence

Intuition:

- When rewards $f(x)$ high, increase the probability of actions we took, vice versa. *趋利避害*
- need SO MANY data to trial and error


what is the future from 2019?

- Actor-Critic
- Model-based RL
- Imitation Learning
- Inverse RL
- Adversarial RL, which is a part of GANs


