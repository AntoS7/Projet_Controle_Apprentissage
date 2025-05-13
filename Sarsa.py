import numpy as np
import random
from collections import defaultdict

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, bins=(10,10,10,10)):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # nombre d’actions à partir de l’espace Gym
        self.n_actions = env.action_space.n
        # Q-table indexée par états discrétisés (ici, tuple d’indices)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        # bornes pour discrétiser l’observation continue
        self.obs_low  = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.bins = bins

    def discretize(self, obs):
        """Convertit obs continu en tuple d’indices."""
        ratios = (obs - self.obs_low) / (self.obs_high - self.obs_low)
        indices = (ratios * (np.array(self.bins) - 1)).astype(int)
        return tuple(np.clip(indices, 0, np.array(self.bins)-1))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def train(self, num_episodes=100, max_steps=200):
        for ep in range(num_episodes):
            obs = self.env.reset()
            state = self.discretize(obs)
            action = self.choose_action(state)

            for _ in range(max_steps):
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.discretize(next_obs)
                next_action = self.choose_action(next_state)

                # mise à jour SARSA
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error  = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error

                state, action = next_state, next_action
                if done:
                    break