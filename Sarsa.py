import random
from collections import defaultdict

class SarsaAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=3000):
        """
        actions : liste ou iterable des actions possibles
        alpha  : taux d'apprentissage
        gamma  : facteur d'actualisation
        epsilon_* : paramètres de l'exploration ε-greedy
        """
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma

        # Q-table par défaut à 0
        self.Q = defaultdict(float)  # clé = (state, action)

        # paramètres ε-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

    def choose_action(self, state):
        """ε-greedy : choisir random avec prob ε, sinon argmax_a Q[state,a]"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Sinon choix du meilleur
        q_vals = [self.Q[(state, a)] for a in self.actions]
        max_q = max(q_vals)
        # en cas d'égalité on randomise entre les meilleurs
        best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update_epsilon(self):
        """Décroissance linéaire d’ε jusqu'à ε_end"""
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

    def learn(self, state, action, reward, next_state, next_action):
        """
        Mise à jour SARSA :
        Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
        """
        sa = (state, action)
        sa_next = (next_state, next_action)
        td_target = reward + self.gamma * self.Q[sa_next]
        td_error  = td_target - self.Q[sa]
        self.Q[sa] += self.alpha * td_error

    def train(self, env, num_episodes=1000, max_steps_per_episode=180):
        """
        Boucle d'entraînement complète
        env : objet avec reset() et step(action)
        """
        for ep in range(num_episodes):
            state = env.reset()
            action = self.choose_action(state)
            for t in range(max_steps_per_episode):
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(next_state)

                # Mise à jour SARSA
                self.learn(state, action, reward, next_state, next_action)

                # avancer
                state, action = next_state, next_action

                # atualização ε
                self.update_epsilon()

                if done:
                    break