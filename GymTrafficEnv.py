import gym
from gym import spaces
import numpy as np
from TrafficEnv import TrafficEnv

class GymTrafficEnv(gym.Env):
    """
    Wrapper OpenAI Gym autour de votre TrafficEnv (SUMO).
    Permet d'utiliser toute la chaîne d'outils RL compatible Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_config):
        super().__init__()
        # Instancie votre environnement SUMO existant
        self.env = TrafficEnv(sumo_config)

        # Définition des espaces : à adapter selon votre TrafficEnv
        n_actions = len(self.env.reset())  # ici 4 par défaut, mais adaptez si besoin
        n_obs     = len(self.env.reset())
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(n_obs,),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return (
            np.array(next_state, dtype=np.float32),
            reward,
            done,
            info
        )

    def render(self, mode='human'):
        try:
            self.env.render()
        except Exception:
            pass

    def close(self):
        self.env.close()