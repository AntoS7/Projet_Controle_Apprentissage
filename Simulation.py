import csv
import random
from GymTrafficEnv import GymTrafficEnv
from Sarsa import SarsaAgent
import os
from datetime import datetime
from Mpc import MPCController
from gym.wrappers import Monitor


def make_env(sumo_config):
    """Crée et renvoie un environnement GymTrafficEnv."""
    return GymTrafficEnv(sumo_config)

def safe_render(env):
    """Lance la simulation SUMO en mode graphique (GUI) si possible."""
    try:
        env.render()
    except Exception as e:
        print(f"Erreur lors du rendu : {e}")
        print("Lancement de SUMO en mode GUI...")
        env.close()
        env.render(mode='human')
        print("Rendu SUMO en mode GUI terminé.")

class CSVLogger:
    """Helper pour enregistrer des lignes dans un fichier CSV."""
    def __init__(self, filepath, headers):
        self.csvfile = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(headers)

    def log(self, row):
        self.writer.writerow(row)

    def close(self):
        self.csvfile.close()

# Hyperparamètres SARSA
alpha = 0.1        # Taux d'apprentissage
gamma = 0.99       # Facteur d'actualisation
epsilon = 0.1      # Taux d'exploration (epsilon-greedy)
num_episodes = 1000
max_steps_per_episode = 180

# Répertoire de logs et horodatage
LOG_DIR = "logs"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)

# Constantes pour les expériences
SIM_EPISODES    = num_episodes
GS_EPISODES     = 500
EVAL_EPISODES   = 30
ALPHAS          = [0.05, 0.1, 0.2]
GAMMAS          = [0.9, 0.95, 0.99]
EPSILON_PAIRS   = [(1.0, 0.1), (0.9, 0.05)]  # (ε_start, ε_end)

def train_and_log(env, agent, log_filename):
    """Entraîne l'agent et génère un CSV de performance."""
    safe_render(env)
    state = env.reset()
    done = False
    step = 0
    logger = CSVLogger(os.path.join(LOG_DIR, log_filename),
                       ['step', 'state', 'action', 'reward'])
    while not done and step < max_steps_per_episode:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        logger.log([step, state, action, reward])
        state = next_state
        step += 1
    logger.close()

def run_baseline(env, actions, log_filename):
    """Exécute la politique aléatoire et logge dans un CSV."""
    safe_render(env)
    state = env.reset()
    done = False
    step = 0
    logger = CSVLogger(os.path.join(LOG_DIR, log_filename),
                       ['step', 'state', 'action', 'reward'])
    while not done and step < max_steps_per_episode:
        action = random.choice(actions)
        next_state, reward, done, _ = env.step(action)
        logger.log([step, state, action, reward])
        state = next_state
        step += 1
    logger.close()

def grid_search(sumo_config, actions):
    """Recherche la meilleure configuration d'hyperparamètres SARSA."""
    best_config = None
    best_score  = float('inf')
    for alpha_val in ALPHAS:
        for gamma_val in GAMMAS:
            for eps_start, eps_end in EPSILON_PAIRS:
                env_h = make_env(sumo_config)
                agent_h = SarsaAgent(actions,
                                     alpha=alpha_val,
                                     gamma=gamma_val,
                                     epsilon_start=eps_start,
                                     epsilon_end=eps_end,
                                     epsilon_decay_steps=3000)
                agent_h.train(env_h, num_episodes=GS_EPISODES, max_steps_per_episode=max_steps_per_episode)
                total_wait = 0
                for _ in range(EVAL_EPISODES):
                    state = env_h.reset()
                    done = False
                    while not done:
                        action = agent_h.select_action(state)
                        next_state, reward, done, _ = env_h.step(action)
                        total_wait += -reward
                        state = next_state
                avg_wait = total_wait / EVAL_EPISODES
                print(f"α={alpha_val}, γ={gamma_val}, ε0={eps_start}->{eps_end} → avg_wait={avg_wait:.2f}s")
                if avg_wait < best_score:
                    best_score  = avg_wait
                    best_config = (alpha_val, gamma_val, eps_start, eps_end)
                env_h.close()
    return best_config, best_score

def main():
    #TODO : Remplacer par le chemin vers votre fichier de configuration SUMO
    sumo_config = "path/to/sumo_config.sumocfg"
    
    #TODO : Vérifiez que les actions correspondent à votre environnement
    # Exemple : 4 actions pour 4 phases de feu
    actions = [0, 1, 2, 3]

    # 1. Entraînement SARSA + log
    env = Monitor(
        make_env(sumo_config),
        directory=os.path.join(LOG_DIR, f"monitor_simulation_{TIMESTAMP}"),
        video_callable=lambda ep: True,
        force=True
    )
    agent = SarsaAgent(actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent.train(env, num_episodes=SIM_EPISODES, max_steps_per_episode=max_steps_per_episode)
    train_and_log(env, agent, f"simulation_{TIMESTAMP}.csv")
    env.close()

    # 2. Baseline aléatoire + log
    env_baseline = Monitor(
        make_env(sumo_config),
        directory=os.path.join(LOG_DIR, f"monitor_baseline_{TIMESTAMP}"),
        video_callable=lambda ep: True,
        force=True
    )
    run_baseline(env_baseline, actions, f"baseline_{TIMESTAMP}.csv")
    env_baseline.close()

    # 3. Contrôle MPC via la classe MPCController
    mpc = MPCController(H=12, lam=1.0, log_dir=LOG_DIR, max_steps=max_steps_per_episode)
    mpc.run_simulation(sumo_config, f"mpc_{TIMESTAMP}.csv")

    # 4. Grid-search hyperparamètres
    best_conf, best_score = grid_search(sumo_config, actions)
    print("Meilleure config :", best_conf, "avec avg_wait =", best_score)


if __name__ == "__main__":
    main()