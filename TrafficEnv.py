import traci

class TrafficEnv:
    def __init__(self, sumo_cfg, max_steps=180):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps

    def reset(self):
        # Lancer SUMO (headless ou GUI)
        traci.start(["sumo", "-c", self.sumo_cfg])
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        # Appliquer l'action : changer la phase du feu
        self._apply_action(action)
        # Faire avancer la simu d'un pas
        traci.simulationStep()
        self.step_count += 1
        # Lire l'état suivant et la reward
        state = self._get_state()
        reward = self._compute_reward(state)
        done = (self.step_count >= self.max_steps)
        return state, reward, done, {}

    def _get_state(self):
        # Exemple : compter les véhicules en file sur chaque voie
        q_N = traci.lane.getLastStepVehicleNumber("lane_N")
        q_E = traci.lane.getLastStepVehicleNumber("lane_E")
        q_S = traci.lane.getLastStepVehicleNumber("lane_S")
        q_W = traci.lane.getLastStepVehicleNumber("lane_W")
        t_phase = traci.trafficlight.getPhaseDuration("TL_ID")
        return (q_N, q_E, q_S, q_W, t_phase)

    def _apply_action(self, action):
        # Mappe action → phase ID puis envoi TraCI
        phase_id = action  
        traci.trafficlight.setPhase("TL_ID", phase_id)

    def close(self):
        traci.close()

    def render(self):
        """
        Lance la simulation SUMO en mode graphique (GUI).
        """
        # Si une session TraCI est déjà ouverte, on la ferme d'abord
        try:
            traci.close()
        except Exception:
            pass
        # Démarrage de SUMO en mode GUI
        traci.start(["sumo-gui", "-c", self.sumo_cfg])