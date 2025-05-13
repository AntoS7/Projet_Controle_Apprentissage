import os
import csv
import numpy as np
import cvxpy as cp
from TrafficEnv import TrafficEnv
from Simulation import safe_render, LOG_DIR, max_steps_per_episode

class MPCController:
    def __init__(self, H=12, lam=1.0, log_dir=LOG_DIR, max_steps=max_steps_per_episode):
        self.H = H
        self.lam = lam
        self.log_dir = log_dir
        self.max_steps = max_steps

    def solve(self, q0, arrivals, S):
        n = q0.shape[0]
        U = cp.Variable((self.H, n))
        Q = [None] * (self.H + 1)
        Q[0] = q0
        constraints = []
        cost = 0
        for k in range(self.H):
            q_next = Q[k] + arrivals[k] - S @ U[k]
            Q[k + 1] = q_next
            constraints += [
                q_next >= 0,
                U[k] >= 0,
                U[k] <= 1
            ]
            cost += cp.norm1(q_next)
            if k > 0:
                cost += self.lam * cp.norm1(U[k] - U[k - 1])
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)
        return U.value[0]

    def run_simulation(self, sumo_config, log_filename):
        env = TrafficEnv(sumo_config)
        safe_render(env)

        state = env.reset()
        done = False
        step = 0
        q = np.array(state)

        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, log_filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'state', 'action', 'reward'])

            while not done and step < self.max_steps:
                predicted_arrivals = np.random.poisson(lam=2, size=(self.H, len(state)))
                service_matrix = np.eye(len(state))
                u0 = self.solve(q, predicted_arrivals, service_matrix)
                action = int(np.argmax(u0))
                next_state, reward, done, _ = env.step(action)
                writer.writerow([step, list(q), action, reward])
                q = np.array(next_state)
                step += 1

        env.close()
