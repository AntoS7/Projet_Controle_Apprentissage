"""
SARSA Agent for Traffic Light Control.

This module implements a SARSA (State-Action-Reward-State-Action) agent
for controlling traffic lights in the SUMO simulation environment.
"""

import numpy as np
import random
from typing import Dict, Tuple, Optional, Literal
import pickle
import os
import math


class SarsaAgent:
    """
    SARSA agent for traffic light control.
    
    SARSA is an on-policy temporal difference learning algorithm that learns
    the action-value function Q(s,a) for the current policy.
    """
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 epsilon_decay_strategy: Literal['exponential', 'linear', 'cosine', 'step'] = 'exponential',
                 total_episodes: int = 1000,
                 step_episodes: Optional[list] = None,
                 step_values: Optional[list] = None):
        """
        Initialize SARSA agent with enhanced epsilon decay strategies.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space (per intersection)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate (for exponential decay)
            epsilon_decay_strategy: Strategy for epsilon decay ('exponential', 'linear', 'cosine', 'step')
            total_episodes: Total number of training episodes (for linear/cosine decay)
            step_episodes: Episode numbers for step decay [50, 100, 200]
            step_values: Epsilon values for step decay [0.5, 0.2, 0.05]
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Enhanced epsilon decay parameters
        self.epsilon_decay_strategy = epsilon_decay_strategy
        self.initial_epsilon = epsilon
        self.total_episodes = total_episodes
        self.current_episode = 0
        
        # Step decay parameters
        self.step_episodes = step_episodes or [int(total_episodes * 0.25), 
                                             int(total_episodes * 0.5), 
                                             int(total_episodes * 0.75)]
        self.step_values = step_values or [0.5, 0.2, 0.05]
        
        # Ensure step_values includes epsilon_min as final value
        if self.step_values[-1] > self.epsilon_min:
            self.step_values.append(self.epsilon_min)
        
        # Q-table for storing state-action values
        self.q_table = {}
        
        # Current state and action (for SARSA updates)
        self.current_state = None
        self.current_action = None
        
        # Statistics
        self.total_updates = 0
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """
        Discretize continuous state for Q-table lookup.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state as tuple
        """
        discretized = []
        
        # Process state vector (9 values per intersection, 3 intersections = 27 total)
        for i, value in enumerate(state):
            position = i % 9  # Position within intersection state
            
            if position < 4:  # Queue lengths (0-3)
                # Discretize queue lengths into bins
                bins = [0, 2, 5, 10, 20, 50]
                discrete_val = min(len(bins)-1, max(0, np.digitize(value, bins)))
            elif position < 8:  # Phase encoding (4-7) - already 0 or 1
                discrete_val = int(value)
            else:  # Time since phase change (8)
                # Discretize time into bins
                bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
                discrete_val = min(len(bins)-1, max(0, np.digitize(value, bins)))
            
            discretized.append(discrete_val)
        
        return tuple(discretized)
    
    def _get_action_key(self, actions: np.ndarray) -> tuple:
        """Convert action array to tuple for Q-table key."""
        return tuple(actions.astype(int))
    
    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action array for all intersections
        """
        discrete_state = self._discretize_state(state)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action = np.array([random.randint(0, self.action_size - 1) 
                             for _ in range(3)])  # 3 intersections
            self.exploration_count += 1
        else:
            # Exploitation: greedy action
            action = self._get_greedy_action(discrete_state)
            self.exploitation_count += 1
        
        # Store current state-action for SARSA update
        if training:
            self.current_state = discrete_state
            self.current_action = self._get_action_key(action)
        
        return action
    
    def _get_greedy_action(self, discrete_state: tuple) -> np.ndarray:
        """Get greedy action for given state."""
        if discrete_state not in self.q_table:
            # Initialize Q-values for new state
            self.q_table[discrete_state] = {}
        
        # Find best action combination
        best_action = None
        best_value = float('-inf')
        
        # If no actions tried yet, return random action
        if not self.q_table[discrete_state]:
            return np.array([random.randint(0, self.action_size - 1) 
                           for _ in range(3)])
        
        # Find action with highest Q-value
        for action_key, q_value in self.q_table[discrete_state].items():
            if q_value > best_value:
                best_value = q_value
                best_action = action_key
        
        if best_action is None:
            # Fallback to random action
            return np.array([random.randint(0, self.action_size - 1) 
                           for _ in range(3)])
        
        return np.array(best_action)
    
    def update(self, reward: float, next_state: np.ndarray, 
               next_action: np.ndarray, done: bool):
        """
        Update Q-values using SARSA algorithm.
        
        Args:
            reward: Reward received
            next_state: Next state
            next_action: Next action
            done: Whether episode is finished
        """
        if self.current_state is None or self.current_action is None:
            return
        
        # Discretize next state
        discrete_next_state = self._discretize_state(next_state)
        next_action_key = self._get_action_key(next_action)
        
        # Initialize Q-tables if needed
        if self.current_state not in self.q_table:
            self.q_table[self.current_state] = {}
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = {}
        
        # Get current Q-value
        if self.current_action not in self.q_table[self.current_state]:
            self.q_table[self.current_state][self.current_action] = 0.0
        current_q = self.q_table[self.current_state][self.current_action]
        
        # Get next Q-value (SARSA uses the actual next action)
        if done:
            next_q = 0.0
        else:
            if next_action_key not in self.q_table[discrete_next_state]:
                self.q_table[discrete_next_state][next_action_key] = 0.0
            next_q = self.q_table[discrete_next_state][next_action_key]
        
        # SARSA update: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        target = reward + self.discount_factor * next_q
        self.q_table[self.current_state][self.current_action] += \
            self.learning_rate * (target - current_q)
        
        self.total_updates += 1
        
        # Enhanced epsilon decay
        self._update_epsilon()
    
    def _update_epsilon(self):
        """Update epsilon using the selected decay strategy."""
        if self.epsilon <= self.epsilon_min:
            self.epsilon = self.epsilon_min
            return
        
        if self.epsilon_decay_strategy == 'exponential':
            # Classic exponential decay: ε = ε * decay_rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        elif self.epsilon_decay_strategy == 'linear':
            # Linear decay: ε = ε_initial - (ε_initial - ε_min) * (episode / total_episodes)
            decay_progress = min(1.0, self.current_episode / self.total_episodes)
            self.epsilon = self.initial_epsilon - (self.initial_epsilon - self.epsilon_min) * decay_progress
            
        elif self.epsilon_decay_strategy == 'cosine':
            # Cosine annealing: ε = ε_min + 0.5 * (ε_initial - ε_min) * (1 + cos(π * episode / total_episodes))
            decay_progress = min(1.0, self.current_episode / self.total_episodes)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
            self.epsilon = self.epsilon_min + (self.initial_epsilon - self.epsilon_min) * cosine_factor
            
        elif self.epsilon_decay_strategy == 'step':
            # Step decay: ε drops to specific values at predefined episodes
            current_step = 0
            for i, step_episode in enumerate(self.step_episodes):
                if self.current_episode >= step_episode:
                    current_step = i + 1
                else:
                    break
            
            if current_step < len(self.step_values):
                self.epsilon = max(self.epsilon_min, self.step_values[current_step])
            else:
                self.epsilon = self.epsilon_min
        
        # Ensure epsilon doesn't go below minimum
        self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def start_episode(self):
        """Call this at the start of each episode to track episode count for decay."""
        self.current_episode += 1
    
    def get_epsilon_schedule_info(self) -> Dict:
        """Get information about the current epsilon schedule."""
        return {
            'strategy': self.epsilon_decay_strategy,
            'current_epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_min': self.epsilon_min,
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'decay_progress': min(1.0, self.current_episode / self.total_episodes),
            'step_episodes': self.step_episodes if self.epsilon_decay_strategy == 'step' else None,
            'step_values': self.step_values if self.epsilon_decay_strategy == 'step' else None
        }
    
    def end_episode(self, final_reward: float):
        """
        Handle end of episode.
        
        Args:
            final_reward: Final reward of episode
        """
        if self.current_state is not None and self.current_action is not None:
            # Update with terminal reward
            if self.current_state not in self.q_table:
                self.q_table[self.current_state] = {}
            if self.current_action not in self.q_table[self.current_state]:
                self.q_table[self.current_state][self.current_action] = 0.0
            
            current_q = self.q_table[self.current_state][self.current_action]
            self.q_table[self.current_state][self.current_action] += \
                self.learning_rate * (final_reward - current_q)
        
        # Reset for next episode
        self.current_state = None
        self.current_action = None
    
    def save(self, filepath: str):
        """Save agent to file."""
        agent_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'current_episode': self.current_episode,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_decay_strategy': self.epsilon_decay_strategy,
                'initial_epsilon': self.initial_epsilon,
                'total_episodes': self.total_episodes,
                'step_episodes': self.step_episodes,
                'step_values': self.step_values
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"Agent saved to {filepath}")
        print(f"Epsilon decay strategy: {self.epsilon_decay_strategy}")
        print(f"Current epsilon: {self.epsilon:.4f}")
        print(f"Episode: {self.current_episode}/{self.total_episodes}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        if not os.path.exists(filepath):
            print(f"No saved agent found at {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            self.q_table = agent_data['q_table']
            self.epsilon = agent_data['epsilon']
            self.total_updates = agent_data.get('total_updates', 0)
            self.exploration_count = agent_data.get('exploration_count', 0)
            self.exploitation_count = agent_data.get('exploitation_count', 0)
            self.current_episode = agent_data.get('current_episode', 0)
            
            # Load enhanced epsilon decay parameters if available
            hyperparams = agent_data.get('hyperparameters', {})
            self.epsilon_decay_strategy = hyperparams.get('epsilon_decay_strategy', 'exponential')
            self.initial_epsilon = hyperparams.get('initial_epsilon', self.initial_epsilon)
            self.total_episodes = hyperparams.get('total_episodes', self.total_episodes)
            self.step_episodes = hyperparams.get('step_episodes', self.step_episodes)
            self.step_values = hyperparams.get('step_values', self.step_values)
            
            print(f"Agent loaded from {filepath}")
            print(f"Q-table size: {len(self.q_table)} states")
            print(f"Total updates: {self.total_updates}")
            print(f"Current epsilon: {self.epsilon:.4f}")
            print(f"Epsilon strategy: {self.epsilon_decay_strategy}")
            print(f"Episode: {self.current_episode}/{self.total_episodes}")
            return True
            
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        total_actions = self.exploration_count + self.exploitation_count
        exploration_rate = (self.exploration_count / total_actions 
                          if total_actions > 0 else 0)
        
        # Calculate Q-table statistics
        q_values = []
        for state_actions in self.q_table.values():
            q_values.extend(state_actions.values())
        
        epsilon_info = self.get_epsilon_schedule_info()
        
        return {
            'q_table_size': len(self.q_table),
            'total_state_actions': sum(len(sa) for sa in self.q_table.values()),
            'total_updates': self.total_updates,
            'current_epsilon': self.epsilon,
            'exploration_rate': exploration_rate,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'avg_q_value': np.mean(q_values) if q_values else 0,
            'max_q_value': np.max(q_values) if q_values else 0,
            'min_q_value': np.min(q_values) if q_values else 0,
            'epsilon_schedule': epsilon_info
        }
    
    def print_statistics(self):
        """Print agent statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("SARSA AGENT STATISTICS")
        print("="*60)
        print(f"Q-table size: {stats['q_table_size']} states")
        print(f"Total state-action pairs: {stats['total_state_actions']}")
        print(f"Total updates: {stats['total_updates']}")
        print(f"Actions taken - Explore: {stats['exploration_count']}, "
              f"Exploit: {stats['exploitation_count']}")
        print(f"Exploration rate: {stats['exploration_rate']:.3f}")
        
        # Epsilon schedule information
        eps_info = stats['epsilon_schedule']
        print(f"\nEPSILON SCHEDULE ({eps_info['strategy'].upper()}):")
        print(f"Current ε: {eps_info['current_epsilon']:.4f}")
        print(f"Initial ε: {eps_info['initial_epsilon']:.4f}")
        print(f"Minimum ε: {eps_info['epsilon_min']:.4f}")
        print(f"Episode: {eps_info['current_episode']}/{eps_info['total_episodes']}")
        print(f"Decay progress: {eps_info['decay_progress']:.1%}")
        
        if eps_info['strategy'] == 'step' and eps_info['step_episodes']:
            print(f"Step episodes: {eps_info['step_episodes']}")
            print(f"Step values: {[f'{v:.3f}' for v in eps_info['step_values']]}")
        
        if stats['total_state_actions'] > 0:
            print(f"\nQ-VALUES:")
            print(f"Average: {stats['avg_q_value']:.2f}")
            print(f"Maximum: {stats['max_q_value']:.2f}")
            print(f"Minimum: {stats['min_q_value']:.2f}")
        print("="*60)


class AdaptiveSarsaAgent(SarsaAgent):
    """
    Adaptive SARSA agent with dynamic learning rate and exploration.
    
    This variant adjusts learning rate and exploration based on performance.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive SARSA agent."""
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self.initial_learning_rate = self.learning_rate
        self.initial_epsilon = self.epsilon
        self.performance_window = 10
        self.recent_rewards = []
        self.best_avg_reward = float('-inf')
        self.episodes_since_improvement = 0
        self.adaptation_threshold = 5
    
    def adapt_parameters(self, episode_reward: float):
        """
        Adapt learning parameters based on recent performance.
        
        Args:
            episode_reward: Reward from completed episode
        """
        self.recent_rewards.append(episode_reward)
        
        # Keep only recent rewards
        if len(self.recent_rewards) > self.performance_window:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= self.performance_window:
            avg_reward = np.mean(self.recent_rewards)
            
            if avg_reward > self.best_avg_reward:
                # Performance improved
                self.best_avg_reward = avg_reward
                self.episodes_since_improvement = 0
            else:
                self.episodes_since_improvement += 1
                
                # If no improvement for several episodes, increase exploration
                if self.episodes_since_improvement >= self.adaptation_threshold:
                    self.epsilon = min(0.5, self.epsilon * 1.1)
                    self.learning_rate = min(0.3, self.learning_rate * 1.05)
                    self.episodes_since_improvement = 0
                    print(f"Adapted: epsilon={self.epsilon:.3f}, "
                          f"lr={self.learning_rate:.3f}")


if __name__ == "__main__":
    # Example usage
    print("SARSA Agent for Traffic Light Control")
    print("="*40)
    
    # Create agent
    agent = SarsaAgent(state_size=27, action_size=4)
    
    # Simulate some updates
    dummy_state = np.random.rand(27)
    dummy_action = np.array([0, 1, 2])
    dummy_reward = -5.0
    dummy_next_state = np.random.rand(27)
    dummy_next_action = np.array([1, 0, 1])
    
    # Get action
    action = agent.get_action(dummy_state)
    print(f"Action: {action}")
    
    # Update
    agent.update(dummy_reward, dummy_next_state, dummy_next_action, False)
    
    # Print statistics
    agent.print_statistics()
