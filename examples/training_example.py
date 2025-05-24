"""
Training example for traffic light control using reinforcement learning.

This script demonstrates how to train an RL agent on the traffic environment.
Note: This is a simplified example. For production use, consider using 
stable-baselines3 or other RL libraries.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.traffic_env import TrafficEnvironment
from network.network_generator import generate_sumo_network
from network.traffic_generator import generate_traffic_flows
from utils.sumo_utils import create_sumo_config

# Try to import SARSA agent (might not exist yet)
try:
    from environment.sarsa_agent import SarsaAgent
    SARSA_AVAILABLE = True
except ImportError:
    SARSA_AVAILABLE = False
    print("SARSA agent not available. Using Q-learning only.")


class SimpleQAgent:
    """
    Simple Q-learning agent for traffic light control.
    
    This is a basic implementation for demonstration purposes.
    For better performance, use more sophisticated algorithms.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        """
        Initialize Q-learning agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for Q-updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.discount_factor = 0.95
        
        # Initialize Q-table (simplified - discretize continuous states)
        self.q_table = {}
        
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for Q-table."""
        # Simple discretization - bin values into ranges
        discretized = []
        for i, value in enumerate(state):
            if i < 12:  # Queue lengths (0-4 for each intersection)
                bins = [0, 5, 10, 20, 50, 100]
                discretized.append(min(len(bins)-1, max(0, np.digitize(value, bins))))
            elif i < 24:  # Phase encodings (already 0 or 1)
                discretized.append(int(value))
            else:  # Time values
                bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
                discretized.append(min(len(bins)-1, max(0, np.digitize(value, bins))))
        
        return tuple(discretized)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action using epsilon-greedy policy."""
        discrete_state = self._discretize_state(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action (exploration)
            return np.array([random.randint(0, 3) for _ in range(3)])
        else:
            # Greedy action (exploitation)
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.zeros((4, 4, 4))  # 3 intersections, 4 actions each
            
            q_values = self.q_table[discrete_state]
            # Find best action combination
            best_action = np.unravel_index(np.argmax(q_values), q_values.shape)
            return np.array(best_action)
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-values."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Initialize Q-values if not seen before
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros((4, 4, 4))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros((4, 4, 4))
        
        # Current Q-value
        current_q = self.q_table[discrete_state][action[0], action[1], action[2]]
        
        # Next max Q-value
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[discrete_next_state])
            target = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[discrete_state][action[0], action[1], action[2]] += \
            self.learning_rate * (target - current_q)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def setup_training_environment():
    """Setup environment for training."""
    print("Setting up training environment...")
    
    # Generate shorter simulation for training
    network_file = generate_sumo_network("config")
    routes_file = generate_traffic_flows("config", duration=900)  # 15 minutes
    config_file = create_sumo_config(
        network_file="network.net.xml",
        routes_file="routes.rou.xml",
        output_file="config/simulation.sumocfg",
        end=900
    )
    
    return config_file


def train_agent(episodes: int = 100, max_steps: int = 500):
    """Train Q-learning agent."""
    print(f"Training agent for {episodes} episodes...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=max_steps
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    agent = SimpleQAgent(state_size=state_size, action_size=64)  # 4^3 action combinations
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_queues = []
    
    try:
        for episode in range(episodes):
            print(f"\\nEpisode {episode + 1}/{episodes}")
            
            # Reset environment
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            
            while True:
                # Get action from agent
                action = agent.get_action(state)
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Update agent
                agent.update(state, action, reward, next_state, terminated or truncated)
                
                # Update metrics
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Store episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            avg_queues.append(total_queues / steps if steps > 0 else 0)
            
            # Print episode summary
            print(f"  Steps: {steps}, Reward: {total_reward:.2f}, "
                  f"Avg Queues: {avg_queues[-1]:.1f}, Epsilon: {agent.epsilon:.3f}")
            
            # Print running average every 10 episodes
            if (episode + 1) % 10 == 0:
                recent_rewards = episode_rewards[-10:]
                recent_queues = avg_queues[-10:]
                print(f"  Last 10 episodes - Avg Reward: {np.mean(recent_rewards):.2f}, "
                      f"Avg Queues: {np.mean(recent_queues):.1f}")
    
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    return agent, episode_rewards, episode_lengths, avg_queues


def evaluate_agent(agent: SimpleQAgent, episodes: int = 10):
    """Evaluate trained agent."""
    print(f"\\nEvaluating agent over {episodes} episodes...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=500
    )
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    evaluation_rewards = []
    evaluation_queues = []
    
    try:
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            
            while True:
                action = agent.get_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                
                if terminated or truncated:
                    break
            
            avg_queue = total_queues / steps if steps > 0 else 0
            evaluation_rewards.append(total_reward)
            evaluation_queues.append(avg_queue)
            
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
                  f"Avg Queues={avg_queue:.1f}")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        env.close()
        # Restore original epsilon
        agent.epsilon = original_epsilon
    
    if evaluation_rewards:
        print(f"\\nEvaluation Results:")
        print(f"  Average Reward: {np.mean(evaluation_rewards):.2f} ± {np.std(evaluation_rewards):.2f}")
        print(f"  Average Queues: {np.mean(evaluation_queues):.1f} ± {np.std(evaluation_queues):.1f}")
    
    return evaluation_rewards, evaluation_queues


def plot_training_results(episode_rewards, avg_queues):
    """Plot training results."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(episode_rewards)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot average queues
        ax2.plot(avg_queues)
        ax2.set_title('Average Queue Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Queues')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        
        print("Training plots saved as 'training_results.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")


def train_single_agent(env, agent, episodes, agent_name):
    """Train a single agent and return rewards."""
    rewards = []
    
    try:
        for episode in range(episodes):
            # Reset environment
            state, info = env.reset()
            total_reward = 0
            
            if agent_name == "SARSA":
                # SARSA training loop
                action = agent.get_action(state, training=True)
                
                while True:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_action = agent.get_action(next_state, training=True)
                    agent.update(reward, next_state, next_action, terminated or truncated)
                    
                    total_reward += reward
                    state = next_state
                    action = next_action
                    
                    if terminated or truncated:
                        agent.end_episode(reward)
                        break
            else:
                # Q-learning training loop
                while True:
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    agent.update(state, action, reward, next_state, terminated or truncated)
                    
                    total_reward += reward
                    state = next_state
                    
                    if terminated or truncated:
                        break
            
            rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"  {agent_name} Episode {episode + 1}: Avg Reward = {avg_reward:.2f}")
                
    except Exception as e:
        print(f"Error training {agent_name}: {e}")
    
    return rewards


def compare_q_learning_and_sarsa():
    """Compare Q-learning and SARSA agents."""
    if not SARSA_AVAILABLE:
        print("SARSA agent not available for comparison")
        return
    
    print("\\n" + "="*60)
    print("COMPARING Q-LEARNING AND SARSA AGENTS")
    print("="*60)
    
    # Setup environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=300
    )
    
    state_size = env.observation_space.shape[0]
    episodes = 30
    
    # Train Q-learning agent
    print("\\n--- Training Q-learning Agent ---")
    q_agent = SimpleQAgent(state_size=state_size, action_size=64)
    q_rewards = train_single_agent(env, q_agent, episodes, "Q-learning")
    
    # Train SARSA agent
    print("\\n--- Training SARSA Agent ---")
    sarsa_agent = SarsaAgent(state_size=state_size, action_size=4)
    sarsa_rewards = train_single_agent(env, sarsa_agent, episodes, "SARSA")
    
    env.close()
    
    # Compare results
    print(f"\\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if q_rewards and sarsa_rewards:
        q_avg = np.mean(q_rewards[-10:])  # Last 10 episodes
        sarsa_avg = np.mean(sarsa_rewards[-10:])
        
        print(f"Q-learning average reward (last 10): {q_avg:.2f}")
        print(f"SARSA average reward (last 10): {sarsa_avg:.2f}")
        
        if sarsa_avg > q_avg:
            print("SARSA performed better!")
        elif q_avg > sarsa_avg:
            print("Q-learning performed better!")
        else:
            print("Both agents performed similarly.")


def main():
    """Main training function."""
    print("Traffic Light Control - Training Example")
    print("=" * 50)
    
    # Setup environment
    try:
        setup_training_environment()
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return
    
    # Train agent
    print("\\nStarting training...")
    agent, rewards, lengths, queues = train_agent(episodes=50, max_steps=300)
    
    if rewards:
        # Plot results
        plot_training_results(rewards, queues)
        
        # Evaluate agent
        evaluate_agent(agent, episodes=5)
        
        print(f"\\nTraining completed!")
        print(f"Q-table size: {len(agent.q_table)} states")
        print(f"Final epsilon: {agent.epsilon:.3f}")
    else:
        print("No training data collected")


if __name__ == "__main__":
    main()
