#!/usr/bin/env python3
"""
Real SUMO Training Script for SARSA Agent

This script provides comprehensive training of SARSA agents using actual SUMO simulation.
It includes proper SUMO integration, performance monitoring, and visualization.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.traffic_env import TrafficEnvironment
from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent
from network.network_generator import generate_sumo_network
from network.traffic_generator import generate_traffic_flows


class SumoTrainingManager:
    """Manages SARSA training with SUMO simulation."""
    
    def __init__(self, config_dir="config", models_dir="models", results_dir="results"):
        self.config_dir = config_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories
        for dir_path in [self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Training metrics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'episode_lengths': [],
            'avg_queues': [],
            'phase_changes': [],
            'epsilon': [],
            'learning_rate': []
        }
    
    def setup_environment(self, use_gui=False, max_steps=600):
        """Setup SUMO environment for training."""
        print("Setting up SUMO environment...")
        
        # Generate network if needed
        sumo_cfg = os.path.join(self.config_dir, "simulation.sumocfg")
        network_file = os.path.join(self.config_dir, "network.net.xml")
        
        if not os.path.exists(network_file):
            print("Generating SUMO network...")
            generate_sumo_network(self.config_dir)
            generate_traffic_flows(self.config_dir)
        
        # Create environment
        try:
            env = TrafficEnvironment(
                sumo_cfg_file=sumo_cfg,
                use_gui=use_gui,
                max_steps=max_steps
            )
            print(f"Environment created successfully!")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            return env
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Make sure SUMO is properly installed and accessible.")
            raise
    
    def create_agent(self, agent_type="standard", **kwargs):
        """Create SARSA agent with specified parameters."""
        default_params = {
            'state_size': 27,  # 9 per intersection * 3 intersections
            'action_size': 4,  # 4 traffic phases
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 0.8,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995
        }
        
        # Update with provided parameters
        params = {**default_params, **kwargs}
        
        if agent_type == "adaptive":
            agent = AdaptiveSarsaAgent(**params)
        else:
            agent = SarsaAgent(**params)
        
        print(f"Created {agent_type} SARSA agent with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        return agent
    
    def train_agent(self, agent, env, episodes=100, save_interval=20, 
                   save_prefix="sarsa_agent", verbose=True):
        """Train SARSA agent on SUMO environment."""
        print(f"\nStarting training for {episodes} episodes...")
        print(f"Agent type: {type(agent).__name__}")
        
        start_time = time.time()
        best_avg_reward = float('-inf')
        best_episode = 0
        
        # Performance tracking window
        recent_rewards = deque(maxlen=10)
        
        try:
            for episode in range(episodes):
                episode_start = time.time()
                
                # Reset environment
                state, info = env.reset()
                total_reward = 0
                steps = 0
                total_queues = 0
                phase_changes = 0
                
                # Get initial action
                action = agent.get_action(state, training=True)
                
                if verbose and (episode + 1) % 10 == 0:
                    print(f"\nEpisode {episode + 1}/{episodes}")
                
                # Episode loop
                done = False
                while not done:
                    # Take step in environment
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    # Track metrics
                    if 'avg_queue_length' in info:
                        total_queues += info['avg_queue_length']
                    if 'phase_changes' in info:
                        phase_changes += info['phase_changes']
                    
                    if done or truncated:
                        # End of episode
                        agent.end_episode(reward)
                        break
                    else:
                        # Get next action and update agent (SARSA)
                        next_action = agent.get_action(next_state, training=True)
                        agent.update(reward, next_state, next_action, False)
                        
                        # Move to next step
                        state = next_state
                        action = next_action
                
                # Episode completed
                episode_time = time.time() - episode_start
                avg_queue = total_queues / max(1, steps)
                
                # Update training history
                self.training_history['episodes'].append(episode + 1)
                self.training_history['rewards'].append(total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['avg_queues'].append(avg_queue)
                self.training_history['phase_changes'].append(phase_changes)
                self.training_history['epsilon'].append(agent.epsilon)
                
                # Handle adaptive agent
                if hasattr(agent, 'adapt_parameters'):
                    agent.adapt_parameters(total_reward)
                    self.training_history['learning_rate'].append(agent.learning_rate)
                else:
                    self.training_history['learning_rate'].append(agent.learning_rate)
                
                # Track performance
                recent_rewards.append(total_reward)
                current_avg = np.mean(recent_rewards)
                
                if current_avg > best_avg_reward:
                    best_avg_reward = current_avg
                    best_episode = episode + 1
                
                # Verbose logging
                if verbose and (episode + 1) % 10 == 0:
                    print(f"  Reward: {total_reward:.1f}, Steps: {steps}, "
                          f"Avg Queue: {avg_queue:.1f}, Changes: {phase_changes}")
                    print(f"  Epsilon: {agent.epsilon:.3f}, "
                          f"Recent Avg: {current_avg:.1f}, Time: {episode_time:.1f}s")
                
                # Save agent periodically
                if (episode + 1) % save_interval == 0:
                    save_path = os.path.join(self.models_dir, f"{save_prefix}_ep{episode + 1}.pkl")
                    agent.save(save_path)
                    print(f"  Agent saved to {save_path}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        finally:
            # Always close environment
            env.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best avg reward: {best_avg_reward:.1f} (episode {best_episode})")
        
        # Save final agent
        final_save_path = os.path.join(self.models_dir, f"{save_prefix}_final.pkl")
        agent.save(final_save_path)
        
        return agent, self.training_history
    
    def evaluate_agent(self, agent_path, episodes=10, use_gui=False):
        """Evaluate trained agent."""
        print(f"\nEvaluating agent: {agent_path}")
        
        # Load agent
        agent = SarsaAgent(state_size=27, action_size=4, epsilon=0.0)  # No exploration
        if not agent.load(agent_path):
            print("Failed to load agent!")
            return None
        
        # Setup environment
        env = self.setup_environment(use_gui=use_gui, max_steps=300)
        
        results = {
            'rewards': [],
            'episode_lengths': [],
            'avg_queues': [],
            'phase_changes': []
        }
        
        try:
            for episode in range(episodes):
                state, info = env.reset()
                total_reward = 0
                steps = 0
                total_queues = 0
                phase_changes = 0
                
                done = False
                while not done:
                    action = agent.get_action(state, training=False)
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    if 'avg_queue_length' in info:
                        total_queues += info['avg_queue_length']
                    if 'phase_changes' in info:
                        phase_changes += info['phase_changes']
                    
                    state = next_state
                    
                    if done or truncated:
                        break
                
                # Store results
                results['rewards'].append(total_reward)
                results['episode_lengths'].append(steps)
                results['avg_queues'].append(total_queues / max(1, steps))
                results['phase_changes'].append(phase_changes)
                
                print(f"Eval Episode {episode + 1}: Reward={total_reward:.1f}, "
                      f"Steps={steps}, Avg Queue={total_queues/max(1,steps):.1f}")
        
        finally:
            env.close()
        
        # Print summary
        print("\n--- EVALUATION SUMMARY ---")
        print(f"Average reward: {np.mean(results['rewards']):.1f} ± {np.std(results['rewards']):.1f}")
        print(f"Average steps: {np.mean(results['episode_lengths']):.1f}")
        print(f"Average queue length: {np.mean(results['avg_queues']):.1f}")
        print(f"Average phase changes: {np.mean(results['phase_changes']):.1f}")
        
        return results
    
    def plot_training_results(self, save_plots=True):
        """Plot training results."""
        if not self.training_history['episodes']:
            print("No training data to plot!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SARSA Agent Training Results (Real SUMO)', fontsize=16)
        
        episodes = self.training_history['episodes']
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(episodes, self.training_history['rewards'], alpha=0.7, label='Episode Reward')
        # Moving average
        window = min(20, len(self.training_history['rewards']))
        if window > 1:
            moving_avg = [np.mean(self.training_history['rewards'][max(0, i-window):i+1]) 
                         for i in range(len(self.training_history['rewards']))]
            axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Average Queue Length
        axes[0, 1].plot(episodes, self.training_history['avg_queues'], color='orange')
        axes[0, 1].set_title('Average Queue Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Queue Length')
        axes[0, 1].grid(True)
        
        # Plot 3: Phase Changes
        axes[0, 2].plot(episodes, self.training_history['phase_changes'], color='green')
        axes[0, 2].set_title('Phase Changes per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Total Phase Changes')
        axes[0, 2].grid(True)
        
        # Plot 4: Epsilon Decay
        axes[1, 0].plot(episodes, self.training_history['epsilon'], color='purple')
        axes[1, 0].set_title('Epsilon (Exploration Rate)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Plot 5: Learning Rate (for adaptive agent)
        axes[1, 1].plot(episodes, self.training_history['learning_rate'], color='brown')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        # Plot 6: Episode Length
        axes[1, 2].plot(episodes, self.training_history['episode_lengths'], color='cyan')
        axes[1, 2].set_title('Episode Length')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f'training_results_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {plot_path}")
        
        plt.show()
    
    def save_training_log(self, agent_info, training_params):
        """Save training log with parameters and results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.results_dir, f'training_log_{timestamp}.json')
        
        log_data = {
            'timestamp': timestamp,
            'agent_info': agent_info,
            'training_params': training_params,
            'training_history': self.training_history,
            'summary': {
                'total_episodes': len(self.training_history['episodes']),
                'final_reward': self.training_history['rewards'][-1] if self.training_history['rewards'] else 0,
                'best_reward': max(self.training_history['rewards']) if self.training_history['rewards'] else 0,
                'final_epsilon': self.training_history['epsilon'][-1] if self.training_history['epsilon'] else 0,
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training log saved to: {log_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SARSA agent with real SUMO simulation')
    parser.add_argument('--agent', choices=['standard', 'adaptive'], default='standard',
                       help='Type of SARSA agent to use')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=600,
                       help='Maximum steps per episode')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI for visualization')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.8,
                       help='Initial epsilon for exploration')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for final evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SARSA AGENT TRAINING WITH REAL SUMO SIMULATION")
    print("="*60)
    print(f"Agent type: {args.agent}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Initial epsilon: {args.epsilon}")
    print(f"GUI mode: {args.gui}")
    
    # Create training manager
    manager = SumoTrainingManager()
    
    try:
        # Setup environment
        env = manager.setup_environment(use_gui=args.gui, max_steps=args.max_steps)
        
        # Create agent
        agent = manager.create_agent(
            agent_type=args.agent,
            learning_rate=args.lr,
            epsilon=args.epsilon
        )
        
        # Training parameters
        training_params = {
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'use_gui': args.gui,
            'agent_type': args.agent
        }
        
        # Train agent
        agent, history = manager.train_agent(
            agent, env, episodes=args.episodes,
            save_prefix=f"{args.agent}_sarsa"
        )
        
        # Show agent statistics
        agent.print_statistics()
        
        # Plot results
        manager.plot_training_results()
        
        # Save training log
        agent_info = {
            'type': type(agent).__name__,
            'parameters': agent.__dict__
        }
        manager.save_training_log(agent_info, training_params)
        
        # Final evaluation
        if args.eval_episodes > 0:
            print(f"\n{'='*60}")
            print("FINAL EVALUATION")
            print("="*60)
            
            final_agent_path = os.path.join(manager.models_dir, f"{args.agent}_sarsa_final.pkl")
            eval_results = manager.evaluate_agent(
                final_agent_path, 
                episodes=args.eval_episodes,
                use_gui=args.gui
            )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Models saved in: {manager.models_dir}")
        print(f"Results saved in: {manager.results_dir}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
