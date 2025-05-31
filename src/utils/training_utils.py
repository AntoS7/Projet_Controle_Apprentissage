"""
Training Utilities Module

This module provides modular functions for training script refactoring,
breaking down large main() functions into smaller, reusable components.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import pickle
import json

# Add src to path if needed
if 'src' not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent
from environment.enhanced_sarsa_agent import EnhancedSarsaAgent
from environment.traffic_env import TrafficEnvironment


class TrainingSetupManager:
    """Manages setup and initialization for training scripts."""
    
    def __init__(self, config_dir: str = "config", models_dir: str = "models"):
        self.config_dir = config_dir
        self.models_dir = models_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        for directory in [self.config_dir, self.models_dir, "logs", "results"]:
            os.makedirs(directory, exist_ok=True)
    
    def parse_training_arguments(self, script_name: str) -> argparse.Namespace:
        """Parse command line arguments for training scripts."""
        parser = argparse.ArgumentParser(description=f'{script_name} SARSA Training')
        
        parser.add_argument('--gui', action='store_true',
                           help='Enable SUMO GUI for visual simulation')
        parser.add_argument('--episodes', type=int, default=30,
                           help='Number of training episodes (default: 30)')
        parser.add_argument('--fast', action='store_true',
                           help='Fast mode with fewer episodes')
        parser.add_argument('--agent', choices=['standard', 'adaptive'], default='standard',
                           help='Type of SARSA agent to use')
        parser.add_argument('--lr', type=float, default=0.1,
                           help='Learning rate')
        parser.add_argument('--epsilon', type=float, default=0.8,
                           help='Initial epsilon for exploration')
        parser.add_argument('--max_steps', type=int, default=400,
                           help='Maximum steps per episode')
        
        return parser.parse_args()
    
    def print_training_header(self, script_name: str, args: argparse.Namespace):
        """Print formatted training header."""
        print("=" * 60)
        print(f"{script_name.upper()} SARSA TRAINING")
        print("=" * 60)
        
        if args.gui:
            print("GUI MODE: Visual simulation enabled")
            print("   - Episodes will pause between runs for observation")
            print("   - Press Enter to automatically start the next episode")
            print("   - Slower training but visual feedback")
        else:
            print("HEADLESS MODE: Fast training without visualization")
        
        if hasattr(args, 'fast') and args.fast:
            print("FAST MODE: Running reduced episodes")
        
        print(f"Agent type: {args.agent}")
        print(f"Episodes: {args.episodes}")
        print(f"Learning rate: {args.lr}")
        print(f"Initial epsilon: {args.epsilon}")
    
    def setup_environment(self, use_gui: bool = False, max_steps: int = 400) -> TrafficEnvironment:
        """Setup and return traffic environment."""
        try:
            env = TrafficEnvironment(
                config_file=os.path.join(self.config_dir, "simulation.sumocfg"),
                use_gui=use_gui,
                max_steps=max_steps
            )
            print("Environment setup completed")
            return env
        except Exception as e:
            print(f"Environment setup failed: {e}")
            raise
    
    def create_agent(self, agent_type: str, state_size: int = 27, action_size: int = 4,
                    learning_rate: float = 0.1, epsilon: float = 0.8, 
                    total_episodes: int = 100, epsilon_decay_strategy: str = 'exponential') -> SarsaAgent:
        """Create and return SARSA agent with enhanced epsilon decay."""
        if agent_type.lower() == "enhanced":
            agent = EnhancedSarsaAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                epsilon=epsilon,
                total_episodes=total_episodes,
                epsilon_decay_strategy=epsilon_decay_strategy,
                use_double_sarsa=True,
                use_replay_buffer=True,
                adaptive_lr=True
            )
            print(f"Enhanced SARSA agent created with {epsilon_decay_strategy} epsilon decay")
        elif agent_type.lower() == "adaptive":
            agent = AdaptiveSarsaAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                epsilon=epsilon,
                total_episodes=total_episodes,
                epsilon_decay_strategy=epsilon_decay_strategy
            )
            print(f"Adaptive SARSA agent created with {epsilon_decay_strategy} epsilon decay")
        else:
            agent = SarsaAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                epsilon=epsilon,
                total_episodes=total_episodes,
                epsilon_decay_strategy=epsilon_decay_strategy
            )
            print(f"Standard SARSA agent created with {epsilon_decay_strategy} epsilon decay")
        
        return agent


class TrainingExecutor:
    """Executes training episodes and manages training loop."""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def run_training_episode(self, agent: SarsaAgent, env: TrafficEnvironment,
                           episode: int, max_steps: int = 400) -> Tuple[float, float, int]:
        """Run a single training episode."""
        state, info = env.reset()
        total_reward = 0
        total_queue = 0
        steps = 0
        
        # Initialize action
        action = agent.get_action(state)
        
        for step in range(max_steps):
            # Take action and observe result
            next_state, reward, done, truncated, info = env.step(action)
            
            # Get next action
            next_action = agent.get_action(next_state)
            
            # Update agent (SARSA update)
            agent.update(state, action, reward, next_state, next_action, done)
            
            # Update state and action
            state = next_state
            action = next_action
            
            total_reward += reward
            total_queue += info.get('queue_length', 0)
            steps = step + 1
            
            if done or truncated:
                break
        
        avg_queue = total_queue / steps if steps > 0 else 0
        
        if self.logger:
            self.logger.log_episode(episode, total_reward, avg_queue, steps)
        
        return total_reward, avg_queue, steps
    
    def train_agent(self, agent: SarsaAgent, env: TrafficEnvironment,
                   episodes: int, max_steps: int = 400,
                   use_gui: bool = False) -> Tuple[SarsaAgent, List[float], List[float]]:
        """Train agent for specified number of episodes."""
        rewards = []
        queues = []
        
        print(f"\nStarting training for {episodes} episodes...")
        print("-" * 50)
        
        try:
            for episode in range(episodes):
                if use_gui and episode > 0:
                    input(f"\nPress Enter to start Episode {episode + 1}/{episodes}...")
                
                # Track episode for enhanced epsilon decay
                agent.start_episode()
                
                # Run episode
                reward, queue, steps = self.run_training_episode(
                    agent, env, episode + 1, max_steps
                )
                
                rewards.append(reward)
                queues.append(queue)
                
                # Print episode summary
                if episode % 5 == 0 or episode == episodes - 1:
                    avg_reward = np.mean(rewards[-5:]) if len(rewards) >= 5 else np.mean(rewards)
                    avg_queue = np.mean(queues[-5:]) if len(queues) >= 5 else np.mean(queues)
                    print(f"Episode {episode + 1:3d}: Reward={reward:6.1f}, "
                          f"Queue={queue:4.1f}, Steps={steps:3d}, "
                          f"Avg(5)={avg_reward:6.1f}, ε={agent.epsilon:.4f}")
                
                # Enhanced epsilon decay is now handled automatically in agent.update()
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            if self.logger:
                self.logger.error(f"Training failed: {e}")
        
        return agent, rewards, queues


class TrainingPostProcessor:
    """Handles post-training operations like saving, plotting, and evaluation."""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    def save_agent(self, agent: SarsaAgent, filename: str, 
                   training_info: Optional[Dict] = None) -> str:
        """Save trained agent to file."""
        filepath = os.path.join(self.models_dir, filename)
        
        try:
            # Save agent
            agent.save(filepath)
            
            # Save training info if provided
            if training_info:
                info_filepath = filepath.replace('.pkl', '_info.json')
                with open(info_filepath, 'w') as f:
                    json.dump(training_info, f, indent=2)
            
            print(f"Agent saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Failed to save agent: {e}")
            return ""
    
    def plot_training_results(self, rewards: List[float], queues: List[float],
                             title: str = "Training Results", show: bool = True,
                             save_path: Optional[str] = None) -> str:
        """Plot training results and optionally save."""
        if not rewards:
            print("No training data to plot")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot rewards
        ax1.plot(rewards, 'b-', alpha=0.7, label='Episode Reward')
        ax1.set_title(f'{title} - Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot moving average for rewards
        if len(rewards) >= 10:
            moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            ax1.plot(range(9, len(rewards)), moving_avg, 'r-', alpha=0.7, 
                    label='Moving Average (10)')
            ax1.legend()
        
        # Plot queues
        ax2.plot(queues, 'g-', alpha=0.7, label='Average Queue Length')
        ax2.set_title(f'{title} - Queue Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Queue Length')
        ax2.grid(True, alpha=0.3)
        
        # Plot moving average for queues
        if len(queues) >= 10:
            moving_avg_q = np.convolve(queues, np.ones(10)/10, mode='valid')
            ax2.plot(range(9, len(queues)), moving_avg_q, 'orange', alpha=0.7,
                    label='Moving Average (10)')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plot_path = save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"training_results_{timestamp}.png")
        
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {plot_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
            plot_path = ""
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def print_training_summary(self, agent: SarsaAgent, rewards: List[float],
                              queues: List[float], training_time: float = 0):
        """Print comprehensive training summary."""
        if not rewards:
            print("No training data to summarize")
            return
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        # Agent statistics
        print(f"Agent Type: {type(agent).__name__}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        print(f"Q-table Size: {len(agent.q_table)} states")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Training Time: {training_time:.2f} seconds" if training_time > 0 else "")
        print(f"  Final Reward: {rewards[-1]:.2f}")
        print(f"  Best Reward: {max(rewards):.2f}")
        print(f"  Average Reward (last 10): {np.mean(rewards[-10:]):.2f}")
        print(f"  Average Queue (last 10): {np.mean(queues[-10:]):.2f}")
        print(f"  Improvement: {rewards[-1] - rewards[0]:.2f}")
        
        # Learning progress
        if len(rewards) >= 20:
            early_avg = np.mean(rewards[:10])
            late_avg = np.mean(rewards[-10:])
            improvement_pct = ((late_avg - early_avg) / abs(early_avg)) * 100
            print(f"  Learning Improvement: {improvement_pct:.1f}%")
        
        print("=" * 60)
    
    def evaluate_agent(self, agent_path: str, env: TrafficEnvironment,
                      episodes: int = 5, use_gui: bool = False) -> Dict[str, float]:
        """Evaluate saved agent performance."""
        try:
            # Load agent
            with open(agent_path, 'rb') as f:
                agent = pickle.load(f)
            
            print(f"\nEvaluating agent: {agent_path}")
            print(f"Evaluation episodes: {episodes}")
            
            # Disable exploration for evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0  # Pure exploitation
            
            eval_rewards = []
            eval_queues = []
            
            for episode in range(episodes):
                state, info = env.reset()
                total_reward = 0
                total_queue = 0
                steps = 0
                
                action = agent.get_action(state)
                
                for step in range(400):  # Max steps
                    next_state, reward, done, truncated, info = env.step(action)
                    next_action = agent.get_action(next_state)
                    
                    state = next_state
                    action = next_action
                    total_reward += reward
                    total_queue += info.get('queue_length', 0)
                    steps = step + 1
                    
                    if done or truncated:
                        break
                
                avg_queue = total_queue / steps if steps > 0 else 0
                eval_rewards.append(total_reward)
                eval_queues.append(avg_queue)
                
                print(f"  Eval Episode {episode + 1}: Reward={total_reward:.1f}, Queue={avg_queue:.1f}")
            
            # Restore original epsilon
            agent.epsilon = original_epsilon
            
            # Calculate evaluation metrics
            results = {
                'mean_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'mean_queue': np.mean(eval_queues),
                'std_queue': np.std(eval_queues),
                'episodes': episodes
            }
            
            print(f"\nEvaluation Results:")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean Queue: {results['mean_queue']:.2f} ± {results['std_queue']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {}


def run_complete_training_pipeline(script_name: str = "Complete Training") -> Dict[str, Any]:
    """
    Complete training pipeline that can be used by any training script.
    Returns training results and metadata.
    """
    # Setup
    setup_manager = TrainingSetupManager()
    args = setup_manager.parse_training_arguments(script_name)
    
    # Adjust episodes for fast mode
    if hasattr(args, 'fast') and args.fast:
        args.episodes = min(args.episodes, 10)
    
    setup_manager.print_training_header(script_name, args)
    
    # Initialize components
    training_executor = TrainingExecutor()
    post_processor = TrainingPostProcessor()
    
    try:
        # Setup environment and agent
        env = setup_manager.setup_environment(args.gui, args.max_steps)
        agent = setup_manager.create_agent(args.agent, learning_rate=args.lr, epsilon=args.epsilon)
        
        # Train
        start_time = datetime.now()
        agent, rewards, queues = training_executor.train_agent(
            agent, env, args.episodes, args.max_steps, args.gui
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Post-process
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save agent
        agent_filename = f"{script_name.lower().replace(' ', '_')}_sarsa_{args.agent}_{timestamp}.pkl"
        training_info = {
            'script': script_name,
            'agent_type': args.agent,
            'episodes': args.episodes,
            'learning_rate': args.lr,
            'initial_epsilon': args.epsilon,
            'final_epsilon': agent.epsilon,
            'training_time': training_time,
            'timestamp': timestamp
        }
        
        agent_path = post_processor.save_agent(agent, agent_filename, training_info)
        
        # Plot results
        plot_path = post_processor.plot_training_results(
            rewards, queues, f"{script_name} Results"
        )
        
        # Print summary
        post_processor.print_training_summary(agent, rewards, queues, training_time)
        
        # Return results
        return {
            'success': True,
            'agent': agent,
            'agent_path': agent_path,
            'rewards': rewards,
            'queues': queues,
            'training_time': training_time,
            'plot_path': plot_path,
            'training_info': training_info
        }
        
    except Exception as e:
        print(f"\nTraining pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    
    finally:
        try:
            env.close()
        except:
            pass
