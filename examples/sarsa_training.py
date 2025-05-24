"""
SARSA Agent Training Example for Traffic Light Control.

This script demonstrates how to train and evaluate a SARSA agent
for controlling traffic lights in the SUMO simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.traffic_env import TrafficEnvironment
from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent
from network.network_generator import generate_sumo_network
from network.traffic_generator import generate_traffic_flows
from utils.sumo_utils import create_sumo_config


def setup_training_environment():
    """Setup environment for SARSA training."""
    print("Setting up training environment...")
    
    # Generate network and traffic
    network_file = generate_sumo_network("config")
    routes_file = generate_traffic_flows("config", duration=1200)  # 20 minutes
    config_file = create_sumo_config(
        network_file="network.net.xml",
        routes_file="routes.rou.xml",
        output_file="config/simulation.sumocfg",
        end=1200
    )
    
    print(f"Environment setup complete: {config_file}")
    return config_file


def train_sarsa_agent(agent_type="standard", episodes=100, max_steps=600, save_path="models/sarsa_agent.pkl"):
    """
    Train SARSA agent on traffic environment.
    
    Args:
        agent_type: "standard" or "adaptive"
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_path: Path to save trained agent
    """
    print(f"Training {agent_type} SARSA agent for {episodes} episodes...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=max_steps
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    if agent_type == "adaptive":
        agent = AdaptiveSarsaAgent(
            state_size=state_size,
            action_size=4,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon=0.9,
            epsilon_min=0.05,
            epsilon_decay=0.998
        )
    else:
        agent = SarsaAgent(
            state_size=state_size,
            action_size=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.8,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_queues = []
    phase_changes_history = []
    
    # Performance tracking
    best_reward = float('-inf')
    best_episode = 0
    
    try:
        for episode in range(episodes):
            start_time = time.time()
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            # Reset environment
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            total_phase_changes = 0
            
            # Get initial action
            action = agent.get_action(state, training=True)
            
            while True:
                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Get next action (SARSA uses actual next action)
                next_action = agent.get_action(next_state, training=True)
                
                # Update agent with SARSA
                agent.update(reward, next_state, next_action, terminated or truncated)
                
                # Update metrics
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                total_phase_changes += info.get('phase_changes', 0)
                
                # Move to next state-action
                state = next_state
                action = next_action
                
                if terminated or truncated:
                    # End episode
                    agent.end_episode(reward)
                    break
            
            # Adaptive agent parameter adjustment
            if isinstance(agent, AdaptiveSarsaAgent):
                agent.adapt_parameters(total_reward)
            
            # Store episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            avg_queue = total_queues / steps if steps > 0 else 0
            avg_queues.append(avg_queue)
            phase_changes_history.append(total_phase_changes)
            
            # Track best performance
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode + 1
            
            # Episode summary
            episode_time = time.time() - start_time
            print(f"  Steps: {steps}, Reward: {total_reward:.2f}, "
                  f"Avg Queues: {avg_queue:.1f}, Phase Changes: {total_phase_changes}")
            print(f"  Epsilon: {agent.epsilon:.3f}, Time: {episode_time:.1f}s")
            
            # Print running statistics every 20 episodes
            if (episode + 1) % 20 == 0:
                recent_rewards = episode_rewards[-20:]
                recent_queues = avg_queues[-20:]
                print(f"\n  === Last 20 episodes summary ===")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"  Avg Queues: {np.mean(recent_queues):.1f} ± {np.std(recent_queues):.1f}")
                print(f"  Best Episode: {best_episode} (Reward: {best_reward:.2f})")
                
                # Print agent statistics
                if (episode + 1) % 40 == 0:
                    agent.print_statistics()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    # Save trained agent
    if episode_rewards:
        try:
            agent.save(save_path)
        except Exception as e:
            print(f"Error saving agent: {e}")
    
    return agent, episode_rewards, episode_lengths, avg_queues, phase_changes_history


def evaluate_sarsa_agent(agent_path="models/sarsa_agent.pkl", episodes=10, use_gui=False):
    """
    Evaluate trained SARSA agent.
    
    Args:
        agent_path: Path to saved agent
        episodes: Number of evaluation episodes
        use_gui: Whether to show SUMO GUI
    """
    print(f"\nEvaluating SARSA agent for {episodes} episodes...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=use_gui,
        max_steps=600
    )
    
    # Load agent
    state_size = env.observation_space.shape[0]
    agent = SarsaAgent(state_size=state_size, action_size=4)
    
    if not agent.load(agent_path):
        print("Could not load agent, using random policy")
        agent = None
    
    evaluation_results = {
        'rewards': [],
        'avg_queues': [],
        'phase_changes': [],
        'episode_lengths': []
    }
    
    try:
        for episode in range(episodes):
            print(f"\nEvaluation Episode {episode + 1}/{episodes}")
            
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            total_phase_changes = 0
            
            while True:
                if agent is not None:
                    # Use trained agent (no exploration)
                    action = agent.get_action(state, training=False)
                else:
                    # Random policy fallback
                    action = env.action_space.sample()
                
                state, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                total_phase_changes += info.get('phase_changes', 0)
                
                if terminated or truncated:
                    break
            
            # Store results
            avg_queue = total_queues / steps if steps > 0 else 0
            evaluation_results['rewards'].append(total_reward)
            evaluation_results['avg_queues'].append(avg_queue)
            evaluation_results['phase_changes'].append(total_phase_changes)
            evaluation_results['episode_lengths'].append(steps)
            
            print(f"  Reward: {total_reward:.2f}, Avg Queues: {avg_queue:.1f}, "
                  f"Phase Changes: {total_phase_changes}, Steps: {steps}")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        env.close()
    
    # Print evaluation summary
    if evaluation_results['rewards']:
        print(f"\n{'='*50}")
        print("EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Episodes: {len(evaluation_results['rewards'])}")
        print(f"Average Reward: {np.mean(evaluation_results['rewards']):.2f} ± "
              f"{np.std(evaluation_results['rewards']):.2f}")
        print(f"Average Queues: {np.mean(evaluation_results['avg_queues']):.1f} ± "
              f"{np.std(evaluation_results['avg_queues']):.1f}")
        print(f"Average Phase Changes: {np.mean(evaluation_results['phase_changes']):.1f} ± "
              f"{np.std(evaluation_results['phase_changes']):.1f}")
        print(f"Average Episode Length: {np.mean(evaluation_results['episode_lengths']):.1f}")
        print(f"{'='*50}")
    
    return evaluation_results


def compare_agents():
    """Compare SARSA agent with random and fixed policies."""
    print("\n" + "="*60)
    print("COMPARING CONTROL STRATEGIES")
    print("="*60)
    
    strategies = {
        "SARSA Agent": lambda: evaluate_sarsa_agent("models/sarsa_agent.pkl", episodes=5),
        "Random Policy": lambda: evaluate_random_policy(episodes=5),
        "Fixed Policy": lambda: evaluate_fixed_policy(episodes=5)
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\n--- Evaluating {name} ---")
        try:
            results[name] = strategy()
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = None
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Strategy':<15} {'Avg Reward':<12} {'Avg Queues':<12} {'Phase Changes':<15}")
    print("-" * 60)
    
    for name, result in results.items():
        if result and result['rewards']:
            avg_reward = np.mean(result['rewards'])
            avg_queues = np.mean(result['avg_queues'])
            avg_changes = np.mean(result['phase_changes'])
            print(f"{name:<15} {avg_reward:<12.2f} {avg_queues:<12.1f} {avg_changes:<15.1f}")
        else:
            print(f"{name:<15} {'Failed':<12} {'Failed':<12} {'Failed':<15}")


def evaluate_random_policy(episodes=5):
    """Evaluate random policy for comparison."""
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=600
    )
    
    results = {'rewards': [], 'avg_queues': [], 'phase_changes': [], 'episode_lengths': []}
    
    try:
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            total_phase_changes = 0
            
            while True:
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                total_phase_changes += info.get('phase_changes', 0)
                
                if terminated or truncated:
                    break
            
            results['rewards'].append(total_reward)
            results['avg_queues'].append(total_queues / steps if steps > 0 else 0)
            results['phase_changes'].append(total_phase_changes)
            results['episode_lengths'].append(steps)
    
    finally:
        env.close()
    
    return results


def evaluate_fixed_policy(episodes=5):
    """Evaluate fixed policy (no phase changes) for comparison."""
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=False,
        max_steps=600
    )
    
    results = {'rewards': [], 'avg_queues': [], 'phase_changes': [], 'episode_lengths': []}
    
    try:
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            total_queues = 0
            total_phase_changes = 0
            
            # Fixed action (no changes)
            fixed_action = np.array([0, 0, 0])
            
            while True:
                state, reward, terminated, truncated, info = env.step(fixed_action)
                
                total_reward += reward
                steps += 1
                total_queues += info.get('total_queues', 0)
                total_phase_changes += info.get('phase_changes', 0)
                
                if terminated or truncated:
                    break
            
            results['rewards'].append(total_reward)
            results['avg_queues'].append(total_queues / steps if steps > 0 else 0)
            results['phase_changes'].append(total_phase_changes)
            results['episode_lengths'].append(steps)
    
    finally:
        env.close()
    
    return results


def plot_training_results(rewards, queues, phase_changes):
    """Plot training results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Moving average of rewards
        window = min(10, len(rewards))
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Average queue lengths
        axes[1, 0].plot(queues)
        axes[1, 0].set_title('Average Queue Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Queues')
        axes[1, 0].grid(True)
        
        # Phase changes
        axes[1, 1].plot(phase_changes)
        axes[1, 1].set_title('Phase Changes per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Phase Changes')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('sarsa_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training plots saved as 'sarsa_training_results.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    """Main function for SARSA agent training and evaluation."""
    print("SARSA Agent for Traffic Light Control")
    print("=" * 50)
    
    # Setup environment
    try:
        setup_training_environment()
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train standard SARSA agent
    print("\n--- Training Standard SARSA Agent ---")
    agent, rewards, lengths, queues, changes = train_sarsa_agent(
        agent_type="standard",
        episodes=80,
        max_steps=500,
        save_path="models/sarsa_agent.pkl"
    )
    
    if rewards:
        # Plot training results
        plot_training_results(rewards, queues, changes)
        
        # Final agent statistics
        agent.print_statistics()
        
        # Evaluate trained agent
        print("\n--- Evaluating Trained Agent ---")
        evaluate_sarsa_agent("models/sarsa_agent.pkl", episodes=5, use_gui=False)
        
        # Compare with other strategies
        compare_agents()
        
        print("\n--- Training Adaptive SARSA Agent ---")
        adaptive_agent, _, _, _, _ = train_sarsa_agent(
            agent_type="adaptive",
            episodes=60,
            max_steps=500,
            save_path="models/adaptive_sarsa_agent.pkl"
        )
        
        if adaptive_agent:
            print("\n--- Evaluating Adaptive Agent ---")
            evaluate_sarsa_agent("models/adaptive_sarsa_agent.pkl", episodes=5)
        
    print(f"\nTraining completed! Models saved in 'models/' directory")


if __name__ == "__main__":
    main()
