#!/usr/bin/env python3
"""
Quick SUMO Training Script for SARSA Agent

This script provides a streamlined way to train SARSA agents with SUMO,
including fallback options when SUMO is not available.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent


def check_sumo_available():
    """Check if SUMO is properly installed."""
    try:
        import traci
        import sumo
        return True
    except ImportError:
        return False


def train_with_sumo(use_gui=False, episodes=20):
    """Train SARSA agent with real SUMO simulation."""
    try:
        from environment.traffic_env import TrafficEnvironment
        
        print(f"Training with real SUMO simulation {'(GUI)' if use_gui else '(Headless)'}...")
        
        # Create environment
        env = TrafficEnvironment(
            sumo_cfg_file="config/simulation.sumocfg",
            use_gui=use_gui,
            max_steps=300,
            keep_sumo_alive=use_gui  # Keep SUMO alive between episodes in GUI mode
        )
        
        # Create SARSA agent
        agent = SarsaAgent(
            state_size=27,
            action_size=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.8,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        # Training parameters
        rewards_history = []
        queues_history = []
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state, info = env.reset()
            total_reward = 0
            total_queues = 0
            steps = 0
            
            # Get initial action
            action = agent.get_action(state, training=True)
            
            done = False
            while not done:
                # Take step
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Add small delay for GUI visualization
                if use_gui and steps % 10 == 0:
                    time.sleep(0.05)
                
                # Track queues
                if 'avg_queue_length' in info:
                    total_queues += info['avg_queue_length']
                
                if done or truncated:
                    agent.end_episode(reward)
                    break
                else:
                    # SARSA update
                    next_action = agent.get_action(next_state, training=True)
                    agent.update(reward, next_state, next_action, False)
                    
                    state = next_state
                    action = next_action
            
            # Store metrics
            rewards_history.append(total_reward)
            queues_history.append(total_queues / max(1, steps))
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                avg_queue = np.mean(queues_history[-10:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.1f}, "
                      f"Avg Queue = {avg_queue:.1f}, Epsilon = {agent.epsilon:.3f}")
        
        # Close environment
        env.close()
        
        # Save agent
        os.makedirs("models", exist_ok=True)
        agent.save("models/sumo_trained_sarsa.pkl")
        
        # Plot results
        plot_sumo_results(rewards_history, queues_history)
        
        return agent, rewards_history, queues_history
        
    except Exception as e:
        print(f"SUMO training failed: {e}")
        return None, [], []


def train_with_mock_environment():
    """Train SARSA agent with mock traffic environment."""
    print("Training with mock traffic environment...")
    
    # Simple mock environment
    class MockSumoEnvironment:
        def __init__(self):
            self.state_size = 27
            self.max_steps = 200
            self.current_step = 0
            
        def reset(self):
            self.current_step = 0
            # Random initial state (queue lengths, phases, timing)
            state = np.zeros(27)
            for i in range(3):  # 3 intersections
                base_idx = i * 9
                state[base_idx:base_idx+4] = np.random.uniform(0, 15, 4)  # Queues
                state[base_idx+4:base_idx+8] = [1, 0, 0, 0]  # Phase 0 active
                state[base_idx+8] = 0  # Time since change
            return state, {}
        
        def step(self, actions):
            self.current_step += 1
            
            # Simulate traffic dynamics
            state = np.zeros(27)
            total_reward = 0
            
            for i in range(3):
                base_idx = i * 9
                action = int(actions[i])
                
                # Generate new queue lengths based on action
                if action == 0:  # North-South
                    queues = [max(0, np.random.uniform(2, 8)), np.random.uniform(5, 15), 
                             max(0, np.random.uniform(2, 8)), np.random.uniform(5, 15)]
                elif action == 1:  # East-West  
                    queues = [np.random.uniform(5, 15), max(0, np.random.uniform(2, 8)),
                             np.random.uniform(5, 15), max(0, np.random.uniform(2, 8))]
                else:  # Other phases
                    queues = [np.random.uniform(3, 12) for _ in range(4)]
                
                state[base_idx:base_idx+4] = queues
                
                # Set phase encoding
                phase_encoding = [0, 0, 0, 0]
                phase_encoding[action] = 1
                state[base_idx+4:base_idx+8] = phase_encoding
                
                # Time since change
                state[base_idx+8] = min(10.0, self.current_step * 0.1)
                
                # Reward: negative sum of queues
                total_reward -= sum(queues)
            
            # Info with mock metrics
            info = {
                'avg_queue_length': np.mean([state[i*9:(i*9)+4].mean() for i in range(3)]),
                'phase_changes': np.random.randint(0, 3)
            }
            
            done = self.current_step >= self.max_steps
            return state, total_reward, done, False, info
        
        def close(self):
            pass
    
    # Create mock environment and agent
    env = MockSumoEnvironment()
    agent = AdaptiveSarsaAgent(
        state_size=27,
        action_size=4,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.998
    )
    
    # Training
    episodes = 80
    rewards_history = []
    queues_history = []
    
    print(f"Starting mock training for {episodes} episodes...")
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        total_queues = 0
        steps = 0
        
        action = agent.get_action(state, training=True)
        
        done = False
        while not done:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if 'avg_queue_length' in info:
                total_queues += info['avg_queue_length']
            
            if done or truncated:
                agent.end_episode(reward)
                break
            else:
                next_action = agent.get_action(next_state, training=True)
                agent.update(reward, next_state, next_action, False)
                
                state = next_state
                action = next_action
        
        # Adaptive parameter adjustment
        agent.adapt_parameters(total_reward)
        
        rewards_history.append(total_reward)
        queues_history.append(total_queues / max(1, steps))
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.1f}, "
                  f"Epsilon = {agent.epsilon:.3f}, LR = {agent.learning_rate:.3f}")
    
    env.close()
    
    # Save agent
    os.makedirs("models", exist_ok=True)
    agent.save("models/mock_trained_sarsa.pkl")
    
    return agent, rewards_history, queues_history


def plot_sumo_results(rewards, queues):
    """Plot training results."""
    plt.figure(figsize=(15, 5))
    
    # Rewards plot
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.7, label='Episode Reward')
    if len(rewards) > 10:
        window = min(10, len(rewards))
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Queue plot
    plt.subplot(1, 3, 2)
    plt.plot(queues, color='orange')
    plt.title('Average Queue Length')
    plt.xlabel('Episode')
    plt.ylabel('Avg Queue Length')
    plt.grid(True)
    
    # Reward distribution
    plt.subplot(1, 3, 3)
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_results_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"Results saved to: training_results_{timestamp}.png")
    plt.show()


def demonstrate_trained_agent(agent_path):
    """Demonstrate a trained agent's behavior."""
    print(f"\nDemonstrating trained agent: {agent_path}")
    
    # Load agent
    agent = SarsaAgent(state_size=27, action_size=4, epsilon=0.0)
    if not agent.load(agent_path):
        print("Could not load agent!")
        return
    
    # Test on different scenarios
    scenarios = {
        "Heavy Traffic": [15, 12, 18, 10] * 3 + [1, 0, 0, 0] * 3 + [2.5] * 3,
        "Light Traffic": [3, 2, 4, 1] * 3 + [0, 1, 0, 0] * 3 + [1.0] * 3,
        "Rush Hour": [20, 25, 15, 30] * 3 + [0, 0, 1, 0] * 3 + [4.0] * 3
    }
    
    print("\nAgent decisions for different traffic scenarios:")
    print("-" * 60)
    
    for scenario_name, state_values in scenarios.items():
        state = np.array(state_values, dtype=np.float32)
        action = agent.get_action(state, training=False)
        
        # Extract queue info for first intersection
        queues = state[0:4]
        current_phase = np.argmax(state[4:8])
        time_since_change = state[8]
        
        print(f"\n{scenario_name}:")
        print(f"  Queues: {queues}")
        print(f"  Current phase: {current_phase}, Time: {time_since_change:.1f}s")
        print(f"  Agent action: {action}")
        print(f"  Phase decisions: {[f'Int{i+1}->Phase{action[i]}' for i in range(3)]}")
    
    agent.print_statistics()


def main():
    """Main function to run SUMO training."""
    parser = argparse.ArgumentParser(description='Quick SUMO SARSA Training')
    parser.add_argument('--gui', action='store_true', 
                       help='Enable SUMO GUI for visual simulation')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of training episodes (default: 20)')
    
    args = parser.parse_args()
    
    print("SARSA Agent Training with SUMO Integration")
    print("=" * 50)
    
    if args.gui:
        print("🎮 GUI MODE: Visual simulation enabled")
    else:
        print("⚡ HEADLESS MODE: Fast training")
    
    # Check SUMO availability
    sumo_available = check_sumo_available()
    print(f"SUMO available: {sumo_available}")
    
    if sumo_available:
        print(f"\n1. Attempting real SUMO training ({args.episodes} episodes)...")
        agent, rewards, queues = train_with_sumo(
            use_gui=args.gui, 
            episodes=args.episodes
        )
        
        if agent is not None:
            print("✓ Real SUMO training completed successfully!")
            agent.print_statistics()
            
            # Demonstrate trained agent
            demonstrate_trained_agent("models/sumo_trained_sarsa.pkl")
        else:
            print("✗ Real SUMO training failed, falling back to mock environment")
            agent, rewards, queues = train_with_mock_environment()
    else:
        print("\n1. SUMO not available, using mock environment...")
        agent, rewards, queues = train_with_mock_environment()
    
    if agent is not None:
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Final performance:")
        print(f"  Average reward (last 10): {np.mean(rewards[-10:]):.1f}")
        print(f"  Average queue (last 10): {np.mean(queues[-10:]):.1f}")
        print(f"  Total episodes: {len(rewards)}")
        
        agent.print_statistics()
        
        print("\nAgent successfully trained and saved!")
        print("Models saved in: ./models/")


if __name__ == "__main__":
    main()
