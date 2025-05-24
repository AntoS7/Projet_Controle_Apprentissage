#!/usr/bin/env python3
"""
Test script for SARSA agent without requiring SUMO installation.

This script demonstrates the SARSA agent functionality using a mock environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent


class MockTrafficEnvironment:
    """Mock traffic environment for testing SARSA agent without SUMO."""
    
    def __init__(self):
        self.num_intersections = 3
        self.state_size = 27  # 9 values per intersection
        self.action_size = 4  # 4 phases per intersection
        self.current_step = 0
        self.max_steps = 100
        
        # Initialize state (queue lengths, phase encoding, time since change)
        self.state = np.zeros(self.state_size)
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        # Random initial queue lengths (0-20 vehicles)
        for i in range(self.num_intersections):
            base_idx = i * 9
            self.state[base_idx:base_idx+4] = np.random.uniform(0, 20, 4)  # Queue lengths
            self.state[base_idx+4:base_idx+8] = [1, 0, 0, 0]  # Initial phase (phase 0)
            self.state[base_idx+8] = 0  # Time since phase change
        
        return self.state.copy(), {}
    
    def step(self, actions):
        """Take a step in the environment."""
        self.current_step += 1
        
        # Simulate traffic dynamics
        reward = 0
        phase_changes = 0
        
        for i in range(self.num_intersections):
            base_idx = i * 9
            action = int(actions[i])
            
            # Get current phase
            current_phase_encoding = self.state[base_idx+4:base_idx+8]
            current_phase = np.argmax(current_phase_encoding)
            
            # Check if phase changed
            if action != current_phase:
                phase_changes += 1
                # Set new phase encoding
                self.state[base_idx+4:base_idx+8] = 0
                self.state[base_idx+4+action] = 1
                # Reset time since phase change
                self.state[base_idx+8] = 0
            else:
                # Increment time since phase change
                self.state[base_idx+8] = min(10.0, self.state[base_idx+8] + 0.1)
            
            # Simulate queue evolution based on phase
            queues = self.state[base_idx:base_idx+4]
            
            # Simple traffic simulation
            # Active phase reduces its queues, others may increase
            for j in range(4):
                if j == action:
                    # Active direction: reduce queue
                    self.state[base_idx+j] = max(0, queues[j] - np.random.uniform(2, 5))
                else:
                    # Inactive directions: may increase
                    self.state[base_idx+j] = min(50, queues[j] + np.random.uniform(0, 2))
            
            # Reward: negative sum of queues
            reward -= np.sum(self.state[base_idx:base_idx+4])
        
        # Penalty for frequent phase changes
        reward -= phase_changes * 5
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return self.state.copy(), reward, done, False, {"phase_changes": phase_changes}


def test_sarsa_agent():
    """Test SARSA agent with mock environment."""
    print("Testing SARSA Agent for Traffic Light Control")
    print("=" * 60)
    
    # Create environment and agent
    env = MockTrafficEnvironment()
    agent = SarsaAgent(
        state_size=27,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.8,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Training
    episodes = 50
    rewards_history = []
    
    print(f"Training for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        # Get initial action
        action = agent.get_action(state)
        
        while True:
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                # End of episode
                agent.end_episode(reward)
                break
            else:
                # Get next action and update agent
                next_action = agent.get_action(next_state)
                agent.update(reward, next_state, next_action, False)
                
                # Move to next step
                state = next_state
                action = next_action
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}: Avg Reward (last 10) = {avg_reward:.2f}, "
                  f"Steps = {steps}, Epsilon = {agent.epsilon:.3f}")
    
    # Print final statistics
    agent.print_statistics()
    
    # Plot results
    if len(rewards_history) > 0:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards_history)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Moving average
        window = min(10, len(rewards_history))
        moving_avg = [np.mean(rewards_history[max(0, i-window):i+1]) 
                     for i in range(len(rewards_history))]
        plt.plot(moving_avg, color='red', label=f'Moving Average ({window})')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(rewards_history, bins=20, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sarsa_training_results.png', dpi=150)
        print(f"\nTraining results saved to: sarsa_training_results.png")
        plt.show()
    
    return agent, rewards_history


def test_adaptive_sarsa():
    """Test adaptive SARSA agent."""
    print("\n" + "=" * 60)
    print("Testing Adaptive SARSA Agent")
    print("=" * 60)
    
    env = MockTrafficEnvironment()
    agent = AdaptiveSarsaAgent(
        state_size=27,
        action_size=4,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.998
    )
    
    episodes = 30
    rewards_history = []
    
    print(f"Training adaptive agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        action = agent.get_action(state)
        
        while True:
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                agent.end_episode(reward)
                break
            else:
                next_action = agent.get_action(next_state)
                agent.update(reward, next_state, next_action, False)
                state = next_state
                action = next_action
        
        rewards_history.append(total_reward)
        agent.adapt_parameters(total_reward)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}, LR = {agent.learning_rate:.3f}")
    
    agent.print_statistics()
    return agent, rewards_history


if __name__ == "__main__":
    # Test standard SARSA agent
    standard_agent, standard_rewards = test_sarsa_agent()
    
    # Test adaptive SARSA agent
    adaptive_agent, adaptive_rewards = test_adaptive_sarsa()
    
    # Compare performance
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Standard SARSA - Final avg reward: {np.mean(standard_rewards[-10:]):.2f}")
    print(f"Adaptive SARSA - Final avg reward: {np.mean(adaptive_rewards[-10:]):.2f}")
    print("=" * 60)
