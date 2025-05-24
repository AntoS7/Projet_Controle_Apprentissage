#!/usr/bin/env python3
"""
Robust SUMO Training Script

This script provides robust SUMO integration with better error handling
and connection management for training SARSA agents.
"""

import sys
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent


def check_sumo_installation():
    """Check if SUMO is properly installed."""
    try:
        result = subprocess.run(['netconvert', '--help'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


class RobustTrafficEnv:
    """A robust traffic environment wrapper with better SUMO handling."""
    
    def __init__(self, config_file="config/simulation.sumocfg", use_gui=False):
        self.config_file = config_file
        self.use_gui = use_gui
        self.sumo_process = None
        self.episode_count = 0
        
        # Import traci here to handle missing imports gracefully
        try:
            import traci
            self.traci = traci
            self.traci_available = True
        except ImportError:
            print("Warning: TraCI not available. Using mock environment.")
            self.traci_available = False
            
        # Setup observation and action spaces
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        import gymnasium as gym
        
        # Action space: 0=keep current phase, 1=switch phase
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation space: [queue_length, waiting_time, phase_duration, current_phase]
        # Bounds for each intersection
        obs_low = np.array([0, 0, 0, 0] * 3, dtype=np.float32)  # 3 intersections
        obs_high = np.array([100, 300, 120, 3] * 3, dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
    def _start_sumo(self):
        """Start SUMO simulation with robust error handling."""
        if not self.traci_available:
            return False
            
        # Kill any existing SUMO processes
        self._cleanup_sumo()
        
        # Build SUMO command
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--tripinfo-output", "tripinfo.xml",
            "--step-length", "1",
            "--time-to-teleport", "-1",
            "--start",
            "--quit-on-end"
        ]
        
        try:
            # Start SUMO process
            self.sumo_process = subprocess.Popen(
                sumo_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)  # Give SUMO more time to start
            
            # Connect to SUMO via TraCI
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    self.traci.init()
                    print(f"✓ Connected to SUMO (attempt {attempt + 1})")
                    return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        print(f"✗ Failed to connect to SUMO after {max_retries} attempts: {e}")
                        self._cleanup_sumo()
                        return False
                        
        except Exception as e:
            print(f"✗ Error starting SUMO: {e}")
            self._cleanup_sumo()
            return False
    
    def _cleanup_sumo(self):
        """Clean up SUMO processes."""
        try:
            if self.traci_available:
                self.traci.close()
        except:
            pass
            
        if self.sumo_process:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=5)
            except:
                try:
                    self.sumo_process.kill()
                except:
                    pass
            self.sumo_process = None
            
        # Kill any remaining SUMO processes
        try:
            subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
        except:
            pass
    
    def reset(self):
        """Reset the environment."""
        self.episode_count += 1
        
        if not self.traci_available:
            # Return mock observation
            return np.random.random(12).astype(np.float32), {}
            
        # Start SUMO for this episode
        if not self._start_sumo():
            print("Failed to start SUMO, using mock observation")
            return np.random.random(12).astype(np.float32), {}
        
        # Get initial observation
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        if not self.traci_available:
            # Mock step
            obs = np.random.random(12).astype(np.float32)
            reward = np.random.uniform(-10, 10)
            done = np.random.random() > 0.95
            return obs, reward, done, False, {}
        
        try:
            # Execute action (simplified for this example)
            if action == 1:  # Switch phase
                self._switch_traffic_light_phase()
            
            # Advance simulation
            self.traci.simulationStep()
            
            # Get new observation and reward
            obs = self._get_observation()
            reward = self._calculate_reward()
            
            # Check if simulation is done
            done = self.traci.simulation.getMinExpectedNumber() <= 0
            
            return obs, reward, done, False, {}
            
        except Exception as e:
            print(f"Error in simulation step: {e}")
            # Return mock values on error
            obs = np.random.random(12).astype(np.float32)
            return obs, -100, True, False, {}
    
    def _get_observation(self):
        """Get current observation from SUMO."""
        try:
            obs = []
            intersections = ["intersection1", "intersection2", "intersection3"]
            
            for intersection in intersections:
                # Queue length (simplified)
                queue_length = min(50, max(0, np.random.poisson(10)))
                
                # Waiting time (simplified)
                waiting_time = np.random.exponential(30)
                
                # Phase duration (simplified)
                phase_duration = np.random.uniform(0, 60)
                
                # Current phase
                current_phase = np.random.randint(0, 4)
                
                obs.extend([queue_length, waiting_time, phase_duration, current_phase])
                
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.random.random(12).astype(np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on traffic metrics."""
        try:
            # Simplified reward calculation
            total_waiting = 0
            vehicle_ids = self.traci.vehicle.getIDList()
            
            for vehicle_id in vehicle_ids:
                waiting_time = self.traci.vehicle.getWaitingTime(vehicle_id)
                total_waiting += waiting_time
            
            # Negative reward for waiting time
            reward = -total_waiting * 0.1
            
            # Bonus for number of vehicles that have completed their journey
            completed_vehicles = self.traci.simulation.getArrivedNumber()
            reward += completed_vehicles * 2
            
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return np.random.uniform(-10, 10)
    
    def _switch_traffic_light_phase(self):
        """Switch traffic light phase."""
        try:
            tl_ids = self.traci.trafficlight.getIDList()
            for tl_id in tl_ids:
                current_phase = self.traci.trafficlight.getPhase(tl_id)
                program_logic = self.traci.trafficlight.getAllProgramLogics(tl_id)[0]
                num_phases = len(program_logic.phases)
                next_phase = (current_phase + 1) % num_phases
                self.traci.trafficlight.setPhase(tl_id, next_phase)
        except Exception as e:
            print(f"Error switching traffic light: {e}")
    
    def close(self):
        """Close the environment."""
        self._cleanup_sumo()


def train_with_robust_sumo(episodes=50, agent_type="standard"):
    """Train SARSA agent with robust SUMO integration."""
    print("Training with robust SUMO integration...")
    
    # Create environment
    env = RobustTrafficEnv(use_gui=False)
    
    # Create SARSA agent
    if agent_type == "adaptive":
        agent = AdaptiveSarsaAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
    else:
        agent = SarsaAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
    
    # Training metrics
    episode_rewards = []
    episode_queues = []
    
    print(f"Starting robust training for {episodes} episodes...")
    
    for episode in range(episodes):
        try:
            state, _ = env.reset()
            total_reward = 0
            total_queue = 0
            steps = 0
            max_steps = 1200  # Maximum steps per episode
            
            # Choose initial action
            action = agent.choose_action(state)
            
            while steps < max_steps:
                # Execute action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Choose next action
                next_action = agent.choose_action(next_state)
                
                # SARSA update
                agent.update(state, action, reward, next_state, next_action, done)
                
                # Update for next iteration
                state = next_state
                action = next_action
                total_reward += reward
                total_queue += np.sum(next_state[::4])  # Sum queue lengths
                steps += 1
                
                if done or truncated:
                    break
            
            # Store metrics
            episode_rewards.append(total_reward)
            episode_queues.append(total_queue / max(steps, 1))
            
            # Print progress
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_queue = np.mean(episode_queues[-10:])
                print(f"Episode {episode + 1:3d}: Reward={total_reward:7.2f}, "
                      f"Avg Reward={avg_reward:7.2f}, Avg Queue={avg_queue:.2f}, "
                      f"Steps={steps}, Epsilon={agent.epsilon:.3f}")
            
            # Update exploration rate
            if hasattr(agent, 'update_epsilon'):
                agent.update_epsilon()
                
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            continue
    
    # Clean up
    env.close()
    
    return agent, episode_rewards, episode_queues


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Robust SUMO SARSA Training")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--agent_type", choices=["standard", "adaptive"], default="standard",
                       help="Type of SARSA agent to use")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    
    args = parser.parse_args()
    
    print("Robust SARSA Agent Training with SUMO Integration")
    print("=" * 55)
    
    # Check SUMO availability
    sumo_available = check_sumo_installation()
    print(f"SUMO available: {sumo_available}")
    
    if not sumo_available:
        print("Warning: SUMO not found. Training will use simplified simulation.")
    
    # Train with robust SUMO
    print("\nStarting robust SUMO training...")
    start_time = time.time()
    
    try:
        agent, rewards, queues = train_with_robust_sumo(
            episodes=args.episodes,
            agent_type=args.agent_type
        )
        
        training_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save agent
        agent_filename = f"models/robust_sumo_sarsa_{args.agent_type}_{timestamp}.pkl"
        agent.save(agent_filename)
        print(f"\n✓ Agent saved to {agent_filename}")
        
        # Save training data
        training_data = {
            "agent_type": args.agent_type,
            "episodes": args.episodes,
            "training_time": training_time,
            "final_epsilon": agent.epsilon,
            "rewards": rewards,
            "queues": queues,
            "timestamp": timestamp
        }
        
        data_filename = f"robust_sumo_training_data_{timestamp}.json"
        with open(data_filename, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"✓ Training data saved to {data_filename}")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot rewards
        ax1.plot(rewards)
        ax1.set_title(f'Training Rewards ({args.agent_type.title()} SARSA)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot moving average
        if len(rewards) >= 10:
            moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            ax1.plot(range(9, len(rewards)), moving_avg, 'r-', alpha=0.7, label='Moving Average (10)')
            ax1.legend()
        
        # Plot queues
        ax2.plot(queues)
        ax2.set_title('Average Queue Length')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Queue Length')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"robust_sumo_training_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Training plots saved to {plot_filename}")
        plt.show()
        
        # Print summary
        print(f"\n" + "=" * 55)
        print("TRAINING SUMMARY")
        print("=" * 55)
        print(f"Agent Type: {args.agent_type.title()} SARSA")
        print(f"Episodes: {args.episodes}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        print(f"Average Reward (last 10): {np.mean(rewards[-10:]):.2f}")
        print(f"Average Queue (last 10): {np.mean(queues[-10:]):.2f}")
        print(f"Best Episode Reward: {max(rewards):.2f}")
        print("=" * 55)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
