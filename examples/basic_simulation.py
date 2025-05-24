"""
Basic simulation example for traffic light control.

This script demonstrates how to run a basic simulation with the traffic environment.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.traffic_env import TrafficEnvironment
from network.network_generator import generate_sumo_network
from network.traffic_generator import generate_traffic_flows
from utils.sumo_utils import create_sumo_config


def setup_simulation():
    """Setup SUMO simulation files."""
    print("Setting up SUMO simulation files...")
    
    # Generate network
    network_file = generate_sumo_network("config")
    print(f"Generated network: {network_file}")
    
    # Generate traffic flows
    routes_file = generate_traffic_flows("config", duration=1800)  # 30 minutes
    print(f"Generated routes: {routes_file}")
    
    # Create SUMO configuration
    config_file = create_sumo_config(
        network_file="network.net.xml",
        routes_file="routes.rou.xml", 
        output_file="config/simulation.sumocfg",
        end=1800
    )
    print(f"Generated config: {config_file}")
    
    return config_file


def run_random_simulation(steps: int = 1000, use_gui: bool = False):
    """Run simulation with random actions."""
    print(f"\\nRunning random simulation for {steps} steps...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=use_gui,
        max_steps=steps
    )
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        
        total_reward = 0
        step_count = 0
        
        # Run simulation
        for step in range(steps):
            # Take random action
            actions = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Total Queues={info.get('total_queues', 0)}, "
                      f"Phase Changes={info.get('phase_changes', 0)}")
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\\nSimulation completed!")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward: {total_reward/step_count:.2f}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()


def run_fixed_phase_simulation(steps: int = 1000, use_gui: bool = False):
    """Run simulation with fixed phases (no changes)."""
    print(f"\\nRunning fixed phase simulation for {steps} steps...")
    
    # Create environment
    env = TrafficEnvironment(
        sumo_cfg_file="config/simulation.sumocfg",
        use_gui=use_gui,
        max_steps=steps
    )
    
    try:
        # Reset environment
        obs, info = env.reset()
        
        total_reward = 0
        step_count = 0
        
        # Fixed actions (no phase changes)
        fixed_actions = np.array([0, 0, 0])  # All intersections stay in phase 0
        
        # Run simulation
        for step in range(steps):
            # Use fixed actions
            obs, reward, terminated, truncated, info = env.step(fixed_actions)
            
            total_reward += reward
            step_count += 1
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Total Queues={info.get('total_queues', 0)}")
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\\nFixed phase simulation completed!")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward: {total_reward/step_count:.2f}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()


def compare_strategies():
    """Compare different control strategies."""
    print("\\n" + "="*50)
    print("COMPARING TRAFFIC CONTROL STRATEGIES")
    print("="*50)
    
    strategies = [
        ("Random Actions", run_random_simulation),
        ("Fixed Phases", run_fixed_phase_simulation)
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"\\n--- {strategy_name} ---")
        strategy_func(steps=500, use_gui=False)


def main():
    """Main function."""
    print("Traffic Light Control - Basic Simulation Example")
    print("=" * 50)
    
    # Setup simulation files
    try:
        setup_simulation()
    except Exception as e:
        print(f"Error setting up simulation: {e}")
        print("\\nNote: You may need to install SUMO and ensure it's in your PATH")
        print("Visit: https://sumo.dlr.de/docs/Installing/index.html")
        return
    
    # Run demonstrations
    try:
        # Single simulation with GUI (if available)
        print("\\n--- Single Simulation (Random Actions) ---")
        run_random_simulation(steps=300, use_gui=False)  # Set to True for GUI
        
        # Compare strategies
        compare_strategies()
        
    except Exception as e:
        print(f"Error running simulations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
