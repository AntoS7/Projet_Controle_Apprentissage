#!/usr/bin/env python3
"""
Complete SUMO Setup and Training Script

This script sets up the complete SUMO environment and trains SARSA agents,
handling all necessary file generation and SUMO configuration.
"""

import sys
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.sarsa_agent import SarsaAgent, AdaptiveSarsaAgent


def check_sumo_installation():
    """Check if SUMO is properly installed."""
    try:
        # Try to run netconvert to check installation
        result = subprocess.run(['netconvert', '--help'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def generate_simple_sumo_network():
    """Generate a simple SUMO network directly without external dependencies."""
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    
    print("Generating simple SUMO network files...")
    
    # Create nodes file
    nodes_content = '''<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="intersection_1" x="0" y="0" type="traffic_light"/>
    <node id="intersection_2" x="500" y="0" type="traffic_light"/>
    <node id="intersection_3" x="250" y="433" type="traffic_light"/>
    
    <node id="north_1" x="0" y="200" type="priority"/>
    <node id="south_1" x="0" y="-200" type="priority"/>
    <node id="east_1" x="-200" y="0" type="priority"/>
    <node id="west_1" x="200" y="0" type="priority"/>
    
    <node id="north_2" x="500" y="200" type="priority"/>
    <node id="south_2" x="500" y="-200" type="priority"/>
    <node id="east_2" x="300" y="0" type="priority"/>
    <node id="west_2" x="700" y="0" type="priority"/>
    
    <node id="north_3" x="250" y="633" type="priority"/>
    <node id="south_3" x="250" y="233" type="priority"/>
    <node id="east_3" x="50" y="433" type="priority"/>
    <node id="west_3" x="450" y="433" type="priority"/>
</nodes>'''
    
    # Create edges file
    edges_content = '''<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Intersection 1 connections -->
    <edge id="north_1_to_int1" from="north_1" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="int1_to_south_1" from="intersection_1" to="south_1" numLanes="2" speed="13.89"/>
    <edge id="east_1_to_int1" from="east_1" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="int1_to_west_1" from="intersection_1" to="west_1" numLanes="2" speed="13.89"/>
    
    <!-- Intersection 2 connections -->
    <edge id="north_2_to_int2" from="north_2" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="int2_to_south_2" from="intersection_2" to="south_2" numLanes="2" speed="13.89"/>
    <edge id="east_2_to_int2" from="east_2" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="int2_to_west_2" from="intersection_2" to="west_2" numLanes="2" speed="13.89"/>
    
    <!-- Intersection 3 connections -->
    <edge id="north_3_to_int3" from="north_3" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="int3_to_south_3" from="intersection_3" to="south_3" numLanes="2" speed="13.89"/>
    <edge id="east_3_to_int3" from="east_3" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="int3_to_west_3" from="intersection_3" to="west_3" numLanes="2" speed="13.89"/>
    
    <!-- Inter-intersection connections -->
    <edge id="int1_to_int2" from="intersection_1" to="intersection_2" numLanes="2" speed="13.89"/>
    <edge id="int2_to_int1" from="intersection_2" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="int1_to_int3" from="intersection_1" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="int3_to_int1" from="intersection_3" to="intersection_1" numLanes="2" speed="13.89"/>
    <edge id="int2_to_int3" from="intersection_2" to="intersection_3" numLanes="2" speed="13.89"/>
    <edge id="int3_to_int2" from="intersection_3" to="intersection_2" numLanes="2" speed="13.89"/>
</edges>'''
    
    # Create routes file with traffic flows
    routes_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle types -->
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
    
    <!-- Routes through the network -->
    <route id="north_south_1" edges="north_1_to_int1 int1_to_south_1"/>
    <route id="east_west_1" edges="east_1_to_int1 int1_to_west_1"/>
    <route id="north_south_2" edges="north_2_to_int2 int2_to_south_2"/>
    <route id="east_west_2" edges="east_2_to_int2 int2_to_west_2"/>
    <route id="north_south_3" edges="north_3_to_int3 int3_to_south_3"/>
    <route id="east_west_3" edges="east_3_to_int3 int3_to_west_3"/>
    
    <!-- Inter-intersection routes -->
    <route id="int1_to_int2_route" edges="north_1_to_int1 int1_to_int2 int2_to_south_2"/>
    <route id="int2_to_int3_route" edges="north_2_to_int2 int2_to_int3 int3_to_south_3"/>
    <route id="int3_to_int1_route" edges="north_3_to_int3 int3_to_int1 int1_to_south_1"/>
    
    <!-- Traffic flows -->
    <flow id="flow_ns1" route="north_south_1" begin="0" end="1200" vehsPerHour="300" type="passenger"/>
    <flow id="flow_ew1" route="east_west_1" begin="0" end="1200" vehsPerHour="200" type="passenger"/>
    <flow id="flow_ns2" route="north_south_2" begin="0" end="1200" vehsPerHour="250" type="passenger"/>
    <flow id="flow_ew2" route="east_west_2" begin="0" end="1200" vehsPerHour="180" type="passenger"/>
    <flow id="flow_ns3" route="north_south_3" begin="0" end="1200" vehsPerHour="200" type="passenger"/>
    <flow id="flow_ew3" route="east_west_3" begin="0" end="1200" vehsPerHour="150" type="passenger"/>
    
    <!-- Inter-intersection flows -->
    <flow id="flow_12" route="int1_to_int2_route" begin="0" end="1200" vehsPerHour="100" type="passenger"/>
    <flow id="flow_23" route="int2_to_int3_route" begin="0" end="1200" vehsPerHour="80" type="passenger"/>
    <flow id="flow_31" route="int3_to_int1_route" begin="0" end="1200" vehsPerHour="90" type="passenger"/>
</routes>'''
    
    # Write files
    with open(os.path.join(config_dir, "network.nod.xml"), 'w') as f:
        f.write(nodes_content)
    
    with open(os.path.join(config_dir, "network.edg.xml"), 'w') as f:
        f.write(edges_content)
    
    with open(os.path.join(config_dir, "routes.rou.xml"), 'w') as f:
        f.write(routes_content)
    
    print("✓ Basic network files created")
    return True


def build_sumo_network():
    """Build the SUMO network using netconvert."""
    config_dir = "config"
    
    if not check_sumo_installation():
        print("⚠ SUMO not found or not properly installed")
        return False
    
    try:
        print("Building SUMO network with netconvert...")
        
        # Run netconvert to create network
        cmd = [
            'netconvert',
            '--node-files', os.path.join(config_dir, 'network.nod.xml'),
            '--edge-files', os.path.join(config_dir, 'network.edg.xml'),
            '--output-file', os.path.join(config_dir, 'network.net.xml'),
            '--tls.guess', 'true',
            '--tls.cycle.time', '60'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Network built successfully")
            return True
        else:
            print(f"✗ netconvert failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ netconvert timed out")
        return False
    except Exception as e:
        print(f"✗ Error running netconvert: {e}")
        return False


def create_sumo_config():
    """Create SUMO configuration file."""
    config_dir = "config"
    
    config_content = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1200"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>'''
    
    config_path = os.path.join(config_dir, "simulation.sumocfg")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("✓ SUMO configuration created")
    return config_path


def train_with_real_sumo(use_gui=False, episodes=30, gui_options=None):
    """Train SARSA agent with real SUMO simulation."""
    print("\n" + "="*50)
    print("TRAINING WITH REAL SUMO SIMULATION")
    if use_gui:
        print("GUI MODE ENABLED - Visual simulation")
        if gui_options:
            print(f"   Speed: {gui_options.get('speed', 1.0)}x")
            print(f"   Step mode: {'ON' if gui_options.get('step_mode') else 'OFF'}")
            print(f"   Auto-start: {'ON' if gui_options.get('auto_start', True) else 'OFF'}")
    else:
        print("HEADLESS MODE - Fast training")
    print("="*50)
    
    try:
        from environment.traffic_env import TrafficEnvironment
        
        # Create environment with enhanced GUI options
        env = TrafficEnvironment(
            sumo_cfg_file="config/simulation.sumocfg",
            use_gui=use_gui,
            max_steps=400,
            keep_sumo_alive=use_gui  # Keep SUMO alive between episodes in GUI mode
        )
        
        # Create SARSA agent  
        agent = SarsaAgent(
            state_size=27,
            action_size=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.7,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        print(f"Starting SUMO training for {episodes} episodes...")
        rewards_history = []
        queues_history = []
        
        # GUI performance tracking
        if use_gui:
            episode_start_times = []
            performance_metrics = {
                'avg_rewards': [],
                'avg_queues': [],
                'learning_progress': []
            }
        
        for episode in range(episodes):
            episode_start_time = time.time()
            print(f"Episode {episode + 1}/{episodes}")
            
            state, info = env.reset()
            total_reward = 0
            total_queues = 0
            steps = 0
            
            # Get initial action
            action = agent.get_action(state, training=True)
            
            done = False
            while not done:
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Track queues
                if 'avg_queue_length' in info:
                    total_queues += info['avg_queue_length']
                
                # Enhanced GUI visualization controls
                if use_gui:
                    step_delay = 0.1 / gui_options.get('speed', 1.0) if gui_options else 0.1
                    
                    # Step-by-step mode
                    if gui_options and gui_options.get('step_mode') and steps % gui_options.get('pause_freq', 50) == 0:
                        print(f"     Step {steps}: Reward={reward:.1f}, Queue={info.get('avg_queue_length', 0):.1f}")
                        print("     Press Enter to continue...")
                        input()
                    elif steps % 5 == 0:  # Regular slow down
                        time.sleep(step_delay)
                
                if done or truncated:
                    agent.end_episode(reward)
                    break
                else:
                    # SARSA update
                    next_action = agent.get_action(next_state, training=True)
                    agent.update(reward, next_state, next_action, False)
                    
                    state = next_state
                    action = next_action
            
            avg_queue = total_queues / max(1, steps)
            rewards_history.append(total_reward)
            queues_history.append(avg_queue)
            
            # Enhanced episode summary for GUI
            episode_time = time.time() - episode_start_time
            print(f"  Reward: {total_reward:.1f}, Avg Queue: {avg_queue:.1f}, "
                  f"Steps: {steps}, Epsilon: {agent.epsilon:.3f}, Time: {episode_time:.1f}s")
            
            # GUI performance tracking
            if use_gui:
                episode_start_times.append(episode_start_time)
                recent_rewards = rewards_history[-min(5, len(rewards_history)):]
                recent_queues = queues_history[-min(5, len(queues_history)):]
                
                performance_metrics['avg_rewards'].append(np.mean(recent_rewards))
                performance_metrics['avg_queues'].append(np.mean(recent_queues))
                performance_metrics['learning_progress'].append(agent.epsilon)
                
                print(f"  Recent Performance: Avg Reward={np.mean(recent_rewards):.1f}, "
                      f"Avg Queue={np.mean(recent_queues):.1f}")
            
            # Enhanced episode transitions for GUI
            if use_gui and episode < episodes - 1:
                auto_start = not gui_options or gui_options.get('auto_start', True)
                if auto_start:
                    print("  Press Enter to continue to next episode...")
                    input()
                    print("  Starting next episode...")
                    env.start_simulation()
                else:
                    print("  Manual mode: Configure SUMO GUI and press Enter when ready...")
                    input()
        
        env.close()
        
        # Save agent
        os.makedirs("models", exist_ok=True)
        model_name = f"real_sumo_sarsa_{'gui' if use_gui else 'headless'}.pkl"
        agent.save(f"models/{model_name}")
        
        # Plot results
        title = f"Real SUMO Training ({'GUI' if use_gui else 'Headless'})"
        plot_training_results(rewards_history, queues_history, title)
        
        print("✓ Real SUMO training completed successfully!")
        return agent, rewards_history, queues_history
        
    except Exception as e:
        print(f"✗ Real SUMO training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []


def plot_training_results(rewards, queues, title):
    """Plot training results."""
    plt.figure(figsize=(12, 4))
    
    # Rewards plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'b-', alpha=0.7, label='Episode Reward')
    if len(rewards) > 5:
        window = min(5, len(rewards))
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    plt.title(f'{title} - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Queue plot
    plt.subplot(1, 2, 2)
    plt.plot(queues, 'g-', alpha=0.7, label='Avg Queue Length')
    plt.title(f'{title} - Queue Performance')
    plt.xlabel('Episode')
    plt.ylabel('Avg Queue Length')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot to plots directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(plots_dir, f'sumo_training_{timestamp}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {filename}")
    plt.show()


def main():
    """Main function to setup and run SUMO training."""
    parser = argparse.ArgumentParser(description='Complete SUMO SARSA Training')
    parser.add_argument('--gui', action='store_true', 
                       help='Enable SUMO GUI for visual simulation')
    parser.add_argument('--episodes', type=int, default=30,
                       help='Number of training episodes (default: 30)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode with fewer episodes (10)')
    # Enhanced GUI options
    parser.add_argument('--gui-speed', type=float, default=1.0,
                       help='GUI simulation speed multiplier (0.1=slow, 2.0=fast)')
    parser.add_argument('--gui-step', action='store_true',
                       help='Step-by-step GUI mode with manual control')
    parser.add_argument('--gui-pause-freq', type=int, default=50,
                       help='Pause frequency in GUI mode (steps between pauses)')
    parser.add_argument('--gui-auto-start', action='store_true', default=True,
                       help='Automatically start next episodes in GUI mode')
    
    args = parser.parse_args()
    
    # Adjust episodes for fast mode
    if args.fast:
        episodes = 10
        print("FAST MODE: Running 10 episodes")
    else:
        episodes = args.episodes
    
    print("COMPLETE SUMO SETUP AND SARSA TRAINING")
    print("="*60)
    
    if args.gui:
        print("GUI MODE: Visual simulation enabled")
        print(f"   - Simulation speed: {args.gui_speed}x")
        if args.gui_step:
            print("   - Step-by-step mode: Manual control enabled")
        else:
            print("   - Episodes will pause between runs for observation")
        if args.gui_auto_start:
            print("   - Auto-start: Press Enter to automatically start episodes")
        print("   - Enhanced visual feedback and controls")
    else:
        print("HEADLESS MODE: Fast training without visualization")
    
    # Step 1: Generate network files
    print("\n1. Setting up SUMO network...")
    if not generate_simple_sumo_network():
        print("✗ Failed to generate network files")
        return
    
    # Step 2: Build network with netconvert
    print("\n2. Building SUMO network...")
    if build_sumo_network():
        print("✓ SUMO network built successfully")
    else:
        print("⚠ Could not build network with netconvert, but continuing...")
    
    # Step 3: Create configuration
    print("\n3. Creating SUMO configuration...")
    config_path = create_sumo_config()
    
    # Step 4: Check if network file exists
    network_file = "config/network.net.xml"
    if os.path.exists(network_file):
        print(f"✓ Network file exists: {network_file}")
        
        # Step 5: Train with real SUMO
        gui_options = {
            'speed': args.gui_speed,
            'step_mode': args.gui_step,
            'pause_freq': args.gui_pause_freq,
            'auto_start': args.gui_auto_start
        }
        
        agent, rewards, queues = train_with_real_sumo(
            use_gui=args.gui, 
            episodes=episodes,
            gui_options=gui_options
        )
        
        if agent is not None:
            print("\n" + "="*60)
            print("REAL SUMO TRAINING COMPLETED!")
            print("="*60)
            agent.print_statistics()
            
            if len(rewards) > 0:
                print(f"Final performance:")
                print(f"  Average reward (last 5): {np.mean(rewards[-5:]):.1f}")
                print(f"  Average queue (last 5): {np.mean(queues[-5:]):.1f}")
                print(f"  Best reward: {max(rewards):.1f}")
                print(f"  Improvement: {rewards[-1] - rewards[0]:.1f}")
        else:
            print("✗ SUMO training failed")
    else:
        print(f"✗ Network file not found: {network_file}")
        print("This might be because SUMO/netconvert is not properly installed.")


if __name__ == "__main__":
    main()
