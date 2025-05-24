#!/usr/bin/env python3
"""
SUMO GUI Demonstration Script

This script demonstrates how to use SUMO with GUI for visual simulation
and provides an interactive way to observe the SARSA agent learning.
"""

import sys
import os
import time
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.sarsa_agent import SarsaAgent


def demo_gui_training(episodes=5, fast_mode=False):
    """Demonstrate SARSA training with GUI visualization."""
    print("🎮 SUMO GUI DEMONSTRATION")
    print("=" * 50)
    print("This will show the SARSA agent learning to control traffic lights")
    print("with visual feedback through SUMO GUI.")
    print()
    print("Instructions:")
    print("- The GUI will show the traffic network")
    print("- Red/Green lights show current traffic light states")
    print("- Vehicle colors indicate speed (red=slow, green=fast)")
    print("- Queue lengths are visible as vehicle accumulations")
    print("- Press Enter between episodes to automatically start the next episode")
    print()
    
    try:
        from environment.traffic_env import TrafficEnvironment
        
        # Create environment with GUI
        env = TrafficEnvironment(
            sumo_cfg_file="config/simulation.sumocfg",
            use_gui=True,
            max_steps=200 if fast_mode else 400,
            keep_sumo_alive=True  # Keep SUMO alive between episodes for continuous viewing
        )
        
        # Create SARSA agent
        agent = SarsaAgent(
            state_size=27,
            action_size=4,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon=0.9,  # High exploration for demonstration
            epsilon_min=0.1,
            epsilon_decay=0.98
        )
        
        print(f"Starting GUI demonstration with {episodes} episodes...")
        print("Press Ctrl+C to stop early if needed.")
        print()
        
        for episode in range(episodes):
            print(f"🚦 Episode {episode + 1}/{episodes}")
            print("   Watch the GUI to see how the agent learns!")
            
            state, info = env.reset()
            total_reward = 0
            steps = 0
            
            # Get initial action
            action = agent.get_action(state, training=True)
            
            done = False
            while not done:
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Slow down for better visualization
                if not fast_mode:
                    time.sleep(0.1)
                
                if done or truncated:
                    agent.end_episode(reward)
                    break
                else:
                    # SARSA update
                    next_action = agent.get_action(next_state, training=True)
                    agent.update(reward, next_state, next_action, False)
                    
                    state = next_state
                    action = next_action
                
                # Print progress every 50 steps
                if steps % 50 == 0:
                    avg_queue = info.get('avg_queue_length', 0)
                    print(f"     Step {steps}: Reward={reward:.1f}, Queue={avg_queue:.1f}")
            
            avg_queue = info.get('avg_queue_length', 0)
            print(f"   ✓ Episode completed: Reward={total_reward:.1f}, "
                  f"Steps={steps}, Epsilon={agent.epsilon:.3f}")
            
            if episode < episodes - 1:
                print("   Press Enter to continue to next episode (or Ctrl+C to stop)...")
                try:
                    input()
                    # Automatically start the simulation for the next episode
                    print("   🎬 Starting next episode...")
                    env.start_simulation()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping demonstration...")
                    break
        
        env.close()
        print("\n✅ GUI demonstration completed!")
        
        # Save the demonstration agent
        os.makedirs("models", exist_ok=True)
        agent.save("models/demo_gui_sarsa.pkl")
        print("📁 Demo agent saved to models/demo_gui_sarsa.pkl")
        
    except Exception as e:
        print(f"❌ GUI demonstration failed: {e}")
        import traceback
        traceback.print_exc()


def demo_single_episode():
    """Run a single episode for quick GUI testing."""
    print("🔍 SINGLE EPISODE GUI TEST")
    print("=" * 30)
    
    try:
        from environment.traffic_env import TrafficEnvironment
        
        env = TrafficEnvironment(
            sumo_cfg_file="config/simulation.sumocfg",
            use_gui=True,
            max_steps=100,
            keep_sumo_alive=True  # Keep SUMO alive for single episode demo
        )
        
        # Simple random agent for testing
        print("Running single episode with random actions...")
        state, info = env.reset()
        
        for step in range(100):
            # Random action
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            
            time.sleep(0.2)  # Slow for observation
            
            if step % 20 == 0:
                print(f"Step {step}: Action={action}, Reward={reward:.1f}")
            
            if done or truncated:
                break
        
        env.close()
        print("✅ Single episode test completed!")
        
    except Exception as e:
        print(f"❌ Single episode test failed: {e}")


def main():
    """Main function for GUI demonstration."""
    parser = argparse.ArgumentParser(description='SUMO GUI Demonstration')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to demonstrate (default: 3)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode with shorter episodes')
    parser.add_argument('--single', action='store_true',
                       help='Run single episode test')
    
    args = parser.parse_args()
    
    if args.single:
        demo_single_episode()
    else:
        demo_gui_training(episodes=args.episodes, fast_mode=args.fast)


if __name__ == "__main__":
    main()
