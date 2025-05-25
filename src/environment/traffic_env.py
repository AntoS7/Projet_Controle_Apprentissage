"""
Traffic Environment for Reinforcement Learning.

This module implements a Gymnasium-compatible environment for traffic light control
using SUMO simulation with three interconnected intersections.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import os
import sys
from typing import Dict, List, Tuple, Optional
import subprocess
import time

from .intersection import Intersection, TrafficPhase


class TrafficEnvironment(gym.Env):
    """
    Traffic light control environment with three intersections.
    
    State space: Combined state of all three intersections
    Action space: Discrete actions for each intersection's phase
    Reward: Global queue minimization with phase change penalties
    """
    
    def __init__(self, 
                 sumo_cfg_file: str = "config/simulation.sumocfg",
                 use_gui: bool = False,
                 step_length: float = 1.0,
                 max_steps: int = 3600,
                 keep_sumo_alive: bool = True):
        """
        Initialize traffic environment.
        
        Args:
            sumo_cfg_file: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            step_length: Simulation step length in seconds
            max_steps: Maximum steps per episode
            keep_sumo_alive: Whether to keep SUMO running between episodes
        """
        super().__init__()
        
        self.sumo_cfg_file = sumo_cfg_file
        self.use_gui = use_gui
        self.step_length = step_length
        self.max_steps = max_steps
        self.keep_sumo_alive = keep_sumo_alive
        
        # Initialize intersections
        self.intersections = {
            'intersection_1': Intersection('intersection_1'),
            'intersection_2': Intersection('intersection_2'), 
            'intersection_3': Intersection('intersection_3')
        }
        
        self.intersection_ids = list(self.intersections.keys())
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Environment state
        self.current_step = 0
        self.sumo_running = False
        self.sumo_proc = None  # Initialize SUMO process reference
        self.previous_total_queues = {}
        
        # Reward parameters
        self.phase_change_penalty = -2.0  # Penalty for changing phase
        self.global_queue_weight = -1.0   # Weight for global queue length
        
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: one action per intersection (4 possible phases each)
        self.action_space = spaces.MultiDiscrete([4] * len(self.intersections))
        
        # Observation space: state vector for each intersection
        # Each intersection state: 4 queues + 4 phase encoding + 1 time = 9 values
        single_intersection_dim = 9
        total_obs_dim = single_intersection_dim * len(self.intersections)
        
        # Queue lengths can be 0-100, time is normalized, phases are 0-1
        single_intersection = [100.0] * 4 + [1.0] * 4 + [10.0]  # 9 values per intersection
        high = np.array([single_intersection for _ in range(len(self.intersections))]).flatten()
        
        single_intersection_low = [0.0] * 9  # 9 values per intersection
        low = np.array([single_intersection_low for _ in range(len(self.intersections))]).flatten()
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _start_sumo(self):
        """Start SUMO simulation with robust TraCI connection."""
        if self.sumo_running:
            self._close_sumo()
        
        try:
            # Use sumolib to find the correct SUMO binary
            import sumolib
            sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
            print(f"Using SUMO binary: {sumo_binary}")
            
            # Build command arguments
            sumo_cmd = [sumo_binary]
            sumo_cmd.extend(['-c', self.sumo_cfg_file])
            sumo_cmd.extend(['--step-length', str(self.step_length)])
            sumo_cmd.extend(['--waiting-time-memory', '1000'])
            sumo_cmd.extend(['--time-to-teleport', '-1'])
            
            # Use TraCI to start SUMO automatically (handles port assignment)
            print("Starting SUMO simulation...")
            traci.start(sumo_cmd)
            self.sumo_running = True
            print("Connected to SUMO successfully")
            
        except Exception as e:
            print(f"Failed to start SUMO with sumolib: {e}")
            print("Trying fallback method...")
            
            # Fallback: manual port assignment
            import socket
            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    s.listen(1)
                    port = s.getsockname()[1]
                return port
            
            port = find_free_port()
            
            # Try different SUMO binary paths
            possible_paths = [
                '/opt/homebrew/opt/sumo/bin/sumo',
                '/opt/homebrew/opt/sumo/share/sumo/bin/sumo',
                '/usr/local/bin/sumo',
                'sumo'  # System PATH
            ]
            
            sumo_binary = None
            for path in possible_paths:
                if path == 'sumo' or os.path.exists(path):
                    sumo_binary = path
                    break
            
            if not sumo_binary:
                raise RuntimeError("No working SUMO binary found")
            
            if self.use_gui:
                sumo_binary = sumo_binary.replace('sumo', 'sumo-gui')
            
            sumo_cmd = [sumo_binary]
            sumo_cmd.extend(['-c', self.sumo_cfg_file])
            sumo_cmd.extend(['--step-length', str(self.step_length)])
            sumo_cmd.extend(['--waiting-time-memory', '1000'])
            sumo_cmd.extend(['--time-to-teleport', '-1'])
            sumo_cmd.extend(['--remote-port', str(port)])
            
            # Start SUMO
            print(f"Starting SUMO on port {port} with binary: {sumo_binary}")
            self.sumo_proc = subprocess.Popen(sumo_cmd)
            time.sleep(3)  # Give SUMO more time to start
            
            # Connect to SUMO via TraCI with retry logic
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    traci.init(port=port, numRetries=3, host="localhost")
                    print(f"Connected to SUMO on port {port}")
                    self.sumo_running = True
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{max_retries}: Connection failed - {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise Exception(f"Failed to connect to SUMO after {max_retries} attempts")
        
        # Initialize traffic lights
        self._init_traffic_lights()
    
    def _reset_simulation_state(self):
        """Reset the simulation state without closing SUMO."""
        if not self.sumo_running:
            return
        
        try:
            # Load the simulation again to reset vehicle positions and traffic
            traci.load(['-c', self.sumo_cfg_file,
                       '--step-length', str(self.step_length),
                       '--waiting-time-memory', '1000',
                       '--time-to-teleport', '-1'])
            
            # Re-initialize traffic lights after reload
            self._init_traffic_lights()
            
            print("Reset simulation state (SUMO still running)")
            
        except Exception as e:
            print(f"Warning: Failed to reset simulation state: {e}")
            print("Falling back to full restart...")
            self._start_sumo()
    
    def _close_sumo(self):
        """Close SUMO simulation."""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
        
        # Terminate the SUMO process if it exists
        if hasattr(self, 'sumo_proc') and self.sumo_proc:
            try:
                self.sumo_proc.terminate()
                self.sumo_proc.wait(timeout=5)
            except:
                try:
                    self.sumo_proc.kill()
                except:
                    pass
            self.sumo_proc = None
    
    def _init_traffic_lights(self):
        """Initialize traffic light programs in SUMO."""
        # Get traffic light IDs from SUMO
        tl_ids = traci.trafficlight.getIDList()
        
        # Map our intersection IDs to SUMO traffic light IDs
        if len(tl_ids) >= 3:
            self.sumo_tl_mapping = {
                'intersection_1': tl_ids[0],
                'intersection_2': tl_ids[1], 
                'intersection_3': tl_ids[2]
            }
        else:
            # If not enough traffic lights, create mapping for available ones
            self.sumo_tl_mapping = {}
            for i, tl_id in enumerate(tl_ids):
                if i < len(self.intersection_ids):
                    self.sumo_tl_mapping[self.intersection_ids[i]] = tl_id
    
    def _get_queue_lengths(self) -> Dict[str, Dict[str, int]]:
        """Get queue lengths for all intersections from SUMO."""
        queue_data = {}
        
        for int_id, tl_id in self.sumo_tl_mapping.items():
            # Get lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # Group lanes by direction (approximate)
            direction_lanes = {
                'north': [lane for lane in controlled_lanes if 'n' in lane.lower()],
                'south': [lane for lane in controlled_lanes if 's' in lane.lower()],
                'east': [lane for lane in controlled_lanes if 'e' in lane.lower()],
                'west': [lane for lane in controlled_lanes if 'w' in lane.lower()]
            }
            
            # If direction detection fails, use first 4 lanes
            if not any(direction_lanes.values()):
                all_lanes = list(set(controlled_lanes))[:4]
                direction_lanes = {
                    'north': [all_lanes[0]] if len(all_lanes) > 0 else [],
                    'south': [all_lanes[1]] if len(all_lanes) > 1 else [],
                    'east': [all_lanes[2]] if len(all_lanes) > 2 else [],
                    'west': [all_lanes[3]] if len(all_lanes) > 3 else []
                }
            
            # Count vehicles waiting in each direction
            queues = {}
            for direction, lanes in direction_lanes.items():
                total_queue = 0
                for lane in lanes:
                    try:
                        # Get number of vehicles waiting on this lane
                        total_queue += traci.lane.getLastStepHaltingNumber(lane)
                    except:
                        pass
                queues[direction] = total_queue
            
            queue_data[int_id] = queues
            
        return queue_data
    
    def _set_traffic_light_phase(self, intersection_id: str, phase: TrafficPhase):
        """Set traffic light phase in SUMO."""
        if intersection_id not in self.sumo_tl_mapping:
            return
            
        tl_id = self.sumo_tl_mapping[intersection_id]
        
        # Map our phases to SUMO phases (simplified)
        sumo_phases = {
            TrafficPhase.NORTH_SOUTH_GREEN: 0,
            TrafficPhase.NORTH_SOUTH_YELLOW: 1,
            TrafficPhase.EAST_WEST_GREEN: 2, 
            TrafficPhase.EAST_WEST_YELLOW: 3
        }
        
        try:
            sumo_phase = sumo_phases[phase]
            traci.trafficlight.setPhase(tl_id, sumo_phase)
        except:
            pass
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        observations = []
        
        for intersection in self.intersections.values():
            obs = intersection.get_state()
            observations.append(obs)
            
        return np.concatenate(observations).astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on global queue lengths and phase changes."""
        total_reward = 0.0
        current_total_queues = {}
        
        # Calculate queue-based rewards for each intersection
        for int_id, intersection in self.intersections.items():
            current_queue = intersection.get_total_queue_length()
            current_total_queues[int_id] = current_queue
            
            # Reward for queue reduction
            if int_id in self.previous_total_queues:
                prev_queue = self.previous_total_queues[int_id]
                queue_reward = self.global_queue_weight * (current_queue - prev_queue)
                total_reward += queue_reward
            
            # Penalty for having queues
            total_reward += self.global_queue_weight * current_queue * 0.1
        
        # Store current queues for next step
        self.previous_total_queues = current_total_queues
        
        return total_reward
    
    def reset(self, seed: Optional[int] = None, force_restart: bool = False) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Start SUMO only if not running or if forced restart
        if not self.sumo_running or force_restart or not self.keep_sumo_alive:
            self._start_sumo()
        else:
            # If keeping SUMO alive, just reset the simulation state
            self._reset_simulation_state()
        
        # Reset intersections
        for intersection in self.intersections.values():
            intersection.current_phase = TrafficPhase.NORTH_SOUTH_GREEN
            intersection.time_since_phase_change = 0
            intersection.queues = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # Reset environment state
        self.current_step = 0
        self.previous_total_queues = {}
        
        # Let simulation run for a few steps to generate initial traffic
        for _ in range(10):
            if self.sumo_running:
                traci.simulationStep()
        
        # Update initial queue states
        if self.sumo_running:
            initial_queues = self._get_queue_lengths()
            for int_id, queues in initial_queues.items():
                if int_id in self.intersections:
                    self.intersections[int_id].update_queues(queues)
        
        observation = self._get_observation()
        info = {"step": self.current_step}
        
        return observation, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            actions: Array of actions for each intersection
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if not self.sumo_running:
            raise RuntimeError("SUMO not running. Call reset() first.")
        
        phase_changes = 0
        
        # Apply actions to intersections
        for i, (int_id, intersection) in enumerate(self.intersections.items()):
            if i < len(actions):
                new_phase = TrafficPhase(actions[i])
                
                # Check if phase change is requested and allowed
                if (new_phase != intersection.current_phase and 
                    intersection.can_change_phase()):
                    
                    if intersection.set_phase(new_phase):
                        phase_changes += 1
                        # Update SUMO traffic light
                        self._set_traffic_light_phase(int_id, new_phase)
        
        # Step SUMO simulation
        traci.simulationStep()
        
        # Update intersection states
        current_queues = self._get_queue_lengths()
        for int_id, queues in current_queues.items():
            if int_id in self.intersections:
                self.intersections[int_id].update_queues(queues)
                self.intersections[int_id].step()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Add penalty for phase changes
        reward += self.phase_change_penalty * phase_changes
        
        # Check if episode is done
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "phase_changes": phase_changes,
            "total_queues": sum(int.get_total_queue_length() 
                              for int in self.intersections.values())
        }
        
        return observation, reward, terminated, truncated, info
    
    def start_simulation(self):
        """Start the SUMO simulation (useful for GUI control)."""
        if self.sumo_running:
            try:
                # In GUI mode, ensure simulation is running
                if self.use_gui:
                    # Get current simulation state
                    sim_time = traci.simulation.getTime()
                    min_expected = traci.simulation.getMinExpectedNumber()
                    
                    # If simulation is paused or not started, this will resume it
                    # SUMO GUI automatically starts when we begin stepping
                    print(f"Simulation resumed at time {sim_time:.1f}s")
                    return True
            except Exception as e:
                print(f"Warning: Could not start simulation: {e}")
                return False
        else:
            print("Warning: SUMO not running. Call reset() first.")
            return False
    
    def pause_simulation(self):
        """Pause the SUMO simulation (useful for GUI control)."""
        if self.sumo_running and self.use_gui:
            try:
                # In GUI mode, we can't directly pause, but we can stop stepping
                # The GUI will naturally pause when we stop calling simulationStep()
                print("⏸️ Simulation paused (will resume on next step)")
                return True
            except Exception as e:
                print(f"Warning: Could not pause simulation: {e}")
                return False
        return False
    
    def close(self):
        """Close the environment."""
        self._close_sumo()
    
    def render(self, mode: str = 'human'):
        """Render the environment (SUMO handles visualization)."""
        if mode == 'human' and self.use_gui:
            # SUMO GUI is already rendering
            pass
        elif mode == 'rgb_array':
            # Could implement screenshot capture here
            pass
