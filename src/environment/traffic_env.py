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
from pathlib import Path
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
        
        # Enhanced reward parameters for better learning and traffic optimization
        self.phase_change_penalty = -2.0      # Moderate penalty for changing phase (optimized for responsiveness)
        self.global_queue_weight = -2.2       # Enhanced weight for global queue reduction
        self.flow_bonus_weight = 0.5          # Enhanced bonus for vehicles passing through
        self.waiting_time_penalty = -0.18     # Enhanced penalty for vehicle waiting time
        self.throughput_bonus = 0.6           # Enhanced bonus for intersection throughput
        self.congestion_penalty = -3.5        # Stronger penalty for severe congestion
        self.efficiency_bonus = 0.8           # Enhanced bonus for efficient phase timing
        
        # Advanced reward parameters (fine-tuned for better performance)
        self.queue_reduction_bonus = 2.5      # Enhanced bonus for actively reducing queues
        self.flow_momentum_bonus = 0.4        # Enhanced bonus for maintaining consistent flow
        self.coordination_bonus = 2.0         # Enhanced bonus for network-wide coordination
        self.stability_bonus = 0.7            # Enhanced bonus for stable traffic patterns
        self.emergency_penalty = -6.0         # Stronger penalty for critical congestion
        self.green_efficiency_bonus = 1.2     # Enhanced bonus for effective green light usage
        
        # New performance optimization parameters
        self.peak_hour_multiplier = 1.3       # Multiplier during high traffic periods
        self.smooth_flow_bonus = 0.8          # Bonus for smooth traffic flow patterns
        self.adaptive_threshold_bonus = 1.0   # Bonus for adaptive traffic management
        
        # Performance tracking for advanced rewards
        self.previous_throughput = {int_id: 0 for int_id in self.intersection_ids}
        self.phase_duration_tracker = {int_id: 0 for int_id in self.intersection_ids}
        self.throughput_history = {int_id: [] for int_id in self.intersection_ids}  # Track throughput trends
        self.queue_history = {int_id: [] for int_id in self.intersection_ids}      # Track queue trends
        self.phase_change_history = {int_id: 0 for int_id in self.intersection_ids} # Track phase changes
        
        # Enhanced performance tracking for optimization
        self.flow_consistency_tracker = {int_id: [] for int_id in self.intersection_ids}  # Track flow consistency
        self.congestion_prevention_score = {int_id: 0 for int_id in self.intersection_ids}  # Proactive congestion prevention
        self.network_harmony_score = 0     # Overall network coordination score
        self.peak_traffic_detector = []    # Detect peak traffic periods for adaptive rewards
        
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
        """
        Calculate enhanced multi-objective reward function for traffic optimization.
        
        Features:
        1. Progressive queue management with momentum tracking
        2. Enhanced throughput rewards with flow consistency
        3. Adaptive phase timing with green light efficiency
        4. Network-wide coordination and stability bonuses
        5. Emergency congestion prevention
        6. Real-time traffic flow optimization
        
        Returns:
            Float reward value (higher is better, typically -100 to +50)
        """
        total_reward = 0.0
        current_total_queues = {}
        
        # Collect comprehensive metrics for all intersections
        total_queue_length = 0
        total_throughput = 0
        total_waiting_time = 0
        severe_congestion_count = 0
        critical_congestion_count = 0
        active_intersections = 0
        
        # Network-wide performance tracking
        queue_improvements = 0
        throughput_improvements = 0
        stable_intersections = 0
        
        for int_id, intersection in self.intersections.items():
            current_queue = intersection.get_total_queue_length()
            current_total_queues[int_id] = current_queue
            total_queue_length += current_queue
            
            if current_queue > 0:
                active_intersections += 1
            
            # Update queue history for trend analysis
            if int_id not in self.queue_history:
                self.queue_history[int_id] = []
            self.queue_history[int_id].append(current_queue)
            if len(self.queue_history[int_id]) > 10:
                self.queue_history[int_id].pop(0)  # Keep only last 10 steps
            
            # ===== 1. ENHANCED QUEUE MANAGEMENT =====
            queue_reward = 0.0
            
            if int_id in self.previous_total_queues:
                prev_queue = self.previous_total_queues[int_id]
                queue_change = prev_queue - current_queue  # Positive when queue reduces
                
                # Base queue change reward with momentum
                if queue_change > 0:
                    # Reward for queue reduction with increasing returns
                    queue_reward += self.queue_reduction_bonus * (queue_change ** 1.1)
                    queue_improvements += 1
                elif queue_change < 0:
                    # Penalty for queue increase with escalating cost
                    queue_reward += self.global_queue_weight * (abs(queue_change) ** 1.2)
                
                # Progressive queue penalty based on current size
                if current_queue > 0:
                    if current_queue <= 5:
                        queue_penalty = -0.1 * current_queue
                    elif current_queue <= 15:
                        queue_penalty = -0.3 * current_queue
                    elif current_queue <= 25:
                        queue_penalty = -0.7 * current_queue
                    else:
                        queue_penalty = -1.5 * current_queue  # Heavy penalty for long queues
                    
                    queue_reward += queue_penalty
            
            total_reward += queue_reward
            
            # ===== 2. ENHANCED THROUGHPUT REWARDS =====
            throughput_reward = 0.0
            
            try:
                current_throughput = self._get_intersection_throughput(int_id)
                
                # Update throughput history
                if int_id not in self.throughput_history:
                    self.throughput_history[int_id] = []
                self.throughput_history[int_id].append(current_throughput)
                if len(self.throughput_history[int_id]) > 10:
                    self.throughput_history[int_id].pop(0)
                
                # Base throughput reward
                if current_throughput > 0:
                    throughput_reward += self.throughput_bonus * (current_throughput ** 0.8)
                    
                    # Flow momentum bonus for consistent throughput
                    if len(self.throughput_history[int_id]) >= 3:
                        recent_avg = np.mean(self.throughput_history[int_id][-3:])
                        if recent_avg > 0 and current_throughput >= recent_avg * 0.8:
                            momentum_bonus = self.flow_momentum_bonus * recent_avg
                            throughput_reward += momentum_bonus
                
                # Throughput improvement tracking
                if int_id in self.previous_throughput:
                    improvement = current_throughput - self.previous_throughput[int_id]
                    if improvement > 0:
                        throughput_improvements += 1
                        throughput_reward += 0.5 * improvement
                
                self.previous_throughput[int_id] = current_throughput
                total_throughput += current_throughput
                
            except Exception:
                pass  # Skip if TRACI data unavailable
            
            total_reward += throughput_reward
            
            # ===== 3. ENHANCED ADAPTIVE PHASE TIMING REWARDS =====
            phase_reward = 0.0
            
            self.phase_duration_tracker[int_id] += 1
            phase_duration = self.phase_duration_tracker[int_id]
            current_phase = intersection.current_phase
            
            # Advanced green light efficiency: reward based on vehicles served during green
            if (current_phase.name.endswith('GREEN') and current_throughput > 0):
                # Calculate efficiency ratio with improved formula
                queue_to_throughput_ratio = current_throughput / max(1, current_queue)
                efficiency_ratio = min(queue_to_throughput_ratio, 2.0)  # Cap at 2.0 for numerical stability
                
                # Base green efficiency bonus
                green_efficiency = self.green_efficiency_bonus * efficiency_ratio
                
                # Additional bonus for high-efficiency green phases
                if efficiency_ratio > 1.2:  # Very efficient green light usage
                    green_efficiency += 0.5 * (efficiency_ratio - 1.2)
                
                phase_reward += green_efficiency
                
                # Bonus for clearing queues during green phase
                if int_id in self.previous_total_queues:
                    queue_cleared = max(0, self.previous_total_queues[int_id] - current_queue)
                    if queue_cleared > 0:
                        phase_reward += 0.3 * queue_cleared  # Reward for queue clearance
            
            # Enhanced optimal phase duration rewards with dynamic thresholds
            queue_urgency_factor = min(current_queue / 20.0, 1.0)  # Scale urgency based on queue size
            
            if 3 <= phase_duration <= 8:  # Short, responsive timing
                phase_reward += self.efficiency_bonus * (0.3 + 0.2 * (1 - queue_urgency_factor))
            elif 8 <= phase_duration <= 20:  # Standard optimal timing
                phase_reward += self.efficiency_bonus * (0.5 + 0.1 * queue_urgency_factor)
            elif 20 <= phase_duration <= 35:  # Extended timing when needed
                if current_queue > 10:  # Justified by high queue
                    phase_reward += self.efficiency_bonus * (0.2 + 0.2 * queue_urgency_factor)
                else:
                    phase_reward -= 0.3 * (1 - queue_urgency_factor)  # Penalty for unnecessary long phases
            else:  # Excessive phase duration
                excess_penalty = 0.8 * (phase_duration - 35) if phase_duration > 35 else 0
                phase_reward -= excess_penalty * (1 + queue_urgency_factor)  # Stronger penalty when queues are high
            
            total_reward += phase_reward
            
            # ===== 4. CONGESTION CLASSIFICATION AND PENALTIES =====
            if current_queue > 30:  # Critical congestion
                critical_congestion_count += 1
                critical_penalty = self.emergency_penalty * ((current_queue - 30) ** 1.3)
                total_reward += critical_penalty
            elif current_queue > 20:  # Severe congestion
                severe_congestion_count += 1
                severe_penalty = self.congestion_penalty * (current_queue - 20)
                total_reward += severe_penalty
            
            # ===== 5. WAITING TIME OPTIMIZATION =====
            try:
                avg_waiting_time = self._get_average_waiting_time(int_id)
                if avg_waiting_time > 0:
                    # Progressive waiting time penalty
                    if avg_waiting_time <= 30:
                        waiting_penalty = self.waiting_time_penalty * avg_waiting_time
                    else:
                        waiting_penalty = self.waiting_time_penalty * (avg_waiting_time ** 1.3)
                    
                    total_reward += waiting_penalty
                    total_waiting_time += avg_waiting_time
            except Exception:
                pass
            
            # ===== 6. STABILITY TRACKING =====
            if len(self.queue_history[int_id]) >= 5:
                queue_variance = np.var(self.queue_history[int_id][-5:])
                if queue_variance < 4.0:  # Low variance indicates stability
                    stable_intersections += 1
        
        # ===== 7. NETWORK-WIDE COORDINATION BONUSES =====
        
        # Multi-intersection flow coordination
        if total_throughput > 0:
            # Network flow bonus with logarithmic scaling
            network_flow_bonus = self.flow_bonus_weight * np.log(1 + total_throughput)
            total_reward += network_flow_bonus
            
            # Coordination bonus for multiple intersections flowing well
            if throughput_improvements >= 2:
                coordination_reward = self.coordination_bonus * throughput_improvements
                total_reward += coordination_reward
        
        # Network stability bonus
        if stable_intersections >= 2:
            stability_reward = self.stability_bonus * stable_intersections
            total_reward += stability_reward
        
        # Enhanced emergency-free network bonus with performance tiers
        if critical_congestion_count == 0:
            if severe_congestion_count == 0:
                if total_queue_length < 15:
                    # Exceptional network performance
                    total_reward += 7.0 + (2.0 * total_throughput / max(1, total_queue_length + 1))
                elif total_queue_length < 30:
                    # Excellent network performance
                    total_reward += 4.0 + (1.0 * total_throughput / max(1, total_queue_length + 1))
                elif total_queue_length < 50:
                    # Good network performance
                    total_reward += 2.0
            elif severe_congestion_count == 1 and total_queue_length < 60:
                # Acceptable with minor issues - scaled by network size
                total_reward += 1.0 * (1 - severe_congestion_count / len(self.intersection_ids))
            elif severe_congestion_count <= 2 and total_queue_length < 80:
                # Manageable congestion
                total_reward += 0.5 * (1 - severe_congestion_count / len(self.intersection_ids))
        else:
            # Penalty scaling based on critical congestion severity
            critical_penalty_multiplier = min(critical_congestion_count / len(self.intersection_ids), 1.0)
            total_reward -= 3.0 * critical_penalty_multiplier
        
        # ===== 8. BALANCED QUEUE DISTRIBUTION =====
        if len(current_total_queues) > 1 and total_queue_length > 0:
            queue_values = list(current_total_queues.values())
            queue_std = np.std(queue_values)
            queue_mean = np.mean(queue_values)
            
            if queue_mean > 0:
                balance_ratio = queue_std / queue_mean
                if balance_ratio < 0.4:  # Well-balanced queues
                    balance_bonus = 2.0 * (0.4 - balance_ratio)
                    total_reward += balance_bonus
                elif balance_ratio > 1.5:  # Very unbalanced queues
                    balance_penalty = -1.0 * (balance_ratio - 1.5)
                    total_reward += balance_penalty
        
        # ===== 9. ENHANCED EFFICIENCY METRICS =====
        if active_intersections > 0:
            # Overall network efficiency
            efficiency_ratio = total_throughput / max(1, total_queue_length + 1)
            if efficiency_ratio > 0.3:  # High efficiency threshold
                efficiency_bonus = 3.0 * (efficiency_ratio - 0.3)
                total_reward += efficiency_bonus
            
            # Peak traffic detection and adaptive rewards
            total_traffic_density = total_queue_length + total_throughput
            self.peak_traffic_detector.append(total_traffic_density)
            if len(self.peak_traffic_detector) > 20:
                self.peak_traffic_detector.pop(0)
            
            # Adaptive reward multiplier during peak hours
            if len(self.peak_traffic_detector) >= 10:
                avg_traffic = np.mean(self.peak_traffic_detector[-10:])
                if total_traffic_density > avg_traffic * 1.5:  # Peak traffic detected
                    total_reward *= self.peak_hour_multiplier
        
        # ===== 10. PROACTIVE CONGESTION PREVENTION =====
        prevention_bonus = 0.0
        for int_id in self.intersection_ids:
            current_queue = current_total_queues[int_id]
            
            # Reward early intervention to prevent congestion
            if 8 <= current_queue <= 15:  # Moderate queue - good time to act
                if int_id in self.previous_total_queues:
                    prev_queue = self.previous_total_queues[int_id]
                    if current_queue < prev_queue:  # Successfully preventing escalation
                        prevention_bonus += self.adaptive_threshold_bonus * (15 - current_queue) / 7
                        self.congestion_prevention_score[int_id] += 1
                    elif current_queue > prev_queue:  # Failing to prevent escalation
                        prevention_bonus -= 0.5
                        self.congestion_prevention_score[int_id] = max(0, self.congestion_prevention_score[int_id] - 1)
        
        total_reward += prevention_bonus
        
        # ===== 11. NETWORK HARMONY AND FLOW CONSISTENCY =====
        harmony_bonus = 0.0
        consistent_intersections = 0
        
        for int_id in self.intersection_ids:
            # Track flow consistency
            if int_id in self.previous_throughput:
                current_throughput = self.previous_throughput[int_id]
                if int_id not in self.flow_consistency_tracker:
                    self.flow_consistency_tracker[int_id] = []
                
                self.flow_consistency_tracker[int_id].append(current_throughput)
                if len(self.flow_consistency_tracker[int_id]) > 8:
                    self.flow_consistency_tracker[int_id].pop(0)
                
                # Reward consistent flow patterns
                if len(self.flow_consistency_tracker[int_id]) >= 5:
                    flow_std = np.std(self.flow_consistency_tracker[int_id][-5:])
                    flow_mean = np.mean(self.flow_consistency_tracker[int_id][-5:])
                    
                    if flow_mean > 0 and flow_std / flow_mean < 0.4:  # Low coefficient of variation
                        consistent_intersections += 1
                        harmony_bonus += self.smooth_flow_bonus * (flow_mean / max(1, flow_std))
        
        # Network-wide harmony bonus
        if consistent_intersections >= 2:
            network_harmony = consistent_intersections / len(self.intersection_ids)
            self.network_harmony_score = network_harmony
            harmony_bonus += self.coordination_bonus * network_harmony
        
        total_reward += harmony_bonus
        
        # Store current state for next iteration
        self.previous_total_queues = current_total_queues
        
        # Ensure reward is bounded for numerical stability
        total_reward = max(-200.0, min(100.0, total_reward))
        
        return total_reward
    
    def _get_intersection_throughput(self, intersection_id: str) -> int:
        """Get number of vehicles that passed through intersection in last step."""
        try:
            # Get all edges connected to this intersection
            tl_id = intersection_id
            if traci.trafficlight.getIDList() and tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                throughput = 0
                for lane in controlled_lanes:
                    # Count vehicles that passed through (left the lane)
                    throughput += traci.lane.getLastStepVehicleNumber(lane)
                return throughput
            return 0
        except:
            return 0
    
    def _get_average_waiting_time(self, intersection_id: str) -> float:
        """Get average waiting time for vehicles near intersection."""
        try:
            tl_id = intersection_id
            if traci.trafficlight.getIDList() and tl_id in traci.trafficlight.getIDList():
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                total_waiting = 0
                total_vehicles = 0
                
                for lane in controlled_lanes:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for veh_id in vehicles:
                        try:
                            waiting_time = traci.vehicle.getWaitingTime(veh_id)
                            total_waiting += waiting_time
                            total_vehicles += 1
                        except:
                            continue
                
                return total_waiting / max(1, total_vehicles)
            return 0.0
        except:
            return 0.0
    
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
        
        # Reset environment state with enhanced tracking
        self.current_step = 0
        self.previous_total_queues = {}
        self.previous_throughput = {int_id: 0 for int_id in self.intersection_ids}
        self.phase_duration_tracker = {int_id: 0 for int_id in self.intersection_ids}
        
        # Reset enhanced tracking variables
        self.throughput_history = {int_id: [] for int_id in self.intersection_ids}
        self.queue_history = {int_id: [] for int_id in self.intersection_ids}
        self.phase_change_history = {int_id: 0 for int_id in self.intersection_ids}
        self.flow_consistency_tracker = {int_id: [] for int_id in self.intersection_ids}
        self.congestion_prevention_score = {int_id: 0 for int_id in self.intersection_ids}
        self.network_harmony_score = 0
        self.peak_traffic_detector = []
        
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
                        # Reset phase duration tracker when phase changes
                        self.phase_duration_tracker[int_id] = 0
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
        
        # Enhanced info with detailed metrics
        total_queues = sum(int.get_total_queue_length() for int in self.intersections.values())
        total_throughput = sum(self.previous_throughput.values())
        
        info = {
            "step": self.current_step,
            "phase_changes": phase_changes,
            "total_queues": total_queues,
            "total_throughput": total_throughput,
            "individual_queues": {int_id: self.intersections[int_id].get_total_queue_length() 
                                for int_id in self.intersection_ids},
            "phase_durations": dict(self.phase_duration_tracker),
            "network_efficiency": total_throughput / max(1, total_queues + 1),  # Throughput per queue unit
            "congestion_level": "high" if total_queues > 50 else "medium" if total_queues > 20 else "low",
            "network_harmony_score": getattr(self, 'network_harmony_score', 0),  # Network coordination score
            "congestion_prevention_scores": dict(self.congestion_prevention_score),  # Proactive prevention scores
            "peak_traffic_detected": len(self.peak_traffic_detector) >= 10 and 
                                   (total_queues + total_throughput) > np.mean(self.peak_traffic_detector[-10:]) * 1.5 
                                   if len(self.peak_traffic_detector) >= 10 else False,
            "flow_consistency": {int_id: np.std(history[-5:]) / max(1, np.mean(history[-5:])) 
                               for int_id, history in self.flow_consistency_tracker.items() 
                               if len(history) >= 5},  # Coefficient of variation for flow consistency
            "reward_components": {
                "queue_management": "enhanced",
                "throughput_optimization": "enhanced", 
                "phase_timing": "adaptive",
                "network_coordination": "multi-objective",
                "congestion_prevention": "proactive"
            }
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
