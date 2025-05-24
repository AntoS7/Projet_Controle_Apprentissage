"""
Intersection model for traffic light control.

This module defines the Intersection class that models a single traffic light
intersection with multiple phases and lane queues.
"""

import numpy as np
from typing import List, Dict, Tuple
from enum import Enum


class TrafficPhase(Enum):
    """Traffic light phases for a 4-way intersection."""
    NORTH_SOUTH_GREEN = 0    # North-South green, East-West red
    NORTH_SOUTH_YELLOW = 1   # North-South yellow, East-West red
    EAST_WEST_GREEN = 2      # East-West green, North-South red
    EAST_WEST_YELLOW = 3     # East-West yellow, North-South red


class Intersection:
    """
    Models a single traffic intersection with traffic lights.
    
    Each intersection has:
    - Queue lengths for different incoming lanes
    - Current traffic light phase
    - Time since last phase change
    - Ability to change phases with minimum timing constraints
    """
    
    def __init__(self, intersection_id: str, min_phase_duration: int = 10):
        """
        Initialize intersection.
        
        Args:
            intersection_id: Unique identifier for this intersection
            min_phase_duration: Minimum time (seconds) a phase must be active
        """
        self.id = intersection_id
        self.min_phase_duration = min_phase_duration
        
        # Current state
        self.current_phase = TrafficPhase.NORTH_SOUTH_GREEN
        self.time_since_phase_change = 0
        
        # Queue lengths for each direction (vehicles waiting)
        self.queues = {
            'north': 0,
            'south': 0,
            'east': 0,
            'west': 0
        }
        
        # Traffic light program - defines which lanes have green light
        self.phase_config = {
            TrafficPhase.NORTH_SOUTH_GREEN: {
                'north': 'green',
                'south': 'green', 
                'east': 'red',
                'west': 'red'
            },
            TrafficPhase.NORTH_SOUTH_YELLOW: {
                'north': 'yellow',
                'south': 'yellow',
                'east': 'red', 
                'west': 'red'
            },
            TrafficPhase.EAST_WEST_GREEN: {
                'north': 'red',
                'south': 'red',
                'east': 'green',
                'west': 'green'
            },
            TrafficPhase.EAST_WEST_YELLOW: {
                'north': 'red',
                'south': 'red', 
                'east': 'yellow',
                'west': 'yellow'
            }
        }
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        Returns:
            State vector containing:
            - Queue lengths (4 values)
            - Current phase (4 one-hot encoded values)
            - Time since phase change (1 normalized value)
        """
        # Queue lengths
        queue_state = np.array([
            self.queues['north'],
            self.queues['south'], 
            self.queues['east'],
            self.queues['west']
        ])
        
        # One-hot encode current phase
        phase_state = np.zeros(4)
        phase_state[self.current_phase.value] = 1.0
        
        # Normalized time since phase change (normalize by min_phase_duration)
        time_state = np.array([self.time_since_phase_change / self.min_phase_duration])
        
        return np.concatenate([queue_state, phase_state, time_state])
    
    def can_change_phase(self) -> bool:
        """Check if phase can be changed based on minimum duration."""
        return self.time_since_phase_change >= self.min_phase_duration
    
    def set_phase(self, new_phase: TrafficPhase) -> bool:
        """
        Attempt to set new traffic light phase.
        
        Args:
            new_phase: Desired traffic phase
            
        Returns:
            True if phase was changed, False if not allowed yet
        """
        if new_phase == self.current_phase:
            return False
            
        if not self.can_change_phase():
            return False
            
        self.current_phase = new_phase
        self.time_since_phase_change = 0
        return True
    
    def update_queues(self, new_queues: Dict[str, int]):
        """Update queue lengths from SUMO simulation."""
        self.queues.update(new_queues)
    
    def step(self, dt: int = 1):
        """
        Update intersection state by one time step.
        
        Args:
            dt: Time step duration in seconds
        """
        self.time_since_phase_change += dt
    
    def get_total_queue_length(self) -> int:
        """Get total number of vehicles waiting at this intersection."""
        return sum(self.queues.values())
    
    def get_green_lanes(self) -> List[str]:
        """Get list of lanes currently having green light."""
        current_config = self.phase_config[self.current_phase]
        return [lane for lane, light in current_config.items() if light == 'green']
    
    def get_phase_reward(self, previous_total_queue: int) -> float:
        """
        Calculate reward for current phase.
        
        Args:
            previous_total_queue: Queue length from previous time step
            
        Returns:
            Reward value (negative for more queues, positive for fewer)
        """
        current_total_queue = self.get_total_queue_length()
        
        # Base reward: reduction in total queue length
        queue_reduction_reward = previous_total_queue - current_total_queue
        
        # Penalty for very long queues (non-linear penalty)
        queue_penalty = -0.1 * (current_total_queue ** 1.5)
        
        return queue_reduction_reward + queue_penalty
    
    def __str__(self) -> str:
        """String representation of intersection state."""
        return (f"Intersection {self.id}: Phase={self.current_phase.name}, "
                f"Queues={self.queues}, Time={self.time_since_phase_change}")
