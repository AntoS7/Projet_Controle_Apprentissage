"""
Centralized logging configuration for the traffic control project.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
import os


class TrafficControlLogger:
    """Centralized logger for traffic control training."""
    
    def __init__(self, name: str = "traffic_control", level: int = logging.INFO):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler (logs directory)
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f'logs/training_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        console_handler.setFormatter(console_format)
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def episode_start(self, episode: int, total_episodes: int):
        """Log episode start."""
        self.info("Episode %d/%d started", episode, total_episodes)
    
    def episode_end(self, episode: int, reward: float, steps: int, epsilon: float):
        """Log episode completion."""
        self.info("Episode %d completed - Reward: %.1f, Steps: %d, Epsilon: %.3f", 
                 episode, reward, steps, epsilon)
    
    def training_summary(self, total_time: float, best_reward: float, final_epsilon: float):
        """Log training summary."""
        self.info("Training completed in %.1f minutes", total_time / 60)
        self.info("Best reward: %.1f, Final epsilon: %.3f", best_reward, final_epsilon)
    
    def agent_statistics(self, q_table_size: int, exploration_rate: float, updates: int):
        """Log agent statistics."""
        self.debug("Agent stats - Q-table size: %d, Exploration: %.3f, Updates: %d",
                  q_table_size, exploration_rate, updates)


# Global logger instance
logger = TrafficControlLogger()


def get_logger(name: Optional[str] = None) -> TrafficControlLogger:
    """
    Get logger instance.
    
    Args:
        name: Optional logger name
        
    Returns:
        Logger instance
    """
    if name:
        return TrafficControlLogger(name)
    return logger
