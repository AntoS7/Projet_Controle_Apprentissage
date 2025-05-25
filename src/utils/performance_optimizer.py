"""
Performance Optimization Utilities

This module provides utilities for optimizing Q-table efficiency, memory usage,
and training performance in the SARSA traffic control system.
"""

import numpy as np
import pickle
import gzip
import os
from typing import Dict, Tuple, Any, Optional, List
from collections import defaultdict, deque
import gc
import psutil


class QTableOptimizer:
    """Optimizes Q-table storage and access patterns."""
    
    def __init__(self, compression_threshold: int = 10000):
        """
        Initialize Q-table optimizer.
        
        Args:
            compression_threshold: Number of entries before compression is considered
        """
        self.compression_threshold = compression_threshold
        self.access_counter = defaultdict(int)
        self.last_cleanup_size = 0
        
    def optimize_q_table(self, q_table: Dict[Tuple, float], 
                        min_visits: int = 2) -> Dict[Tuple, float]:
        """
        Optimize Q-table by removing rarely visited states.
        
        Args:
            q_table: Original Q-table dictionary
            min_visits: Minimum visits required to keep a state-action pair
            
        Returns:
            Optimized Q-table
        """
        original_size = len(q_table)
        
        # Track access patterns if enabled
        if hasattr(self, 'access_counter') and self.access_counter:
            optimized = {
                key: value for key, value in q_table.items()
                if self.access_counter[key] >= min_visits
            }
        else:
            # Fallback: keep all entries with non-zero values
            optimized = {
                key: value for key, value in q_table.items()
                if abs(value) > 1e-6
            }
        
        optimized_size = len(optimized)
        reduction = (original_size - optimized_size) / original_size * 100
        
        print(f"Q-table optimization: {original_size} -> {optimized_size} entries "
              f"({reduction:.1f}% reduction)")
        
        return optimized
    
    def compress_q_table(self, q_table: Dict[Tuple, float], 
                        filepath: str) -> str:
        """
        Save Q-table with compression.
        
        Args:
            q_table: Q-table to compress and save
            filepath: Path to save compressed file
            
        Returns:
            Path to compressed file
        """
        compressed_path = filepath + '.gz'
        
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compare file sizes
        try:
            original_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            compressed_size = os.path.getsize(compressed_path)
            
            if original_size > 0:
                compression_ratio = compressed_size / original_size * 100
                print(f"Q-table compression: {compression_ratio:.1f}% of original size")
        except OSError:
            pass
        
        return compressed_path
    
    def load_compressed_q_table(self, filepath: str) -> Dict[Tuple, float]:
        """
        Load compressed Q-table.
        
        Args:
            filepath: Path to compressed Q-table file
            
        Returns:
            Loaded Q-table
        """
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def track_access(self, state_action_key: Tuple):
        """Track access to state-action pairs for optimization."""
        self.access_counter[state_action_key] += 1
    
    def should_cleanup(self, current_size: int) -> bool:
        """Determine if Q-table cleanup should be performed."""
        size_increase = current_size - self.last_cleanup_size
        return (current_size > self.compression_threshold and 
                size_increase > self.compression_threshold * 0.5)


class MemoryMonitor:
    """Monitors and optimizes memory usage during training."""
    
    def __init__(self, warning_threshold: float = 80.0, 
                 critical_threshold: float = 90.0):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Memory usage percentage to trigger warnings
            critical_threshold: Memory usage percentage to trigger cleanup
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_history = deque(maxlen=100)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        usage = {
            'process_mb': memory_info.rss / 1024 / 1024,
            'system_percent': system_memory.percent,
            'available_gb': system_memory.available / 1024 / 1024 / 1024
        }
        
        self.memory_history.append(usage['system_percent'])
        return usage
    
    def check_memory_status(self) -> Tuple[str, Dict[str, float]]:
        """
        Check memory status and return recommendations.
        
        Returns:
            Tuple of (status, memory_info) where status is 'ok', 'warning', or 'critical'
        """
        memory_info = self.get_memory_usage()
        system_percent = memory_info['system_percent']
        
        if system_percent >= self.critical_threshold:
            return 'critical', memory_info
        elif system_percent >= self.warning_threshold:
            return 'warning', memory_info
        else:
            return 'ok', memory_info
    
    def suggest_cleanup(self, memory_info: Dict[str, float]) -> List[str]:
        """Suggest memory cleanup actions based on current usage."""
        suggestions = []
        
        if memory_info['system_percent'] > self.warning_threshold:
            suggestions.append("Consider reducing Q-table size")
            suggestions.append("Enable Q-table compression")
            
        if memory_info['system_percent'] > self.critical_threshold:
            suggestions.append("Force garbage collection")
            suggestions.append("Reduce episode buffer sizes")
            suggestions.append("Save and restart training")
            
        if memory_info['available_gb'] < 1.0:
            suggestions.append("Critical: Very low available memory")
            
        return suggestions
    
    def force_cleanup(self):
        """Force memory cleanup operations."""
        # Force garbage collection
        gc.collect()
        
        # Additional Python-specific cleanup
        import sys
        sys.intern("cleanup")  # Force string interning cleanup
        
    def get_memory_trend(self) -> str:
        """Analyze memory usage trend over recent history."""
        if len(self.memory_history) < 10:
            return "insufficient_data"
        
        recent = list(self.memory_history)[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 1.0:
            return "increasing"
        elif trend < -1.0:
            return "decreasing"
        else:
            return "stable"


class PerformanceProfiler:
    """Profiles training performance and identifies bottlenecks."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.episode_metrics = []
        
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        import time
        start_time = time.time()
        return start_time
    
    def end_timer(self, operation: str, start_time: float):
        """End timing and record duration."""
        import time
        duration = time.time() - start_time
        self.timing_data[operation].append(duration)
    
    def record_episode_metrics(self, episode: int, steps: int, 
                             reward: float, q_table_size: int):
        """Record metrics for a training episode."""
        self.episode_metrics.append({
            'episode': episode,
            'steps': steps,
            'reward': reward,
            'q_table_size': q_table_size,
            'timestamp': __import__('time').time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary report."""
        summary = {}
        
        # Timing analysis
        for operation, times in self.timing_data.items():
            if times:
                summary[f'{operation}_avg_time'] = np.mean(times)
                summary[f'{operation}_total_time'] = np.sum(times)
                summary[f'{operation}_max_time'] = np.max(times)
        
        # Episode analysis
        if self.episode_metrics:
            rewards = [ep['reward'] for ep in self.episode_metrics]
            q_sizes = [ep['q_table_size'] for ep in self.episode_metrics]
            
            summary['avg_reward'] = np.mean(rewards)
            summary['reward_trend'] = self._calculate_trend(rewards)
            summary['q_table_growth_rate'] = self._calculate_growth_rate(q_sizes)
            summary['episodes_per_second'] = self._calculate_episode_rate()
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 10:
            return "insufficient_data"
        
        trend = np.polyfit(range(len(values)), values, 1)[0]
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_growth_rate(self, q_sizes: List[int]) -> float:
        """Calculate Q-table growth rate."""
        if len(q_sizes) < 2:
            return 0.0
        
        return (q_sizes[-1] - q_sizes[0]) / len(q_sizes)
    
    def _calculate_episode_rate(self) -> float:
        """Calculate episodes per second."""
        if len(self.episode_metrics) < 2:
            return 0.0
        
        first_time = self.episode_metrics[0]['timestamp']
        last_time = self.episode_metrics[-1]['timestamp']
        duration = last_time - first_time
        
        if duration > 0:
            return len(self.episode_metrics) / duration
        return 0.0
    
    def identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for slow operations
        for operation, times in self.timing_data.items():
            if times and np.mean(times) > 1.0:  # Operations taking > 1 second
                bottlenecks.append(f"Slow {operation}: {np.mean(times):.2f}s average")
        
        # Check Q-table growth
        if self.episode_metrics:
            q_sizes = [ep['q_table_size'] for ep in self.episode_metrics]
            growth_rate = self._calculate_growth_rate(q_sizes)
            
            if growth_rate > 100:  # More than 100 new entries per episode
                bottlenecks.append(f"High Q-table growth: {growth_rate:.1f} entries/episode")
        
        return bottlenecks


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.q_optimizer = QTableOptimizer()
        self.memory_monitor = MemoryMonitor()
        self.profiler = PerformanceProfiler()
        
    def optimize_training_session(self, agent, episode: int) -> Dict[str, Any]:
        """
        Perform comprehensive optimization for a training session.
        
        Args:
            agent: SARSA agent instance
            episode: Current episode number
            
        Returns:
            Optimization results and recommendations
        """
        results = {}
        
        # Memory monitoring
        memory_status, memory_info = self.memory_monitor.check_memory_status()
        results['memory_status'] = memory_status
        results['memory_info'] = memory_info
        
        # Q-table optimization
        if hasattr(agent, 'q_table'):
            q_table_size = len(agent.q_table)
            
            # Optimize if needed
            if self.q_optimizer.should_cleanup(q_table_size):
                optimized_q_table = self.q_optimizer.optimize_q_table(agent.q_table)
                agent.q_table = optimized_q_table
                results['q_table_optimized'] = True
                results['new_q_table_size'] = len(optimized_q_table)
            else:
                results['q_table_optimized'] = False
                results['new_q_table_size'] = q_table_size
        
        # Memory cleanup if critical
        if memory_status == 'critical':
            self.memory_monitor.force_cleanup()
            results['memory_cleanup_performed'] = True
        else:
            results['memory_cleanup_performed'] = False
        
        # Performance analysis
        performance_summary = self.profiler.get_performance_summary()
        bottlenecks = self.profiler.identify_bottlenecks()
        
        results['performance_summary'] = performance_summary
        results['bottlenecks'] = bottlenecks
        
        return results
    
    def get_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Memory recommendations
        if results['memory_status'] != 'ok':
            memory_suggestions = self.memory_monitor.suggest_cleanup(results['memory_info'])
            recommendations.extend(memory_suggestions)
        
        # Performance recommendations
        if 'bottlenecks' in results and results['bottlenecks']:
            recommendations.append("Performance bottlenecks detected:")
            recommendations.extend(results['bottlenecks'])
        
        # Q-table recommendations
        if 'new_q_table_size' in results and results['new_q_table_size'] > 50000:
            recommendations.append("Consider reducing state discretization")
            recommendations.append("Enable more frequent Q-table optimization")
        
        return recommendations
