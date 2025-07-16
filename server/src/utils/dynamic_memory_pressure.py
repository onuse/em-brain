"""
Dynamic Memory Pressure System

The brain should grow until it hits real performance constraints, then adapt.
This system learns actual hardware limits through runtime experience rather than
conservative estimates.

Biological inspiration: Real brains develop their capacity through experience,
not through hardcoded limits. They adapt to their actual computational resources.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetrics:
    """Real-time performance measurements."""
    cycle_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    experience_count: int
    timestamp: float


class DynamicMemoryPressure:
    """
    Discovers actual hardware limits through runtime experience.
    
    The system starts with optimistic limits and gradually learns where
    the real performance boundaries are by monitoring actual cycle times,
    memory usage, and system responsiveness.
    """
    
    def __init__(self, target_cycle_time_ms: float = 100.0):
        """
        Initialize dynamic memory pressure system.
        
        Args:
            target_cycle_time_ms: Desired brain cycle time in milliseconds
        """
        self.target_cycle_time_ms = target_cycle_time_ms
        self.degradation_threshold_ms = 200.0  # 200ms = getting slow
        self.critical_threshold_ms = 400.0  # 400ms = concerning for robotics
        
        # Performance history for learning
        self.performance_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
        
        # Dynamic limits (start optimistic, learn through experience)
        self.current_experience_limit = 1000000  # Start with 1M experiences
        self.current_memory_limit_mb = psutil.virtual_memory().available / (1024 * 1024) * 0.9  # 90% of available
        
        # Learning state
        self.last_adjustment_time = time.time()
        self.consecutive_good_cycles = 0
        self.consecutive_bad_cycles = 0
        self.best_performance_point = None  # (experience_count, cycle_time_ms)
        
        # Performance trend tracking
        self.performance_trend = "stable"  # "improving", "stable", "degrading"
        
        print(f"🧠 Dynamic Memory Pressure initialized:")
        print(f"   Target cycle time: {target_cycle_time_ms}ms")
        print(f"   Initial experience limit: {self.current_experience_limit:,}")
        print(f"   Initial memory limit: {self.current_memory_limit_mb:.1f} MB")
    
    def record_cycle_performance(self, cycle_time_ms: float, experience_count: int):
        """
        Record a brain cycle's performance for dynamic learning.
        
        Args:
            cycle_time_ms: Time taken for this cycle
            experience_count: Number of experiences in the system
        """
        with self.lock:
            # Get current system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            metrics = PerformanceMetrics(
                cycle_time_ms=cycle_time_ms,
                memory_usage_mb=memory_info.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                experience_count=experience_count,
                timestamp=time.time()
            )
            
            self.performance_history.append(metrics)
            
            # Learn from this cycle
            self._update_performance_assessment(metrics)
            self._adapt_limits_if_needed(metrics)
    
    def _update_performance_assessment(self, metrics: PerformanceMetrics):
        """Update our assessment of current performance trend."""
        if metrics.cycle_time_ms <= self.target_cycle_time_ms:
            self.consecutive_good_cycles += 1
            self.consecutive_bad_cycles = 0
            
            # Track best performance point
            if (self.best_performance_point is None or 
                metrics.experience_count > self.best_performance_point[0]):
                self.best_performance_point = (metrics.experience_count, metrics.cycle_time_ms)
                
        elif metrics.cycle_time_ms <= self.degradation_threshold_ms:
            # Acceptable but not optimal
            self.consecutive_good_cycles = max(0, self.consecutive_good_cycles - 1)
            self.consecutive_bad_cycles = 0
            
        else:
            # Performance is degrading
            self.consecutive_good_cycles = 0
            self.consecutive_bad_cycles += 1
    
    def _adapt_limits_if_needed(self, metrics: PerformanceMetrics):
        """Adapt limits based on current performance trend."""
        now = time.time()
        
        # Only adjust every 30 seconds to avoid oscillation
        if now - self.last_adjustment_time < 30.0:
            return
        
        # Performance is consistently good - we can grow
        if self.consecutive_good_cycles >= 50:  # Be more conservative about growing
            self._increase_limits(metrics)
            self.last_adjustment_time = now
            
        # Performance is degrading - we need to shrink
        elif self.consecutive_bad_cycles >= 3:  # React quickly to degradation
            self._decrease_limits(metrics)
            self.last_adjustment_time = now
    
    def _increase_limits(self, metrics: PerformanceMetrics):
        """Increase limits when performance is consistently good."""
        # Increase experience limit by 10% (more conservative for robotics)
        new_limit = int(self.current_experience_limit * 1.1)
        
        # But don't exceed memory capacity
        estimated_memory_mb = (new_limit * 200) / (1024 * 1024)  # 200 bytes per experience
        if estimated_memory_mb <= self.current_memory_limit_mb:
            old_limit = self.current_experience_limit
            self.current_experience_limit = new_limit
            
            print(f"🚀 Growing brain capacity: {old_limit:,} → {new_limit:,} experiences")
            print(f"   Performance: {metrics.cycle_time_ms:.1f}ms cycles, {metrics.experience_count:,} experiences")
            
            self.consecutive_good_cycles = 0  # Reset counter
        else:
            print(f"⚠️  Cannot grow further: would exceed memory limit ({estimated_memory_mb:.1f} MB)")
    
    def _decrease_limits(self, metrics: PerformanceMetrics):
        """Decrease limits when performance is consistently bad."""
        # Decrease experience limit by 20% (react strongly to degradation)
        if metrics.cycle_time_ms >= self.critical_threshold_ms:
            # Critical performance - reduce aggressively
            new_limit = int(self.current_experience_limit * 0.7)
        else:
            # Moderate degradation - reduce moderately
            new_limit = int(self.current_experience_limit * 0.85)
        
        new_limit = max(10000, new_limit)  # Never go below 10k for hundreds of thousands target
        
        if new_limit != self.current_experience_limit:
            old_limit = self.current_experience_limit
            self.current_experience_limit = new_limit
            
            print(f"⚠️  Reducing brain capacity: {old_limit:,} → {new_limit:,} experiences")
            print(f"   Performance degraded: {metrics.cycle_time_ms:.1f}ms cycles (target: {self.target_cycle_time_ms:.1f}ms)")
            
            self.consecutive_bad_cycles = 0  # Reset counter
    
    def get_current_experience_limit(self) -> int:
        """Get the current dynamic experience limit."""
        return self.current_experience_limit
    
    def get_current_memory_limit_mb(self) -> float:
        """Get the current dynamic memory limit in MB."""
        return self.current_memory_limit_mb
    
    def should_trigger_cleanup(self, current_experience_count: int) -> bool:
        """
        Check if we should trigger cleanup based on dynamic limits.
        
        Args:
            current_experience_count: Current number of experiences
            
        Returns:
            True if cleanup should be triggered
        """
        # Trigger cleanup when we exceed the learned limit
        return current_experience_count > self.current_experience_limit
    
    def get_cleanup_target(self, current_experience_count: int) -> int:
        """
        Get the target experience count after cleanup.
        
        Args:
            current_experience_count: Current number of experiences
            
        Returns:
            Target experience count after cleanup
        """
        # Clean up to 90% of the learned limit
        return int(self.current_experience_limit * 0.9)
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        with self.lock:
            if not self.performance_history:
                return {}
            
            recent_cycles = list(self.performance_history)[-20:]  # Last 20 cycles
            
            avg_cycle_time = sum(m.cycle_time_ms for m in recent_cycles) / len(recent_cycles)
            avg_memory_usage = sum(m.memory_usage_mb for m in recent_cycles) / len(recent_cycles)
            avg_cpu_usage = sum(m.cpu_usage_percent for m in recent_cycles) / len(recent_cycles)
            
            return {
                'current_experience_limit': self.current_experience_limit,
                'current_memory_limit_mb': self.current_memory_limit_mb,
                'avg_cycle_time_ms': avg_cycle_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_cpu_usage_percent': avg_cpu_usage,
                'target_cycle_time_ms': self.target_cycle_time_ms,
                'performance_trend': self.performance_trend,
                'consecutive_good_cycles': self.consecutive_good_cycles,
                'consecutive_bad_cycles': self.consecutive_bad_cycles,
                'best_performance_point': self.best_performance_point
            }


# Global instance
_dynamic_memory_pressure = None


def get_dynamic_memory_pressure() -> DynamicMemoryPressure:
    """Get the global dynamic memory pressure instance."""
    global _dynamic_memory_pressure
    if _dynamic_memory_pressure is None:
        _dynamic_memory_pressure = DynamicMemoryPressure()
    return _dynamic_memory_pressure


def record_brain_cycle_performance(cycle_time_ms: float, experience_count: int):
    """Record a brain cycle's performance for dynamic learning."""
    get_dynamic_memory_pressure().record_cycle_performance(cycle_time_ms, experience_count)


def get_dynamic_experience_limit() -> int:
    """Get the current dynamic experience limit."""
    return get_dynamic_memory_pressure().get_current_experience_limit()


def should_trigger_cleanup(current_experience_count: int) -> bool:
    """Check if we should trigger cleanup based on dynamic limits."""
    return get_dynamic_memory_pressure().should_trigger_cleanup(current_experience_count)


def get_cleanup_target(current_experience_count: int) -> int:
    """Get the target experience count after cleanup."""
    return get_dynamic_memory_pressure().get_cleanup_target(current_experience_count)