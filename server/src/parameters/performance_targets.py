"""
Performance Targets Configuration

Defines acceptable performance for different environments.
Development machines can have much slower performance than deployment.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Environment(Enum):
    """Execution environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    EMBEDDED = "embedded"


@dataclass
class PerformanceTargets:
    """Performance targets for different environments."""
    
    # Cycle time targets in milliseconds
    max_cycle_time_ms: float
    target_cycle_time_ms: float
    min_cycle_time_ms: float
    
    # Memory targets in MB
    max_memory_mb: float
    target_memory_mb: float
    
    # Other constraints
    max_cpu_percent: float
    max_gpu_memory_mb: Optional[float]
    
    @classmethod
    def for_environment(cls, env: Environment) -> 'PerformanceTargets':
        """Get performance targets for specific environment."""
        
        if env == Environment.DEVELOPMENT:
            # Relaxed targets for development machines
            # Jonas's M1 MacBook - anything under 1200ms is fine
            return cls(
                max_cycle_time_ms=1200.0,
                target_cycle_time_ms=750.0,
                min_cycle_time_ms=50.0,
                max_memory_mb=2000.0,
                target_memory_mb=500.0,
                max_cpu_percent=80.0,
                max_gpu_memory_mb=4000.0
            )
            
        elif env == Environment.TESTING:
            # Moderate targets for CI/testing
            return cls(
                max_cycle_time_ms=500.0,
                target_cycle_time_ms=150.0,
                min_cycle_time_ms=50.0,
                max_memory_mb=1000.0,
                target_memory_mb=300.0,
                max_cpu_percent=60.0,
                max_gpu_memory_mb=2000.0
            )
            
        elif env == Environment.PRODUCTION:
            # Strict targets for production deployment
            # Target machine is ~10x faster
            return cls(
                max_cycle_time_ms=650.0,
                target_cycle_time_ms=250.0,
                min_cycle_time_ms=50.0,
                max_memory_mb=20000.0,
                target_memory_mb=16000.0,
                max_cpu_percent=100.0,
                max_gpu_memory_mb=7500.0
            )
            
        elif env == Environment.EMBEDDED:
            # Very strict targets for embedded systems
            return cls(
                max_cycle_time_ms=100.0,
                target_cycle_time_ms=50.0,
                min_cycle_time_ms=10.0,
                max_memory_mb=200.0,
                target_memory_mb=100.0,
                max_cpu_percent=30.0,
                max_gpu_memory_mb=None  # No GPU
            )
            
        else:
            # Default to development
            return cls.for_environment(Environment.DEVELOPMENT)
    
    def is_cycle_time_acceptable(self, cycle_time_ms: float) -> bool:
        """Check if cycle time is within acceptable range."""
        return cycle_time_ms <= self.max_cycle_time_ms
        
    def get_cycle_time_rating(self, cycle_time_ms: float) -> str:
        """Get human-readable rating for cycle time."""
        if cycle_time_ms <= self.target_cycle_time_ms:
            return "EXCELLENT"
        elif cycle_time_ms <= self.max_cycle_time_ms:
            return "ACCEPTABLE"
        else:
            return "SLOW"


# Default to development environment
DEFAULT_TARGETS = PerformanceTargets.for_environment(Environment.DEVELOPMENT)


def get_current_targets() -> PerformanceTargets:
    """Get performance targets for current environment."""
    # Could read from env var or config file
    import os
    env_name = os.environ.get('BRAIN_ENV', 'development')
    
    try:
        env = Environment(env_name.lower())
    except ValueError:
        env = Environment.DEVELOPMENT
        
    return PerformanceTargets.for_environment(env)