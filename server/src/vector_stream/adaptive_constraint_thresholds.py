#!/usr/bin/env python3
"""
Adaptive Constraint Thresholds System

Implements dynamic constraint adaptation based on processing load and system behavior.
Thresholds adapt to avoid constraint storms while maintaining sensitivity to real issues.

Key principles:
- Learn from constraint frequency to adjust sensitivity
- Adapt to system load patterns over time
- Prevent constraint storms through dynamic dampening
- Maintain responsiveness to genuine constraint conditions
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from .stream_types import StreamType, ConstraintType


@dataclass
class ThresholdConfig:
    """Configuration for an adaptive constraint threshold."""
    base_threshold: float       # Initial/baseline threshold
    min_threshold: float        # Minimum allowed threshold
    max_threshold: float        # Maximum allowed threshold
    adaptation_rate: float      # How quickly threshold adapts (0.0-1.0)
    decay_rate: float          # How quickly threshold returns to baseline
    target_frequency: float     # Target constraint events per second
    
    # Current state
    current_threshold: float = None
    last_update_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.current_threshold is None:
            self.current_threshold = self.base_threshold


@dataclass
class LoadStatistics:
    """Statistics about system load and constraint patterns."""
    avg_processing_time: float
    avg_resource_usage: float
    constraint_frequency: float
    load_trend: str  # "increasing", "decreasing", "stable"
    last_update_time: float


class AdaptiveConstraintThresholds:
    """
    Manages adaptive constraint thresholds based on system behavior.
    
    This system learns from constraint propagation patterns and adjusts
    thresholds to maintain optimal sensitivity without constraint storms.
    """
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        
        # Threshold configurations for each constraint type
        self._threshold_configs = self._initialize_threshold_configs()
        
        # Statistics tracking
        self._load_history = deque(maxlen=100)  # Last 100 load measurements
        self._constraint_history = defaultdict(lambda: deque(maxlen=50))  # Last 50 events per type
        self._update_interval = 5.0  # Update thresholds every 5 seconds
        self._last_adaptation_time = time.time()
        
        # System state tracking
        self._system_load_stats = LoadStatistics(
            avg_processing_time=0.0,
            avg_resource_usage=0.0,
            constraint_frequency=0.0,
            load_trend="stable",
            last_update_time=time.time()
        )
        
        if not quiet_mode:
            print(f"🔧 AdaptiveConstraintThresholds initialized")
            print(f"   Update interval: {self._update_interval}s")
            print(f"   Tracking {len(self._threshold_configs)} constraint types")
    
    def _initialize_threshold_configs(self) -> Dict[ConstraintType, ThresholdConfig]:
        """Initialize threshold configurations for each constraint type."""
        configs = {
            ConstraintType.PROCESSING_LOAD: ThresholdConfig(
                base_threshold=0.7,     # 70% of budget (more conservative than current 30%)
                min_threshold=0.4,      # Don't go below 40%
                max_threshold=0.95,     # Don't exceed 95%
                adaptation_rate=0.1,    # Moderate adaptation speed
                decay_rate=0.02,        # Slow return to baseline
                target_frequency=2.0    # Target 2 events per second
            ),
            
            ConstraintType.RESOURCE_SCARCITY: ThresholdConfig(
                base_threshold=0.3,     # 30% resource shortage
                min_threshold=0.1,      # Don't go below 10%
                max_threshold=0.6,      # Don't exceed 60%
                adaptation_rate=0.15,   # Faster adaptation for resources
                decay_rate=0.03,        # Moderate return to baseline
                target_frequency=1.5    # Target 1.5 events per second
            ),
            
            ConstraintType.URGENCY_SIGNAL: ThresholdConfig(
                base_threshold=0.6,     # 60% activation + low resources
                min_threshold=0.4,      # Don't go below 40%
                max_threshold=0.9,      # Don't exceed 90%
                adaptation_rate=0.2,    # Fast adaptation for urgency
                decay_rate=0.05,        # Quick return to baseline
                target_frequency=0.5    # Target 0.5 events per second (rare)
            ),
            
            ConstraintType.INTERFERENCE: ThresholdConfig(
                base_threshold=0.6,     # 60% of budget with low output
                min_threshold=0.4,      # Don't go below 40%
                max_threshold=0.9,      # Don't exceed 90%
                adaptation_rate=0.1,    # Moderate adaptation
                decay_rate=0.02,        # Slow return to baseline
                target_frequency=1.0    # Target 1 event per second
            ),
            
            ConstraintType.COHERENCE_PRESSURE: ThresholdConfig(
                base_threshold=0.2,     # 20% activation variance
                min_threshold=0.1,      # Don't go below 10%
                max_threshold=0.4,      # Don't exceed 40%
                adaptation_rate=0.05,   # Slow adaptation for coherence
                decay_rate=0.01,        # Very slow return to baseline
                target_frequency=0.8    # Target 0.8 events per second
            ),
            
            ConstraintType.ENERGY_DEPLETION: ThresholdConfig(
                base_threshold=2.0,     # 2ms increase in processing time
                min_threshold=1.0,      # Don't go below 1ms
                max_threshold=5.0,      # Don't exceed 5ms
                adaptation_rate=0.08,   # Slow adaptation for energy
                decay_rate=0.015,       # Slow return to baseline
                target_frequency=0.3    # Target 0.3 events per second (very rare)
            )
        }
        
        return configs
    
    def update_load_measurement(self, stream_type: StreamType, processing_time: float, 
                              budget_ms: float, attention_used: float, energy_used: float):
        """Update load statistics from a processing cycle."""
        current_time = time.time()
        
        # Calculate load metrics
        load_ratio = processing_time / budget_ms if budget_ms > 0 else 1.0
        resource_usage = (attention_used + energy_used) / 2.0
        
        # Add to history
        self._load_history.append({
            'timestamp': current_time,
            'stream_type': stream_type,
            'load_ratio': load_ratio,
            'resource_usage': resource_usage,
            'processing_time': processing_time
        })
        
        # Update system statistics periodically
        if current_time - self._last_adaptation_time > self._update_interval:
            self._update_system_statistics()
            self._adapt_thresholds()
            self._last_adaptation_time = current_time
    
    def log_constraint_event(self, constraint_type: ConstraintType, intensity: float):
        """Log a constraint event for threshold adaptation."""
        current_time = time.time()
        
        self._constraint_history[constraint_type].append({
            'timestamp': current_time,
            'intensity': intensity
        })
    
    def get_threshold(self, constraint_type: ConstraintType) -> float:
        """Get the current adaptive threshold for a constraint type."""
        config = self._threshold_configs.get(constraint_type)
        if config:
            return config.current_threshold
        else:
            # Fallback to default values if constraint type not configured
            defaults = {
                ConstraintType.PROCESSING_LOAD: 0.7,
                ConstraintType.RESOURCE_SCARCITY: 0.3,
                ConstraintType.URGENCY_SIGNAL: 0.6,
                ConstraintType.INTERFERENCE: 0.6,
                ConstraintType.COHERENCE_PRESSURE: 0.2,
                ConstraintType.ENERGY_DEPLETION: 2.0
            }
            return defaults.get(constraint_type, 0.5)
    
    def get_all_thresholds(self) -> Dict[ConstraintType, float]:
        """Get all current adaptive thresholds."""
        return {
            constraint_type: config.current_threshold 
            for constraint_type, config in self._threshold_configs.items()
        }
    
    def _update_system_statistics(self):
        """Update system-wide load statistics."""
        if not self._load_history:
            return
        
        current_time = time.time()
        recent_window = 30.0  # Last 30 seconds
        
        # Filter recent measurements
        recent_measurements = [
            m for m in self._load_history 
            if current_time - m['timestamp'] < recent_window
        ]
        
        if not recent_measurements:
            return
        
        # Calculate statistics
        avg_processing_time = np.mean([m['processing_time'] for m in recent_measurements])
        avg_resource_usage = np.mean([m['resource_usage'] for m in recent_measurements])
        
        # Calculate constraint frequency
        total_constraints = sum(len(history) for history in self._constraint_history.values())
        constraint_frequency = total_constraints / recent_window if recent_window > 0 else 0.0
        
        # Determine load trend
        if len(recent_measurements) >= 10:
            recent_loads = [m['load_ratio'] for m in recent_measurements[-10:]]
            early_loads = [m['load_ratio'] for m in recent_measurements[:10]]
            
            recent_avg = np.mean(recent_loads)
            early_avg = np.mean(early_loads)
            
            if recent_avg > early_avg * 1.1:
                load_trend = "increasing"
            elif recent_avg < early_avg * 0.9:
                load_trend = "decreasing"
            else:
                load_trend = "stable"
        else:
            load_trend = "stable"
        
        # Update statistics
        self._system_load_stats = LoadStatistics(
            avg_processing_time=avg_processing_time,
            avg_resource_usage=avg_resource_usage,
            constraint_frequency=constraint_frequency,
            load_trend=load_trend,
            last_update_time=current_time
        )
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on current system behavior."""
        current_time = time.time()
        
        for constraint_type, config in self._threshold_configs.items():
            # Calculate actual constraint frequency for this type
            recent_events = [
                event for event in self._constraint_history[constraint_type]
                if current_time - event['timestamp'] < 30.0  # Last 30 seconds
            ]
            actual_frequency = len(recent_events) / 30.0
            
            # Calculate adaptation direction
            frequency_error = actual_frequency - config.target_frequency
            
            # Adapt threshold based on frequency error
            if frequency_error > 0:
                # Too many constraints - increase threshold (make less sensitive)
                adaptation = config.adaptation_rate * frequency_error
                new_threshold = config.current_threshold + adaptation
            else:
                # Too few constraints - decrease threshold (make more sensitive)
                adaptation = config.adaptation_rate * abs(frequency_error)
                new_threshold = config.current_threshold - adaptation
            
            # Apply decay toward baseline
            decay_factor = config.decay_rate
            baseline_pull = (config.base_threshold - config.current_threshold) * decay_factor
            new_threshold += baseline_pull
            
            # Clamp to limits
            new_threshold = max(config.min_threshold, min(config.max_threshold, new_threshold))
            
            # Update threshold if change is significant
            if abs(new_threshold - config.current_threshold) > 0.01:
                old_threshold = config.current_threshold
                config.current_threshold = new_threshold
                config.last_update_time = current_time
                
                if not self.quiet_mode:
                    print(f"🔧 Adapted {constraint_type.value} threshold: {old_threshold:.3f} → {new_threshold:.3f}")
                    print(f"   Frequency: {actual_frequency:.2f}/s (target: {config.target_frequency:.2f}/s)")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about threshold adaptation."""
        current_time = time.time()
        
        # Calculate recent constraint frequencies
        constraint_frequencies = {}
        for constraint_type, history in self._constraint_history.items():
            recent_events = [
                event for event in history
                if current_time - event['timestamp'] < 30.0
            ]
            constraint_frequencies[constraint_type.value] = len(recent_events) / 30.0
        
        # Get current thresholds vs baselines
        threshold_adaptations = {}
        for constraint_type, config in self._threshold_configs.items():
            adaptation_percent = ((config.current_threshold - config.base_threshold) / 
                                config.base_threshold * 100)
            threshold_adaptations[constraint_type.value] = {
                'current': config.current_threshold,
                'baseline': config.base_threshold,
                'adaptation_percent': adaptation_percent
            }
        
        return {
            'system_load_stats': {
                'avg_processing_time': self._system_load_stats.avg_processing_time,
                'avg_resource_usage': self._system_load_stats.avg_resource_usage,
                'constraint_frequency': self._system_load_stats.constraint_frequency,
                'load_trend': self._system_load_stats.load_trend
            },
            'constraint_frequencies': constraint_frequencies,
            'threshold_adaptations': threshold_adaptations,
            'total_load_measurements': len(self._load_history),
            'last_adaptation_time': self._last_adaptation_time
        }
    
    def reset_adaptations(self):
        """Reset all thresholds to their baseline values."""
        for config in self._threshold_configs.values():
            config.current_threshold = config.base_threshold
            config.last_update_time = time.time()
        
        self._constraint_history.clear()
        self._load_history.clear()
        self._last_adaptation_time = time.time()
        
        if not self.quiet_mode:
            print("🔧 Reset all adaptive thresholds to baseline values")


# Factory function
def create_adaptive_thresholds(quiet_mode: bool = False) -> AdaptiveConstraintThresholds:
    """Create an adaptive constraint thresholds system."""
    return AdaptiveConstraintThresholds(quiet_mode)


if __name__ == "__main__":
    # Example usage
    thresholds = create_adaptive_thresholds(quiet_mode=False)
    
    # Simulate some load measurements
    for i in range(10):
        thresholds.update_load_measurement(
            StreamType.SENSORY, 
            processing_time=15.0 + i * 2,  # Increasing processing time
            budget_ms=25.0,
            attention_used=0.3,
            energy_used=0.4
        )
        
        # Simulate constraint events
        if i % 3 == 0:
            thresholds.log_constraint_event(ConstraintType.PROCESSING_LOAD, 0.8)
    
    # Get current thresholds
    print(f"\nCurrent thresholds:")
    for constraint_type, threshold in thresholds.get_all_thresholds().items():
        print(f"  {constraint_type.value}: {threshold:.3f}")
    
    # Get adaptation statistics
    stats = thresholds.get_adaptation_stats()
    print(f"\nAdaptation statistics:")
    print(f"  System load trend: {stats['system_load_stats']['load_trend']}")
    print(f"  Constraint frequency: {stats['system_load_stats']['constraint_frequency']:.2f}/s")