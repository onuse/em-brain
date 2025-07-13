"""
Memory-Pressure Based Decay (Phase 2 Lite)

Adapts activation decay rates based on memory pressure and utility.
This is safe because it only adjusts within bounded ranges and maintains
all the stability guarantees of the original system.

Key insight: Decay rate should be smart about memory needs rather than fixed.
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque

from ..cognitive_constants import CognitiveCapacityConstants


class MemoryPressureDecay:
    """
    Manages decay rates that adapt to memory pressure and utility patterns.
    
    Strategy: Instead of fixed decay rate (e.g., 0.02), use decay that responds
    to how full memory is and how useful memories are.
    """
    
    def __init__(self, base_decay_rate: float = 0.02):
        """
        Initialize memory pressure-based decay.
        
        Args:
            base_decay_rate: Starting decay rate (safe default)
        """
        self.base_decay_rate = base_decay_rate
        self.current_decay_rate = base_decay_rate
        
        # Memory pressure tracking
        self.memory_usage_history = deque(maxlen=20)
        self.utility_history = deque(maxlen=50)
        self.decay_effectiveness = deque(maxlen=30)
        
        # Adaptation parameters (conservative)
        self.max_decay_multiplier = 3.0  # Max 3x base decay rate
        self.min_decay_multiplier = 0.3  # Min 0.3x base decay rate
        self.adaptation_rate = 0.01  # Biologically plausible (was 0.05)
        
        print(f"🧠 MemoryPressureDecay initialized - base rate: {base_decay_rate}")
    
    def get_adaptive_decay_rate(self,
                               current_memory_usage: float,
                               average_memory_utility: float,
                               working_memory_size: int) -> float:
        """
        Calculate adaptive decay rate based on memory conditions.
        
        Args:
            current_memory_usage: Memory usage (0.0-1.0)
            average_memory_utility: How useful current memories are (0.0-1.0)  
            working_memory_size: Current working memory size
            
        Returns:
            Adaptive decay rate
        """
        # Track memory conditions
        self.memory_usage_history.append(current_memory_usage)
        self.utility_history.append(average_memory_utility)
        
        # Calculate memory pressure factors
        memory_pressure = self._calculate_memory_pressure(working_memory_size)
        utility_factor = self._calculate_utility_factor(average_memory_utility)
        stability_factor = self._calculate_stability_factor()
        
        # Combine factors to get decay multiplier
        decay_multiplier = self._combine_factors(memory_pressure, utility_factor, stability_factor)
        
        # Apply adaptive decay rate
        target_decay_rate = self.base_decay_rate * decay_multiplier
        
        # Smooth adaptation to prevent oscillation
        rate_change = (target_decay_rate - self.current_decay_rate) * self.adaptation_rate
        self.current_decay_rate += rate_change
        
        # Apply safety bounds
        self.current_decay_rate = np.clip(
            self.current_decay_rate,
            self.base_decay_rate * self.min_decay_multiplier,
            self.base_decay_rate * self.max_decay_multiplier
        )
        
        return self.current_decay_rate
    
    def _calculate_memory_pressure(self, working_memory_size: int) -> float:
        """Calculate memory pressure based on working memory size."""
        
        # Define memory pressure based on cognitive capacity constants (hardware-adaptive)
        min_size = CognitiveCapacityConstants.MIN_WORKING_MEMORY_SIZE
        max_size = CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE
        target_size = CognitiveCapacityConstants.get_working_memory_limit()
        
        if working_memory_size <= target_size:
            # Below target: low pressure (can afford slow decay)
            pressure = 0.0
        elif working_memory_size >= max_size:
            # At capacity: high pressure (need fast decay)
            pressure = 1.0
        else:
            # Between target and max: linear pressure increase
            pressure = (working_memory_size - target_size) / (max_size - target_size)
        
        return pressure
    
    def _calculate_utility_factor(self, average_utility: float) -> float:
        """Calculate how utility affects decay (high utility = slower decay)."""
        
        # High utility memories should decay slower
        # Low utility memories should decay faster
        
        if average_utility > 0.7:
            # High utility: reduce decay (keep useful memories)
            utility_factor = 0.7
        elif average_utility < 0.3:
            # Low utility: increase decay (clear useless memories)
            utility_factor = 1.5
        else:
            # Medium utility: normal decay
            utility_factor = 1.0
        
        return utility_factor
    
    def _calculate_stability_factor(self) -> float:
        """Calculate stability factor to prevent oscillation."""
        
        if len(self.memory_usage_history) < 5:
            return 1.0  # Default stable
        
        # Check for memory usage oscillation
        recent_usage = list(self.memory_usage_history)[-5:]
        usage_variance = np.var(recent_usage)
        
        if usage_variance > 0.1:
            # High variance: be more conservative with changes
            stability_factor = 0.8
        else:
            # Low variance: normal adaptation
            stability_factor = 1.0
        
        return stability_factor
    
    def _combine_factors(self, memory_pressure: float, utility_factor: float, stability_factor: float) -> float:
        """Combine all factors into final decay multiplier."""
        
        # Memory pressure increases decay rate
        pressure_multiplier = 1.0 + memory_pressure * 1.0  # Up to 2x base rate
        
        # Utility factor modifies decay rate
        utility_multiplier = utility_factor
        
        # Stability factor dampens changes
        combined_multiplier = (pressure_multiplier * utility_multiplier) * stability_factor
        
        # Apply bounds
        return np.clip(combined_multiplier, self.min_decay_multiplier, self.max_decay_multiplier)
    
    def get_decay_statistics(self) -> Dict[str, Any]:
        """Get statistics about decay adaptation."""
        
        return {
            'base_decay_rate': self.base_decay_rate,
            'current_decay_rate': self.current_decay_rate,
            'decay_multiplier': self.current_decay_rate / self.base_decay_rate,
            'recent_memory_pressure': np.mean(list(self.memory_usage_history)[-5:]) if self.memory_usage_history else 0.0,
            'recent_utility': np.mean(list(self.utility_history)[-5:]) if self.utility_history else 0.5,
            'adaptation_active': abs(self.current_decay_rate - self.base_decay_rate) > 0.001,
            'decay_bounds': (self.base_decay_rate * self.min_decay_multiplier, 
                           self.base_decay_rate * self.max_decay_multiplier)
        }
    
    def reset_decay_adaptation(self):
        """Reset to base decay rate."""
        self.current_decay_rate = self.base_decay_rate
        self.memory_usage_history.clear()
        self.utility_history.clear()
        self.decay_effectiveness.clear()
        print(f"🔄 Memory pressure decay reset to base rate: {self.base_decay_rate}")


def calculate_memory_usage(current_activations: Dict[str, float], 
                          total_experiences: int) -> float:
    """
    Calculate current memory usage as a proportion of capacity.
    
    Args:
        current_activations: Dict of experience_id -> activation_level
        total_experiences: Total number of experiences in storage
        
    Returns:
        Memory usage (0.0-1.0)
    """
    active_memories = len(current_activations)
    max_capacity = CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE
    
    # Memory usage based on active working memory
    working_memory_usage = active_memories / max_capacity
    
    # Also consider total experience storage (for future extensions)
    # This could be used if we add limits on total experience storage
    storage_usage = min(1.0, total_experiences / 1000)  # Example: 1000 experience soft limit
    
    # Combine (working memory is more important for immediate pressure)
    overall_usage = working_memory_usage * 0.8 + storage_usage * 0.2
    
    return min(1.0, overall_usage)


def calculate_average_utility(current_activations: Dict[str, float],
                            experiences: Dict[str, Any]) -> float:
    """
    Calculate average utility of currently active memories.
    
    Args:
        current_activations: Dict of experience_id -> activation_level
        experiences: Dict of experience_id -> experience_object
        
    Returns:
        Average utility (0.0-1.0)
    """
    if not current_activations:
        return 0.5  # Default neutral utility
    
    utilities = []
    for exp_id, activation_level in current_activations.items():
        if exp_id in experiences:
            exp = experiences[exp_id]
            # Get utility from experience (prediction_utility or similar attribute)
            utility = getattr(exp, 'prediction_utility', 0.5)
            # Weight by activation level (more active = more important)
            weighted_utility = utility * activation_level
            utilities.append(weighted_utility)
    
    if utilities:
        return np.mean(utilities)
    else:
        return 0.5  # Default neutral utility