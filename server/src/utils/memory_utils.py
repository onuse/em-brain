"""
Memory Utility Functions

Simple utility functions for memory calculation and management.
These were extracted from the removed Phase 2 memory_pressure_decay system.
"""

from typing import Dict, Any
from ..cognitive_constants import CognitiveCapacityConstants
from .hardware_adaptation import get_adaptive_cognitive_limits


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
    # Get adaptive limits based on current hardware
    limits = get_adaptive_cognitive_limits()
    max_capacity = limits.get('working_memory_limit', CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE)
    max_storage = limits.get('similarity_search_limit', 500) * 4  # Same calculation as storage system
    
    # Memory usage based on active working memory
    working_memory_usage = active_memories / max_capacity
    
    # Storage usage based on adaptive hardware limits
    storage_usage = min(1.0, total_experiences / max_storage)
    
    # Return combined usage (weighted toward working memory)
    return working_memory_usage * 0.7 + storage_usage * 0.3


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
            
            # Weight utility by activation level
            weighted_utility = utility * activation_level
            utilities.append(weighted_utility)
    
    if utilities:
        return sum(utilities) / len(utilities)
    return 0.5