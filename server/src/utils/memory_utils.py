"""
Memory Utility Functions

Simple utility functions for memory calculation and management.
These were extracted from the removed Phase 2 memory_pressure_decay system.
"""

from typing import Dict, Any
from ..cognitive_constants import CognitiveCapacityConstants


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