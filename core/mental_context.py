"""
MentalContext - Represents the current state of the robot's mind.
What the robot is "thinking about" right now.
"""

from typing import List
from dataclasses import dataclass, field
from .experience_node import ExperienceNode


@dataclass
class MentalContext:
    """
    Represents the current state of the robot's mind.
    What the robot is "thinking about" right now.
    """
    context_vector: List[float]           # Compressed representation of current mental state
    recent_experiences: List[str]         # Node IDs of recent experiences
    active_predictions: List[str]         # Node IDs currently being predicted
    attention_focus: List[int]            # Indices into context_vector that have attention
    
    # Derived properties
    uncertainty_level: float = 0.0       # How uncertain the current predictions are
    novelty_level: float = 0.0           # How novel the current situation is
    
    def update_from_experience(self, new_experience: ExperienceNode, decay_rate: float = 0.1):
        """Update mental context based on new experience."""
        # Add the new experience to recent experiences
        self.recent_experiences.append(new_experience.node_id)
        
        # Keep only recent experiences (sliding window)
        if len(self.recent_experiences) > 10:
            self.recent_experiences = self.recent_experiences[-5:]
        
        # Update context vector with weighted blend of new experience
        if new_experience.mental_context and len(new_experience.mental_context) == len(self.context_vector):
            for i in range(len(self.context_vector)):
                # Blend current context with new experience context
                self.context_vector[i] = (
                    self.context_vector[i] * (1 - decay_rate) + 
                    new_experience.mental_context[i] * decay_rate
                )
        
        # Update novelty based on prediction error
        self.novelty_level = min(1.0, new_experience.prediction_error / 5.0)
    
    def get_context_vector(self) -> List[float]:
        """Get the current context vector."""
        return self.context_vector.copy()
    
    def set_attention_focus(self, indices: List[int]):
        """Set which parts of the context vector are currently receiving attention."""
        self.attention_focus = [i for i in indices if 0 <= i < len(self.context_vector)]
    
    def get_attention_weighted_context(self) -> List[float]:
        """Get context vector with attention weights applied."""
        if not self.attention_focus:
            return self.context_vector.copy()
        
        weighted_context = self.context_vector.copy()
        attention_boost = 1.5
        
        for i in self.attention_focus:
            if i < len(weighted_context):
                weighted_context[i] *= attention_boost
        
        return weighted_context
    
    def clear_predictions(self):
        """Clear all active predictions."""
        self.active_predictions.clear()
    
    def add_prediction(self, node_id: str):
        """Add a node ID to active predictions."""
        if node_id not in self.active_predictions:
            self.active_predictions.append(node_id)
    
    def remove_prediction(self, node_id: str):
        """Remove a node ID from active predictions."""
        if node_id in self.active_predictions:
            self.active_predictions.remove(node_id)
    
    def calculate_uncertainty(self, prediction_errors: List[float]) -> float:
        """Calculate uncertainty level based on recent prediction errors."""
        if not prediction_errors:
            return 0.0
        
        # Uncertainty increases with both magnitude and variance of errors
        avg_error = sum(prediction_errors) / len(prediction_errors)
        variance = sum((e - avg_error) ** 2 for e in prediction_errors) / len(prediction_errors)
        
        # Normalize to 0-1 range
        self.uncertainty_level = min(1.0, (avg_error + variance) / 10.0)
        return self.uncertainty_level
    
    def is_context_novel(self, threshold: float = 0.8) -> bool:
        """Check if current context represents a novel situation."""
        return self.novelty_level >= threshold
    
    def is_context_uncertain(self, threshold: float = 0.7) -> bool:
        """Check if current context is highly uncertain."""
        return self.uncertainty_level >= threshold