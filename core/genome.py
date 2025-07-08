"""
GenomeData - The robot's "DNA" containing core drives and behavioral parameters.
Stored as data rather than code for easy experimentation.
"""

from dataclasses import dataclass


@dataclass
class GenomeData:
    """
    The robot's "DNA" - core drives and behavioral parameters.
    Stored as data rather than code for easy experimentation.
    """
    # Core drives (0.0 to 1.0)
    collective_survival_drive: float = 0.8
    curiosity_drive: float = 0.7
    competence_drive: float = 0.6
    social_bonding_drive: float = 0.5
    
    # Learning parameters
    base_learning_rate: float = 0.01
    memory_decay_rate: float = 0.001
    similarity_threshold: float = 0.7
    merge_threshold: float = 0.5
    
    # Behavioral tendencies
    exploration_bias: float = 0.3         # Tendency to try new things vs exploit known patterns
    deliberation_preference: float = 1.0   # How much thinking time to use
    social_sensitivity: float = 0.8       # How quickly to recognize other agents
    
    # Safety constraints
    max_motor_change_per_step: float = 0.1  # Prevent sudden dangerous movements
    emergency_stop_threshold: float = 10.0  # Stop if prediction error exceeds this
    
    def get_drive(self, drive_name: str, default: float = 0.5) -> float:
        """Get a drive value by name."""
        return getattr(self, f"{drive_name}_drive", default)
    
    def should_explore(self, current_uncertainty: float = 0.0) -> bool:
        """Determine if the robot should explore based on drives and uncertainty."""
        # Higher uncertainty increases exploration tendency
        exploration_factor = self.exploration_bias + (current_uncertainty * 0.3)
        return exploration_factor > 0.5
    
    def should_deliberate(self, complexity: float = 0.5) -> bool:
        """Determine if the robot should spend time deliberating."""
        # More complex situations require more deliberation
        deliberation_factor = self.deliberation_preference * (0.5 + complexity * 0.5)
        return deliberation_factor > 0.6
    
    def get_learning_rate(self, prediction_accuracy: float = 0.5) -> float:
        """Get adaptive learning rate based on prediction accuracy."""
        # Learn faster when predictions are poor
        accuracy_factor = 1.0 - prediction_accuracy
        return self.base_learning_rate * (1.0 + accuracy_factor)
    
    def is_emergency_situation(self, prediction_error: float) -> bool:
        """Check if prediction error indicates an emergency."""
        return prediction_error > self.emergency_stop_threshold
    
    def get_motor_change_limit(self) -> float:
        """Get the maximum allowed motor command change per step."""
        return self.max_motor_change_per_step
    
    def adapt_parameters(self, success_rate: float):
        """Adapt parameters based on overall success rate."""
        # If success rate is low, increase exploration and learning
        if success_rate < 0.3:
            self.exploration_bias = min(1.0, self.exploration_bias + 0.1)
            self.base_learning_rate = min(0.1, self.base_learning_rate + 0.005)
        
        # If success rate is high, can afford to be more conservative
        elif success_rate > 0.8:
            self.exploration_bias = max(0.1, self.exploration_bias - 0.05)
            self.base_learning_rate = max(0.001, self.base_learning_rate - 0.002)
    
    def get_similarity_threshold(self, context_novelty: float = 0.0) -> float:
        """Get adaptive similarity threshold based on context novelty."""
        # In novel situations, be more lenient about similarity
        novelty_adjustment = context_novelty * 0.2
        return max(0.1, self.similarity_threshold - novelty_adjustment)
    
    def get_merge_threshold(self, memory_pressure: float = 0.0) -> float:
        """Get adaptive merge threshold based on memory pressure."""
        # Under memory pressure, merge more aggressively
        pressure_adjustment = memory_pressure * 0.3
        return max(0.1, self.merge_threshold - pressure_adjustment)