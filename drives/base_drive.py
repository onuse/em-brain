"""
Base class for all robot drives/motivations.
Provides the interface that all drives must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DriveContext:
    """Context information available to all drives for decision making."""
    current_sensory: List[float]         # Current sensor readings
    robot_health: float                  # Robot health [0.0-1.0]
    robot_energy: float                  # Robot energy [0.0-1.0]
    robot_position: tuple               # (x, y) position in world
    robot_orientation: int              # 0=North, 1=East, 2=South, 3=West
    recent_experiences: List[Any]       # Recent experience nodes
    prediction_errors: List[float]      # Recent prediction errors
    time_since_last_food: int           # Steps since last food
    time_since_last_damage: int         # Steps since last damage
    threat_level: str                   # "normal", "alert", "danger", "critical"
    step_count: int                     # Total simulation steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'robot_health': self.robot_health,
            'robot_energy': self.robot_energy,
            'robot_position': self.robot_position,
            'robot_orientation': self.robot_orientation,
            'threat_level': self.threat_level,
            'step_count': self.step_count,
            'sensory_data_length': len(self.current_sensory),
            'recent_experiences_count': len(self.recent_experiences),
            'prediction_errors_count': len(self.prediction_errors)
        }


@dataclass
class ActionEvaluation:
    """A drive's evaluation of a potential action."""
    drive_name: str                     # Which drive generated this evaluation
    action_score: float                 # Score for this action [0.0-1.0]
    confidence: float                   # How confident the drive is [0.0-1.0]
    reasoning: str                      # Human-readable explanation
    urgency: float = 0.5               # How urgent this drive considers the action [0.0-1.0]
    expected_pain: float = 0.0         # Expected pain if this action is taken [-1.0 to 0.0]
    expected_pleasure: float = 0.0     # Expected pleasure if this action is taken [0.0 to 1.0]
    
    def weighted_score(self, drive_weight: float) -> float:
        """Calculate final weighted score for this action."""
        return self.action_score * drive_weight * self.confidence
    
    def get_pain_pleasure_bias(self) -> float:
        """Get the pain/pleasure bias for this action evaluation."""
        return self.expected_pleasure + self.expected_pain  # Pain is negative, pleasure is positive


class BaseDrive(ABC):
    """
    Abstract base class for all robot drives/motivations.
    
    Each drive evaluates potential actions and provides scores indicating
    how well each action satisfies that particular drive's goals.
    """
    
    def __init__(self, name: str, base_weight: float = 0.33):
        """
        Initialize the drive.
        
        Args:
            name: Human-readable name for this drive
            base_weight: Default importance weight [0.0-1.0]
        """
        self.name = name
        self.base_weight = base_weight
        self.current_weight = base_weight
        self.activation_history = []  # Track when this drive was most active
        self.total_evaluations = 0
        
        # Sensor discovery - initially empty, discovered through brainstem interaction
        self.discovered_sensors = {}  # purpose -> [sensor_indices]
        self.sensor_semantics = None  # Will be set via initialize_with_brainstem
        
    @abstractmethod
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> ActionEvaluation:
        """
        Evaluate how well a potential action satisfies this drive.
        
        Args:
            action: Motor action to evaluate {"forward_motor": 0.5, "turn_motor": 0.0, ...}
            context: Current situation context
            
        Returns:
            ActionEvaluation with score and reasoning
        """
        pass
    
    @abstractmethod
    def update_drive_state(self, context: DriveContext, world_graph=None) -> float:
        """
        Update internal drive state based on current context.
        
        Args:
            context: Current situation context
            world_graph: Optional world graph for advanced drives
            
        Returns:
            Updated drive weight based on current needs [0.0-1.0]
        """
        pass
    
    def get_drive_info(self) -> Dict[str, Any]:
        """Get information about this drive's current state."""
        return {
            'name': self.name,
            'base_weight': self.base_weight,
            'current_weight': self.current_weight,
            'total_evaluations': self.total_evaluations,
            'activation_history_length': len(self.activation_history)
        }
    
    def reset_drive(self):
        """Reset drive to initial state."""
        self.current_weight = self.base_weight
        self.activation_history.clear()
        self.total_evaluations = 0
    
    def log_activation(self, score: float, step: int):
        """Log when this drive was highly active."""
        if score > 0.7:  # High activation threshold
            self.activation_history.append({
                'step': step,
                'activation_score': score,
                'weight': self.current_weight
            })
            
            # Keep only recent history
            if len(self.activation_history) > 100:
                self.activation_history = self.activation_history[-50:]
    
    def evaluate_experience_significance(self, experience, context: DriveContext) -> float:
        """
        Evaluate how significant this experience is for this drive.
        
        Args:
            experience: ExperienceNode to evaluate
            context: Current drive context
            
        Returns:
            Significance multiplier (1.0 = normal, >1.0 = more important, <1.0 = less important)
        """
        # Base implementation - drives should override for domain-specific significance
        return 1.0
    
    def evaluate_experience_valence(self, experience, context: DriveContext) -> float:
        """
        Evaluate whether this experience was good (+) or bad (-) for this drive.
        
        This is the "pain/pleasure" signal - drives should override this to provide
        domain-specific feedback about whether outcomes were positive or negative.
        
        Args:
            experience: ExperienceNode to evaluate
            context: Current drive context
            
        Returns:
            Valence score: +1.0 = very good, 0.0 = neutral, -1.0 = very bad (pain)
        """
        # Base implementation - drives should override for domain-specific valence
        return 0.0
    
    def evaluate_action_pain_pleasure(self, action: Dict[str, float], context: DriveContext) -> tuple[float, float]:
        """
        Evaluate the expected pain and pleasure of an action for this drive.
        
        This allows drives to provide predictive emotional feedback about actions
        based on their learned associations and current state.
        
        Args:
            action: Motor action to evaluate
            context: Current drive context
            
        Returns:
            Tuple of (expected_pain, expected_pleasure) where:
            - expected_pain: negative value [-1.0 to 0.0] representing anticipated suffering
            - expected_pleasure: positive value [0.0 to 1.0] representing anticipated satisfaction
        """
        # Base implementation - drives should override for domain-specific pain/pleasure prediction
        return 0.0, 0.0
    
    def learn_pain_pleasure_association(self, action: Dict[str, float], context: DriveContext, 
                                      outcome_pain: float, outcome_pleasure: float):
        """
        Learn from pain/pleasure outcomes to improve future action predictions.
        
        This allows drives to build their own associative learning about which actions
        lead to satisfying or dissatisfying outcomes for their specific domain.
        
        Args:
            action: The action that was taken
            context: The context when the action was taken
            outcome_pain: The actual pain experienced [-1.0 to 0.0]
            outcome_pleasure: The actual pleasure experienced [0.0 to 1.0]
        """
        # Base implementation - drives can override for domain-specific learning
        pass
    
    def record_action_execution(self, action: Dict[str, float], context: DriveContext):
        """
        Record that an action was executed to update drive-specific learning.
        
        This allows drives to track action familiarity, success rates, and other
        execution-related learning for their specific domain.
        
        Args:
            action: The action that was executed
            context: The context when the action was executed
        """
        # Base implementation - drives can override for domain-specific recording
        pass
    
    def get_current_mood_contribution(self, context: DriveContext) -> Dict[str, float]:
        """
        Get this drive's contribution to overall robot mood.
        
        Returns:
            Dictionary with mood dimensions and their values [-1.0 to +1.0]
        """
        # Base implementation - drives should override for domain-specific mood
        return {
            'satisfaction': 0.0,  # How satisfied this drive currently is
            'urgency': 0.0,       # How urgent this drive's needs are
            'confidence': 0.0     # How confident this drive is about current state
        }
    
    def initialize_with_brainstem(self, sensor_semantics):
        """
        Initialize drive with sensor semantics from brainstem.
        
        This allows the drive to discover which sensors are relevant to its purposes
        without hardcoding sensor indices.
        """
        from core.sensor_semantics import SensorPurpose
        
        self.sensor_semantics = sensor_semantics
        self.discovered_sensors = {}
        
        # Discover sensors relevant to this drive's purposes
        relevant_purposes = self._get_relevant_sensor_purposes()
        
        for purpose in relevant_purposes:
            sensor_indices = sensor_semantics.get_sensors_for_purpose(purpose)
            if sensor_indices:
                self.discovered_sensors[purpose.value] = sensor_indices
                print(f"{self.name} drive discovered {len(sensor_indices)} sensors for {purpose.value}: {sensor_indices}")
    
    def _get_relevant_sensor_purposes(self):
        """
        Return the sensor purposes that this drive cares about.
        
        Subclasses should override this to specify which purposes are relevant.
        """
        # Base implementation returns empty - subclasses should override
        return []
    
    def get_sensors_for_purpose(self, purpose_name: str) -> List[int]:
        """Get sensor indices that serve a specific purpose for this drive."""
        return self.discovered_sensors.get(purpose_name, [])
    
    def extract_sensor_values(self, context: DriveContext, purpose_name: str) -> List[float]:
        """Extract sensor values for a specific purpose from the context."""
        if not hasattr(context, 'current_sensory') or not context.current_sensory:
            return []
        
        sensor_indices = self.get_sensors_for_purpose(purpose_name)
        values = []
        
        for idx in sensor_indices:
            if 0 <= idx < len(context.current_sensory):
                values.append(context.current_sensory[idx])
        
        return values
    
    def get_sensor_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about sensor discovery for this drive."""
        return {
            'discovered_purposes': list(self.discovered_sensors.keys()),
            'total_discovered_sensors': sum(len(sensors) for sensors in self.discovered_sensors.values()),
            'sensor_mappings': self.discovered_sensors.copy(),
            'has_sensor_semantics': self.sensor_semantics is not None
        }