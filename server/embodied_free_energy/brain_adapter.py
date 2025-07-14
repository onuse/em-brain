"""
Brain Adapter for Embodied Free Energy System

Provides a clean interface between the embodied Free Energy system
and the existing 4-system minimal brain, allowing seamless integration
without modifying the core brain architecture.
"""

from typing import Any, List, Dict
import numpy as np


class EmbodiedBrainAdapter:
    """
    Adapter that makes the 4-system brain compatible with embodied Free Energy.
    
    This preserves the brain's purity while providing the prediction interface
    that the embodied system needs.
    """
    
    def __init__(self, brain):
        """
        Initialize adapter with the 4-system brain.
        
        Args:
            brain: MinimalBrain instance (4-system architecture)
        """
        self.brain = brain
        
    def predict(self, sensory_input: Any, action: Any) -> 'PredictionOutcome':
        """
        Predict outcome of taking action in given sensory context.
        
        Converts embodied Free Energy prediction requests into format
        the 4-system brain can handle.
        """
        # Convert sensory input to format brain expects
        sensory_vector = self._sensory_to_vector(sensory_input)
        
        # Convert action to format brain expects  
        action_vector = self._action_to_vector(action)
        
        try:
            # Get brain's prediction
            predicted_action, brain_state = self.brain.process_sensory_input(
                sensory_vector, 
                action_dimensions=len(action_vector) if action_vector else 2
            )
            
            # Extract confidence from brain state
            confidence = brain_state.get('prediction_confidence', 0.7)
            
            # Return prediction in embodied system format
            return PredictionOutcome(
                action=action,
                predicted_sensory_outcome=predicted_action,
                confidence=confidence,
                brain_state=brain_state
            )
            
        except Exception as e:
            # Graceful fallback if brain interface changes
            return PredictionOutcome(
                action=action,
                predicted_sensory_outcome=None,
                confidence=0.5,
                brain_state={'error': str(e)}
            )
    
    def _sensory_to_vector(self, sensory_input: Any) -> List[float]:
        """Convert various sensory input formats to vector."""
        
        if isinstance(sensory_input, (list, tuple, np.ndarray)):
            return list(sensory_input)
        
        elif hasattr(sensory_input, '__dict__'):
            # Object with attributes - extract numeric values
            values = []
            for key, value in sensory_input.__dict__.items():
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif key in ['battery', 'obstacle_distance', 'location']:
                    # Handle common robot state attributes
                    if isinstance(value, (list, tuple)):
                        values.extend([float(v) for v in value])
                    else:
                        values.append(float(value))
            return values if values else [0.5, 0.5, 0.0, 0.0]
        
        elif isinstance(sensory_input, dict):
            # Dictionary format
            values = []
            for key in ['battery', 'obstacle_distance', 'x', 'y', 'heading']:
                if key in sensory_input:
                    value = sensory_input[key]
                    if isinstance(value, (list, tuple)):
                        values.extend([float(v) for v in value])
                    else:
                        values.append(float(value))
            return values if values else [0.5, 0.5, 0.0, 0.0]
        
        else:
            # Default sensory input
            return [0.5, 0.5, 0.0, 0.0]
    
    def _action_to_vector(self, action: Any) -> List[float]:
        """Convert action dictionary to vector format."""
        
        if isinstance(action, dict):
            action_type = action.get('type', 'stop')
            
            if action_type == 'move':
                # Movement: [speed, direction_encoding]
                speed = action.get('speed', 0.5)
                direction = action.get('direction', 'forward')
                
                # Encode direction as angle
                direction_angles = {
                    'forward': 0.0,
                    'backward': 1.0,  # 180 degrees
                    'left': 0.75,     # 270 degrees  
                    'right': 0.25     # 90 degrees
                }
                direction_encoding = direction_angles.get(direction, 0.0)
                
                return [speed, direction_encoding]
            
            elif action_type == 'rotate':
                # Rotation: [0, normalized_angle]
                angle = action.get('angle', 0)
                normalized_angle = (angle + 180) / 360  # Normalize to 0-1
                return [0.0, normalized_angle]
            
            elif action_type == 'stop':
                # Stop: minimal values
                duration = action.get('duration', 1.0)
                return [0.0, duration / 10.0]  # Normalize duration
            
            elif action_type == 'seek_charger':
                # Energy seeking: encoded as specific pattern
                urgency = action.get('urgency', 'moderate')
                urgency_values = {'low': 0.3, 'moderate': 0.6, 'high': 0.9}
                urgency_value = urgency_values.get(urgency, 0.6)
                return [urgency_value, 0.8]  # High second value indicates energy-seeking
            
            else:
                # Unknown action type
                return [0.0, 0.0]
        
        elif isinstance(action, (list, tuple, np.ndarray)):
            # Already in vector format
            return list(action)
        
        else:
            # Default action
            return [0.0, 0.0]


class PredictionOutcome:
    """
    Outcome of brain prediction suitable for embodied Free Energy calculations.
    """
    
    def __init__(self, action: Any, predicted_sensory_outcome: Any, 
                 confidence: float, brain_state: Dict[str, Any]):
        self.action = action
        self.predicted_sensory_outcome = predicted_sensory_outcome
        self.confidence = confidence
        self.brain_state = brain_state
        
        # For compatibility with existing motivation system interfaces
        self.similarity_score = confidence
    
    def __str__(self):
        return f"Prediction(action={self.action}, confidence={self.confidence:.3f})"


class StateExtractor:
    """
    Utility class for extracting structured information from various state formats.
    """
    
    @staticmethod
    def extract_battery_level(state: Any) -> float:
        """Extract battery level from various state formats."""
        
        if hasattr(state, 'battery'):
            return float(state.battery)
        elif hasattr(state, 'battery_percentage'):
            return float(state.battery_percentage)
        elif isinstance(state, dict):
            return float(state.get('battery', state.get('battery_percentage', 0.7)))
        else:
            return 0.7  # Default reasonable value
    
    @staticmethod
    def extract_obstacle_distance(state: Any) -> float:
        """Extract obstacle distance from various state formats."""
        
        if hasattr(state, 'obstacle_distance'):
            return float(state.obstacle_distance)
        elif hasattr(state, 'distance_to_obstacle'):
            return float(state.distance_to_obstacle)
        elif isinstance(state, dict):
            return float(state.get('obstacle_distance', 
                                 state.get('distance_to_obstacle', 50.0)))
        else:
            return 50.0  # Default safe distance
    
    @staticmethod
    def extract_location(state: Any) -> tuple:
        """Extract position from various state formats."""
        
        if hasattr(state, 'location') and isinstance(state.location, (list, tuple)):
            return tuple(state.location[:2])
        elif hasattr(state, 'x') and hasattr(state, 'y'):
            return (float(state.x), float(state.y))
        elif isinstance(state, dict):
            if 'location' in state and isinstance(state['location'], (list, tuple)):
                return tuple(state['location'][:2])
            elif 'x' in state and 'y' in state:
                return (float(state['x']), float(state['y']))
        
        return (0.0, 0.0)  # Default origin