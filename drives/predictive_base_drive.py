"""
Predictive Base Drive - Enhanced drive that evaluates actions based on predicted consequences.

Extends the base drive system to use sensory prediction, transforming drives from
evaluating actions by their properties to evaluating them by their imagined outcomes.

Before: "This action seems safe"
After: "This action will lead to safety"
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from prediction.sensory_predictor import SensoryPredictor, SensoryPrediction


@dataclass
class PredictiveActionEvaluation(ActionEvaluation):
    """Action evaluation enhanced with sensory prediction."""
    sensory_prediction: Optional[SensoryPrediction] = None
    prediction_confidence: float = 0.0
    consequence_score: float = 0.0  # Score based on predicted consequences
    immediate_score: float = 0.0    # Score based on immediate action properties
    
    def __post_init__(self):
        """Calculate combined score from immediate and predictive components."""
        if self.sensory_prediction:
            # Weight prediction vs immediate based on prediction confidence
            prediction_weight = self.prediction_confidence
            immediate_weight = 1.0 - prediction_weight
            
            self.action_score = (self.consequence_score * prediction_weight + 
                                self.immediate_score * immediate_weight)
        else:
            # Fall back to immediate scoring if no prediction available
            self.action_score = self.immediate_score


class PredictiveBaseDrive(BaseDrive):
    """
    Enhanced base drive that evaluates actions based on their predicted consequences.
    
    This transforms drives from reactive to predictive, allowing them to avoid
    bad outcomes before experiencing them.
    """
    
    def __init__(self, name: str, base_weight: float = 1.0, predictor: Optional[SensoryPredictor] = None):
        super().__init__(name, base_weight)
        self.predictor = predictor
        self.enable_prediction = True  # Can be toggled for comparison
        
        # Prediction performance tracking
        self.prediction_successes = 0
        self.prediction_attempts = 0
        self.prediction_benefit_score = 0.0  # How much prediction helped
        
    def set_predictor(self, predictor: SensoryPredictor):
        """Set the sensory predictor for this drive."""
        self.predictor = predictor
    
    def evaluate_action(self, action: Dict[str, float], context: DriveContext) -> PredictiveActionEvaluation:
        """
        Evaluate action using both immediate properties and predicted consequences.
        
        This is the key transformation: instead of just looking at the action,
        we imagine its future and evaluate that future.
        """
        # Get immediate evaluation (original behavior)
        immediate_score = self.evaluate_immediate_action(action, context)
        
        # Get predictive evaluation if predictor available
        prediction = None
        consequence_score = immediate_score  # Default to immediate if no prediction
        prediction_confidence = 0.0
        
        if self.predictor and self.enable_prediction:
            try:
                # Predict sensory consequences of this action
                current_sensors = self._extract_current_sensors(context)
                prediction = self.predictor.predict_sensory_outcome(
                    action=action,
                    current_context=getattr(context, 'mental_context', []),
                    current_sensors=current_sensors
                )
                
                # Evaluate the predicted future
                if prediction and prediction.confidence > 0.2:
                    consequence_score = self.evaluate_predicted_consequences(prediction, context)
                    prediction_confidence = prediction.confidence
                    self.prediction_attempts += 1
                    
            except Exception as e:
                # Graceful fallback if prediction fails
                print(f"Warning: Prediction failed for {self.name}: {e}")
                prediction = None
        
        # Create enhanced evaluation
        evaluation = PredictiveActionEvaluation(
            drive_name=self.name,
            action_score=0.0,  # Will be calculated in __post_init__
            confidence=self._calculate_confidence(immediate_score, prediction),
            urgency=self._calculate_urgency(context),
            reasoning=self._generate_predictive_reasoning(action, prediction, immediate_score, consequence_score),
            sensory_prediction=prediction,
            prediction_confidence=prediction_confidence,
            consequence_score=consequence_score,
            immediate_score=immediate_score
        )
        
        return evaluation
    
    def evaluate_immediate_action(self, action: Dict[str, float], context: DriveContext) -> float:
        """
        Evaluate action based on its immediate properties (original behavior).
        
        This is the fallback when prediction is unavailable or unreliable.
        Subclasses should override this instead of evaluate_action.
        """
        # Default implementation - subclasses should override
        return 0.5
    
    def evaluate_predicted_consequences(self, prediction: SensoryPrediction, context: DriveContext) -> float:
        """
        Evaluate action based on its predicted sensory consequences.
        
        This is where the magic happens: drives can now evaluate imagined futures
        instead of just immediate actions.
        
        Subclasses should override this to implement drive-specific consequence evaluation.
        """
        # Default implementation - subclasses should override
        return 0.5
    
    def _extract_current_sensors(self, context: DriveContext) -> Dict[str, float]:
        """Extract current sensory readings from context."""
        current_sensors = {}
        
        if hasattr(context, 'current_sensory') and context.current_sensory:
            # Convert list to dictionary
            for i, value in enumerate(context.current_sensory):
                current_sensors[str(i)] = value
        
        return current_sensors
    
    def _calculate_confidence(self, immediate_score: float, prediction: Optional[SensoryPrediction]) -> float:
        """Calculate confidence in evaluation."""
        base_confidence = 0.7  # Base confidence in immediate evaluation
        
        if prediction and prediction.confidence > 0.5:
            # High-confidence predictions increase overall confidence
            return min(1.0, base_confidence + (prediction.confidence * 0.3))
        elif prediction and prediction.confidence > 0.2:
            # Low-confidence predictions maintain base confidence
            return base_confidence
        else:
            # No prediction slightly reduces confidence
            return base_confidence * 0.9
    
    def _calculate_urgency(self, context: DriveContext) -> float:
        """Calculate urgency (can be overridden by subclasses)."""
        return 0.5  # Default urgency
    
    def _generate_predictive_reasoning(self, action: Dict[str, float], 
                                     prediction: Optional[SensoryPrediction],
                                     immediate_score: float, 
                                     consequence_score: float) -> str:
        """Generate human-readable reasoning for the evaluation."""
        if prediction and prediction.confidence > 0.3:
            if consequence_score > immediate_score + 0.1:
                return f"Prediction shows positive outcome (confidence: {prediction.confidence:.2f})"
            elif consequence_score < immediate_score - 0.1:
                return f"Prediction shows negative outcome (confidence: {prediction.confidence:.2f})"
            else:
                return f"Prediction confirms immediate assessment (confidence: {prediction.confidence:.2f})"
        else:
            return "No reliable prediction available, using immediate assessment"
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction usage and effectiveness."""
        stats = {
            'predictor_enabled': self.enable_prediction,
            'predictor_available': self.predictor is not None,
            'prediction_attempts': self.prediction_attempts,
            'prediction_successes': self.prediction_successes,
            'prediction_benefit_score': self.prediction_benefit_score
        }
        
        if self.prediction_attempts > 0:
            stats['prediction_success_rate'] = self.prediction_successes / self.prediction_attempts
        
        return stats
    
    def record_prediction_outcome(self, prediction_was_beneficial: bool, benefit_amount: float = 0.0):
        """Record whether a prediction helped make a better decision."""
        if prediction_was_beneficial:
            self.prediction_successes += 1
            self.prediction_benefit_score += benefit_amount
    
    def toggle_prediction(self, enabled: Optional[bool] = None):
        """Toggle prediction on/off for comparison testing."""
        if enabled is None:
            self.enable_prediction = not self.enable_prediction
        else:
            self.enable_prediction = enabled
        
        status = "enabled" if self.enable_prediction else "disabled"
        print(f"🔮 Sensory prediction {status} for {self.name}")
    
    def get_drive_info(self) -> Dict[str, Any]:
        """Get comprehensive drive information including prediction stats."""
        base_info = super().get_drive_info()
        prediction_info = self.get_prediction_statistics()
        
        return {**base_info, 'prediction_stats': prediction_info}