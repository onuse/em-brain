"""
Prediction-Gated Learning - Learn Only From Surprises

The brain only updates when predictions fail. This prevents overfitting
and ensures learning focuses on what's not yet understood.
"""

import torch
from typing import Optional, Dict


class PredictionGatedLearning:
    """
    Gate learning based on prediction accuracy.
    
    Core principle: Only learn when surprised.
    This prevents constant churning and allows stable skills to form.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device):
        """
        Initialize prediction-gated learning.
        
        Args:
            field_shape: Shape of the field tensor
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Track prediction accuracy over time
        self.prediction_history = []
        self.surprise_threshold = 0.01  # Below this error, don't learn (lowered for sensitivity)
        self.max_surprise = 1.0  # Above this error, something's very wrong
        
        # Learning rate modulation based on surprise
        self.base_learning_rate = 0.1
        self.surprise_amplification = 5.0  # How much surprise boosts learning
        
        # Confidence tracking
        self.confidence = torch.ones(field_shape, device=device) * 0.5
        self.confidence_decay = 0.99  # How fast confidence fades without reinforcement
        
    def should_learn(self, prediction_error: float) -> bool:
        """
        Determine if learning should occur based on prediction error.
        
        Args:
            prediction_error: How wrong the prediction was
            
        Returns:
            Whether learning should be triggered
        """
        # Add to history
        self.prediction_history.append(prediction_error)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
        
        # Don't learn if prediction was good enough
        if prediction_error < self.surprise_threshold:
            return False
            
        # Don't learn if error is suspiciously high (likely noise/glitch)
        if prediction_error > self.max_surprise:
            return False
            
        return True
    
    def compute_learning_rate(self, prediction_error: float) -> float:
        """
        Compute adaptive learning rate based on surprise level.
        
        High surprise = high learning rate (big update needed)
        Low surprise = low learning rate (minor adjustment)
        
        Args:
            prediction_error: Current prediction error
            
        Returns:
            Adapted learning rate
        """
        # Normalize error to [0, 1] range
        normalized_error = min(prediction_error / self.max_surprise, 1.0)
        
        # Compute surprise factor (0 = expected, 1 = very surprised)
        if len(self.prediction_history) > 10:
            recent_average = sum(self.prediction_history[-10:]) / 10
            surprise = abs(prediction_error - recent_average) / (recent_average + 1e-6)
        else:
            surprise = normalized_error
        
        # Adaptive learning rate
        learning_rate = self.base_learning_rate * (1 + surprise * self.surprise_amplification)
        
        # Clamp to reasonable range
        return min(learning_rate, 0.5)
    
    def update_confidence(self, field: torch.Tensor, prediction_error: float) -> None:
        """
        Update confidence map based on prediction success.
        
        Regions that predict well gain confidence.
        Confident regions resist change.
        
        Args:
            field: Current field state
            prediction_error: How wrong the prediction was
        """
        # Decay confidence everywhere (use it or lose it)
        self.confidence *= self.confidence_decay
        
        # Boost confidence where predictions were good
        if prediction_error < self.surprise_threshold:
            # Find active regions (they made the prediction)
            active_mask = torch.abs(field) > field.abs().mean()
            
            # Boost their confidence
            confidence_boost = (1.0 - prediction_error) * 0.1
            self.confidence[active_mask] = torch.clamp(
                self.confidence[active_mask] + confidence_boost,
                0.0, 1.0
            )
    
    def gate_learning(self, 
                      field_update: torch.Tensor, 
                      prediction_error: float,
                      field: torch.Tensor) -> torch.Tensor:
        """
        Apply prediction-gated learning to a field update.
        
        Args:
            field_update: Proposed change to field
            prediction_error: Current prediction error
            field: Current field state
            
        Returns:
            Gated field update (may be zero if no learning should occur)
        """
        # Check if we should learn at all
        if not self.should_learn(prediction_error):
            # No learning - return zero update
            return torch.zeros_like(field_update)
        
        # Compute adaptive learning rate
        learning_rate = self.compute_learning_rate(prediction_error)
        
        # Update confidence map
        self.update_confidence(field, prediction_error)
        
        # Apply confidence-based gating
        # High confidence regions resist change
        resistance = self.confidence
        gated_update = field_update * learning_rate * (1.0 - resistance * 0.8)
        
        return gated_update
    
    def get_learning_stats(self) -> Dict[str, float]:
        """
        Get statistics about recent learning.
        
        Returns:
            Dictionary with learning metrics
        """
        if not self.prediction_history:
            return {
                'average_error': 0.0,
                'learning_rate': self.base_learning_rate,
                'confidence': 0.5,
                'should_learn': False
            }
        
        recent_errors = self.prediction_history[-10:] if len(self.prediction_history) >= 10 else self.prediction_history
        avg_error = sum(recent_errors) / len(recent_errors)
        
        return {
            'average_error': avg_error,
            'learning_rate': self.compute_learning_rate(avg_error),
            'confidence': self.confidence.mean().item(),
            'should_learn': avg_error > self.surprise_threshold
        }
    
    def reset_confidence(self):
        """Reset confidence to neutral state."""
        self.confidence = torch.ones(self.field_shape, device=self.device) * 0.5
        self.prediction_history = []