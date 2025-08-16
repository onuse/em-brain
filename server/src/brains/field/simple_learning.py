"""
Simple Learning System

Prediction error creates field tension, not complex learning rules.
The key insight: error makes the field uncomfortable, driving change.
"""

import torch


class SimpleLearning:
    """
    Convert prediction errors into field modifications.
    
    Core principle: Prediction error literally creates discomfort (turbulence)
    in the field, which naturally drives the system to reduce error.
    """
    
    def __init__(self, device: torch.device):
        """Initialize learning system."""
        self.device = device
        self.error_history = []
        
    def error_to_field_tension(self, error: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """
        Convert prediction error into field tension/turbulence.
        
        High error → field becomes turbulent/uncomfortable
        Low error → field remains calm
        
        Args:
            error: Prediction error per sensor
            field: Current field state
            
        Returns:
            Field update that creates appropriate tension
        """
        # Compute error magnitude
        error_magnitude = torch.abs(error).mean().item()
        self.error_history.append(error_magnitude)
        
        # Three types of discomfort from error:
        
        # 1. HEAT - Random activation proportional to error
        # Like the field getting "agitated" when wrong
        heat = torch.randn_like(field) * error_magnitude * 0.1
        
        # 2. WAVES - Oscillations that grow with error
        # Like the field "vibrating" with confusion
        phase_shift = error_magnitude * torch.randn_like(field)
        waves = torch.sin(field + phase_shift) * error_magnitude * 0.05
        
        # 3. SHARPENING - Gradients become more extreme
        # Like the field getting "tense" 
        sharpening = torch.sign(field) * error_magnitude * 0.02
        
        # Combine all three discomforts
        field_tension = heat + waves + sharpening
        
        return field_tension
    
    def get_learning_state(self) -> str:
        """Interpret current learning state."""
        if not self.error_history:
            return "No learning yet"
        
        recent_error = self.error_history[-1] if self.error_history else 0
        
        if recent_error < 0.1:
            return "Predicting well - field calm"
        elif recent_error < 0.3:
            return "Moderate error - field mildly turbulent"
        elif recent_error < 0.5:
            return "High error - field agitated"
        else:
            return "Very high error - field highly turbulent"
    
    def should_explore(self) -> bool:
        """
        Decide if the system should explore based on learning progress.
        
        If we're not learning (error not decreasing), explore more.
        """
        if len(self.error_history) < 10:
            return True  # Always explore initially
        
        # Check if error is decreasing
        recent_errors = self.error_history[-5:]
        older_errors = self.error_history[-10:-5]
        
        recent_avg = sum(recent_errors) / len(recent_errors)
        older_avg = sum(older_errors) / len(older_errors)
        
        # If error isn't improving, explore
        improvement = older_avg - recent_avg
        return improvement < 0.01  # Not improving much