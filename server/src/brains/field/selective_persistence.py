"""
Selective Persistence - The Memory Foundation

Patterns that successfully predict the future become stable.
This creates memory without explicit storage - successful patterns simply persist.
"""

import torch
from typing import Optional


class SelectivePersistence:
    """
    Enable memory through prediction-based stability.
    
    Core principle: Patterns that predict well decay slower.
    This creates natural memory consolidation without explicit storage.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device):
        """
        Initialize selective persistence mechanism.
        
        Args:
            field_shape: Shape of the field tensor
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Pattern stability map - tracks how stable each region should be
        # 1.0 = highly stable (good predictions), 0.5 = unstable (poor predictions)
        self.pattern_stability = torch.ones(field_shape, device=device)
        
        # Running average of local prediction accuracy
        self.prediction_confidence = torch.ones(field_shape, device=device)
        
        # Parameters
        self.stability_learning_rate = 0.05  # How fast stability adapts
        self.min_stability = 0.5  # Minimum stability (even bad patterns don't disappear instantly)
        self.max_stability = 0.99  # Maximum stability (even good patterns slowly fade)
        
    def update_stability(self, field: torch.Tensor, prediction_error: float) -> None:
        """
        Update pattern stability based on prediction success.
        
        Args:
            field: Current field state
            prediction_error: How wrong the last prediction was (0 = perfect, 1 = terrible)
        """
        # Compute local field activity (which patterns are active)
        field_activity = torch.abs(field)
        activity_normalized = field_activity / (field_activity.mean() + 1e-6)
        
        # Active regions that predicted poorly lose stability
        # Active regions that predicted well gain stability
        stability_update = 1.0 - prediction_error  # Good prediction = high value
        
        # Only update stability where patterns are active
        # This prevents inactive regions from gaining false confidence
        self.prediction_confidence = (
            0.95 * self.prediction_confidence + 
            0.05 * stability_update * activity_normalized
        )
        
        # Update stability map based on prediction confidence
        target_stability = (
            self.min_stability + 
            (self.max_stability - self.min_stability) * self.prediction_confidence
        )
        
        # Smooth adaptation (avoid sudden changes)
        self.pattern_stability = (
            (1 - self.stability_learning_rate) * self.pattern_stability +
            self.stability_learning_rate * target_stability
        )
        
        # Ensure bounds
        self.pattern_stability = torch.clamp(
            self.pattern_stability, 
            self.min_stability, 
            self.max_stability
        )
    
    def apply_selective_decay(self, field: torch.Tensor, base_decay: float = 0.995) -> torch.Tensor:
        """
        Apply spatially-varying decay based on pattern stability.
        
        Stable patterns (good predictors) decay slowly.
        Unstable patterns (poor predictors) decay quickly.
        
        Args:
            field: Current field state
            base_decay: Base decay rate for unstable patterns
            
        Returns:
            Field with selective decay applied
        """
        # Compute adaptive decay rate
        # Stable regions: decay = ~0.999 (very slow)
        # Unstable regions: decay = base_decay (normal)
        adaptive_decay = base_decay + (1.0 - base_decay) * self.pattern_stability
        
        # Apply spatially-varying decay
        field = field * adaptive_decay
        
        return field
    
    def get_memory_map(self) -> torch.Tensor:
        """
        Get a map of where memories are forming.
        
        High values indicate regions with stable, persistent patterns.
        Low values indicate volatile, reactive regions.
        """
        return self.pattern_stability
    
    def get_memory_utilization(self) -> float:
        """
        Measure how much of the field has developed stable memories.
        
        Returns:
            Fraction of field with high stability (0-1)
        """
        high_stability = (self.pattern_stability > 0.8).float()
        return high_stability.mean().item()
    
    def reset_region(self, x: int, y: int, z: int, radius: int = 3):
        """
        Reset stability in a specific region (useful for forgetting).
        
        Args:
            x, y, z: Center of region to reset
            radius: Radius of reset region
        """
        # Create mask for region
        xx, yy, zz = torch.meshgrid(
            torch.arange(self.field_shape[0], device=self.device),
            torch.arange(self.field_shape[1], device=self.device),
            torch.arange(self.field_shape[2], device=self.device),
            indexing='ij'
        )
        
        distance = torch.sqrt((xx - x)**2 + (yy - y)**2 + (zz - z)**2)
        mask = distance <= radius
        
        # Reset stability in this region
        self.pattern_stability[mask] = self.min_stability
        self.prediction_confidence[mask] = 0.5