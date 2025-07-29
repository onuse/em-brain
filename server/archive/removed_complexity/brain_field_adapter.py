#!/usr/bin/env python3
"""
Brain Field Adapter

Adapts the DynamicUnifiedFieldBrain to work with EnhancedFieldDynamics
by implementing the FieldImplementation interface.
"""

import torch
from typing import Dict, List, Optional, Any
from .field_types import UnifiedFieldExperience, FieldDimension


class BrainFieldAdapter:
    """
    Adapter that allows DynamicUnifiedFieldBrain to work with EnhancedFieldDynamics.
    """
    
    def __init__(self, brain):
        """
        Initialize adapter with reference to brain.
        
        Args:
            brain: DynamicUnifiedFieldBrain instance
        """
        self.brain = brain
        self.field_device = brain.device
    
    def imprint_experience(self, experience: UnifiedFieldExperience) -> None:
        """Imprint an experience into the field."""
        # Use the brain's existing imprint method
        self.brain._imprint_unified_experience(experience)
    
    def evolve_field(self, dt: float = 0.1, current_input_stream: Optional[List[float]] = None) -> None:
        """Evolve the field dynamics over time."""
        # This is handled by the brain's main evolution loop
        pass
    
    def compute_field_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute gradients across field dimensions."""
        # Return the brain's gradient flows
        return self.brain.gradient_flows
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive field statistics."""
        # Calculate basic field statistics
        total_activation = torch.sum(torch.abs(self.brain.unified_field)).item()
        mean_activation = torch.mean(torch.abs(self.brain.unified_field)).item()
        
        return {
            'total_activation': total_activation,
            'mean_activation': mean_activation,
            'field_energy': total_activation / self.brain.unified_field.numel(),
            'topology_regions': len(self.brain.topology_regions),
            'brain_cycles': self.brain.brain_cycles
        }
    
    def get_field_state_summary(self) -> Dict[str, Any]:
        """Get current field state summary."""
        return {
            'field_shape': self.brain.unified_field.shape,
            'device': str(self.brain.device),
            'prediction_confidence': self.brain._current_prediction_confidence,
            'spontaneous_enabled': self.brain.spontaneous_enabled,
            'enhanced_dynamics_enabled': getattr(self.brain, 'enhanced_dynamics_enabled', False)
        }
    
    def get_implementation_type(self) -> str:
        """Return the implementation type for debugging/logging."""
        return "DynamicUnifiedFieldBrain"