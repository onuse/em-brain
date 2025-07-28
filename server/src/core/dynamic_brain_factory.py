"""
Dynamic Brain Factory implementation.

Creates brain instances with dynamic dimensions, only when needed.
This replaces the old BrainFactory that created brains at startup.
"""

import time
from typing import Dict, Any, Optional

from ..core.interfaces import IBrainFactory, IBrain
from .dynamic_dimension_calculator import DynamicDimensionCalculator
from ..parameters.cognitive_config import get_cognitive_config


class DynamicBrainWrapper(IBrain):
    """
    Wrapper around DynamicUnifiedFieldBrain that implements the IBrain interface.
    """
    
    def __init__(self, unified_field_brain):
        self.brain = unified_field_brain
        self.creation_time = time.time()
        self.total_cycles = 0
    
    def process_field_dynamics(self, field_input) -> Any:
        """Process field dynamics and return field output."""
        # Convert field input to sensory format for the brain
        
        import torch
        
        # Ensure field_input is a tensor
        if not isinstance(field_input, torch.Tensor):
            field_input = torch.tensor(field_input, dtype=torch.float32)
        
        # Get dimensions
        field_dims = self.get_field_dimensions()
        sensory_dim = self.brain.expected_sensory_dim
        motor_dim = self.brain.expected_motor_dim
        
        # Convert field tensor to sensory list
        # Take the first sensory_dim values from field space
        if len(field_input) >= sensory_dim:
            sensory_input = field_input[:sensory_dim].tolist()
        else:
            # Pad if needed
            sensory_input = field_input.tolist() + [0.0] * (sensory_dim - len(field_input))
        
        # Process through brain
        action_output, brain_state = self.brain.process_robot_cycle(sensory_input)
        
        # Convert motor output back to field space
        field_output = torch.zeros(field_dims)
        
        # Place motor outputs in first motor_dim positions
        for i in range(min(len(action_output), field_dims)):
            field_output[i] = action_output[i]
        
        self.total_cycles += 1
        
        return field_output
    
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        # Return the target field dimensions we want to expose
        if hasattr(self.brain, 'target_field_dimensions'):
            return self.brain.target_field_dimensions
        # Fall back to total_dimensions (the actual internal dimensions)
        return self.brain.total_dimensions
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get brain state - delegate to wrapped brain"""
        if hasattr(self.brain, 'get_brain_state'):
            return self.brain.get_brain_state()
        return {}
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get field statistics - delegate to wrapped brain"""
        if hasattr(self.brain, 'get_field_statistics'):
            return self.brain.get_field_statistics()
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get brain state for persistence."""
        return {
            'field_dimensions': self.get_field_dimensions(),
            'unified_field': self.brain.unified_field.cpu().numpy().tolist(),
            'total_cycles': self.total_cycles,
            'brain_cycles': self.brain.brain_cycles,
            'creation_time': self.creation_time
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load brain state from persistence."""
        import torch
        
        if 'unified_field' in state:
            field_data = torch.tensor(state['unified_field'])
            if field_data.shape == self.brain.unified_field.shape:
                self.brain.unified_field = field_data.to(self.brain.device)
        
        if 'total_cycles' in state:
            self.total_cycles = state['total_cycles']
        
        if 'brain_cycles' in state:
            self.brain.brain_cycles = state['brain_cycles']


class DynamicBrainFactory(IBrainFactory):
    """
    Creates brain instances with dynamic dimensions.
    
    This factory creates brain instances only when requested,
    with dimensions tailored to the specific robot profile.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.quiet_mode = self.config.get('quiet_mode', False)
        # Simple brain option removed - using only dynamic unified brain
        # Dynamic brain is now the only option
        
        # Load cognitive configuration
        self.cognitive_config = get_cognitive_config(quiet=True)
        self.brain_config = self.cognitive_config.brain_config
        
        # Initialize dimension calculator
        self.dimension_calculator = DynamicDimensionCalculator(
            complexity_factor=self.config.get('complexity_factor', 6.0)
        )
        
        # Always use the dynamic brain - non-dynamic brain archived
        from ..brains.field.dynamic_unified_brain import DynamicUnifiedFieldBrain
        self.DynamicUnifiedFieldBrain = DynamicUnifiedFieldBrain
    
    def create(self, field_dimensions: int, spatial_resolution: int,
               sensory_dim: Optional[int] = None, motor_dim: Optional[int] = None) -> IBrain:
        """Create a new brain with specified dimensions."""
        
        # Dynamic brain requires sensory and motor dimensions from robot
        if sensory_dim and motor_dim:
            # Calculate conceptual dimensions based on robot profile
            conceptual_dims = self.dimension_calculator.calculate_dimensions(
                sensory_dim, motor_dim
            )
            
            # Select appropriate tensor configuration
            tensor_shape = self.dimension_calculator.select_tensor_configuration(
                len(conceptual_dims), spatial_resolution
            )
            
            # Create dimension mapping
            dimension_mapping = self.dimension_calculator.create_dimension_mapping(
                conceptual_dims, tensor_shape
            )
            
            print(f"🏗️  Creating dynamic brain: {len(conceptual_dims)}D conceptual → {len(tensor_shape)}D tensor")
            print(f"   Robot interface: {sensory_dim}D sensors → {motor_dim}D motors")
            
            # Create dynamic brain
            brain = self.DynamicUnifiedFieldBrain(
                field_dimensions=conceptual_dims,
                tensor_shape=tensor_shape,
                dimension_mapping=dimension_mapping,
                spatial_resolution=spatial_resolution,
                temporal_window=self.config.get('temporal_window', 10.0),
                field_evolution_rate=self.config.get('field_evolution_rate', self.brain_config.field_evolution_rate),
                constraint_discovery_rate=self.config.get('constraint_discovery_rate', self.brain_config.constraint_discovery_rate),
                quiet_mode=self.quiet_mode,
                enable_attention=self.config.get('enable_attention', None),
                enable_emergent_navigation=self.config.get('emergent_navigation', None),
                pattern_motor=self.config.get('pattern_motor', None),
                pattern_attention=self.config.get('pattern_attention', None)
            )
            
            # Set robot interface dimensions
            brain.expected_sensory_dim = sensory_dim
            brain.expected_motor_dim = motor_dim
            
            # Apply blended reality if enabled
            if hasattr(brain, 'blended_reality_enabled') and brain.blended_reality_enabled:
                from ..brains.field.blended_reality import integrate_blended_reality
                brain = integrate_blended_reality(brain)
                if not self.quiet_mode:
                    print(f"   Blended reality: Integrated")
            
            if not self.quiet_mode:
                print(f"✅ Created dynamic unified field brain")
            return DynamicBrainWrapper(brain)
        
        # No fallback - dynamic brain requires robot to specify its capabilities
        raise ValueError(
            "Dynamic brain requires sensory_dim and motor_dim to be specified. "
            "The robot must tell the brain about its capabilities."
        )
