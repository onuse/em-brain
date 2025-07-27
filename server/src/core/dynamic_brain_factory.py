"""
Dynamic Brain Factory implementation.

Creates brain instances with dynamic dimensions, only when needed.
This replaces the old BrainFactory that created brains at startup.
"""

import time
from typing import Dict, Any, Optional

from ..core.interfaces import IBrainFactory, IBrain
from .dynamic_dimension_calculator import DynamicDimensionCalculator


class DynamicBrainWrapper(IBrain):
    """
    Wrapper around UnifiedFieldBrain that implements the IBrain interface.
    
    This allows us to use the existing UnifiedFieldBrain implementation
    while adapting it to the new dynamic architecture.
    """
    
    def __init__(self, unified_field_brain):
        self.brain = unified_field_brain
        self.creation_time = time.time()
        self.total_cycles = 0
    
    def process_field_dynamics(self, field_input) -> Any:
        """Process field dynamics and return field output."""
        # The UnifiedFieldBrain expects sensory input and returns (action, state)
        # We need to adapt this to pure field dynamics
        
        # For now, we'll convert field input to sensory format
        # This is a temporary adapter until UnifiedFieldBrain is refactored
        
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
        self.use_simple_brain = self.config.get('use_simple_brain', False)
        self.use_dynamic_brain = self.config.get('use_dynamic_brain', True)
        
        # Initialize dimension calculator
        self.dimension_calculator = DynamicDimensionCalculator(
            complexity_factor=self.config.get('complexity_factor', 6.0)
        )
        
        if self.use_simple_brain:
            # Use our clean simple implementation
            from .simple_field_brain import SimpleFieldBrain
            self.BrainClass = SimpleFieldBrain
        else:
            # Import here to avoid circular dependency
            if self.use_dynamic_brain:
                # Check if we should use the full-featured version
                if self.config.get('use_full_features', True):
                    from ..brains.field.dynamic_unified_brain_full import DynamicUnifiedFieldBrain
                else:
                    from ..brains.field.dynamic_unified_brain import DynamicUnifiedFieldBrain
                self.DynamicUnifiedFieldBrain = DynamicUnifiedFieldBrain
            else:
                from ..brains.field.core_brain import UnifiedFieldBrain
                self.UnifiedFieldBrain = UnifiedFieldBrain
    
    def create(self, field_dimensions: int, spatial_resolution: int,
               sensory_dim: Optional[int] = None, motor_dim: Optional[int] = None) -> IBrain:
        """Create a new brain with specified dimensions."""
        
        # Use dynamic brain if enabled
        if self.use_dynamic_brain and sensory_dim and motor_dim:
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
                field_evolution_rate=self.config.get('field_evolution_rate', 0.1),
                constraint_discovery_rate=self.config.get('constraint_discovery_rate', 0.15),
                quiet_mode=self.quiet_mode
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
            
            print(f"✅ Created dynamic unified field brain")
            return DynamicBrainWrapper(brain)
        
        # Otherwise use the compatibility mode
        print(f"🏗️  Creating {field_dimensions}D brain with {spatial_resolution}³ spatial resolution")
        if sensory_dim and motor_dim:
            print(f"   Robot interface: {sensory_dim}D sensors → {motor_dim}D motors")
        
        if self.use_simple_brain:
            # Create simple brain that properly supports dynamic dimensions
            brain = self.BrainClass(
                field_dimensions=field_dimensions,
                spatial_resolution=spatial_resolution,
                sensory_dim=sensory_dim or field_dimensions,
                motor_dim=motor_dim or max(2, field_dimensions // 8)
            )
            print(f"✅ Created {field_dimensions}D simple field brain")
            return brain
        
        # Otherwise create UnifiedFieldBrain with workarounds
        # For now, we create a UnifiedFieldBrain with its current interface
        # In the future, this should be refactored to accept dynamic dimensions
        
        # Create brain with current interface
        brain = self.UnifiedFieldBrain(
            spatial_resolution=spatial_resolution,
            temporal_window=self.config.get('temporal_window', 10.0),
            field_evolution_rate=self.config.get('field_evolution_rate', 0.1),
            constraint_discovery_rate=self.config.get('constraint_discovery_rate', 0.15),
            quiet_mode=self.quiet_mode
        )
        
        # Override the hardcoded dimensions
        # This is a temporary hack until UnifiedFieldBrain is refactored
        brain.total_dimensions = field_dimensions
        
        # Use actual robot dimensions if provided, otherwise use heuristic
        if sensory_dim and motor_dim:
            brain.expected_sensory_dim = sensory_dim
            brain.expected_motor_dim = motor_dim
        else:
            # Fallback heuristic: sensory = 60% of field dims, motor = 15% of field dims
            brain.expected_sensory_dim = max(8, int(field_dimensions * 0.6))
            brain.expected_motor_dim = max(2, int(field_dimensions * 0.15))
        
        # Don't recreate the unified field - let the brain keep its complex structure
        # The brain already created its field with the proper multi-dimensional shape
        # We just need to ensure the dimension mappings work
        
        # Store our target field dimensions as a separate attribute
        # Don't overwrite brain.field_dimensions which is a list of FieldDimension objects
        brain.target_field_dimensions = field_dimensions
        
        # The brain already has its field_dimensions list initialized properly
        # We just need to ensure the total matches what we expect
        if hasattr(brain, 'field_dimensions') and isinstance(brain.field_dimensions, list):
            # Brain already has proper field dimension objects
            pass
        else:
            # Initialize dimension families if the method exists
            if hasattr(brain, '_initialize_dimension_families'):
                brain._initialize_dimension_families()
        
        print(f"✅ Created {field_dimensions}D unified field brain")
        
        # Wrap in adapter
        return DynamicBrainWrapper(brain)