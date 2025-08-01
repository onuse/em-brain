"""
Simplified Brain Factory

Creates simplified 4D tensor brains for GPU optimization.
"""

import time
from typing import Dict, Any, Optional
import torch

from ..core.interfaces import IBrainFactory, IBrain
from ..brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from .simple_dimension_calculator import SimpleDimensionCalculator
from ..parameters.cognitive_config import get_cognitive_config


class SimplifiedBrainWrapper(IBrain):
    """
    Wrapper around SimplifiedUnifiedBrain that implements IBrain interface.
    """
    
    def __init__(self, brain: SimplifiedUnifiedBrain):
        self.brain = brain
        self.creation_time = time.time()
        self.total_cycles = 0
        
    def process_field_dynamics(self, field_input) -> Any:
        """Process field dynamics and return field output."""
        # Convert to list if tensor
        if isinstance(field_input, torch.Tensor):
            sensory_input = field_input.tolist()
        else:
            sensory_input = field_input
            
        # Process through brain
        action_output, brain_state = self.brain.process_robot_cycle(sensory_input)
        
        # Store motor commands for direct access
        self._last_motor_commands = action_output
        self._last_brain_state = brain_state
        
        # Return motor commands as tensor (no padding needed)
        return torch.tensor(action_output, dtype=torch.float32)
    
    def get_motor_commands(self) -> list:
        """Get motor commands directly without field space conversion."""
        return getattr(self, '_last_motor_commands', [])
    
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        # Return actual motor dimensions instead of fake 26D
        return len(self.get_motor_commands()) if hasattr(self, '_last_motor_commands') else 5
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state."""
        # Use the brain's full telemetry method if available
        if hasattr(self.brain, '_create_brain_state'):
            return self.brain._create_brain_state()
        
        # Fallback to basic state
        return {
            'cycle': self.brain.brain_cycles,
            'cycle_time_ms': self.brain._last_cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.brain.unified_field))),
            'device': str(self.brain.device),
            'tensor_shape': self.brain.tensor_shape
        }
    
    def _create_brain_state(self) -> Dict[str, Any]:
        """Delegate to underlying brain's telemetry method."""
        if hasattr(self.brain, '_create_brain_state'):
            return self.brain._create_brain_state()
        return self.get_brain_state()
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get field statistics."""
        field = self.brain.unified_field
        return {
            'field_energy': float(torch.mean(torch.abs(field))),
            'max_activation': float(torch.max(torch.abs(field))),
            'field_variance': float(torch.var(field)),
            'active_regions': torch.sum(torch.abs(field) > 0.1).item()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'unified_field': self.brain.unified_field.cpu().numpy().tolist(),
            'brain_cycles': self.brain.brain_cycles,
            'creation_time': self.creation_time
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from persistence."""
        if 'unified_field' in state:
            field_data = torch.tensor(state['unified_field'])
            if field_data.shape == self.brain.unified_field.shape:
                self.brain.unified_field = field_data.to(self.brain.device)
        
        if 'brain_cycles' in state:
            self.brain.brain_cycles = state['brain_cycles']


class SimplifiedBrainFactory(IBrainFactory):
    """
    Factory for creating simplified 4D tensor brains.
    """
    
    def __init__(self, brain_config: Optional[Dict[str, Any]] = None):
        """Initialize factory."""
        self.brain_config = brain_config or {}
        self.cognitive_config = get_cognitive_config()
        self.calculator = SimpleDimensionCalculator()
        
        # Check if we should use simplified brain
        self.use_simplified = self.brain_config.get('use_simplified_brain', True)
        
        if not self.brain_config.get('quiet_mode', False):
            print("\n" + "="*60)
            print("🧠 SIMPLIFIED BRAIN FACTORY")
            print("="*60)
            print("✅ 4D Tensor Architecture")
            print("✅ GPU Optimized")
            print("✅ All properties emerge from unified field")
            print("="*60 + "\n")
    
    def create(self, 
               field_dimensions: Optional[int] = None,
               spatial_resolution: Optional[int] = None,
               sensory_dim: int = 16,
               motor_dim: int = 5) -> IBrain:
        """
        Create a simplified brain instance.
        
        Args:
            field_dimensions: Ignored - we use fixed 4D
            spatial_resolution: Spatial resolution (default 32)
            sensory_dim: Number of sensors
            motor_dim: Number of motors
            
        Returns:
            Brain instance wrapped in IBrain interface
        """
        # Use provided resolution or default
        if spatial_resolution is None:
            spatial_resolution = self.brain_config.get('field_spatial_resolution', 32)
            
        # Get tensor shape
        tensor_shape, conceptual_dims = self.calculator.calculate_tensor_shape(
            sensory_dim, motor_dim
        )
        
        # Create simplified brain
        brain = SimplifiedUnifiedBrain(
            sensory_dim=sensory_dim,
            motor_dim=motor_dim,
            spatial_resolution=spatial_resolution,
            device=None,  # Auto-select best device
            quiet_mode=self.brain_config.get('quiet_mode', False)
        )
        
        # Enable all predictive processing phases
        brain.enable_hierarchical_prediction(True)  # Phase 3
        brain.enable_action_prediction(True)        # Phase 4
        brain.enable_active_vision(True)           # Phase 5
        
        # Wrap in interface
        return SimplifiedBrainWrapper(brain)
    
    def get_brain_types(self) -> list:
        """Get available brain types."""
        return ['simplified_field']
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get factory configuration."""
        return {
            'type': 'simplified',
            'tensor_architecture': '4D',
            'gpu_optimized': True,
            'spatial_resolution': self.brain_config.get('field_spatial_resolution', 32)
        }