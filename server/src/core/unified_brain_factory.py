"""
Pure Field Brain Factory

Creates PureFieldBrain instances for real intelligence research.
Standardized on the ultimate synthesis - GPU-optimal, biologically-inspired,
emergent field dynamics without architectural complexity.
"""

import time
from typing import Dict, Any, Optional
import torch
import logging

from ..core.interfaces import IBrainFactory, IBrain
from ..brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS
from .simple_dimension_calculator import SimpleDimensionCalculator
from ..parameters.cognitive_config import get_cognitive_config

logger = logging.getLogger(__name__)


class PureFieldBrainWrapper(IBrain):
    """
    Wrapper that implements IBrain interface for PureFieldBrain.
    Standardized on PureFieldBrain - the ultimate synthesis.
    """
    
    def __init__(self, brain: PureFieldBrain):
        self.brain = brain
        self.creation_time = time.time()
        self.total_cycles = 0
        
    def process_field_dynamics(self, field_input) -> Any:
        """Process field dynamics through PureFieldBrain and return field output."""
        # Convert to tensor if needed
        if isinstance(field_input, torch.Tensor):
            sensory_input = field_input
        else:
            sensory_input = torch.tensor(field_input, dtype=torch.float32, device=self.brain.device)
            
        # Process through PureFieldBrain (uses forward method)
        output_tensor = self.brain.forward(sensory_input)
        action_output = output_tensor.cpu().tolist()
        brain_state = {'cycle': self.brain.cycle_count}
        
        # Store for direct access
        self._last_motor_commands = action_output
        self._last_brain_state = brain_state
        
        return output_tensor
    
    def get_motor_commands(self) -> list:
        """Get motor commands directly without field space conversion."""
        return getattr(self, '_last_motor_commands', [])
    
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        # Return actual motor dimensions instead of fake 26D
        return len(self.get_motor_commands()) if hasattr(self, '_last_motor_commands') else 5
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state from PureFieldBrain."""
        # Use the brain's metrics if available
        if hasattr(self.brain, 'metrics'):
            metrics = self.brain.metrics
            return {
                'cycle': self.brain.cycle_count,
                'field_energy': metrics.get('field_energy', 0.0),
                'field_mean': metrics.get('field_mean', 0.0),
                'field_std': metrics.get('field_std', 0.0),
                'prediction_error': metrics.get('prediction_error', 0.0),
                'device': str(self.brain.device),
                'tensor_shape': list(self.brain.field.shape),
                'brain_type': 'pure'
            }
        
        # Fallback to basic state
        field = self.brain.field
        return {
            'cycle': self.brain.cycle_count,
            'field_energy': float(torch.mean(torch.abs(field))),
            'device': str(self.brain.device),
            'tensor_shape': list(field.shape),
            'brain_type': 'pure'
        }
    
    def _create_brain_state(self) -> Dict[str, Any]:
        """Delegate to underlying brain's telemetry method."""
        if hasattr(self.brain, '_create_brain_state'):
            return self.brain._create_brain_state()
        return self.get_brain_state()
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get field statistics from PureFieldBrain."""
        field = self.brain.field
        return {
            'field_energy': float(torch.mean(torch.abs(field))),
            'max_activation': float(torch.max(torch.abs(field))),
            'field_variance': float(torch.var(field)),
            'active_regions': torch.sum(torch.abs(field) > 0.1).item(),
            'brain_type': 'pure'
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        # Use PureFieldBrain's built-in state management
        state_dict = self.brain.get_state_dict()
        return {
            'brain_state': state_dict,
            'brain_cycles': self.brain.cycle_count,
            'creation_time': self.creation_time,
            'brain_type': 'pure'
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from persistence."""
        # Use PureFieldBrain's built-in state loading
        if 'brain_state' in state:
            self.brain.load_state_dict(state['brain_state'])
        elif 'field' in state:
            # Legacy format - convert to PureFieldBrain format
            field_data = torch.tensor(state['field'])
            if field_data.shape == self.brain.field.shape:
                self.brain.field = field_data.to(self.brain.device)
        
        # Set cycle count
        if 'brain_cycles' in state:
            self.brain.cycle_count = state['brain_cycles']
        
        # Set creation time
        if 'creation_time' in state:
            self.creation_time = state['creation_time']


class PureFieldBrainFactory(IBrainFactory):
    """
    Factory for creating PureFieldBrain instances.
    
    Standardized on the ultimate synthesis:
    - GPU-optimal field computation
    - Biologically-inspired dynamics  
    - Emergent intelligence through scale
    - No architectural complexity - pure field dynamics
    """
    
    def __init__(self, brain_config: Optional[Dict[str, Any]] = None):
        """Initialize factory."""
        self.brain_config = brain_config or {}
        self.cognitive_config = get_cognitive_config()
        self.calculator = SimpleDimensionCalculator()
        
        # Log initialization (quiet by default for clean startup)
        if not self.brain_config.get('quiet_mode', False):
            logger.info("PureFieldBrain factory initialized - focused on real intelligence")
    
    def create(self, 
               field_dimensions: Optional[int] = None,
               spatial_resolution: Optional[int] = None,
               sensory_dim: int = 16,
               motor_dim: int = 5) -> IBrain:
        """
        Create a PureFieldBrain instance.
        
        Args:
            field_dimensions: Ignored - PureFieldBrain uses 4D tensors
            spatial_resolution: Spatial resolution (default 32)
            sensory_dim: Number of input sensors
            motor_dim: Number of output motors
            
        Returns:
            PureFieldBrain instance wrapped in IBrain interface
        """
        # Use provided resolution or default
        if spatial_resolution is None:
            spatial_resolution = (self.brain_config.get('spatial_resolution') or 
                                self.brain_config.get('field_spatial_resolution') or 
                                32)
        
        quiet_mode = self.brain_config.get('quiet_mode', False)
        
        # Create PureFieldBrain - the ultimate synthesis
        # Choose scale based on available resources
        import torch
        if torch.cuda.is_available():
            # GPU available - use medium scale for good balance
            scale_config = SCALE_CONFIGS.get('medium')
            if not quiet_mode:
                logger.info(f"🎯 Using MEDIUM scale config on GPU - optimal for real-time learning")
        else:
            # CPU only - use small scale for performance
            scale_config = SCALE_CONFIGS.get('small')
            if not quiet_mode:
                logger.info(f"⚡ Using SMALL scale config on CPU - optimized for performance")
        
        brain = PureFieldBrain(
            input_dim=sensory_dim,
            output_dim=motor_dim,
            scale_config=scale_config,
            device=None,  # Auto-select best device
            aggressive=True  # Use aggressive parameters for real intelligence
        )
        
        if not quiet_mode:
            logger.info("🧠 PureFieldBrain initialized - The ultimate synthesis for real intelligence")
            
        # Wrap in interface
        return PureFieldBrainWrapper(brain)
    
    def get_brain_types(self) -> list:
        """Get available brain types - now only pure."""
        return ['pure']
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get factory configuration."""
        return {
            'brain_type': 'pure',
            'tensor_architecture': '4D',
            'gpu_optimized': True,
            'spatial_resolution': self.brain_config.get('field_spatial_resolution', 32),
            'status': 'THE ULTIMATE SYNTHESIS',
            'notes': 'Real intelligence through pure field dynamics'
        }


# Compatibility aliases for existing code
UnifiedBrainFactory = PureFieldBrainFactory
UnifiedBrainWrapper = PureFieldBrainWrapper