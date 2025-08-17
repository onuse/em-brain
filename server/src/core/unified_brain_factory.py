"""
Unified Field Brain Factory

Creates UnifiedFieldBrain instances with full intelligence mechanisms.
Restored from backup - includes all emergent intelligence systems.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
import torch
import logging

from .interfaces import IBrainFactory, IBrain
from ..brains.field.unified_field_brain import MinimalUnifiedBrain, UnifiedFieldBrain
from .simple_dimension_calculator import SimpleDimensionCalculator
from ..parameters.cognitive_config import get_cognitive_config

logger = logging.getLogger(__name__)


class UnifiedFieldBrainWrapper(IBrain):
    """
    Minimal wrapper to adapt UnifiedFieldBrain to IBrain interface.
    Direct pass-through with no unnecessary abstraction.
    """
    
    def __init__(self, brain: MinimalUnifiedBrain):
        self.brain = brain
        self.creation_time = time.time()
        self.total_cycles = 0
        
        # Expose attributes for parallel injection threads
        self.field = brain.field  # Reference to the field tensor
        self.levels = [brain.field]  # Single level for compatibility
        self.device = brain.device  # Device for tensor operations
        
    @property
    def cycle_count(self):
        """Forward cycle count from wrapped brain."""
        return self.brain.cycle  # TrulyMinimalBrain uses 'cycle'
        
    def process_field_dynamics(self, field_input) -> Any:
        """Process field dynamics through UnifiedFieldBrain."""
        # Convert to list if tensor
        if isinstance(field_input, torch.Tensor):
            sensory_input = field_input.cpu().tolist()
        else:
            sensory_input = field_input
            
        # Process through MinimalUnifiedBrain's process method
        motor_commands, brain_state = self.brain.process(sensory_input)
        
        # Store for direct access
        self._last_motor_commands = motor_commands
        self._last_brain_state = brain_state
        
        # Return as tensor for compatibility
        return torch.tensor(motor_commands, dtype=torch.float32, device=self.brain.device)
    
    def get_motor_commands(self) -> list:
        """Get motor commands directly without field space conversion."""
        return getattr(self, '_last_motor_commands', [])
    
    def get_field_dimensions(self) -> int:
        """Get number of field dimensions."""
        # Return actual motor dimensions instead of fake 26D
        return len(self.get_motor_commands()) if hasattr(self, '_last_motor_commands') else 5
    
    def get_brain_state(self) -> Dict[str, Any]:
        """Get current brain state from UnifiedFieldBrain."""
        # Return the last brain state from process_robot_cycle
        if hasattr(self, '_last_brain_state'):
            return self._last_brain_state
        
        # Fallback to basic state
        field = self.brain.field
        return {
            'cycle': self.brain.cycle,  # TrulyMinimalBrain uses 'cycle'
            'field_energy': float(torch.mean(torch.abs(field))),
            'device': str(self.brain.device),
            'tensor_shape': list(field.shape),
            'brain_type': 'minimal'
        }
    
    def _create_brain_state(self) -> Dict[str, Any]:
        """Delegate to underlying brain's telemetry method."""
        if hasattr(self.brain, '_create_brain_state'):
            return self.brain._create_brain_state()
        return self.get_brain_state()
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get field statistics from UnifiedFieldBrain."""
        field = self.brain.field
        return {
            'field_energy': float(torch.mean(torch.abs(field))),
            'max_activation': float(torch.max(torch.abs(field))),
            'field_variance': float(torch.var(field)),
            'active_regions': torch.sum(torch.abs(field) > 0.1).item(),
            'brain_type': 'minimal'
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        # Get the unified field state
        return {
            'brain_state': {
                'unified_field': self.brain.field.cpu().numpy(),
                'brain_cycles': self.brain.cycle,  # TrulyMinimalBrain uses 'cycle'
                'metrics': getattr(self.brain, 'metrics', {})
            },
            'brain_cycles': self.brain.cycle,  # TrulyMinimalBrain uses 'cycle'
            'creation_time': self.creation_time,
            'brain_type': 'unified'
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from persistence."""
        # Load unified field state
        if 'brain_state' in state:
            brain_state = state['brain_state']
            if 'unified_field' in brain_state:
                field_data = torch.tensor(brain_state['unified_field'])
                if field_data.shape == self.brain.field.shape:
                    self.brain.field = field_data.to(self.brain.device)
            if 'brain_cycles' in brain_state:
                self.brain.cycle = brain_state['brain_cycles']  # TrulyMinimalBrain uses 'cycle'
        elif 'field' in state:
            # Legacy format
            field_data = torch.tensor(state['field'])
            if field_data.shape == self.brain.field.shape:
                self.brain.field = field_data.to(self.brain.device)
        
        # Set creation time
        if 'creation_time' in state:
            self.creation_time = state['creation_time']


class UnifiedFieldBrainFactory(IBrainFactory):
    """
    Factory for creating UnifiedFieldBrain instances.
    
    Complete intelligence system with:
    - Intrinsic tensions driving behavior
    - Prediction error learning
    - Self-modifying dynamics
    - Strategic pattern discovery
    - Emergent sensory mapping
    """
    
    def __init__(self, brain_config: Optional[Dict[str, Any]] = None):
        """Initialize factory."""
        self.brain_config = brain_config or {}
        self.cognitive_config = get_cognitive_config()
        self.calculator = SimpleDimensionCalculator()
        
        # Log initialization (quiet by default for clean startup)
        if not self.brain_config.get('quiet_mode', False):
            logger.info("🧠 UnifiedFieldBrain factory initialized - full intelligence restored")
    
    def create(self, 
               field_dimensions: Optional[int] = None,
               spatial_resolution: Optional[int] = None,
               sensory_dim: int = 16,
               motor_dim: int = 5) -> IBrain:
        """
        Create a UnifiedFieldBrain instance.
        
        Args:
            field_dimensions: Ignored - UnifiedFieldBrain uses 4D tensors
            spatial_resolution: Spatial resolution (default 32)
            sensory_dim: Number of input sensors
            motor_dim: Number of output motors
            
        Returns:
            UnifiedFieldBrain instance wrapped in IBrain interface
        """
        # Use provided resolution or default
        if spatial_resolution is None:
            spatial_resolution = (self.brain_config.get('spatial_resolution') or 
                                self.brain_config.get('field_spatial_resolution') or 
                                32)
        
        quiet_mode = self.brain_config.get('quiet_mode', False)
        
        # Create MinimalUnifiedBrain with essential systems only
        brain = MinimalUnifiedBrain(
            sensory_dim=sensory_dim,
            motor_dim=motor_dim,
            spatial_size=spatial_resolution,  # TrulyMinimalBrain uses spatial_size
            device=None,  # Auto-select best device
            quiet_mode=quiet_mode
        )
        
        # No features to enable - everything emerges!
        
        if not quiet_mode:
            logger.info("🧠 MinimalUnifiedBrain initialized - emergence enabled")
            logger.info("   ✅ Field dynamics (physics-based)")
            logger.info("   ✅ Sensory mapping (emergent)")
            logger.info("   ✅ Motor extraction (gradients)")
            logger.info("   ✅ Prediction (next state)")
            logger.info("   ✅ Learning (error only)")
            
        # Wrap in minimal interface adapter
        return UnifiedFieldBrainWrapper(brain)
    
    def get_brain_types(self) -> list:
        """Get available brain types - unified with full intelligence."""
        return ['unified']
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get factory configuration."""
        return {
            'brain_type': 'unified',
            'tensor_architecture': '4D',
            'gpu_optimized': True,
            'spatial_resolution': self.brain_config.get('field_spatial_resolution', 32),
            'status': 'MINIMAL BRAIN - EMERGENCE ENABLED',
            'notes': 'Reduced to 5 essential systems - everything else emerges'
        }


# Compatibility aliases - keep the same names so server doesn't break
PureFieldBrainFactory = UnifiedFieldBrainFactory
PureFieldBrainWrapper = UnifiedFieldBrainWrapper
UnifiedBrainFactory = UnifiedFieldBrainFactory  # For brain.py import
UnifiedBrainWrapper = UnifiedFieldBrainWrapper  # For consistency