"""
Brain Pool implementation.

Manages brain instances and determines optimal brain configurations
for different robot profiles.
"""

import math
from typing import Dict, Optional
from threading import RLock

from .interfaces import IBrainPool, IBrain, IBrainFactory


class BrainPool(IBrainPool):
    """
    Manages brain instances - separation from creation.
    
    This component determines the optimal brain configuration for
    each robot profile and manages the lifecycle of brain instances.
    """
    
    def __init__(self, brain_factory: IBrainFactory):
        self.brain_factory = brain_factory
        self.brains: Dict[str, IBrain] = {}
        self.brain_configs: Dict[str, Dict] = {}  # profile_key -> config
        self.lock = RLock()  # Thread safety for multi-client support
        
        # Default spatial resolution (can be made configurable)
        self.default_spatial_resolution = 4
    
    def get_brain_for_profile(self, profile_key: str) -> IBrain:
        """Get or create brain for a robot profile."""
        
        with self.lock:
            if profile_key not in self.brains:
                # Determine optimal brain parameters
                config = self._determine_brain_config(profile_key)
                self.brain_configs[profile_key] = config
                
                # Create brain
                brain = self.brain_factory.create(
                    field_dimensions=config['field_dimensions'],
                    spatial_resolution=config['spatial_resolution'],
                    sensory_dim=config['sensory_dim'],
                    motor_dim=config['motor_dim']
                )
                
                self.brains[profile_key] = brain
                
                print(f"🧠 Created {config['field_dimensions']}D brain for profile: {profile_key}")
                print(f"   Spatial resolution: {config['spatial_resolution']}³")
            
            return self.brains[profile_key]
    
    def get_brain_config(self, profile_key: str) -> Optional[Dict]:
        """Get the configuration for a brain profile."""
        return self.brain_configs.get(profile_key)
    
    def get_active_brains(self) -> Dict[str, IBrain]:
        """Get all active brain instances."""
        with self.lock:
            return self.brains.copy()
    
    def _determine_brain_config(self, profile_key: str) -> Dict:
        """
        Calculate optimal brain configuration based on robot profile.
        
        This is where we implement the scaling algorithm that determines
        how complex the brain should be for a given robot.
        """
        
        # Parse profile key (format: "robottype_Xs_Ym")
        parts = profile_key.split('_')
        
        # Extract dimensions if available
        sensory_dim = 16  # default
        motor_dim = 4     # default
        
        for part in parts:
            if part.endswith('s'):
                try:
                    sensory_dim = int(part[:-1])
                except ValueError:
                    pass
            elif part.endswith('m'):
                try:
                    motor_dim = int(part[:-1])
                except ValueError:
                    pass
        
        # Calculate field dimensions based on robot complexity
        field_dims = self._calculate_field_dimensions(sensory_dim, motor_dim)
        
        # Determine spatial resolution based on available resources
        spatial_res = self._calculate_spatial_resolution(field_dims)
        
        return {
            'field_dimensions': field_dims,
            'spatial_resolution': spatial_res,
            'sensory_dim': sensory_dim,
            'motor_dim': motor_dim
        }
    
    def _calculate_field_dimensions(self, sensory_dim: int, motor_dim: int) -> int:
        """
        Calculate unified field dimensions based on robot complexity.
        
        This implements a scaling algorithm that ensures the brain has
        enough capacity to represent the robot's sensorimotor space while
        remaining computationally tractable.
        """
        
        # Base dimensions for core field dynamics
        # (minimum needed for basic intelligence)
        base_dims = 12
        
        # Scale with sensory complexity
        # Use log scale to prevent explosion with high-dimensional sensors
        sensory_factor = math.ceil(math.log2(sensory_dim + 1)) * 3
        
        # Scale with motor complexity
        # Motors typically need less dimensional representation
        motor_factor = math.ceil(math.log2(motor_dim + 1)) * 2
        
        # Add bonus dimensions for rich sensory spaces
        sensory_bonus = 0
        if sensory_dim > 20:
            sensory_bonus = 4
        elif sensory_dim > 10:
            sensory_bonus = 2
        
        # Calculate total
        total_dims = base_dims + sensory_factor + motor_factor + sensory_bonus
        
        # Round to nice number for computational efficiency
        # Clamp between reasonable bounds
        total_dims = max(16, min(64, total_dims))
        
        # Round to multiple of 4 for better tensor operations
        total_dims = ((total_dims + 3) // 4) * 4
        
        return total_dims
    
    def _calculate_spatial_resolution(self, field_dims: int) -> int:
        """
        Calculate spatial resolution based on field dimensions and available resources.
        
        Higher dimensional fields may need lower spatial resolution to fit
        in memory and compute efficiently.
        """
        
        # Memory usage approximation:
        # field_dims * spatial_res^3 * 4 bytes (float32)
        
        # Target memory usage for field (in MB)
        target_memory_mb = 10.0  # Reasonable for most systems
        
        # Calculate maximum spatial resolution
        max_voxels = (target_memory_mb * 1024 * 1024) / (field_dims * 4)
        max_spatial_res = int(math.pow(max_voxels, 1/3))
        
        # Apply bounds
        spatial_res = max(3, min(8, max_spatial_res))
        
        # Prefer powers of 2 when possible
        if spatial_res >= 6:
            spatial_res = 8
        elif spatial_res >= 3:
            spatial_res = 4
        
        return spatial_res
    
    def get_brain_info(self, profile_key: str) -> Optional[Dict]:
        """Get information about a brain configuration."""
        return self.brain_configs.get(profile_key)