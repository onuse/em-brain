"""
Single Brain Pool Implementation

A brain locks to the first brainstem that connects and only accepts
subsequent connections with matching dimensions. This reflects biological
reality where a brain adapts to its body.

Only one brainstem can be connected at a time.
"""

from typing import Optional, Dict, List
from threading import RLock
import time

from .interfaces import IBrainPool, IBrain, IBrainFactory


class SingleBrainPool(IBrainPool):
    """
    Manages a single brain that locks to specific dimensions.
    
    Once a brain is created for a specific sensory/motor configuration,
    it only accepts connections from robots with matching dimensions.
    This is biologically accurate - a brain is adapted to its body.
    """
    
    def __init__(self, brain_factory: IBrainFactory):
        self.brain_factory = brain_factory
        self.brain: Optional[IBrain] = None
        self.locked_dimensions: Optional[Dict[str, int]] = None
        self.lock = RLock()
        
        # Track connection attempts for diagnostics
        self.connection_attempts: List[Dict] = []
        self.successful_connections = 0
        self.rejected_connections = 0
    
    def get_brain_for_profile(self, profile_key: str) -> IBrain:
        """Get brain if dimensions match, or create first brain."""
        
        with self.lock:
            # Parse dimensions from profile key
            sensory_dim, motor_dim = self._parse_profile_dimensions(profile_key)
            
            # Record connection attempt
            self.connection_attempts.append({
                'time': time.time(),
                'profile': profile_key,
                'sensory_dim': sensory_dim,
                'motor_dim': motor_dim,
                'accepted': False  # Will update if accepted
            })
            
            # First connection - create brain and lock dimensions
            if self.brain is None:
                self._create_and_lock_brain(sensory_dim, motor_dim, profile_key)
                self.connection_attempts[-1]['accepted'] = True
                self.successful_connections += 1
                return self.brain
            
            # Subsequent connections - check dimension match
            if (self.locked_dimensions['sensory_dim'] == sensory_dim and
                self.locked_dimensions['motor_dim'] == motor_dim):
                
                print(f"✅ Brain accepting connection from {profile_key} (dimensions match)")
                self.connection_attempts[-1]['accepted'] = True
                self.successful_connections += 1
                return self.brain
            else:
                self.rejected_connections += 1
                raise ValueError(
                    f"❌ Brain dimension mismatch! "
                    f"Brain locked to {self.locked_dimensions['sensory_dim']}s_{self.locked_dimensions['motor_dim']}m, "
                    f"but {profile_key} requires {sensory_dim}s_{motor_dim}m. "
                    f"A brain can only serve robots with matching dimensions."
                )
    
    def _create_and_lock_brain(self, sensory_dim: int, motor_dim: int, profile_key: str):
        """Create the brain and lock it to specific dimensions."""
        
        # Calculate optimal field dimensions for these sensors/motors
        field_dims = self._calculate_field_dimensions(sensory_dim, motor_dim)
        spatial_res = 4  # Standard resolution for all brains
        
        # Create the brain
        self.brain = self.brain_factory.create(
            field_dimensions=field_dims,
            spatial_resolution=spatial_res,
            sensory_dim=sensory_dim,
            motor_dim=motor_dim
        )
        
        # Lock dimensions
        self.locked_dimensions = {
            'sensory_dim': sensory_dim,
            'motor_dim': motor_dim,
            'field_dimensions': field_dims,
            'spatial_resolution': spatial_res
        }
        
        print(f"🧠 Brain created and locked to dimensions:")
        print(f"   Sensory: {sensory_dim} channels")
        print(f"   Motor: {motor_dim} channels") 
        print(f"   Field: {field_dims}D")
        print(f"   First connection: {profile_key}")
    
    def get_brain_config(self, profile_key: str) -> Optional[Dict]:
        """Get brain configuration if it matches the profile."""
        with self.lock:
            if self.locked_dimensions is None:
                return None
                
            # Check if profile matches locked dimensions
            sensory_dim, motor_dim = self._parse_profile_dimensions(profile_key)
            if (self.locked_dimensions['sensory_dim'] == sensory_dim and
                self.locked_dimensions['motor_dim'] == motor_dim):
                return self.locked_dimensions.copy()
            
            return None
    
    def get_active_brains(self) -> Dict[str, IBrain]:
        """Get the single brain if it exists."""
        with self.lock:
            if self.brain is not None and self.locked_dimensions is not None:
                # Create profile key from locked dimensions
                profile = f"locked_{self.locked_dimensions['sensory_dim']}s_{self.locked_dimensions['motor_dim']}m"
                return {profile: self.brain}
            return {}
    
    def get_brain_info(self, profile_key: str) -> Optional[Dict]:
        """Get information about the brain if profile matches."""
        config = self.get_brain_config(profile_key)
        if config:
            config.update({
                'successful_connections': self.successful_connections,
                'rejected_connections': self.rejected_connections,
                'total_attempts': len(self.connection_attempts)
            })
        return config
    
    def _parse_profile_dimensions(self, profile_key: str) -> tuple[int, int]:
        """Extract sensory and motor dimensions from profile key."""
        # Default PiCar-X dimensions
        sensory_dim = 24
        motor_dim = 4
        
        # Parse from profile key (e.g., "picarx_24s_4m")
        parts = profile_key.split('_')
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
        
        return sensory_dim, motor_dim
    
    def _calculate_field_dimensions(self, sensory_dim: int, motor_dim: int) -> int:
        """
        Calculate field dimensions for given sensory/motor configuration.
        
        Uses the same algorithm as before but simplified since we're
        committing to one brain per instance.
        """
        # Base dimensions
        base_dims = 12
        
        # Scale with I/O complexity
        sensory_factor = 4 if sensory_dim <= 10 else 8 if sensory_dim <= 20 else 12
        motor_factor = 2 if motor_dim <= 4 else 4
        
        # Total dimensions
        total = base_dims + sensory_factor + motor_factor
        
        # Round to multiple of 4
        return ((total + 3) // 4) * 4