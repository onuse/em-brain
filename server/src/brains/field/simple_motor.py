"""
Simple Motor Extraction

Field gradients become motor commands. That's it.
No adaptation, no thresholds - just gradients to motors.
"""

import torch
import numpy as np


class SimpleMotorExtraction:
    """
    Extract motor commands from field gradients.
    
    Core principle: Steep gradients = strong motor activation.
    The field's "slopes" literally drive movement.
    """
    
    def __init__(self, motor_dim: int, device: torch.device, field_size: int = 16):
        """
        Initialize motor extraction.
        
        Args:
            motor_dim: Number of motor outputs
            device: Computation device
            field_size: Spatial size of the field (default 16)
        """
        self.motor_dim = motor_dim
        self.device = device
        self.field_size = field_size
        
        # Random but fixed mapping from field regions to motors
        # This determines which parts of the field control which motors
        self.motor_regions = torch.randint(0, field_size, (motor_dim, 3), device=device)
        
    def extract_motors(self, field: torch.Tensor) -> list:
        """
        Extract motor commands from field gradients.
        
        Args:
            field: Current field state [D, H, W, C]
            
        Returns:
            Motor commands as list of floats
        """
        # Compute spatial gradients
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        
        # Gradient magnitude at each point
        gradient_mag = torch.sqrt(dx**2 + dy**2 + dz**2).mean(dim=3)  # Average across channels
        
        # Extract motor commands from specific regions
        motors = []
        for i in range(self.motor_dim):
            x, y, z = self.motor_regions[i]
            
            # Sample gradient in a small region around the motor point
            x_min, x_max = max(0, x-1), min(field.shape[0], x+2)
            y_min, y_max = max(0, y-1), min(field.shape[1], y+2)
            z_min, z_max = max(0, z-1), min(field.shape[2], z+2)
            
            region_gradient = gradient_mag[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Motor command is mean gradient in that region
            motor_value = region_gradient.mean().item()
            
            # Also consider the direction of the gradient for signed motor commands
            # Use x-gradient for forward/backward, y-gradient for left/right
            # Clamp indices to valid range
            x = min(x, dx.shape[0] - 1)
            y = min(y, dx.shape[1] - 1)
            z = min(z, dx.shape[2] - 1)
            
            if i == 0:  # Forward/backward
                direction = dx[x, y, z].mean().item()
            elif i == 1:  # Left/right  
                direction = dy[x, y, z].mean().item()
            else:  # Other motors
                direction = dz[x, y, z].mean().item()
            
            # Combine magnitude and direction
            motor_command = np.tanh(direction * motor_value * 10)  # Scale and bound to [-1, 1]
            motors.append(motor_command)
        
        return motors
    
    def get_motor_state(self, motors: list) -> str:
        """Interpret motor commands as behavior."""
        if not motors:
            return "No motors"
        
        magnitude = np.linalg.norm(motors)
        
        if magnitude < 0.1:
            return "Still"
        elif magnitude < 0.3:
            return "Gentle movement"
        elif magnitude < 0.6:
            return "Active movement"
        else:
            return "Vigorous movement"