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
        
        # Pre-allocate motor tensor on GPU
        motor_values = torch.zeros(self.motor_dim, device=field.device)
        motor_directions = torch.zeros(self.motor_dim, device=field.device)
        
        # Vectorized boundary computation using torch operations
        x_coords = self.motor_regions[:, 0]
        y_coords = self.motor_regions[:, 1]
        z_coords = self.motor_regions[:, 2]
        
        # Compute bounds using torch operations (stays on GPU)
        x_min = torch.clamp(x_coords - 1, min=0)
        x_max = torch.clamp(x_coords + 2, max=field.shape[0])
        y_min = torch.clamp(y_coords - 1, min=0)
        y_max = torch.clamp(y_coords + 2, max=field.shape[1])
        z_min = torch.clamp(z_coords - 1, min=0)
        z_max = torch.clamp(z_coords + 2, max=field.shape[2])
        
        # Clamp indices for direction sampling
        x_clamped = torch.clamp(x_coords, max=dx.shape[0] - 1)
        y_clamped = torch.clamp(y_coords, max=dx.shape[1] - 1)
        z_clamped = torch.clamp(z_coords, max=dx.shape[2] - 1)
        
        # Extract motor commands (still need loop for variable-sized regions)
        for i in range(self.motor_dim):
            # Get region bounds (already computed on GPU)
            region_gradient = gradient_mag[
                x_min[i]:x_max[i], 
                y_min[i]:y_max[i], 
                z_min[i]:z_max[i]
            ]
            
            # Motor command is mean gradient in that region
            motor_values[i] = region_gradient.mean()
            
            # Get direction based on motor index
            if i == 0:  # Forward/backward
                motor_directions[i] = dx[x_clamped[i], y_clamped[i], z_clamped[i]].mean()
            elif i == 1:  # Left/right  
                motor_directions[i] = dy[x_clamped[i], y_clamped[i], z_clamped[i]].mean()
            else:  # Other motors
                motor_directions[i] = dz[x_clamped[i], y_clamped[i], z_clamped[i]].mean()
        
        # Combine magnitude and direction, apply tanh on GPU
        motor_commands = torch.tanh(motor_directions * motor_values * 10)
        
        # Convert to list only at the very end (single CPU transfer)
        return motor_commands.cpu().tolist()
    
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