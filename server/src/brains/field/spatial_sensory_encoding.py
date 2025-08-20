"""
Spatial Sensory Encoding - Preserving Structure in Sensation

Instead of random injection, sensory data is encoded spatially to preserve
relationships between inputs. This allows the brain to learn spatial concepts.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SpatialSensoryEncoding:
    """
    Encode sensory inputs while preserving spatial relationships.
    
    Core principle: Nearby sensory inputs go to nearby field locations.
    This enables learning of spatial concepts like "above", "next to", etc.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device):
        """
        Initialize spatial encoding system.
        
        Args:
            field_shape: Shape of the field tensor [D, H, W, C]
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Create a sensory injection grid
        # This maps 2D sensory input to 3D field locations
        self.setup_injection_grid()
        
    def setup_injection_grid(self):
        """
        Create a regular grid for sensory injection.
        
        This ensures spatial relationships are preserved.
        """
        # Determine grid size based on field dimensions
        # Use the middle layer of the field for primary sensory injection
        depth, height, width, channels = self.field_shape
        
        # Create a 2D grid in the middle layers of the 3D field
        # This gives sensory input a "surface" to project onto
        self.injection_depth = depth // 2  # Middle layer
        
        # Grid resolution (how many sensory points we can handle)
        self.grid_size = min(height, width) // 2
        
        # Pre-compute injection coordinates for efficiency
        grid_coords = []
        step = min(height, width) // self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.injection_depth
                y = i * step + step // 2
                z = j * step + step // 2
                
                # Ensure coordinates are within bounds
                x = min(x, depth - 1)
                y = min(y, height - 1)
                z = min(z, width - 1)
                
                grid_coords.append([x, y, z])
        
        self.injection_coords = torch.tensor(grid_coords, device=self.device)
        
    def encode_visual_field(self, visual_input: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """
        Encode visual input into the field while preserving spatial structure.
        
        Args:
            visual_input: 2D visual input [H, W] or [H, W, C]
            field: Current field state [D, H, W, C]
            
        Returns:
            Field with visual input encoded
        """
        # Ensure visual input is 2D
        if visual_input.dim() == 3:
            # Average across color channels if present
            visual_input = visual_input.mean(dim=2)
        
        # Resize visual input to match our injection grid
        visual_resized = F.interpolate(
            visual_input.unsqueeze(0).unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Flatten for injection
        visual_flat = visual_resized.flatten()
        
        # Inject into field with spatial coherence
        for idx, (x, y, z) in enumerate(self.injection_coords):
            if idx < len(visual_flat):
                # Inject with gaussian influence to preserve continuity
                injection_value = visual_flat[idx]
                
                # Apply to a small region around the injection point
                # This creates smooth gradients rather than point injections
                x_min = max(0, x - 1)
                x_max = min(field.shape[0], x + 2)
                y_min = max(0, y - 1)
                y_max = min(field.shape[1], y + 2)
                z_min = max(0, z - 1)
                z_max = min(field.shape[2], z + 2)
                
                # Gaussian-weighted injection
                for dx in range(x_min, x_max):
                    for dy in range(y_min, y_max):
                        for dz in range(z_min, z_max):
                            distance = ((dx - x)**2 + (dy - y)**2 + (dz - z)**2) ** 0.5
                            weight = torch.exp(-distance / 1.5)
                            
                            # Inject into multiple channels with slight variation
                            # This gives the field multiple "perspectives" on the same input
                            for c in range(min(4, field.shape[3])):
                                field[dx, dy, dz, c] += injection_value * weight * (0.8 + 0.2 * c / 4) * 0.1
        
        return field
    
    def encode_sensor_array(self, sensor_values: List[float], field: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple sensor readings with preserved relationships.
        
        Args:
            sensor_values: List of sensor readings
            field: Current field state
            
        Returns:
            Field with sensors encoded
        """
        # Map sensors to a line in 3D space
        # This preserves their sequential relationship
        num_sensors = len(sensor_values)
        
        if num_sensors > 0:
            # Create a line through the field for sensor values
            depth = self.field_shape[0]
            height = self.field_shape[1]
            
            # Inject along a diagonal line for better spatial distribution
            for i, value in enumerate(sensor_values):
                # Map sensor index to 3D position
                t = i / max(1, num_sensors - 1)  # Normalize to [0, 1]
                
                x = int(depth * 0.3 + depth * 0.4 * t)  # Use middle 40% of depth
                y = int(height * 0.2 + height * 0.6 * t)  # Use middle 60% of height
                z = self.field_shape[2] // 2  # Center width
                
                # Ensure bounds
                x = min(max(0, x), depth - 1)
                y = min(max(0, y), height - 1)
                z = min(max(0, z), self.field_shape[2] - 1)
                
                # Inject with stronger local influence for better spatial differentiation
                injection_strength = 0.2  # Increased from 0.05
                field[x, y, z, :4] += value * injection_strength
                
                # Small neighborhood influence for continuity
                if x > 0:
                    field[x-1, y, z, :4] += value * injection_strength * 0.5
                if x < depth - 1:
                    field[x+1, y, z, :4] += value * injection_strength * 0.5
        
        return field
    
    def get_receptive_fields(self) -> torch.Tensor:
        """
        Return the receptive field map showing where sensory inputs project.
        
        This helps visualize how sensory information is distributed in the field.
        """
        receptive_map = torch.zeros(self.field_shape[:3], device=self.device)
        
        # Mark injection points
        for x, y, z in self.injection_coords:
            receptive_map[x, y, z] = 1.0
            
            # Show influence regions
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < self.field_shape[0] and \
                           0 <= ny < self.field_shape[1] and \
                           0 <= nz < self.field_shape[2]:
                            distance = (dx**2 + dy**2 + dz**2) ** 0.5
                            receptive_map[nx, ny, nz] = max(
                                receptive_map[nx, ny, nz],
                                torch.exp(-distance / 1.5)
                            )
        
        return receptive_map
    
    def encode_multimodal(self, 
                          visual: Optional[torch.Tensor] = None,
                          sensors: Optional[List[float]] = None,
                          field: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode multiple sensory modalities simultaneously.
        
        Args:
            visual: Visual input tensor
            sensors: Other sensor readings
            field: Current field state
            
        Returns:
            Field with all sensory inputs encoded
        """
        if field is None:
            field = torch.zeros(self.field_shape, device=self.device)
            
        if visual is not None:
            field = self.encode_visual_field(visual, field)
            
        if sensors is not None:
            field = self.encode_sensor_array(sensors, field)
            
        return field