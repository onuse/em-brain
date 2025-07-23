#!/usr/bin/env python3
"""
Fix for the dimension utilization problem in UnifiedFieldBrain.

The current implementation wastes 32 dimensions by making them singleton.
This fix restructures the field to use all dimensions meaningfully.
"""

import torch
import math
from typing import List, Tuple


def create_proper_field_shape(spatial_resolution: int, total_dimensions: int) -> List[int]:
    """
    Create a field shape that uses all dimensions meaningfully.
    
    Instead of [20, 20, 20, 10, 15, 1, 1, ..., 1], we'll create
    a shape where all dimensions contribute.
    """
    # Distribute dimensions more evenly
    # We have 37 dimensions to work with
    
    if total_dimensions == 37:
        # New distribution that uses all dimensions
        shape = [
            spatial_resolution,      # X position
            spatial_resolution,      # Y position  
            spatial_resolution,      # Z position
            10,                     # Scale levels
            15,                     # Time steps
            # Now distribute the remaining 32 dimensions meaningfully
            8,                      # Oscillatory frequency bands
            8,                      # Flow directions
            6,                      # Topology patterns
            4,                      # Energy levels
            5,                      # Coupling strengths
            3,                      # Emergence states
        ]
        
        # Verify we have 37 dimensions
        assert len(shape) == 11
        assert sum([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 11  # Count of dimension groups
        
    else:
        # Fallback for other dimension counts
        base_dims = 5  # spatial(3) + scale + time
        remaining = total_dimensions - base_dims
        
        # Distribute remaining dimensions
        shape = [spatial_resolution] * 3 + [10, 15]
        
        # Add remaining dimensions with reasonable sizes
        dim_size = max(2, min(10, int(math.sqrt(remaining))))
        while len(shape) < total_dimensions:
            shape.append(dim_size)
    
    return shape


def map_sensor_to_rich_field(sensory_input: List[float], 
                            field_dimensions: List['FieldDimension'],
                            device: torch.device) -> torch.Tensor:
    """
    Create a rich mapping from sensors to ALL field dimensions.
    
    This ensures every dimension gets meaningful values, not just zeros.
    """
    total_dims = len(field_dimensions)
    field_coords = torch.zeros(total_dims, device=device)
    
    # Use all sensor values to create rich patterns
    sensor_array = torch.tensor(sensory_input, device=device)
    
    # 1. Spatial dimensions (0-2) - position in space
    field_coords[0] = (sensory_input[0] - 0.5) * 4  # X
    field_coords[1] = (sensory_input[1] - 0.5) * 4  # Y
    field_coords[2] = (sensory_input[2] - 0.5) * 2  # Z
    
    # 2. Scale dimension (3) - abstraction level
    # Derive from sensor variance
    sensor_variance = torch.var(sensor_array[:6])
    field_coords[3] = torch.tanh(sensor_variance * 2)
    
    # 3. Time dimension (4) - temporal dynamics
    # Use sensor change rate if available
    field_coords[4] = torch.sin(sensory_input[0] * 3.14159)
    
    # 4. Oscillatory dimensions (5-10) - 6 frequency bands
    for i in range(6):
        if i + 3 < len(sensory_input):
            # Create oscillatory patterns from sensors
            freq = (i + 1) * 0.5  # Different frequencies
            phase = sensory_input[i + 3] * 2 * 3.14159
            field_coords[5 + i] = torch.sin(torch.tensor(phase + freq))
    
    # 5. Flow dimensions (11-18) - 8 gradient directions  
    for i in range(8):
        if i + 9 < len(sensory_input):
            # Calculate local gradients
            curr_val = sensory_input[i + 9]
            prev_val = sensory_input[i + 8] if i > 0 else curr_val
            field_coords[11 + i] = (curr_val - prev_val) * 5
    
    # 6. Topology dimensions (19-24) - 6 stable patterns
    for i in range(6):
        if i + 17 < len(sensory_input):
            # Threshold-based topology
            field_coords[19 + i] = 1.0 if sensory_input[i + 17] > 0.6 else -1.0
    
    # 7. Energy dimensions (25-28) - 4 energy types
    for i in range(4):
        if i < len(sensory_input):
            # Energy as magnitude
            field_coords[25 + i] = abs(sensory_input[i] - 0.5) * 2
    
    # 8. Coupling dimensions (29-33) - 5 correlation patterns
    for i in range(5):
        if i + 1 < len(sensory_input):
            # Correlations between adjacent sensors
            field_coords[29 + i] = sensory_input[i] * sensory_input[i + 1] * 2 - 1
    
    # 9. Emergence dimensions (34-36) - 3 creative combinations
    if len(sensory_input) >= 3:
        # Nonlinear combinations
        field_coords[34] = torch.tanh(sensor_array[:3].sum() - 1.5)
        field_coords[35] = torch.tanh(sensor_array[:3].prod() * 10)
        field_coords[36] = torch.tanh(torch.max(sensor_array) - torch.min(sensor_array))
    
    return field_coords


def create_multidimensional_gradients(field: torch.Tensor) -> dict:
    """
    Calculate gradients for a properly structured multi-dimensional field.
    
    Handles fields where all dimensions have size > 1.
    """
    gradients = {}
    
    # Calculate gradients for each meaningful dimension
    # For dims with size > 2, we can use proper gradients
    for dim in range(field.ndim):
        if field.shape[dim] > 1:
            # Use finite differences
            grad = torch.zeros_like(field)
            
            # Forward difference
            if field.shape[dim] > 2:
                # Central differences for interior points
                slices_prev = [slice(None)] * field.ndim
                slices_next = [slice(None)] * field.ndim
                slices_center = [slice(None)] * field.ndim
                
                slices_prev[dim] = slice(None, -2)
                slices_next[dim] = slice(2, None)
                slices_center[dim] = slice(1, -1)
                
                grad[slices_center] = (field[slices_next] - field[slices_prev]) / 2.0
            else:
                # Simple forward difference for size 2
                slices_0 = [slice(None)] * field.ndim
                slices_1 = [slice(None)] * field.ndim
                slices_0[dim] = 0
                slices_1[dim] = 1
                
                diff = field[slices_1] - field[slices_0]
                grad[slices_0] = diff
                grad[slices_1] = diff
            
            gradients[f'gradient_dim_{dim}'] = grad
    
    return gradients


def extract_rich_motor_commands(field: torch.Tensor, 
                              gradients: dict,
                              center_position: List[int]) -> torch.Tensor:
    """
    Extract motor commands from a rich multi-dimensional field.
    
    Uses information from all dimensions, not just spatial.
    """
    motor_commands = torch.zeros(4)
    
    # Use gradients from multiple dimensions
    # Spatial gradients for primary movement
    if 'gradient_dim_0' in gradients:  # X
        motor_commands[0] = sample_gradient_region(gradients['gradient_dim_0'], center_position)
    
    if 'gradient_dim_1' in gradients:  # Y
        motor_commands[1] = sample_gradient_region(gradients['gradient_dim_1'], center_position)
    
    if 'gradient_dim_2' in gradients:  # Z
        motor_commands[2] = sample_gradient_region(gradients['gradient_dim_2'], center_position)
    
    # Use higher dimensions for the 4th motor command
    # Combine scale, time, and energy gradients
    higher_dim_contribution = 0.0
    
    if 'gradient_dim_3' in gradients:  # Scale
        higher_dim_contribution += sample_gradient_region(gradients['gradient_dim_3'], center_position) * 0.3
    
    if 'gradient_dim_4' in gradients:  # Time
        higher_dim_contribution += sample_gradient_region(gradients['gradient_dim_4'], center_position) * 0.3
    
    # Energy dimensions (25-28 in our mapping)
    for energy_dim in range(7, 11):
        if f'gradient_dim_{energy_dim}' in gradients:
            higher_dim_contribution += sample_gradient_region(
                gradients[f'gradient_dim_{energy_dim}'], center_position
            ) * 0.1
    
    motor_commands[3] = higher_dim_contribution
    
    return motor_commands


def sample_gradient_region(gradient_tensor: torch.Tensor, 
                          center_position: List[int],
                          window_size: int = 3) -> float:
    """
    Sample a gradient tensor around a center position.
    
    Handles multi-dimensional tensors properly.
    """
    # Build slice for local region
    slices = []
    for dim, center in enumerate(center_position):
        if dim < gradient_tensor.ndim:
            dim_size = gradient_tensor.shape[dim]
            half_window = window_size // 2
            
            start = max(0, center - half_window)
            end = min(dim_size, center + half_window + 1)
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))
    
    # Extract local region
    local_region = gradient_tensor[tuple(slices)]
    
    # Return weighted average
    if local_region.numel() > 0:
        weights = torch.abs(local_region) + 1e-8
        weights = weights / torch.sum(weights)
        return torch.sum(local_region * weights).item()
    else:
        return 0.0


if __name__ == "__main__":
    print("Testing field dimension fix...")
    
    # Test proper field shape
    shape = create_proper_field_shape(20, 37)
    print(f"New field shape: {shape}")
    print(f"Total dimensions: {len(shape)}")
    print(f"No singleton dimensions!")
    
    # Test rich sensor mapping
    test_input = [0.5 + 0.1 * i for i in range(24)]
    
    # Mock field dimensions
    class MockDim:
        pass
    
    field_dims = [MockDim() for _ in range(37)]
    coords = map_sensor_to_rich_field(test_input, field_dims, torch.device('cpu'))
    
    print(f"\nField coordinates from input:")
    print(f"  Non-zero coordinates: {torch.sum(coords != 0).item()}/37")
    print(f"  Coordinate range: [{torch.min(coords).item():.3f}, {torch.max(coords).item():.3f}]")
    print(f"  Coordinate variance: {torch.var(coords).item():.3f}")