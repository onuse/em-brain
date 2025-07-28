"""
Dynamic Unified Field Brain

This is a refactored version of UnifiedFieldBrain that accepts dynamic
conceptual dimensions while maintaining the efficient tensor representation.
"""

import torch
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .core_brain import (
    FieldDimension, FieldDynamicsFamily, UnifiedFieldExperience,
    FieldNativeAction
)
from ...core.dynamic_dimension_calculator import DynamicDimensionCalculator


class DynamicUnifiedFieldBrain:
    """
    Unified Field Brain with Dynamic Dimensions
    
    This brain adapts its conceptual dimensions to robot complexity while
    using efficient preset tensor configurations for memory management.
    """
    
    def __init__(self,
                 field_dimensions: List[FieldDimension],
                 tensor_shape: List[int],
                 dimension_mapping: Dict[str, Any],
                 spatial_resolution: int = 4,
                 temporal_window: float = 10.0,
                 field_evolution_rate: float = 0.1,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        """
        Initialize brain with dynamic dimensions.
        
        Args:
            field_dimensions: List of conceptual field dimensions
            tensor_shape: Actual tensor shape for field representation
            dimension_mapping: Mapping between conceptual and tensor dimensions
            spatial_resolution: Resolution for spatial dimensions
            temporal_window: Time window for temporal dynamics
            field_evolution_rate: Rate of field evolution
            device: Torch device for computation
            quiet_mode: Suppress output if True
        """
        self.field_dimensions = field_dimensions
        self.tensor_shape = tensor_shape
        self.dimension_mapping = dimension_mapping
        self.spatial_resolution = spatial_resolution
        self.temporal_window = temporal_window
        self.field_evolution_rate = field_evolution_rate
        self.quiet_mode = quiet_mode
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Create unified field with dynamic tensor shape
        self.unified_field = torch.zeros(tensor_shape, dtype=torch.float32, device=self.device)
        self.unified_field.fill_(0.01)  # Baseline activation
        
        # Field parameters
        self.field_decay_rate = 0.999
        self.field_diffusion_rate = 0.05
        self.gradient_following_strength = 0.5
        
        # Robot interface dimensions (to be set by factory)
        self.expected_sensory_dim = None
        self.expected_motor_dim = None
        
        # Performance tracking
        self.brain_cycles = 0
        
        if not quiet_mode:
            print(f"🧠 Dynamic Unified Field Brain initialized")
            print(f"   Conceptual dimensions: {len(field_dimensions)}D")
            print(f"   Tensor shape: {tensor_shape} ({len(tensor_shape)}D)")
            print(f"   Memory usage: {self._calculate_memory_usage():.1f}MB")
            self._print_dimension_summary()
    
    def _calculate_memory_usage(self) -> float:
        """Calculate field memory usage in MB."""
        elements = 1
        for dim in self.tensor_shape:
            elements *= dim
        return (elements * 4) / (1024 * 1024)
    
    def _print_dimension_summary(self):
        """Print summary of dimension organization."""
        from collections import defaultdict
        family_counts = defaultdict(int)
        for dim in self.field_dimensions:
            family_counts[dim.family] += 1
        
        print("   Conceptual dimension families:")
        for family, count in family_counts.items():
            tensor_range = self.dimension_mapping['family_tensor_ranges'].get(family, (0, 0))
            print(f"      {family.value}: {count}D conceptual → tensor indices {tensor_range}")
    
    def process_robot_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process one robot cycle with dynamic dimensions.
        
        Args:
            sensory_input: Robot sensor values
            
        Returns:
            Tuple of (motor_commands, brain_state)
        """
        cycle_start = time.perf_counter()
        
        # Map sensory input to field coordinates
        field_coords = self._sensory_to_field_coordinates(sensory_input)
        
        # Imprint experience in field
        self._imprint_experience(field_coords)
        
        # Evolve field
        self._evolve_field()
        
        # Calculate gradients and generate actions
        motor_commands = self._field_gradients_to_motor(field_coords)
        
        # Update tracking
        self.brain_cycles += 1
        cycle_time = time.perf_counter() - cycle_start
        
        # Create brain state
        brain_state = {
            'cycle': self.brain_cycles,
            'cycle_time_ms': cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.unified_field))),
            'conceptual_dims': len(self.field_dimensions),
            'tensor_dims': len(self.tensor_shape)
        }
        
        return motor_commands, brain_state
    
    def _sensory_to_field_coordinates(self, sensory_input: List[float]) -> torch.Tensor:
        """Map sensory input to field coordinates using dynamic mapping."""
        # For now, simple mapping - spread sensors across spatial dimensions
        field_coords = torch.zeros(len(self.tensor_shape), device=self.device)
        
        # Map to spatial center
        spatial_center = self.spatial_resolution // 2
        for i in range(min(3, len(self.tensor_shape))):
            field_coords[i] = spatial_center
        
        # Add some variation based on sensors
        if len(sensory_input) > 0:
            # Use first few sensors to modulate position
            for i in range(min(3, len(sensory_input))):
                field_coords[i] += (sensory_input[i] - 0.5) * 2  # Scale to [-1, 1]
        
        return field_coords
    
    def _imprint_experience(self, field_coords: torch.Tensor):
        """Imprint experience at field coordinates."""
        # Convert continuous coords to indices
        indices = []
        for i, coord in enumerate(field_coords):
            idx = int(torch.clamp(coord, 0, self.tensor_shape[i] - 1))
            indices.append(idx)
        
        # Create slice for 3x3x3 region around point
        slices = []
        for i, idx in enumerate(indices[:3]):  # Only for spatial dims
            start = max(0, idx - 1)
            end = min(self.tensor_shape[i], idx + 2)
            slices.append(slice(start, end))
        
        # Add remaining dimensions as single points
        for i in range(3, len(indices)):
            slices.append(indices[i])
        
        # Imprint activation
        self.unified_field[tuple(slices)] += 0.1
    
    def _evolve_field(self):
        """Evolve field dynamics."""
        # Simple decay
        self.unified_field *= self.field_decay_rate
        
        # Add some diffusion in spatial dimensions
        if self.field_diffusion_rate > 0:
            # Simple smoothing as diffusion proxy
            spatial_dims = min(3, len(self.tensor_shape))
            for dim in range(spatial_dims):
                # Shift and average
                shifted_forward = torch.roll(self.unified_field, shifts=1, dims=dim)
                shifted_backward = torch.roll(self.unified_field, shifts=-1, dims=dim)
                self.unified_field += self.field_diffusion_rate * (
                    shifted_forward + shifted_backward - 2 * self.unified_field
                ) / 2
    
    def _field_gradients_to_motor(self, field_coords: torch.Tensor) -> List[float]:
        """Convert field gradients to motor commands."""
        # Simple gradient following in spatial dimensions
        motor_commands = []
        
        # Calculate local gradients
        indices = [int(torch.clamp(coord, 0, self.tensor_shape[i] - 1)) 
                  for i, coord in enumerate(field_coords)]
        
        # Get gradients in X and Y directions (first 2 motors)
        for dim in range(min(2, len(indices))):
            idx = indices[dim]
            if idx > 0 and idx < self.tensor_shape[dim] - 1:
                # Create slices for gradient calculation
                # We need to access a specific point in the multi-dimensional tensor
                slice_before = [slice(None)] * len(indices)
                slice_after = [slice(None)] * len(indices)
                slice_before[dim] = idx - 1
                slice_after[dim] = idx + 1
                
                # Fix other dimensions to current position
                for i in range(len(indices)):
                    if i != dim:
                        slice_before[i] = indices[i]
                        slice_after[i] = indices[i]
                
                # Calculate gradient
                val_before = self.unified_field[tuple(slice_before)]
                val_after = self.unified_field[tuple(slice_after)]
                grad = float(val_after - val_before) / 2.0
                motor_commands.append(grad * self.gradient_following_strength)
            else:
                motor_commands.append(0.0)
        
        # Add zeros for remaining motors
        while len(motor_commands) < (self.expected_motor_dim or 4):
            motor_commands.append(0.0)
        
        return motor_commands