#!/usr/bin/env python3
"""
Adaptive Field Implementation Architecture

This module provides device-adaptive field implementations that optimize
performance based on hardware capabilities while maintaining identical
external interfaces.

Key Design Principles:
1. Abstract base class defines the interface
2. Concrete implementations optimize for specific devices
3. External code sees no implementation differences
4. Automatic device detection and selection
"""

import torch
import numpy as np
import time

# Force float32 as default to prevent MPS float64 issues
torch.set_default_dtype(torch.float32)
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict

# Import field brain components
try:
    from .field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience, 
        FieldNativeAction
    )
except ImportError:
    from field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience,
        FieldNativeAction
    )

# GPU detection and device selection (priority: CUDA > MPS > CPU)
try:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_AVAILABLE = True
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        GPU_AVAILABLE = True
    else:
        DEVICE = torch.device('cpu')
        GPU_AVAILABLE = False
except Exception:
    DEVICE = torch.device('cpu')
    GPU_AVAILABLE = False

def get_field_device(tensor_dims: int) -> torch.device:
    """
    Get appropriate device for field tensors, considering MPS dimension limits.
    
    MPS has a 16-dimension limit, so high-dimensional field tensors must use CPU.
    """
    if DEVICE.type == 'mps' and tensor_dims > 16:
        return torch.device('cpu')  # MPS limitation fallback
    return DEVICE


class FieldImplementation(ABC):
    """
    Abstract base class for field implementations.
    
    This defines the interface that all field implementations must provide,
    ensuring that external code works identically regardless of the internal
    implementation (unified vs multi-tensor).
    """
    
    def __init__(self, 
                 spatial_resolution: int,
                 field_dimensions: List[FieldDimension],
                 field_device: torch.device,
                 quiet_mode: bool = False):
        """Initialize the field implementation."""
        self.spatial_resolution = spatial_resolution
        self.field_dimensions = field_dimensions
        self.field_device = field_device
        self.quiet_mode = quiet_mode
        
        # Common tracking variables
        self.field_evolution_cycles = 0
        self.total_imprints = 0
        self.field_statistics = {}
        
    @abstractmethod
    def imprint_experience(self, experience: UnifiedFieldExperience) -> None:
        """Imprint an experience into the field."""
        pass
    
    @abstractmethod
    def evolve_field(self, dt: float = 0.1, current_input_stream: Optional[List[float]] = None) -> None:
        """Evolve the field dynamics over time."""
        pass
    
    @abstractmethod
    def compute_field_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute gradients across field dimensions."""
        pass
    
    @abstractmethod
    def generate_field_output(self, field_coordinates: torch.Tensor) -> torch.Tensor:
        """Generate output stream from field state."""
        pass
    
    @abstractmethod
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive field statistics."""
        pass
    
    @abstractmethod
    def get_field_state_summary(self) -> Dict[str, Any]:
        """Get current field state summary."""
        pass
    
    def get_implementation_type(self) -> str:
        """Return the implementation type for debugging/logging."""
        return self.__class__.__name__


def select_optimal_field_implementation(field_dimensions: List[FieldDimension],
                                      spatial_resolution: int,
                                      requested_device: torch.device,
                                      quiet_mode: bool = False) -> Tuple[str, torch.device]:
    """
    Select the optimal field implementation and device based on capabilities.
    
    Returns:
        Tuple of (implementation_type, actual_device)
        - Always returns 'unified' to eliminate multi-tensor path confusion
    """
    total_field_dims = len(field_dimensions)
    
    if not quiet_mode:
        print(f"🔧 Using unified field implementation ({total_field_dims}D field)")
    
    # Always use unified implementation - single code path for clarity
    # PyTorch will handle device limitations gracefully (fallback to CPU if needed)
    return 'unified', requested_device


class UnifiedFieldImplementation(FieldImplementation):
    """
    Unified field implementation using a single high-dimensional tensor.
    
    This is optimal for CUDA devices that can handle arbitrary tensor dimensions.
    Uses the original unified field approach for maximum performance.
    """
    
    def __init__(self, spatial_resolution: int, field_dimensions: List[FieldDimension],
                 field_device: torch.device, quiet_mode: bool = False):
        super().__init__(spatial_resolution, field_dimensions, field_device, quiet_mode)
        
        self.total_dimensions = len(field_dimensions)
        
        # Create unified multi-dimensional field - original approach
        field_shape = [spatial_resolution] * 3 + [10] + [15] + [1] * (self.total_dimensions - 5)
        
        # Try to create on requested device, fall back to CPU if dimensions exceed device limits
        try:
            self.unified_field = torch.zeros(field_shape, dtype=torch.float32, device=field_device)
            self.field_device = field_device
        except RuntimeError as e:
            if "MPS supports tensors with dimensions" in str(e):
                # MPS device limitation - fall back to CPU gracefully
                if not quiet_mode:
                    print(f"⚠️ MPS device cannot handle {len(field_shape)}D tensor, falling back to CPU")
                self.unified_field = torch.zeros(field_shape, dtype=torch.float32, device=torch.device('cpu'))
                self.field_device = torch.device('cpu')
            else:
                # Re-raise other errors
                raise
        
        # Field dynamics tracking
        self.field_memory = []
        self.topology_regions = []
        self.constraint_violations = 0
        self.gradient_flows = {}
        
        # Performance optimization
        self.last_evolution_time = time.time()
        self.evolution_cache = {}
        
        if not quiet_mode:
            memory_mb = (self.unified_field.numel() * 4) / (1024 * 1024)
            print(f"🧠 Unified field: {field_shape} = {self.unified_field.numel():,} elements ({memory_mb:.1f}MB)")
    
    def imprint_experience(self, experience: UnifiedFieldExperience) -> None:
        """Imprint experience into unified field using original algorithm."""
        field_coords = experience.field_coordinates
        intensity = experience.field_intensity
        
        # DEBUG: Log suspicious intensity values
        if intensity > 1.0:
            coords_norm = torch.norm(field_coords).item()
            print(f"🔍 UNIFIED IMPRINT: intensity={intensity:.6f}, coords_norm={coords_norm:.6f}, coords_max={torch.max(torch.abs(field_coords)).item():.6f}")
        
        # Map field coordinates to spatial positions
        spatial_coords = field_coords[:3]  # First 3 are spatial
        scale_coord = field_coords[3] if len(field_coords) > 3 else torch.tensor(0.0, device=self.field_device)
        time_coord = field_coords[4] if len(field_coords) > 4 else torch.tensor(0.0, device=self.field_device)
        
        # Convert to field indices
        x_center = int((spatial_coords[0] + 1) * 0.5 * (self.spatial_resolution - 1))
        y_center = int((spatial_coords[1] + 1) * 0.5 * (self.spatial_resolution - 1))
        z_center = int((spatial_coords[2] + 1) * 0.5 * (self.spatial_resolution - 1))
        s_center = int((scale_coord + 1) * 0.5 * 9)  # Scale dimension
        t_center = int((time_coord + 1) * 0.5 * 14)  # Time dimension
        
        # Clamp to valid ranges
        x_center = max(0, min(self.spatial_resolution - 1, x_center))
        y_center = max(0, min(self.spatial_resolution - 1, y_center))
        z_center = max(0, min(self.spatial_resolution - 1, z_center))
        s_center = max(0, min(9, s_center))
        t_center = max(0, min(14, t_center))
        
        # Gaussian imprint with adaptive radius
        spatial_spread = 2.0 + intensity  # Larger experiences create wider imprints
        
        # Calculate total spatial positions for normalization
        dx_range = range(-3, 4)  # 7 positions
        dy_range = range(-3, 4)  # 7 positions  
        dz_range = range(-3, 4)  # 7 positions
        ds_range = range(-1, 2)  # 3 positions
        dt_range = range(-2, 3)  # 5 positions
        total_spatial_positions = len(dx_range) * len(dy_range) * len(dz_range) * len(ds_range) * len(dt_range)
        
        # Pre-normalize intensity by both spatial and dimensional factors (OPTIMIZATION!)
        # Calculate TOTAL field dimensions for proper normalization
        if len(self.unified_field.shape) > 5:
            # Multi-dimensional field: use the total number of dynamics dimensions
            dimension_count = self.unified_field.shape[5]
            # Also account for the field coordinates dimension count (37D field)
            total_field_dimensions = len(field_coords)  # This should be 37 for full field
            normalization_factor = total_spatial_positions * dimension_count * total_field_dimensions
        else:
            # 5D field case
            total_field_dimensions = len(field_coords) if hasattr('field_coords', '__len__') else 37
            normalization_factor = total_spatial_positions * total_field_dimensions
        
        base_intensity = intensity / normalization_factor
        
        for dx in dx_range:
            for dy in dy_range:
                for dz in dz_range:
                    for ds in ds_range:
                        for dt in dt_range:
                            x_pos = x_center + dx
                            y_pos = y_center + dy
                            z_pos = z_center + dz
                            s_pos = s_center + ds
                            t_pos = t_center + dt
                            
                            # Check bounds
                            if (0 <= x_pos < self.spatial_resolution and
                                0 <= y_pos < self.spatial_resolution and
                                0 <= z_pos < self.spatial_resolution and
                                0 <= s_pos < 10 and
                                0 <= t_pos < 15):
                                
                                # Calculate Gaussian weight (pre-normalized for efficiency)
                                distance = math.sqrt(dx**2 + dy**2 + dz**2 + ds**2 + dt**2)
                                weight = base_intensity * math.exp(-(distance**2) / (2 * spatial_spread**2))
                                
                                # Clamp weight to prevent exponential growth
                                weight = min(weight, 10.0)
                                
                                # Apply imprint to all dynamics dimensions (already normalized)
                                if len(self.unified_field.shape) > 5:
                                    # Multi-dimensional case - weight already normalized
                                    for dim_idx in range(self.unified_field.shape[5]):
                                        self.unified_field[x_pos, y_pos, z_pos, s_pos, t_pos, dim_idx] += weight
                                else:
                                    # 5D case
                                    self.unified_field[x_pos, y_pos, z_pos, s_pos, t_pos] += weight
        
        self.total_imprints += 1
    
    def evolve_field(self, dt: float = 0.1, current_input_stream: Optional[List[float]] = None) -> None:
        """Evolve unified field using original hierarchical coarse-to-fine processing."""
        self.field_evolution_cycles += 1
        
        # CRITICAL FIX: Process input stream to create spatial patterns
        if current_input_stream is not None:
            self._inject_input_stream(current_input_stream, dt)
        
        # Biological optimization: decay and diffusion
        # Adaptive decay rate based on field energy to prevent accumulation
        field_energy = torch.sum(self.unified_field).item()
        if field_energy > 100.0:
            decay_rate = 0.98  # Stronger decay for high energy
        elif field_energy > 50.0:
            decay_rate = 0.99  # Moderate decay for medium energy
        else:
            decay_rate = 0.995  # Light decay for low energy
        diffusion_rate = 0.02  # Gentle diffusion for natural spreading
        
        # Apply decay
        self.unified_field *= decay_rate
        
        # Apply diffusion (simplified 3D spatial diffusion)
        if self.spatial_resolution > 1:
            # Create shifted versions for diffusion
            diffusion_kernel = diffusion_rate / 6.0  # 6 spatial neighbors
            
            # X-direction diffusion
            if self.spatial_resolution > 2:
                self.unified_field[1:-1, :, :] += diffusion_kernel * (
                    self.unified_field[2:, :, :] + self.unified_field[:-2, :, :] - 2 * self.unified_field[1:-1, :, :]
                )
            
            # Y-direction diffusion  
            if self.spatial_resolution > 2:
                self.unified_field[:, 1:-1, :] += diffusion_kernel * (
                    self.unified_field[:, 2:, :] + self.unified_field[:, :-2, :] - 2 * self.unified_field[:, 1:-1, :]
                )
            
            # Z-direction diffusion
            if self.spatial_resolution > 2:
                self.unified_field[:, :, 1:-1] += diffusion_kernel * (
                    self.unified_field[:, :, 2:] + self.unified_field[:, :, :-2] - 2 * self.unified_field[:, :, 1:-1]
                )
        
        # Update gradient flows for action generation
        self._update_gradient_flows()
    
    def _update_gradient_flows(self):
        """Update gradient flows for action generation."""
        # Compute spatial gradients for the unified field
        if self.spatial_resolution > 1:
            # X gradient
            grad_x = torch.zeros_like(self.unified_field)
            grad_x[1:, :, :] = self.unified_field[1:, :, :] - self.unified_field[:-1, :, :]
            
            # Y gradient  
            grad_y = torch.zeros_like(self.unified_field)
            grad_y[:, 1:, :] = self.unified_field[:, 1:, :] - self.unified_field[:, :-1, :]
            
            # Z gradient
            grad_z = torch.zeros_like(self.unified_field)
            grad_z[:, :, 1:] = self.unified_field[:, :, 1:] - self.unified_field[:, :, :-1]
            
            # Store gradients for action generation
            self.gradient_flows = {
                'spatial_x': grad_x,
                'spatial_y': grad_y, 
                'spatial_z': grad_z
            }
    
    def _inject_input_stream(self, input_stream: List[float], dt: float):
        """Inject input stream into unified field to create spatial patterns."""
        if not input_stream:
            return
        
        # Convert input to tensor
        input_tensor = torch.tensor(input_stream, dtype=torch.float32, device=self.field_device)
        input_intensity = torch.mean(torch.abs(input_tensor)).item()
        
        # Don't inject if input is too weak
        if input_intensity < 0.01:
            return
        
        # Map input dimensions to field positions
        # Use a spatial grid approach for the unified field
        max_positions = min(self.spatial_resolution, 8)  # Limit for performance
        
        for i, input_value in enumerate(input_stream):
            if abs(input_value) < 0.01:
                continue
                
            # Map input index to spatial position
            x_pos = (i * 7) % max_positions  # Prime number for good distribution
            y_pos = (i * 11) % max_positions 
            z_pos = (i * 13) % max_positions
            
            # Add spatial spread around the position
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        x = (x_pos + dx) % max_positions
                        y = (y_pos + dy) % max_positions
                        z = (z_pos + dz) % max_positions
                        
                        # Distance-based weighting
                        dist = abs(dx) + abs(dy) + abs(dz)
                        weight = 1.0 / (1.0 + dist)
                        
                        # Inject into unified field with proper strength
                        injection_strength = input_value * weight * dt * 0.1  # Same as multi-tensor version
                        
                        # Inject across multiple dimensions of the unified field
                        for dim_offset in range(min(self.unified_field.shape[3], 4)):
                            self.unified_field[x, y, z, dim_offset] += injection_strength
    
    def compute_field_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute gradients across field dimensions."""
        return self.gradient_flows
    
    def generate_field_output(self, field_coordinates: torch.Tensor) -> torch.Tensor:
        """Generate output stream from unified field state."""
        # Use gradient flows to generate output with proper magnitude extraction
        if self.gradient_flows:
            # Extract dominant gradients for output
            grad_x = self.gradient_flows.get('spatial_x', torch.zeros(1, device=self.field_device))
            grad_y = self.gradient_flows.get('spatial_y', torch.zeros(1, device=self.field_device))
            grad_z = self.gradient_flows.get('spatial_z', torch.zeros(1, device=self.field_device))
            
            # Fix: Use max magnitude instead of mean to preserve gradient strength
            grad_x_mag = torch.max(torch.abs(grad_x)) * torch.sign(torch.sum(grad_x))
            grad_y_mag = torch.max(torch.abs(grad_y)) * torch.sign(torch.sum(grad_y))
            grad_z_mag = torch.max(torch.abs(grad_z)) * torch.sign(torch.sum(grad_z))
            
            # Compute output as dominant gradient direction
            output_tensor = torch.stack([
                grad_x_mag,
                grad_y_mag, 
                grad_z_mag,
                torch.norm(torch.stack([grad_x_mag, grad_y_mag, grad_z_mag]))
            ])
            
            return torch.tanh(output_tensor)  # Normalize to [-1, 1]
        else:
            # Exploration-driven fallback: Generate weak random actions instead of zeros
            # This prevents the brain from learning complete stillness
            exploration_strength = 0.1  # Small but non-zero exploration
            random_output = torch.randn(4, device=self.field_device) * exploration_strength
            return torch.tanh(random_output)  # Normalize to [-1, 1]
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive unified field statistics."""
        # Calculate field statistics with overflow protection
        total_activation = torch.sum(self.unified_field).item()
        max_activation = torch.max(self.unified_field).item()
        mean_activation = torch.mean(self.unified_field).item()
        
        # Protect against numerical overflow
        if np.isinf(total_activation) or np.isnan(total_activation) or total_activation > 1e10:
            print(f"⚠️ Field energy overflow detected: {total_activation:.3e} - clamping to safe range")
            # Emergency field normalization
            field_magnitude = torch.mean(torch.abs(self.unified_field)).item()
            if field_magnitude > 100.0:
                self.unified_field = torch.clamp(self.unified_field, -100.0, 100.0)
                total_activation = torch.sum(self.unified_field).item()
                max_activation = torch.max(self.unified_field).item()
                mean_activation = torch.mean(self.unified_field).item()
            else:
                total_activation = min(total_activation, 1000.0) if not (np.isinf(total_activation) or np.isnan(total_activation)) else 10.0
                max_activation = min(max_activation, 100.0) if not (np.isinf(max_activation) or np.isnan(max_activation)) else 1.0
                mean_activation = min(mean_activation, 10.0) if not (np.isinf(mean_activation) or np.isnan(mean_activation)) else 0.1
        
        stats = {
            'implementation_type': 'unified',
            'field_shape': list(self.unified_field.shape),
            'total_elements': self.unified_field.numel(),
            'total_activation': total_activation,
            'max_activation': max_activation,
            'mean_activation': mean_activation,
            'evolution_cycles': self.field_evolution_cycles,
            'total_imprints': self.total_imprints,
            'gradient_flows_count': len(self.gradient_flows),
            'memory_mb': (self.unified_field.numel() * 4) / (1024 * 1024)
        }
        
        # DEBUG: Removed - was causing spam on CUDA machines
        
        return stats
    
    def perform_adaptive_decay_maintenance(self) -> Dict[str, float]:
        """
        Perform adaptive energy decay during maintenance cycles.
        
        This was moved from the main processing loop to improve performance.
        Returns decay statistics for monitoring.
        """
        # Calculate total field energy
        total_field_energy = torch.sum(self.unified_field).item()
        
        # Determine adaptive decay rate based on energy level
        if total_field_energy > 200.0:
            decay_rate = 0.95  # 5% decay for very high energy
            energy_level = "very_high"
        elif total_field_energy > 100.0:
            decay_rate = 0.97  # 3% decay for high energy  
            energy_level = "high"
        elif total_field_energy > 50.0:
            decay_rate = 0.985  # 1.5% decay for medium energy
            energy_level = "medium"
        elif total_field_energy > 25.0:
            decay_rate = 0.992  # 0.8% decay for moderate energy
            energy_level = "moderate"
        else:
            decay_rate = 0.995  # 0.5% decay for low energy
            energy_level = "low"
        
        # Apply decay to unified field
        self.unified_field *= decay_rate
        
        # Calculate energy after decay
        energy_after_decay = torch.sum(self.unified_field).item()
        energy_removed = total_field_energy - energy_after_decay
        
        return {
            'total_energy': total_field_energy,
            'decay_rate': decay_rate,
            'energy_after_decay': energy_after_decay,
            'energy_removed': energy_removed,
            'energy_level': energy_level
        }
    
    def get_field_state_summary(self) -> Dict[str, Any]:
        """Get current unified field state summary."""
        stats = self.get_field_statistics()
        
        return {
            'field_energy': stats['total_activation'],
            'field_complexity': stats['max_activation'],
            'field_utilization': min(1.0, stats['mean_activation']),
            'evolution_cycles': self.field_evolution_cycles,
            'implementation': 'unified_field'
        }


# MultiTensorFieldImplementation removed - UnifiedFieldImplementation is now the only implementation


def create_adaptive_field_implementation(field_dimensions: List[FieldDimension],
                                       spatial_resolution: int,
                                       requested_device: torch.device,
                                       quiet_mode: bool = False) -> FieldImplementation:
    """
    Factory function to create the optimal field implementation for the current device.
    
    This is the main entry point for creating adaptive field implementations.
    """
    impl_type, actual_device = select_optimal_field_implementation(
        field_dimensions, spatial_resolution, requested_device, quiet_mode
    )
    
    # Always create unified implementation (multi-tensor path removed)
    return UnifiedFieldImplementation(
        spatial_resolution, field_dimensions, actual_device, quiet_mode
    )


# Legacy MultiTensorFieldImplementation code removed - UnifiedFieldImplementation now handles all functionality