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
        - 'unified': Use single unified field tensor (optimal for CUDA)
        - 'multi_tensor': Use multiple smaller tensors (required for MPS 16D limit)
    """
    total_field_dims = len(field_dimensions)
    
    if requested_device.type == 'cuda':
        # CUDA can handle any number of dimensions optimally
        return 'unified', requested_device
    elif requested_device.type == 'mps':
        # Check if we need multi-tensor for MPS 16D limit
        if total_field_dims > 16:
            # Use multi-tensor approach to stay on MPS
            if not quiet_mode:
                print(f"🔧 MPS 16D limit detected ({total_field_dims}D field) → Using multi-tensor implementation")
            return 'multi_tensor', requested_device
        else:
            # Can use unified on MPS
            return 'unified', requested_device
    else:
        # Default to unified on CPU
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
        self.unified_field = torch.zeros(field_shape, dtype=torch.float32, device=field_device)
        
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


class MultiTensorFieldImplementation(FieldImplementation):
    """
    Multi-tensor field implementation for MPS compatibility.
    
    Uses multiple smaller tensors to work around MPS 16D limitation while
    preserving unified field semantics through cross-field coupling.
    """
    
    def __init__(self, spatial_resolution: int, field_dimensions: List[FieldDimension],
                 field_device: torch.device, quiet_mode: bool = False):
        super().__init__(spatial_resolution, field_dimensions, field_device, quiet_mode)
        
        # Organize dimensions by family for multi-tensor approach
        self.dimension_families = self._organize_dimensions_by_family()
        
        # Create separate field tensors for each dynamics family
        self.field_tensors = {}
        self._create_field_tensors()
        
        # Cross-field coupling mechanisms
        self.coupling_strengths = self._initialize_coupling_strengths()
        self.field_memories = {}  # Separate memory for each field
        self.gradient_flows = {}
        
        if not quiet_mode:
            total_elements = sum(tensor.numel() for tensor in self.field_tensors.values())
            memory_mb = (total_elements * 4) / (1024 * 1024)
            print(f"🧠 Multi-tensor field: {len(self.field_tensors)} tensors, {total_elements:,} elements ({memory_mb:.1f}MB)")
            for name, tensor in self.field_tensors.items():
                print(f"   {name}: {list(tensor.shape)} ({tensor.numel():,} elements)")
    
    def _organize_dimensions_by_family(self) -> Dict[FieldDynamicsFamily, List[FieldDimension]]:
        """Organize field dimensions by their dynamics family."""
        families = defaultdict(list)
        for dim in self.field_dimensions:
            families[dim.family].append(dim)
        return dict(families)
    
    def _create_field_tensors(self):
        """Create separate field tensors for each dynamics family."""
        spatial_dims = [self.spatial_resolution] * 3  # [x, y, z]
        
        # Core spatio-temporal field (always present)
        self.field_tensors['core'] = torch.zeros(
            spatial_dims + [2],  # scale + time = 2 additional dims = 5D total
            dtype=torch.float32, device=self.field_device
        )
        
        # Create field tensors for each dynamics family
        for family, dims in self.dimension_families.items():
            if family == FieldDynamicsFamily.SPATIAL:
                continue  # Handled in core field
            
            family_dim_count = len(dims)
            if family_dim_count > 0:
                # Create tensor: [spatial_x, spatial_y, spatial_z, family_dynamics...]
                tensor_shape = spatial_dims + [family_dim_count]
                tensor_dims = len(tensor_shape)
                
                if tensor_dims <= 16:  # MPS compatible
                    self.field_tensors[family.value] = torch.zeros(
                        tensor_shape, dtype=torch.float32, device=self.field_device
                    )
                else:
                    # Fallback: split large families into smaller chunks
                    chunk_size = 13  # 3 spatial + 13 dynamics = 16D max
                    chunks = [dims[i:i+chunk_size] for i in range(0, len(dims), chunk_size)]
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_name = f"{family.value}_{chunk_idx}"
                        chunk_shape = spatial_dims + [len(chunk)]
                        self.field_tensors[chunk_name] = torch.zeros(
                            chunk_shape, dtype=torch.float32, device=self.field_device
                        )
    
    def _initialize_coupling_strengths(self) -> Dict[str, float]:
        """Initialize cross-field coupling strengths."""
        return {
            'core_to_dynamics': 0.3,      # Core field influences dynamics
            'dynamics_to_core': 0.2,      # Dynamics influence core field
            'cross_dynamics': 0.1,        # Dynamics families influence each other
            'temporal_coupling': 0.4,     # Temporal momentum coupling
        }
    
    def imprint_experience(self, experience: UnifiedFieldExperience) -> None:
        """Imprint experience across all field tensors with coupling."""
        field_coords = experience.field_coordinates
        intensity = experience.field_intensity
        
        # MAINTENANCE OPTIMIZATION: Store imprint metrics for maintenance analysis
        if not hasattr(self, '_maintenance_imprint_metrics'):
            self._maintenance_imprint_metrics = []
        
        pre_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        
        # Extract spatial coordinates
        spatial_coords = field_coords[:3]
        scale_coord = field_coords[3] if len(field_coords) > 3 else torch.tensor(0.0, device=self.field_device)
        time_coord = field_coords[4] if len(field_coords) > 4 else torch.tensor(0.0, device=self.field_device)
        
        # Convert to field indices
        x_center = int((spatial_coords[0] + 1) * 0.5 * (self.spatial_resolution - 1))
        y_center = int((spatial_coords[1] + 1) * 0.5 * (self.spatial_resolution - 1))
        z_center = int((spatial_coords[2] + 1) * 0.5 * (self.spatial_resolution - 1))
        
        # Clamp to valid ranges
        x_center = max(0, min(self.spatial_resolution - 1, x_center))
        y_center = max(0, min(self.spatial_resolution - 1, y_center))
        z_center = max(0, min(self.spatial_resolution - 1, z_center))
        
        # Imprint into core field first
        self._imprint_into_tensor(self.field_tensors['core'], x_center, y_center, z_center, 
                                 intensity, [scale_coord.item(), time_coord.item()])
        
        # Imprint into dynamics field tensors
        dynamics_start_idx = 5  # After spatial(3) + scale(1) + time(1)
        
        for tensor_name, tensor in self.field_tensors.items():
            if tensor_name == 'core':
                continue
                
            # Extract relevant dynamics coordinates for this tensor
            family_intensity = intensity * (0.8 + 0.4 * torch.rand(1).item())  # Add variation
            
            # Determine dynamics coordinates based on field_coordinates
            if dynamics_start_idx < len(field_coords):
                dynamics_coords = field_coords[dynamics_start_idx:dynamics_start_idx + tensor.shape[3]]
                dynamics_values = [coord.item() for coord in dynamics_coords]
                dynamics_start_idx += len(dynamics_coords)
            else:
                # Use random values if not enough coordinates
                dynamics_values = [torch.rand(1).item() * 2 - 1 for _ in range(tensor.shape[3])]
            
            self._imprint_into_tensor(tensor, x_center, y_center, z_center, 
                                     family_intensity, dynamics_values)
        
        self.total_imprints += 1
        
        # MAINTENANCE OPTIMIZATION: Store imprint energy metrics for analysis
        post_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        energy_change = post_energy - pre_energy
        
        if energy_change > 10.0 or post_energy > 500.0 or intensity > 1.0:
            self._maintenance_imprint_metrics.append({
                'intensity': intensity,
                'pre_energy': pre_energy,
                'post_energy': post_energy,
                'change': energy_change,
                'timestamp': time.time()
            })
            
            # Keep only recent imprint metrics (last 10)
            if len(self._maintenance_imprint_metrics) > 10:
                self._maintenance_imprint_metrics.pop(0)
    
    def _imprint_into_tensor(self, tensor: torch.Tensor, x_center: int, y_center: int, 
                           z_center: int, intensity: float, dynamics_values: List[float]):
        """Imprint experience into a specific field tensor."""
        spatial_spread = 2.0 + intensity
        
        # Calculate total spatial positions for normalization (5x5x5 = 125)
        spatial_range = range(-2, 3)  # 5 positions
        total_spatial_positions = len(spatial_range) ** 3  # 5^3 = 125
        
        # Pre-normalize intensity by both spatial and dimensional factors (OPTIMIZATION!)
        dimension_count = tensor.shape[3]
        base_intensity = intensity / (total_spatial_positions * dimension_count)
        
        for dx in spatial_range:  # Smaller spread for multi-tensor
            for dy in spatial_range:
                for dz in spatial_range:
                    x_pos = x_center + dx
                    y_pos = y_center + dy
                    z_pos = z_center + dz
                    
                    # Check spatial bounds
                    if (0 <= x_pos < self.spatial_resolution and
                        0 <= y_pos < self.spatial_resolution and
                        0 <= z_pos < self.spatial_resolution):
                        
                        # Calculate Gaussian weight (pre-normalized for efficiency)
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        weight = base_intensity * math.exp(-(distance**2) / (2 * spatial_spread**2))
                        
                        # Clamp weight to prevent exponential growth
                        weight = min(weight, 10.0)
                        
                        # Apply to dynamics dimensions (already normalized)
                        for dim_idx in range(dimension_count):
                            if dim_idx < len(dynamics_values):
                                dynamics_influence = dynamics_values[dim_idx]
                                final_weight = weight * (1 + dynamics_influence)
                                # Additional clamp on final weight
                                final_weight = min(final_weight, 20.0)
                                tensor[x_pos, y_pos, z_pos, dim_idx] += final_weight
    
    def evolve_field(self, dt: float = 0.1, current_input_stream: Optional[List[float]] = None) -> None:
        """Evolve all field tensors with cross-field coupling (performance optimized)."""
        # ENERGY DEBUG: Track energy before evolution
        pre_evolution_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        
        self.field_evolution_cycles += 1
        
        # Field evolution should run every cycle for proper learning
        # Previous "optimization" was skipping evolution and preventing learning
        
        # CRITICAL FIX: Inject input stream into field if provided
        if current_input_stream is not None:
            self._inject_input_stream(current_input_stream, dt)
        
        # Phase 1: Individual field evolution (process all tensors for proper learning)
        active_tensors = list(self.field_tensors.items())
        # Always process all tensors to ensure proper learning and field evolution
        
        # ENERGY DEBUG: Track energy before individual tensor evolution
        pre_individual_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        
        # PERFORMANCE OPTIMIZATION: Calculate total energy once per cycle for adaptive decay
        # This prevents expensive recalculation for every tensor evolution
        self._cached_total_energy = pre_individual_energy
        
        for tensor_name, tensor in active_tensors:
            self._evolve_individual_tensor(tensor, dt)
        
        # ENERGY DEBUG: Track energy after individual tensor evolution
        post_individual_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        individual_change = post_individual_energy - pre_individual_energy
        
        # Phase 2: Cross-field coupling (already optimized to run every 3rd cycle)
        pre_coupling_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        self._apply_cross_field_coupling(dt)
        post_coupling_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        coupling_change = post_coupling_energy - pre_coupling_energy
        
        # Phase 3: Update gradient flows (MAINTENANCE OPTIMIZED - run less frequently)
        pre_gradient_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        
        # MAINTENANCE OPTIMIZATION: Reduce gradient update frequency for better performance
        if not hasattr(self, '_gradient_update_counter'):
            self._gradient_update_counter = 0
        self._gradient_update_counter += 1
        
        # CRITICAL FIX: Ensure gradients are calculated early for action generation
        # Always update gradients in first 10 cycles, then every 3rd cycle for performance
        if self._gradient_update_counter <= 10 or self._gradient_update_counter % 3 == 0:
            self._update_multi_tensor_gradients()
        
        post_gradient_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        gradient_change = post_gradient_energy - pre_gradient_energy
        
        # MAINTENANCE OPTIMIZATION: Move expensive debugging to maintenance cycles
        # Store energy metrics for maintenance analysis instead of real-time debugging
        if not hasattr(self, '_maintenance_energy_metrics'):
            self._maintenance_energy_metrics = {
                'recent_changes': [],
                'phase_breakdowns': [],
                'high_energy_events': []
            }
        
        post_evolution_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        energy_change = post_evolution_energy - pre_evolution_energy
        
        # Store metrics for maintenance analysis (much faster than printing)
        if energy_change > 20.0 or post_evolution_energy > 500.0:
            self._maintenance_energy_metrics['recent_changes'].append({
                'pre': pre_evolution_energy,
                'post': post_evolution_energy,
                'change': energy_change,
                'individual': individual_change,
                'coupling': coupling_change,
                'gradients': gradient_change,
                'timestamp': time.time()
            })
            
            # Keep only recent metrics (last 10 significant changes)
            if len(self._maintenance_energy_metrics['recent_changes']) > 10:
                self._maintenance_energy_metrics['recent_changes'].pop(0)
            
            # Flag high energy events for maintenance attention
            if post_evolution_energy > 1000.0:
                self._maintenance_energy_metrics['high_energy_events'].append({
                    'energy': post_evolution_energy,
                    'timestamp': time.time()
                })
    
    def _inject_input_stream(self, input_stream: List[float], dt: float):
        """Inject input stream into field tensors to stimulate evolution."""
        if not input_stream:
            return
        
        # Convert input to tensor
        input_tensor = torch.tensor(input_stream, dtype=torch.float32, device=self.field_device)
        input_intensity = torch.mean(torch.abs(input_tensor)).item()
        
        # Don't inject if input is too weak
        if input_intensity < 0.01:
            return
        
        # Inject input into multiple field tensors with different patterns
        spatial_positions = min(self.spatial_resolution, 5)  # Limit for performance
        
        for i, (tensor_name, tensor) in enumerate(self.field_tensors.items()):
            if i >= len(input_stream):
                break
            
            # Get input value for this tensor
            input_value = input_stream[i % len(input_stream)]
            if abs(input_value) < 0.01:
                continue
            
            # Inject into center region of field
            center_x = spatial_positions // 2
            center_y = spatial_positions // 2
            center_z = spatial_positions // 2
            
            # Add input with spatial spread
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        x = center_x + dx
                        y = center_y + dy
                        z = center_z + dz
                        
                        if (0 <= x < spatial_positions and 
                            0 <= y < spatial_positions and 
                            0 <= z < spatial_positions):
                            
                            # Distance-based weighting
                            dist = abs(dx) + abs(dy) + abs(dz)
                            weight = 1.0 / (1.0 + dist)
                            
                            # Inject into multiple dimensions of this tensor
                            for dim_idx in range(min(tensor.shape[3], 3)):
                                injection_strength = input_value * weight * dt * 0.1
                                tensor[x, y, z, dim_idx] += injection_strength
    
    def _evolve_individual_tensor(self, tensor: torch.Tensor, dt: float):
        """Evolve an individual field tensor (optimized - decay moved to maintenance)."""
        # MAINTENANCE OPTIMIZATION: Decay moved to maintenance thread for better performance
        # Main processing loop now focuses purely on field evolution and dynamics
        
        # AGGRESSIVE OPTIMIZATION: Skip diffusion during stable periods or use simplified diffusion
        if not hasattr(self, '_diffusion_skip_counter'):
            self._diffusion_skip_counter = 0
        
        # Check tensor activity to decide on diffusion complexity
        tensor_activity = torch.sum(torch.abs(tensor)).item()
        
        # FIXED: Only skip diffusion if tensor has significant activity and field is established
        # This prevents skipping during initial field development
        if tensor_activity > 10.0 and hasattr(self, 'field_evolution_cycles') and self.field_evolution_cycles > 50 and self._diffusion_skip_counter % 4 == 0:
            self._diffusion_skip_counter += 1
            return
        
        self._diffusion_skip_counter += 1
        
        # Optimized spatial diffusion - skip if resolution is too small
        if self.spatial_resolution > 4:  # Only apply to meaningful sizes
            diffusion_kernel = 0.02 / 6.0
            
            # Pre-allocate result tensor to avoid repeated memory allocation
            if not hasattr(self, '_diffusion_temps'):
                self._diffusion_temps = {}
            
            temp_key = f'{tensor.shape}'
            if temp_key not in self._diffusion_temps:
                self._diffusion_temps[temp_key] = torch.zeros_like(tensor)
            
            temp = self._diffusion_temps[temp_key]
            temp.zero_()
            
            # Efficient 3D diffusion using tensor slicing (parallel operations)
            if tensor.size(0) > 2:  # X-direction
                temp[1:-1, :, :] = diffusion_kernel * (
                    tensor[2:, :, :] + tensor[:-2, :, :] - 2 * tensor[1:-1, :, :]
                )
            
            if tensor.size(1) > 2:  # Y-direction  
                temp[:, 1:-1, :] += diffusion_kernel * (
                    tensor[:, 2:, :] + tensor[:, :-2, :] - 2 * tensor[:, 1:-1, :]
                )
            
            if tensor.size(2) > 2:  # Z-direction
                temp[:, :, 1:-1] += diffusion_kernel * (
                    tensor[:, :, 2:] + tensor[:, :, :-2] - 2 * tensor[:, :, 1:-1]
                )
            
            # Apply diffusion result
            tensor.add_(temp)
    
    def _apply_cross_field_coupling(self, dt: float):
        """Apply coupling between field tensors to preserve unified semantics (optimized)."""
        core_tensor = self.field_tensors['core']
        
        # Optimized: Only apply coupling every few cycles for performance
        if not hasattr(self, '_coupling_cycle'):
            self._coupling_cycle = 0
        self._coupling_cycle += 1
        
        # Apply coupling every 3rd cycle to reduce computational load
        if self._coupling_cycle % 3 != 0:
            return
        
        # Core field influences all dynamics fields (ENERGY CONSERVING VERSION)
        core_influence = torch.mean(core_tensor, dim=3, keepdim=True)
        coupling_strength = self.coupling_strengths['core_to_dynamics']
        
        # Count dynamics tensors to properly distribute energy
        dynamics_tensor_count = len([t for name, t in self.field_tensors.items() if name != 'core'])
        
        if dynamics_tensor_count > 0:
            # ENERGY CONSERVATION: Transfer energy instead of creating it
            energy_per_tensor = coupling_strength / dynamics_tensor_count  # Distribute core energy
            
            for tensor_name, tensor in self.field_tensors.items():
                if tensor_name == 'core':
                    continue
                
                # Add energy to dynamics tensor
                tensor.add_(core_influence, alpha=energy_per_tensor)
                
            # CRITICAL: Remove transferred energy from core to conserve total energy
            core_tensor.add_(core_influence, alpha=-coupling_strength)
        
        # Dynamics fields influence core field back (ENERGY CONSERVING VERSION)
        if len(self.field_tensors) > 1:  # Only if we have dynamics tensors
            dynamics_influence = torch.zeros_like(core_tensor)
            dynamics_count = 0
            
            # Collect influence and reduce source tensors
            for tensor_name, tensor in self.field_tensors.items():
                if tensor_name == 'core':
                    continue
                
                # Use more efficient mean calculation
                tensor_influence = torch.mean(tensor, dim=3, keepdim=True)
                dynamics_influence += tensor_influence
                dynamics_count += 1
            
            if dynamics_count > 0:
                # ENERGY CONSERVATION: Transfer energy from dynamics to core
                transfer_strength = self.coupling_strengths['dynamics_to_core'] / dynamics_count
                dynamics_influence *= transfer_strength
                
                # Add transferred energy to core
                core_tensor.add_(dynamics_influence)
                
                # CRITICAL: Remove transferred energy from dynamics tensors
                for tensor_name, tensor in self.field_tensors.items():
                    if tensor_name == 'core':
                        continue
                    tensor_influence = torch.mean(tensor, dim=3, keepdim=True)
                    tensor.add_(tensor_influence, alpha=-transfer_strength)
    
    def _update_multi_tensor_gradients(self):
        """Update gradient flows across all field tensors (optimized)."""
        # Optimized: Only update gradients every few cycles
        if not hasattr(self, '_gradient_cycle'):
            self._gradient_cycle = 0
        self._gradient_cycle += 1
        
        # CRITICAL FIX: Always update gradients in early cycles for action generation
        # Only skip gradient updates after the first 10 cycles and only every other cycle
        if self._gradient_cycle > 10 and self._gradient_cycle % 2 != 0:
            return  # Keep previous gradients after initial period
        
        self.gradient_flows = {}
        
        # Compute gradients for each field tensor (optimized)
        for tensor_name, tensor in self.field_tensors.items():
            if self.spatial_resolution > 1:
                # More efficient gradient computation - avoid creating full-size zero tensors
                # X gradient (simplified)
                if tensor.size(0) > 1:
                    grad_x_diff = tensor[1:, :, :] - tensor[:-1, :, :]
                    self.gradient_flows[f'{tensor_name}_grad_x'] = grad_x_diff
                
                # Y gradient (simplified)  
                if tensor.size(1) > 1:
                    grad_y_diff = tensor[:, 1:, :] - tensor[:, :-1, :]
                    self.gradient_flows[f'{tensor_name}_grad_y'] = grad_y_diff
                
                # Z gradient (simplified)
                if tensor.size(2) > 1:
                    grad_z_diff = tensor[:, :, 1:] - tensor[:, :, :-1]
                    self.gradient_flows[f'{tensor_name}_grad_z'] = grad_z_diff
    
    def compute_field_gradients(self) -> Dict[str, torch.Tensor]:
        """Compute gradients across all field tensors."""
        return self.gradient_flows
    
    def generate_field_output(self, field_coordinates: torch.Tensor) -> torch.Tensor:
        """Generate output stream from multi-tensor field state."""
        # Combine gradients from all field tensors
        output_components = []
        
        # Use core field gradients as primary output with proper magnitude extraction
        if 'core_grad_x' in self.gradient_flows:
            # Fix: Use max magnitude instead of mean to preserve gradient strength
            core_grad_x = torch.max(torch.abs(self.gradient_flows['core_grad_x'])) * torch.sign(torch.sum(self.gradient_flows['core_grad_x']))
            core_grad_y = torch.max(torch.abs(self.gradient_flows['core_grad_y'])) * torch.sign(torch.sum(self.gradient_flows['core_grad_y']))
            core_grad_z = torch.max(torch.abs(self.gradient_flows['core_grad_z'])) * torch.sign(torch.sum(self.gradient_flows['core_grad_z']))
            
            # AMPLIFICATION: Scale up gradients for meaningful action generation
            gradient_amplification = 100.0  # Scale tiny gradients to actionable range
            core_grad_x *= gradient_amplification
            core_grad_y *= gradient_amplification  
            core_grad_z *= gradient_amplification
            
            output_components.extend([core_grad_x, core_grad_y, core_grad_z])
        
        # Add influence from dynamics fields
        dynamics_influence = torch.tensor(0.0, device=self.field_device)
        dynamics_count = 0
        
        for grad_name, grad_tensor in self.gradient_flows.items():
            if not grad_name.startswith('core_'):
                dynamics_influence += torch.mean(torch.abs(grad_tensor))
                dynamics_count += 1
        
        if dynamics_count > 0:
            dynamics_influence /= dynamics_count
            # AMPLIFICATION: Scale dynamics influence to match core gradients
            dynamics_influence *= gradient_amplification
            output_components.append(dynamics_influence)
        else:
            output_components.append(torch.tensor(0.0, device=self.field_device))
        
        # Create output tensor
        if output_components:
            output_tensor = torch.stack(output_components)
            return torch.tanh(output_tensor)  # Normalize to [-1, 1]
        else:
            # Exploration-driven fallback: Generate weak random actions instead of zeros
            # This prevents the brain from learning complete stillness
            exploration_strength = 0.1  # Small but non-zero exploration
            random_output = torch.randn(4, device=self.field_device) * exploration_strength
            return torch.tanh(random_output)  # Normalize to [-1, 1]
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive multi-tensor field statistics."""
        total_elements = sum(tensor.numel() for tensor in self.field_tensors.values())
        total_activation = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        max_activation = max(torch.max(tensor).item() for tensor in self.field_tensors.values())
        mean_activation = total_activation / total_elements if total_elements > 0 else 0.0
        
        # Protect against numerical overflow in multi-tensor implementation
        if np.isinf(total_activation) or np.isnan(total_activation) or total_activation > 1e10:
            print(f"⚠️ Multi-tensor field energy overflow detected: {total_activation:.3e} - clamping tensors")
            # Emergency field normalization for all tensors
            for name, tensor in self.field_tensors.items():
                field_magnitude = torch.mean(torch.abs(tensor)).item()
                if field_magnitude > 100.0:
                    self.field_tensors[name] = torch.clamp(tensor, -100.0, 100.0)
            
            # Recalculate after clamping
            total_activation = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
            max_activation = max(torch.max(tensor).item() for tensor in self.field_tensors.values())
            mean_activation = total_activation / total_elements if total_elements > 0 else 0.0
        
        # MAINTENANCE OPTIMIZATION: Store clamping events for maintenance analysis
        if not hasattr(self, '_maintenance_clamp_events'):
            self._maintenance_clamp_events = []
        
        original_total = total_activation
        
        # Additional safety bounds with maintenance logging
        if total_activation > 1000.0:
            self._maintenance_clamp_events.append({
                'type': 'energy_clamp',
                'original': total_activation,
                'clamped': 1000.0,
                'timestamp': time.time()
            })
            total_activation = 1000.0
        elif np.isinf(total_activation) or np.isnan(total_activation):
            self._maintenance_clamp_events.append({
                'type': 'energy_reset',
                'original': str(total_activation),
                'reset': 10.0,
                'timestamp': time.time()
            })
            total_activation = 10.0
            
        if max_activation > 100.0:
            self._maintenance_clamp_events.append({
                'type': 'max_activation_clamp',
                'original': max_activation,
                'clamped': 100.0,
                'timestamp': time.time()
            })
            max_activation = 100.0
        elif np.isinf(max_activation) or np.isnan(max_activation):
            max_activation = 1.0
        
        # Keep only recent clamp events (last 5)
        if len(self._maintenance_clamp_events) > 5:
            self._maintenance_clamp_events.pop(0)
        
        # Per-tensor statistics
        tensor_stats = {}
        for name, tensor in self.field_tensors.items():
            tensor_stats[name] = {
                'shape': list(tensor.shape),
                'elements': tensor.numel(),
                'activation': torch.sum(tensor).item(),
                'max_activation': torch.max(tensor).item()
            }
        
        return {
            'implementation_type': 'multi_tensor',
            'tensor_count': len(self.field_tensors),
            'total_elements': total_elements,
            'total_activation': total_activation,
            'max_activation': max_activation,
            'mean_activation': mean_activation,
            'evolution_cycles': self.field_evolution_cycles,
            'total_imprints': self.total_imprints,
            'gradient_flows_count': len(self.gradient_flows),
            'memory_mb': (total_elements * 4) / (1024 * 1024),
            'tensor_details': tensor_stats
        }
    
    def get_field_state_summary(self) -> Dict[str, Any]:
        """Get current multi-tensor field state summary."""
        stats = self.get_field_statistics()
        
        return {
            'field_energy': stats['total_activation'],
            'field_complexity': stats['max_activation'],
            'field_utilization': min(1.0, stats['mean_activation']),
            'evolution_cycles': self.field_evolution_cycles,
            'implementation': 'multi_tensor_field',
            'tensor_count': stats['tensor_count']
        }


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
    
    if impl_type == 'unified':
        return UnifiedFieldImplementation(
            spatial_resolution, field_dimensions, actual_device, quiet_mode
        )
    else:  # multi_tensor
        return MultiTensorFieldImplementation(
            spatial_resolution, field_dimensions, actual_device, quiet_mode
        )


# Add decay maintenance method to MultiTensorFieldImplementation
def _add_decay_maintenance_to_multi_tensor():
    """Add decay maintenance method to MultiTensorFieldImplementation class."""
    
    def perform_adaptive_decay_maintenance(self) -> Dict[str, float]:
        """
        Perform adaptive energy decay during maintenance cycles.
        
        This was moved from the main processing loop to improve performance.
        Returns decay statistics for monitoring.
        """
        if not hasattr(self, 'field_tensors') or not self.field_tensors:
            return {'total_energy': 0.0, 'decay_rate': 0.0, 'energy_after_decay': 0.0}
        
        # Calculate total field energy
        total_field_energy = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        
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
        
        # Apply decay to all field tensors
        for tensor_name, tensor in self.field_tensors.items():
            tensor.mul_(decay_rate)
        
        # Calculate energy after decay
        energy_after_decay = sum(torch.sum(tensor).item() for tensor in self.field_tensors.values())
        energy_removed = total_field_energy - energy_after_decay
        
        # Store decay statistics for analysis
        if not hasattr(self, '_decay_statistics'):
            self._decay_statistics = []
        
        decay_stats = {
            'timestamp': time.time(),
            'total_energy_before': total_field_energy,
            'energy_after_decay': energy_after_decay,
            'energy_removed': energy_removed,
            'decay_rate': decay_rate,
            'energy_level': energy_level,
            'tensors_processed': len(self.field_tensors)
        }
        
        self._decay_statistics.append(decay_stats)
        
        # Keep only recent decay statistics
        if len(self._decay_statistics) > 20:
            self._decay_statistics.pop(0)
        
        return {
            'total_energy': total_field_energy,
            'decay_rate': decay_rate,
            'energy_after_decay': energy_after_decay,
            'energy_removed': energy_removed,
            'energy_level': energy_level
        }
    
    # Add method to the class
    MultiTensorFieldImplementation.perform_adaptive_decay_maintenance = perform_adaptive_decay_maintenance

# Apply the method addition
_add_decay_maintenance_to_multi_tensor()