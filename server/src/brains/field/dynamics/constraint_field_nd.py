#!/usr/bin/env python3
"""
N-Dimensional Constraint Field Dynamics

This extends constraint-based field dynamics to arbitrary dimensions,
enabling constraint discovery and enforcement in high-dimensional spaces
like the 37D UnifiedFieldBrain.

Key improvements over ConstraintField4D:
- Handles arbitrary number of dimensions
- Efficient sparse constraint representation
- Dimension-aware constraint types
- Scalable constraint discovery
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import math

# Define constraint types locally since constraint_field_dynamics.py was archived
class FieldConstraintType(Enum):
    """Types of constraints that emerge from field dynamics."""
    GRADIENT_FLOW = "gradient_flow"         # Gradients should flow smoothly
    TOPOLOGY_BOUNDARY = "topology_boundary" # Boundaries in field topology
    ACTIVATION_THRESHOLD = "activation_threshold" # Activation level constraints
    TEMPORAL_MOMENTUM = "temporal_momentum" # Temporal consistency
    SCALE_COUPLING = "scale_coupling"       # Multi-scale relationships
    PATTERN_COHERENCE = "pattern_coherence" # Pattern consistency


@dataclass
class FieldConstraintND:
    """A constraint that operates in N-dimensional field space."""
    constraint_type: FieldConstraintType
    dimensions: List[int]  # Which dimensions this constraint affects
    region_centers: torch.Tensor  # Centers of constraint regions [n_regions x n_dims]
    region_radii: torch.Tensor  # Radii for each dimension [n_regions x n_dims]
    strength: float  # Constraint strength
    gradient_direction: Optional[torch.Tensor] = None  # [n_dims] gradient
    threshold_value: Optional[float] = None
    temporal_momentum: Optional[float] = None
    pattern_signature: Optional[torch.Tensor] = None
    discovery_timestamp: float = 0.0
    violation_count: int = 0
    enforcement_success: float = 0.0
    
    def applies_to_point(self, point: torch.Tensor) -> bool:
        """Check if constraint applies to a given field point."""
        if len(self.dimensions) == 0:
            return False
            
        # Extract relevant dimensions
        relevant_point = point[self.dimensions]
        
        # Check if point is within any constraint region
        for i in range(self.region_centers.shape[0]):
            center = self.region_centers[i]
            radii = self.region_radii[i]
            
            # Check if within ellipsoid defined by radii
            normalized_dist = (relevant_point - center) / (radii + 1e-8)
            if torch.norm(normalized_dist) <= 1.0:
                return True
                
        return False
    
    def compute_constraint_force(self, point: torch.Tensor) -> torch.Tensor:
        """Compute constraint force at a given point."""
        force = torch.zeros_like(point)
        
        if not self.applies_to_point(point):
            return force
            
        # Apply constraint based on type
        if self.constraint_type == FieldConstraintType.GRADIENT_FLOW:
            if self.gradient_direction is not None:
                # Apply gradient in specified dimensions
                force[self.dimensions] = self.gradient_direction * self.strength
                
        elif self.constraint_type == FieldConstraintType.ACTIVATION_THRESHOLD:
            if self.threshold_value is not None:
                # Push activation toward threshold
                current_activation = torch.mean(point[self.dimensions])
                if current_activation < self.threshold_value:
                    force[self.dimensions] = self.strength
                else:
                    force[self.dimensions] = -self.strength
                    
        return force


class ConstraintFieldND:
    """
    N-Dimensional Field with Constraint Discovery and Enforcement
    
    Designed to work with arbitrary dimensional fields like the 37D UnifiedFieldBrain.
    """
    
    def __init__(self, 
                 field_shape: List[int],
                 dimension_names: Optional[List[str]] = None,
                 constraint_discovery_rate: float = 0.1,
                 constraint_enforcement_strength: float = 0.3,
                 max_constraints: int = 100,
                 device: Optional[torch.device] = None,
                 quiet_mode: bool = False):
        
        self.field_shape = field_shape
        self.n_dimensions = len(field_shape)
        self.dimension_names = dimension_names or [f"dim_{i}" for i in range(self.n_dimensions)]
        self.constraint_discovery_rate = constraint_discovery_rate
        self.constraint_enforcement_strength = constraint_enforcement_strength
        self.max_constraints = max_constraints
        self.device = device or torch.device('cpu')
        self.quiet_mode = quiet_mode
        
        # Constraint storage
        self.active_constraints: List[FieldConstraintND] = []
        self.constraint_history: List[FieldConstraintND] = []
        
        # Discovery parameters
        self.gradient_threshold = 0.05
        self.pattern_stability_window = 10
        self.min_constraint_lifetime = 50
        
        # Statistics
        self.constraints_discovered = 0
        self.constraints_enforced = 0
        self.discovery_cycles = 0
        
        if not quiet_mode:
            print(f"🔗 ConstraintFieldND initialized")
            print(f"   Field shape: {field_shape} ({self.n_dimensions}D)")
            print(f"   Max constraints: {max_constraints}")
    
    def discover_constraints(self, field: torch.Tensor, 
                           field_gradients: Optional[Dict[str, torch.Tensor]] = None) -> List[FieldConstraintND]:
        """
        Discover constraints from current field state.
        
        Args:
            field: The N-dimensional field tensor
            field_gradients: Optional pre-computed gradients
            
        Returns:
            List of newly discovered constraints
        """
        new_constraints = []
        self.discovery_cycles += 1
        
        # Only discover constraints occasionally
        if np.random.random() > self.constraint_discovery_rate:
            return new_constraints
        
        # 1. Gradient-based constraints
        if field_gradients:
            for dim_name, gradient in field_gradients.items():
                # Find regions with strong, consistent gradients
                grad_magnitude = torch.abs(gradient)
                high_grad_mask = grad_magnitude > self.gradient_threshold
                
                if torch.any(high_grad_mask):
                    # Extract dimension index from gradient name
                    if dim_name.startswith('gradient_dim_'):
                        dim_idx = int(dim_name.split('_')[-1])
                        if dim_idx < self.n_dimensions:
                            # Create gradient flow constraint
                            constraint = self._create_gradient_constraint(
                                field, gradient, dim_idx, high_grad_mask
                            )
                            if constraint:
                                new_constraints.append(constraint)
        
        # 2. Activation threshold constraints - OPTIMIZED
        # Only check spatial dimensions (first 3) for efficiency
        if self.n_dimensions >= 3:
            for dim_idx in range(min(3, self.n_dimensions)):
                # Sample a subset of the field for efficiency
                sample_size = min(1000, field.numel() // 10)
                flat_field = field.flatten()
                sample_indices = torch.randperm(flat_field.size(0))[:sample_size]
                field_sample = flat_field[sample_indices]
                
                # Check if there's significant variation in the sample
                if torch.std(field_sample) > 0.1:
                    # Use sampled mean as threshold
                    threshold = torch.median(field_sample).item()
                    constraint = FieldConstraintND(
                        constraint_type=FieldConstraintType.ACTIVATION_THRESHOLD,
                        dimensions=[dim_idx],
                        region_centers=torch.zeros(1, 1),
                        region_radii=torch.ones(1, 1) * float('inf'),
                        strength=0.3,
                        threshold_value=threshold,
                        discovery_timestamp=time.time()
                    )
                    new_constraints.append(constraint)
                    break  # Only create one threshold constraint per discovery
        
        # 3. Cross-dimensional coupling constraints
        # Disabled for performance - the corrcoef calculation is expensive
        # and the current 11D field doesn't meet the requirements anyway
        pass
        
        # Add to active constraints (with limit)
        for constraint in new_constraints:
            if len(self.active_constraints) >= self.max_constraints:
                # Remove oldest constraint
                self.active_constraints.pop(0)
            self.active_constraints.append(constraint)
            self.constraints_discovered += 1
        
        return new_constraints
    
    def enforce_constraints(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply active constraints to the field.
        
        Args:
            field: The current field state (11D tensor)
            
        Returns:
            Constraint forces to be applied to the field
        """
        constraint_forces = torch.zeros_like(field)
        
        # Remove expired constraints
        current_time = time.time()
        self.active_constraints = [
            c for c in self.active_constraints 
            if current_time - c.discovery_timestamp < self.min_constraint_lifetime
        ]
        
        # Apply each active constraint
        for constraint in self.active_constraints:
            # For gradient flow constraints, apply forces globally
            if constraint.constraint_type == FieldConstraintType.GRADIENT_FLOW:
                if constraint.gradient_direction is not None:
                    # Apply gradient force uniformly across the field
                    # This creates a general flow in the specified direction
                    force_magnitude = float(constraint.gradient_direction[0]) * constraint.strength * self.constraint_enforcement_strength
                    constraint_forces += force_magnitude * 0.1  # Scale down for stability
                    self.constraints_enforced += 1
                            
            elif constraint.constraint_type == FieldConstraintType.ACTIVATION_THRESHOLD:
                if constraint.threshold_value is not None:
                    # Apply threshold constraint - push field values toward threshold
                    below_threshold = field < constraint.threshold_value
                    above_threshold = field >= constraint.threshold_value
                    
                    # Push below-threshold values up, above-threshold values down
                    constraint_forces[below_threshold] += constraint.strength * self.constraint_enforcement_strength
                    constraint_forces[above_threshold] -= constraint.strength * self.constraint_enforcement_strength
                    self.constraints_enforced += 1
                    
            elif constraint.constraint_type == FieldConstraintType.SCALE_COUPLING:
                # Apply coupling between different dimension families
                # This helps coordinate activity across the field
                if len(constraint.dimensions) >= 2:
                    # Simple coupling: average activity should be similar
                    dim1, dim2 = constraint.dimensions[0], constraint.dimensions[1]
                    if dim1 < len(field.shape) and dim2 < len(field.shape):
                        # Calculate mean activation in each dimension
                        mean1 = torch.mean(field.select(dim1, 0))
                        mean2 = torch.mean(field.select(dim2, 0))
                        
                        # Apply force to reduce difference
                        diff = mean1 - mean2
                        constraint_forces += diff * constraint.strength * self.constraint_enforcement_strength * 0.01
                        self.constraints_enforced += 1
        
        return constraint_forces
    
    def _create_gradient_constraint(self, field: torch.Tensor, gradient: torch.Tensor, 
                                  dim_idx: int, high_grad_mask: torch.Tensor) -> Optional[FieldConstraintND]:
        """Create a gradient flow constraint."""
        # Find center of high gradient region
        indices = torch.nonzero(high_grad_mask)
        if len(indices) == 0:
            return None
            
        center = torch.mean(indices.float(), dim=0)
        
        # Estimate region size
        std = torch.std(indices.float(), dim=0) + 1.0
        
        # Create constraint
        constraint = FieldConstraintND(
            constraint_type=FieldConstraintType.GRADIENT_FLOW,
            dimensions=[dim_idx],
            region_centers=center.unsqueeze(0),
            region_radii=std.unsqueeze(0) * 2,  # 2 std deviations
            strength=0.5,
            gradient_direction=torch.mean(gradient[high_grad_mask]).unsqueeze(0),
            discovery_timestamp=time.time()
        )
        
        return constraint
    
    def _create_threshold_constraint(self, field: torch.Tensor, dim_idx: int, 
                                   dim_values: torch.Tensor) -> Optional[FieldConstraintND]:
        """Create an activation threshold constraint."""
        # Find stable threshold value
        threshold = torch.median(dim_values).item()
        
        # Create constraint for entire dimension
        constraint = FieldConstraintND(
            constraint_type=FieldConstraintType.ACTIVATION_THRESHOLD,
            dimensions=[dim_idx],
            region_centers=torch.zeros(1, 1),  # Applies everywhere
            region_radii=torch.ones(1, 1) * float('inf'),  # Infinite radius
            strength=0.3,
            threshold_value=threshold,
            discovery_timestamp=time.time()
        )
        
        return constraint
    
    def _discover_coupling_constraints(self, field: torch.Tensor) -> Optional[FieldConstraintND]:
        """Discover coupling between dimensions."""
        # Simple correlation check between spatial and energy dimensions
        if self.n_dimensions >= 30:  # Ensure we have energy dimensions
            spatial_dims = list(range(3))  # First 3 dimensions
            energy_dims = list(range(25, min(29, self.n_dimensions)))  # Energy dimensions
            
            # Check if high spatial activity correlates with high energy
            spatial_activity = torch.mean(field, dim=tuple(range(3, self.n_dimensions)))
            energy_activity = torch.mean(field, dim=tuple(set(range(self.n_dimensions)) - set(energy_dims)))
            
            correlation = torch.corrcoef(torch.stack([
                spatial_activity.flatten(),
                energy_activity.flatten()
            ]))[0, 1]
            
            if abs(correlation) > 0.5:  # Significant correlation
                constraint = FieldConstraintND(
                    constraint_type=FieldConstraintType.SCALE_COUPLING,
                    dimensions=spatial_dims + energy_dims,
                    region_centers=torch.zeros(1, len(spatial_dims + energy_dims)),
                    region_radii=torch.ones(1, len(spatial_dims + energy_dims)) * float('inf'),
                    strength=0.2 * torch.sign(correlation).item(),
                    discovery_timestamp=time.time()
                )
                return constraint
                
        return None
    
    def _get_constraint_sample_indices(self, field_shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get sample indices for constraint checking."""
        # For efficiency, sample a subset of points
        n_samples = min(100, np.prod(field_shape[:3]))  # Focus on spatial dims
        
        indices = []
        for _ in range(n_samples):
            idx = tuple(np.random.randint(0, s) for s in field_shape)
            indices.append(idx)
            
        return indices
    
    def get_constraint_stats(self) -> Dict[str, Any]:
        """Get constraint system statistics."""
        constraint_types = {}
        for c in self.active_constraints:
            ctype = c.constraint_type.value
            constraint_types[ctype] = constraint_types.get(ctype, 0) + 1
            
        return {
            'active_constraints': len(self.active_constraints),
            'constraints_discovered': self.constraints_discovered,
            'constraints_enforced': self.constraints_enforced,
            'discovery_cycles': self.discovery_cycles,
            'constraint_types': constraint_types,
            'discovery_rate': self.constraint_discovery_rate,
            'enforcement_strength': self.constraint_enforcement_strength
        }