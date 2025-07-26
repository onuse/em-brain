"""
Dynamic Dimension Calculator

Calculates field dimensions based on robot capabilities using logarithmic scaling.
This replaces the hardcoded 37D architecture with truly dynamic dimensions.
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import field types from the brain module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from brains.field.core_brain import FieldDynamicsFamily, FieldDimension


class DynamicDimensionCalculator:
    """
    Calculates brain field dimensions based on robot capabilities.
    
    Uses logarithmic scaling: dimensions = log2(sensors) × complexity_factor
    
    This ensures:
    - Simple robots (8 sensors) → ~12D brain
    - Medium robots (24 sensors) → ~27D brain  
    - Complex robots (64 sensors) → ~36D brain
    - Very complex robots (256 sensors) → ~48D brain
    """
    
    def __init__(self, complexity_factor: float = 6.0, min_dims_per_family: int = 1):
        self.complexity_factor = complexity_factor
        self.min_dims_per_family = min_dims_per_family
        
        # Family distribution weights (must sum to 1.0)
        self.family_weights = {
            FieldDynamicsFamily.SPATIAL: 0.20,      # 20% for position/scale/time
            FieldDynamicsFamily.OSCILLATORY: 0.15,  # 15% for frequencies/rhythms
            FieldDynamicsFamily.FLOW: 0.20,         # 20% for gradients/motion
            FieldDynamicsFamily.TOPOLOGY: 0.15,     # 15% for stable patterns
            FieldDynamicsFamily.ENERGY: 0.10,       # 10% for resource management
            FieldDynamicsFamily.COUPLING: 0.12,     # 12% for correlations
            FieldDynamicsFamily.EMERGENCE: 0.08     # 8% for novelty/creativity
        }
        
    def calculate_dimensions(self, 
                           sensory_dim: int, 
                           motor_dim: int,
                           capabilities: Optional[Dict[str, Any]] = None) -> List[FieldDimension]:
        """
        Calculate field dimensions based on robot profile.
        
        Args:
            sensory_dim: Number of robot sensors
            motor_dim: Number of robot actuators
            capabilities: Optional capability flags (visual, audio, manipulation, etc)
            
        Returns:
            List of FieldDimension objects defining the brain architecture
        """
        # Calculate total dimensions using logarithmic scaling
        base_dims = int(math.log2(max(8, sensory_dim)) * self.complexity_factor)
        
        # Adjust based on motor complexity
        motor_factor = math.log2(max(2, motor_dim)) / 2.0
        total_dims = int(base_dims * (1.0 + motor_factor * 0.1))
        
        # Ensure minimum viable brain
        total_dims = max(12, total_dims)
        
        print(f"📐 Calculating dimensions: {sensory_dim}D sensors → {total_dims}D brain")
        
        # Distribute dimensions across families
        family_dims = self._distribute_dimensions(total_dims, capabilities)
        
        # Create dimension objects
        dimensions = []
        index = 0
        
        for family, count in family_dims.items():
            family_dimensions = self._create_family_dimensions(family, count, index)
            dimensions.extend(family_dimensions)
            index += len(family_dimensions)
            
        return dimensions
    
    def _distribute_dimensions(self, total_dims: int, 
                             capabilities: Optional[Dict[str, Any]]) -> Dict[FieldDynamicsFamily, int]:
        """Distribute total dimensions across physics families."""
        
        distribution = {}
        allocated = 0
        
        # First pass: allocate based on weights
        for family, weight in self.family_weights.items():
            dims = max(self.min_dims_per_family, int(total_dims * weight))
            distribution[family] = dims
            allocated += dims
        
        # Adjust for capabilities if provided
        if capabilities:
            if capabilities.get('visual_processing', False):
                distribution[FieldDynamicsFamily.OSCILLATORY] += 1
                distribution[FieldDynamicsFamily.FLOW] += 1
                allocated += 2
                
            if capabilities.get('manipulation', False):
                distribution[FieldDynamicsFamily.TOPOLOGY] += 1
                distribution[FieldDynamicsFamily.COUPLING] += 1
                allocated += 2
        
        # Handle over/under allocation
        if allocated > total_dims:
            # Remove from largest families first
            while allocated > total_dims:
                largest = max(distribution.items(), key=lambda x: x[1])
                if largest[1] > self.min_dims_per_family:
                    distribution[largest[0]] -= 1
                    allocated -= 1
                else:
                    break
                    
        elif allocated < total_dims:
            # Add to spatial first (most fundamental)
            distribution[FieldDynamicsFamily.SPATIAL] += (total_dims - allocated)
            
        return distribution
    
    def _create_family_dimensions(self, family: FieldDynamicsFamily, 
                                count: int, start_index: int) -> List[FieldDimension]:
        """Create specific dimensions for a physics family."""
        
        dimensions = []
        
        if family == FieldDynamicsFamily.SPATIAL:
            # Always include core spatial dimensions
            core_spatial = [
                ("x", "Robot X position"),
                ("y", "Robot Y position"),
                ("z", "Robot Z position/height")
            ]
            
            # Add additional spatial dimensions if count allows
            extra_spatial = [
                ("scale", "Abstraction level"),
                ("time", "Temporal position"),
                ("rotation", "Orientation angle"),
                ("velocity", "Movement speed")
            ]
            
            dims_to_create = core_spatial[:count]
            if count > len(core_spatial):
                dims_to_create.extend(extra_spatial[:count - len(core_spatial)])
                
            for i, (name, desc) in enumerate(dims_to_create):
                dimensions.append(FieldDimension(
                    f"spatial_{name}", family, start_index + i, -1.0, 1.0, 0.0, desc
                ))
                
        elif family == FieldDynamicsFamily.OSCILLATORY:
            oscillatory_dims = [
                ("primary_freq", "Primary oscillation"),
                ("harmonic_1", "First harmonic"),
                ("rhythm", "Rhythmic patterns"),
                ("phase", "Phase relationships")
            ]
            
            for i in range(min(count, len(oscillatory_dims))):
                name, desc = oscillatory_dims[i]
                dimensions.append(FieldDimension(
                    f"osc_{name}", family, start_index + i, -1.0, 1.0, 0.0, desc
                ))
                
        elif family == FieldDynamicsFamily.FLOW:
            flow_dims = [
                ("gradient_x", "X-direction gradient"),
                ("gradient_y", "Y-direction gradient"),
                ("attention", "Attention flow"),
                ("momentum", "Movement momentum")
            ]
            
            for i in range(min(count, len(flow_dims))):
                name, desc = flow_dims[i]
                dimensions.append(FieldDimension(
                    f"flow_{name}", family, start_index + i, -1.0, 1.0, 0.0, desc
                ))
                
        elif family == FieldDynamicsFamily.TOPOLOGY:
            topology_dims = [
                ("stability", "Pattern stability"),
                ("boundary", "Region boundaries"),
                ("persistence", "Memory persistence")
            ]
            
            for i in range(min(count, len(topology_dims))):
                name, desc = topology_dims[i]
                dimensions.append(FieldDimension(
                    f"topo_{name}", family, start_index + i, 0.0, 1.0, 0.0, desc
                ))
                
        elif family == FieldDynamicsFamily.ENERGY:
            energy_dims = [
                ("motor", "Motor energy"),
                ("cognitive", "Processing resources"),
                ("activation", "Overall activation")
            ]
            
            for i in range(min(count, len(energy_dims))):
                name, desc = energy_dims[i]
                dimensions.append(FieldDimension(
                    f"energy_{name}", family, start_index + i, 0.0, 1.0, 0.5, desc
                ))
                
        elif family == FieldDynamicsFamily.COUPLING:
            coupling_dims = [
                ("sensorimotor", "Sensor-motor binding"),
                ("temporal", "Time correlations"),
                ("spatial", "Spatial relationships")
            ]
            
            for i in range(min(count, len(coupling_dims))):
                name, desc = coupling_dims[i]
                dimensions.append(FieldDimension(
                    f"couple_{name}", family, start_index + i, -1.0, 1.0, 0.0, desc
                ))
                
        elif family == FieldDynamicsFamily.EMERGENCE:
            emergence_dims = [
                ("novelty", "Novel patterns"),
                ("creativity", "Creative combinations")
            ]
            
            for i in range(min(count, len(emergence_dims))):
                name, desc = emergence_dims[i]
                dimensions.append(FieldDimension(
                    f"emerge_{name}", family, start_index + i, 0.0, 1.0, 0.1, desc
                ))
        
        # Fill remaining slots with generic dimensions if needed
        while len(dimensions) < count:
            i = len(dimensions)
            dimensions.append(FieldDimension(
                f"{family.value}_{i}", family, start_index + i, -1.0, 1.0, 0.0, 
                f"Additional {family.value} dimension {i}"
            ))
            
        return dimensions
    
    def select_tensor_configuration(self, conceptual_dims: int, 
                                  spatial_resolution: int) -> List[int]:
        """
        Select appropriate tensor shape based on conceptual dimension count.
        
        This uses preset configurations for manageable memory usage while
        still adapting to robot complexity.
        """
        # Preset tensor configurations
        # Format: [x, y, z, scale, time, ...other compressed dims...]
        
        if conceptual_dims < 20:
            # Small brain for simple robots
            return [
                spatial_resolution,      # X
                spatial_resolution,      # Y  
                spatial_resolution,      # Z
                5,                      # Scale levels
                8,                      # Time steps
                2,                      # Compressed oscillatory
                2,                      # Compressed flow
                1,                      # Compressed topology
                1,                      # Compressed energy
                1,                      # Compressed coupling
                1,                      # Compressed emergence
            ]
        elif conceptual_dims < 40:
            # Medium brain (current default)
            return [
                spatial_resolution,      # X
                spatial_resolution,      # Y
                spatial_resolution,      # Z
                10,                     # Scale levels
                15,                     # Time steps
                3,                      # Compressed oscillatory
                3,                      # Compressed flow
                2,                      # Compressed topology
                2,                      # Compressed energy
                2,                      # Compressed coupling
                2,                      # Compressed emergence
            ]
        elif conceptual_dims < 60:
            # Large brain for complex robots
            return [
                spatial_resolution,      # X
                spatial_resolution,      # Y
                spatial_resolution,      # Z
                15,                     # Scale levels
                20,                     # Time steps
                4,                      # Compressed oscillatory
                4,                      # Compressed flow
                3,                      # Compressed topology
                3,                      # Compressed energy
                3,                      # Compressed coupling
                3,                      # Compressed emergence
            ]
        else:
            # XLarge brain for very complex robots
            return [
                spatial_resolution,      # X
                spatial_resolution,      # Y
                spatial_resolution,      # Z
                20,                     # Scale levels
                25,                     # Time steps
                5,                      # Compressed oscillatory
                5,                      # Compressed flow
                4,                      # Compressed topology
                4,                      # Compressed energy
                4,                      # Compressed coupling
                4,                      # Compressed emergence
            ]
    
    def create_dimension_mapping(self, dimensions: List[FieldDimension], 
                               tensor_shape: List[int]) -> Dict[str, Any]:
        """
        Create mapping between conceptual dimensions and tensor positions.
        
        This allows us to know which tensor dimensions correspond to which
        conceptual physics families.
        """
        mapping = {
            'conceptual_to_tensor': {},  # dim_name -> tensor indices
            'tensor_to_family': {},       # tensor_idx -> family
            'family_tensor_ranges': {},   # family -> (start_idx, end_idx)
        }
        
        # Group dimensions by family
        family_dims = {}
        for dim in dimensions:
            if dim.family not in family_dims:
                family_dims[dim.family] = []
            family_dims[dim.family].append(dim)
        
        # Standard tensor layout:
        # [0-2]: Spatial XYZ
        # [3]: Scale 
        # [4]: Time
        # [5+]: Compressed other families
        
        tensor_idx = 0
        
        # Map spatial dimensions
        if FieldDynamicsFamily.SPATIAL in family_dims:
            spatial_dims = family_dims[FieldDynamicsFamily.SPATIAL]
            start_idx = tensor_idx
            
            # Map spatial dimensions to first 5 tensor positions
            # First 3 get their own positions (X, Y, Z)
            for i, dim in enumerate(spatial_dims[:3]):
                mapping['conceptual_to_tensor'][dim.name] = tensor_idx
                mapping['tensor_to_family'][tensor_idx] = FieldDynamicsFamily.SPATIAL
                tensor_idx += 1
            
            # Next 2 get their own positions if available (scale, time)
            if len(spatial_dims) > 3 and tensor_idx < 5:
                for i, dim in enumerate(spatial_dims[3:5]):
                    mapping['conceptual_to_tensor'][dim.name] = tensor_idx
                    mapping['tensor_to_family'][tensor_idx] = FieldDynamicsFamily.SPATIAL
                    tensor_idx += 1
            
            # Any remaining spatial dims share the last used spatial tensor position
            if len(spatial_dims) > 5:
                last_spatial_idx = tensor_idx - 1
                for dim in spatial_dims[5:]:
                    mapping['conceptual_to_tensor'][dim.name] = last_spatial_idx
            
            mapping['family_tensor_ranges'][FieldDynamicsFamily.SPATIAL] = (start_idx, tensor_idx)
        
        # Map other families to compressed dimensions
        compressed_families = [
            FieldDynamicsFamily.OSCILLATORY,
            FieldDynamicsFamily.FLOW,
            FieldDynamicsFamily.TOPOLOGY,
            FieldDynamicsFamily.ENERGY,
            FieldDynamicsFamily.COUPLING,
            FieldDynamicsFamily.EMERGENCE
        ]
        
        for family in compressed_families:
            if family in family_dims and tensor_idx < len(tensor_shape):
                start_idx = tensor_idx
                family_conceptual_dims = family_dims[family]
                
                # All conceptual dims in this family map to same tensor position
                for dim in family_conceptual_dims:
                    mapping['conceptual_to_tensor'][dim.name] = tensor_idx
                
                mapping['tensor_to_family'][tensor_idx] = family
                mapping['family_tensor_ranges'][family] = (start_idx, tensor_idx + 1)
                tensor_idx += 1
        
        
        return mapping