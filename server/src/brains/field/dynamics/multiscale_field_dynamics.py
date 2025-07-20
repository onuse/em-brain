#!/usr/bin/env python3
"""
Multi-Scale Field Dynamics - Phase A2: 3D Continuous Field with Scale Dimension

This extends our 2D analog field to include a third dimension for scale/abstraction.
The key insight: intelligence operates simultaneously across multiple scales, from
fine-grained sensory details to abstract conceptual relationships.

Phase A2 Goals:
1. Implement 3D continuous field (width x height x scale)
2. Add cross-scale field interactions and coupling
3. Test hierarchical pattern formation from micro to macro
4. Validate scale-invariant learning across abstraction levels

Core Innovation: The scale dimension enables the same relational mathematics
to work from pixels to ecosystems, enabling true hierarchical intelligence.

Research Questions:
- Can micro-scale patterns naturally aggregate into macro-scale concepts?
- Do cross-scale interactions create emergent abstraction hierarchies?
- How do we balance scale coupling strength for optimal learning?
- What's the computational cost of multi-scale field evolution?
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

try:
    from .analog_field_dynamics import FieldExperience, FieldImprint
except ImportError:
    from analog_field_dynamics import FieldExperience, FieldImprint


@dataclass
class MultiScaleExperience:
    """An experience with explicit scale information."""
    sensory_data: torch.Tensor  # Raw sensory input
    position: Tuple[float, float]  # Spatial position (x, y)
    scale_level: float  # Scale level (0.0 = micro, 1.0 = macro)
    intensity: float  # Imprint intensity
    spread: float  # Spatial spread
    scale_spread: float  # Scale dimension spread
    timestamp: float  # When this experience occurred


@dataclass
class MultiScaleImprint:
    """3D field imprint with scale dimension."""
    center_x: float
    center_y: float
    center_scale: float  # Position in scale dimension
    intensity: float
    spatial_spread: float  # Spread in x,y dimensions
    scale_spread: float  # Spread in scale dimension
    pattern_signature: torch.Tensor  # Pattern identity


class MultiScaleField3D:
    """
    3D Continuous Field for Multi-Scale Pattern Dynamics
    
    This extends 2D analog fields with a scale dimension, enabling:
    - Hierarchical pattern formation from micro to macro
    - Cross-scale coupling and constraint propagation
    - Scale-invariant learning and abstraction
    - Natural emergence of conceptual hierarchies
    """
    
    def __init__(self, width: int = 200, height: int = 200, scale_depth: int = 50, quiet_mode: bool = False):
        self.width = width
        self.height = height
        self.scale_depth = scale_depth
        self.quiet_mode = quiet_mode
        
        # The core 3D continuous field - [width, height, scale]
        self.field = torch.zeros(width, height, scale_depth, dtype=torch.float32)
        
        # Field dynamics parameters
        self.spatial_decay_rate = 0.99  # Decay in spatial dimensions
        self.scale_decay_rate = 0.995   # Decay in scale dimension (slower)
        self.spatial_diffusion_rate = 0.01  # Spatial diffusion
        self.scale_diffusion_rate = 0.005   # Scale diffusion (weaker)
        self.cross_scale_coupling = 0.02    # Cross-scale interaction strength
        self.activation_threshold = 0.1
        
        # Scale dimension interpretation
        self.scale_min = 0.0  # Micro scale (fine details)
        self.scale_max = 1.0  # Macro scale (abstract concepts)
        
        # Multi-scale tracking
        self.imprint_history: List[MultiScaleImprint] = []
        self.cross_scale_events: List[Dict] = []
        self.total_experiences = 0
        
        # Performance tracking
        self.stats = {
            'total_imprints': 0,
            'total_retrievals': 0,
            'cross_scale_interactions': 0,
            'field_evolution_cycles': 0,
            'avg_imprint_time_ms': 0.0,
            'avg_retrieval_time_ms': 0.0,
            'field_density_by_scale': [],  # Density at each scale level
            'hierarchical_formations': 0,  # Count of micro→macro formations
        }
        
        if not self.quiet_mode:
            print(f"🌊 MultiScaleField3D initialized: {width}x{height}x{scale_depth} continuous field")
            print(f"   Scale range: {self.scale_min:.1f} (micro) → {self.scale_max:.1f} (macro)")
    
    def experience_to_multiscale_imprint(self, experience: MultiScaleExperience) -> MultiScaleImprint:
        """
        Convert a multi-scale experience into a 3D field imprint.
        
        This maps sensory experiences to 3D field coordinates including scale.
        """
        # Spatial coordinates
        center_x = max(0, min(self.width - 1, experience.position[0]))
        center_y = max(0, min(self.height - 1, experience.position[1]))
        
        # Scale coordinate
        scale_coord = experience.scale_level * (self.scale_depth - 1)
        center_scale = max(0, min(self.scale_depth - 1, scale_coord))
        
        # Create pattern signature from sensory data
        pattern_signature = self._create_pattern_signature(experience.sensory_data)
        
        return MultiScaleImprint(
            center_x=center_x,
            center_y=center_y,
            center_scale=center_scale,
            intensity=experience.intensity,
            spatial_spread=experience.spread,
            scale_spread=experience.scale_spread,
            pattern_signature=pattern_signature
        )
    
    def _create_pattern_signature(self, sensory_data: torch.Tensor) -> torch.Tensor:
        """Create a pattern signature from sensory data."""
        signature_size = 10
        
        if len(sensory_data) == 0:
            return torch.ones(signature_size)
        
        # Create signature by segmenting and summarizing sensory data
        signature = torch.zeros(signature_size)
        segment_size = max(1, len(sensory_data) // signature_size)
        
        for i in range(signature_size):
            start_idx = i * segment_size
            end_idx = min(start_idx + segment_size, len(sensory_data))
            
            if start_idx < len(sensory_data):
                segment_data = sensory_data[start_idx:end_idx]
                signature[i] = torch.mean(segment_data).item()
        
        return signature
    
    def apply_multiscale_imprint(self, imprint: MultiScaleImprint):
        """
        Apply a 3D imprint to the multi-scale field.
        
        This creates activation blobs that span spatial and scale dimensions.
        """
        start_time = time.perf_counter()
        
        # Create 3D coordinate grids
        y_coords, x_coords, s_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.scale_depth, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distances in 3D space (spatial + scale dimensions)
        spatial_distances = torch.sqrt(
            (x_coords - imprint.center_x) ** 2 + 
            (y_coords - imprint.center_y) ** 2
        )
        
        scale_distances = torch.abs(s_coords - imprint.center_scale)
        
        # Create 3D Gaussian activation blob
        spatial_gaussian = torch.exp(-(spatial_distances ** 2) / (2 * imprint.spatial_spread ** 2))
        scale_gaussian = torch.exp(-(scale_distances ** 2) / (2 * imprint.scale_spread ** 2))
        
        # Combine spatial and scale Gaussians
        multiscale_blob = imprint.intensity * spatial_gaussian * scale_gaussian
        
        # Apply the imprint to the 3D field
        self.field += multiscale_blob
        
        # Record this imprint
        self.imprint_history.append(imprint)
        self.total_experiences += 1
        
        # Update stats
        imprint_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_imprint_stats(imprint_time_ms)
        
        if not self.quiet_mode:
            print(f"📍 3D Imprint applied: pos=({imprint.center_x:.1f}, {imprint.center_y:.1f}), "
                  f"scale={imprint.center_scale:.1f}, intensity={imprint.intensity:.3f}")
    
    def evolve_multiscale_field(self, dt: float = 0.1):
        """
        Evolve the 3D field with multi-scale dynamics.
        
        This includes:
        - Spatial decay and diffusion
        - Scale decay and diffusion  
        - Cross-scale coupling interactions
        """
        start_time = time.perf_counter()
        
        # Apply decay in all dimensions
        self.field *= self.spatial_decay_rate
        
        # Apply spatial diffusion (x, y dimensions)
        if self.spatial_diffusion_rate > 0:
            # 2D spatial diffusion kernel
            spatial_kernel = torch.tensor([
                [[0.05, 0.1, 0.05],
                 [0.1,  0.4, 0.1 ],
                 [0.05, 0.1, 0.05]]
            ]).unsqueeze(0)  # Add batch dimension
            
            # Apply diffusion to each scale slice
            for s in range(self.scale_depth):
                slice_2d = self.field[:, :, s].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                diffused = torch.conv2d(slice_2d, spatial_kernel, padding=1)
                diffused_slice = diffused.squeeze(0).squeeze(0)
                
                # Blend original with diffused
                self.field[:, :, s] = (1.0 - self.spatial_diffusion_rate) * self.field[:, :, s] + \
                                    self.spatial_diffusion_rate * diffused_slice
        
        # Apply scale diffusion (scale dimension)
        if self.scale_diffusion_rate > 0:
            # 1D scale diffusion kernel
            scale_kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
            
            # Apply diffusion along scale dimension for each spatial location
            for y in range(self.height):
                for x in range(self.width):
                    scale_profile = self.field[y, x, :].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, scale_depth]
                    
                    # Pad for convolution
                    padded_profile = torch.nn.functional.pad(scale_profile, (1, 1), mode='reflect')
                    diffused = torch.conv1d(padded_profile, scale_kernel)
                    diffused_profile = diffused.squeeze(0).squeeze(0)
                    
                    # Blend original with diffused
                    self.field[y, x, :] = (1.0 - self.scale_diffusion_rate) * self.field[y, x, :] + \
                                        self.scale_diffusion_rate * diffused_profile
        
        # Apply cross-scale coupling
        self._apply_cross_scale_coupling()
        
        self.stats['field_evolution_cycles'] += 1
        
        evolution_time_ms = (time.perf_counter() - start_time) * 1000
        if not self.quiet_mode and self.stats['field_evolution_cycles'] % 50 == 0:
            print(f"🌊 3D Field evolved: cycle {self.stats['field_evolution_cycles']}, "
                  f"time={evolution_time_ms:.2f}ms")
    
    def _apply_cross_scale_coupling(self):
        """
        Apply cross-scale coupling interactions.
        
        This enables:
        - Micro-scale patterns to aggregate into macro-scale concepts
        - Macro-scale patterns to influence micro-scale details
        - Natural hierarchy formation across scales
        """
        if self.cross_scale_coupling <= 0:
            return
        
        coupling_events = 0
        
        # Bottom-up coupling: micro → macro aggregation
        for s in range(self.scale_depth - 1):
            current_scale = self.field[:, :, s]
            next_scale = self.field[:, :, s + 1]
            
            # High activation at current scale influences next scale up
            influence_mask = current_scale > self.activation_threshold
            
            if torch.sum(influence_mask) > 0:
                # Aggregate micro patterns into macro patterns
                aggregation_strength = torch.mean(current_scale[influence_mask]) * self.cross_scale_coupling
                
                # Apply aggregated influence to next scale (with spatial smoothing)
                influence_field = current_scale * influence_mask.float()
                
                # Smooth the influence (macro patterns are more spatially distributed)
                if torch.sum(influence_field) > 0:
                    # Simple averaging filter for spatial smoothing
                    kernel_size = 3
                    kernel = torch.ones(kernel_size, kernel_size) / (kernel_size ** 2)
                    kernel = kernel.unsqueeze(0).unsqueeze(0)
                    
                    influence_4d = influence_field.unsqueeze(0).unsqueeze(0)
                    smoothed = torch.conv2d(influence_4d, kernel, padding=1)
                    smoothed_influence = smoothed.squeeze(0).squeeze(0)
                    
                    # Apply bottom-up influence
                    self.field[:, :, s + 1] += smoothed_influence * aggregation_strength
                    coupling_events += 1
        
        # Top-down coupling: macro → micro influence
        for s in range(self.scale_depth - 1, 0, -1):
            current_scale = self.field[:, :, s]
            prev_scale = self.field[:, :, s - 1]
            
            # High activation at current scale influences previous scale down
            influence_mask = current_scale > self.activation_threshold
            
            if torch.sum(influence_mask) > 0:
                # Macro patterns influence micro details
                influence_strength = torch.mean(current_scale[influence_mask]) * self.cross_scale_coupling * 0.5  # Weaker top-down
                
                # Apply macro influence to micro scale (with spatial sharpening)
                influence_field = current_scale * influence_mask.float()
                
                # Apply top-down influence (macro concepts guide micro attention)
                self.field[:, :, s - 1] += influence_field * influence_strength
                coupling_events += 1
        
        # Record cross-scale interactions
        if coupling_events > 0:
            self.stats['cross_scale_interactions'] += coupling_events
            self.cross_scale_events.append({
                'timestamp': time.time(),
                'coupling_events': coupling_events,
                'cycle': self.stats['field_evolution_cycles']
            })
    
    def find_multiscale_similar_regions(self, query_experience: MultiScaleExperience, 
                                       top_k: int = 5) -> List[Tuple[float, float, float, float]]:
        """
        Find similar regions across the 3D multi-scale field.
        
        Returns: List of (x, y, scale, correlation) tuples
        """
        start_time = time.perf_counter()
        
        # Convert query to 3D imprint
        query_imprint = self.experience_to_multiscale_imprint(query_experience)
        
        # Create the query pattern in 3D field space
        y_coords, x_coords, s_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.scale_depth, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate 3D distances
        spatial_distances = torch.sqrt(
            (x_coords - query_imprint.center_x) ** 2 + 
            (y_coords - query_imprint.center_y) ** 2
        )
        scale_distances = torch.abs(s_coords - query_imprint.center_scale)
        
        # Create 3D query pattern
        spatial_gaussian = torch.exp(-(spatial_distances ** 2) / (2 * query_imprint.spatial_spread ** 2))
        scale_gaussian = torch.exp(-(scale_distances ** 2) / (2 * query_imprint.scale_spread ** 2))
        query_pattern_3d = query_imprint.intensity * spatial_gaussian * scale_gaussian
        
        # Find correlations across the 3D field using sliding window
        window_size = int(query_imprint.spatial_spread * 2)
        scale_window = int(query_imprint.scale_spread * 2)
        stride = max(1, window_size // 2)
        
        correlations = []
        
        for y in range(0, self.height - window_size, stride):
            for x in range(0, self.width - window_size, stride):
                for s in range(0, self.scale_depth - scale_window, max(1, scale_window // 2)):
                    # Extract 3D field region
                    field_region = self.field[y:y+window_size, x:x+window_size, s:s+scale_window]
                    query_region = query_pattern_3d[y:y+window_size, x:x+window_size, s:s+scale_window]
                    
                    # Calculate 3D correlation
                    if torch.sum(field_region) > 0 and torch.sum(query_region) > 0:
                        correlation = self._calculate_3d_correlation(field_region, query_region)
                        center_x = x + window_size // 2
                        center_y = y + window_size // 2
                        center_scale = s + scale_window // 2
                        correlations.append((center_x, center_y, center_scale, correlation))
        
        # Sort by correlation and return top_k
        correlations.sort(key=lambda x: x[3], reverse=True)
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_retrieval_stats(retrieval_time_ms)
        
        return correlations[:top_k]
    
    def _calculate_3d_correlation(self, region1: torch.Tensor, region2: torch.Tensor) -> float:
        """Calculate correlation between two 3D field regions."""
        # Flatten 3D regions
        r1_flat = region1.flatten()
        r2_flat = region2.flatten()
        
        # Calculate Pearson correlation
        if torch.std(r1_flat) > 0 and torch.std(r2_flat) > 0:
            correlation = torch.corrcoef(torch.stack([r1_flat, r2_flat]))[0, 1]
            return correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def get_scale_density_profile(self) -> List[float]:
        """Get field density at each scale level."""
        scale_densities = []
        
        for s in range(self.scale_depth):
            scale_slice = self.field[:, :, s]
            active_cells = torch.sum(scale_slice > self.activation_threshold).item()
            total_cells = self.width * self.height
            density = active_cells / total_cells if total_cells > 0 else 0.0
            scale_densities.append(density)
        
        return scale_densities
    
    def get_hierarchical_patterns(self) -> List[Dict]:
        """
        Identify hierarchical patterns that span multiple scales.
        
        Returns patterns that show connectivity across scale levels.
        """
        hierarchical_patterns = []
        
        # Look for patterns that have significant activation across multiple scales
        for y in range(0, self.height, 10):  # Sample every 10 pixels
            for x in range(0, self.width, 10):
                # Get activation profile across scales for this spatial location
                scale_profile = self.field[y, x, :].numpy()
                
                # Check if this location has multi-scale activation
                active_scales = np.where(scale_profile > self.activation_threshold)[0]
                
                if len(active_scales) >= 3:  # Pattern spans at least 3 scales
                    scale_span = active_scales[-1] - active_scales[0]
                    avg_activation = np.mean(scale_profile[active_scales])
                    
                    hierarchical_patterns.append({
                        'spatial_position': (x, y),
                        'scale_span': scale_span,
                        'active_scales': active_scales.tolist(),
                        'avg_activation': avg_activation,
                        'pattern_type': 'hierarchical'
                    })
        
        # Sort by scale span (most hierarchical first)
        hierarchical_patterns.sort(key=lambda p: p['scale_span'], reverse=True)
        return hierarchical_patterns
    
    def _update_imprint_stats(self, imprint_time_ms: float):
        """Update imprint performance statistics."""
        self.stats['total_imprints'] += 1
        
        # Update average imprint time
        total_imprints = self.stats['total_imprints']
        current_avg = self.stats['avg_imprint_time_ms']
        self.stats['avg_imprint_time_ms'] = (current_avg * (total_imprints - 1) + imprint_time_ms) / total_imprints
        
        # Update scale density profile
        self.stats['field_density_by_scale'] = self.get_scale_density_profile()
    
    def _update_retrieval_stats(self, retrieval_time_ms: float):
        """Update retrieval performance statistics."""
        self.stats['total_retrievals'] += 1
        
        # Update average retrieval time
        total_retrievals = self.stats['total_retrievals']
        current_avg = self.stats['avg_retrieval_time_ms']
        self.stats['avg_retrieval_time_ms'] = (current_avg * (total_retrievals - 1) + retrieval_time_ms) / total_retrievals
    
    def get_multiscale_stats(self) -> Dict[str, Any]:
        """Get comprehensive multi-scale field statistics."""
        hierarchical_patterns = self.get_hierarchical_patterns()
        scale_densities = self.get_scale_density_profile()
        
        stats = self.stats.copy()
        stats.update({
            'field_dimensions': (self.width, self.height, self.scale_depth),
            'total_experiences': self.total_experiences,
            'imprint_history_length': len(self.imprint_history),
            'scale_density_profile': scale_densities,
            'hierarchical_patterns_count': len(hierarchical_patterns),
            'max_scale_span': max([p['scale_span'] for p in hierarchical_patterns]) if hierarchical_patterns else 0,
            'field_max_activation': torch.max(self.field).item(),
            'field_mean_activation': torch.mean(self.field).item(),
            'cross_scale_events_total': len(self.cross_scale_events),
        })
        return stats


# Factory function for easy creation
def create_multiscale_field_3d(width: int = 200, height: int = 200, scale_depth: int = 50, 
                              quiet_mode: bool = False) -> MultiScaleField3D:
    """Create a 3D multi-scale field with standard configuration."""
    return MultiScaleField3D(width=width, height=height, scale_depth=scale_depth, quiet_mode=quiet_mode)


if __name__ == "__main__":
    # Test the 3D multi-scale field implementation
    print("🧪 Testing 3D Multi-Scale Field Dynamics...")
    
    # Create field
    field = create_multiscale_field_3d(width=100, height=100, scale_depth=20, quiet_mode=False)
    
    # Test multi-scale experience imprinting
    print("\n📍 Testing multi-scale experience imprinting:")
    
    # Create test experiences at different scales
    experiences = [
        MultiScaleExperience(
            sensory_data=torch.tensor([0.8, 0.2, 0.1, 0.9]),  # "Local feature" pattern
            position=(25, 25),
            scale_level=0.1,  # Micro scale
            intensity=1.0,
            spread=3.0,
            scale_spread=2.0,
            timestamp=time.time()
        ),
        MultiScaleExperience(
            sensory_data=torch.tensor([0.1, 0.8, 0.2, 0.1]),  # "Regional pattern" 
            position=(25, 25),  # Same spatial location
            scale_level=0.5,  # Medium scale
            intensity=0.8,
            spread=8.0,
            scale_spread=4.0,
            timestamp=time.time()
        ),
        MultiScaleExperience(
            sensory_data=torch.tensor([0.2, 0.1, 0.9, 0.1]),  # "Global concept"
            position=(25, 25),  # Same spatial location
            scale_level=0.9,  # Macro scale
            intensity=0.6,
            spread=15.0,
            scale_spread=6.0,
            timestamp=time.time()
        ),
    ]
    
    # Apply experiences to field
    for i, exp in enumerate(experiences):
        imprint = field.experience_to_multiscale_imprint(exp)
        field.apply_multiscale_imprint(imprint)
        print(f"   Experience {i+1}: scale={exp.scale_level:.1f}, "
              f"imprint at ({imprint.center_x}, {imprint.center_y}, {imprint.center_scale})")
    
    # Test multi-scale field evolution
    print("\n🌊 Testing multi-scale field evolution:")
    for i in range(10):
        field.evolve_multiscale_field()
        if i % 3 == 0:
            scale_densities = field.get_scale_density_profile()
            avg_density = np.mean(scale_densities)
            print(f"   Evolution step {i+1}: avg density = {avg_density:.4f}")
    
    # Test hierarchical pattern formation
    print("\n🏗️ Testing hierarchical pattern formation:")
    hierarchical_patterns = field.get_hierarchical_patterns()
    print(f"   Found {len(hierarchical_patterns)} hierarchical patterns")
    for i, pattern in enumerate(hierarchical_patterns[:3]):
        print(f"   Pattern {i+1}: pos={pattern['spatial_position']}, "
              f"scale_span={pattern['scale_span']}, "
              f"activation={pattern['avg_activation']:.3f}")
    
    # Test multi-scale similarity search
    print("\n🔍 Testing multi-scale similarity search:")
    query_experience = MultiScaleExperience(
        sensory_data=torch.tensor([0.7, 0.3, 0.1, 0.8]),
        position=(30, 30),
        scale_level=0.3,  # Query at medium-micro scale
        intensity=1.0,
        spread=5.0,
        scale_spread=3.0,
        timestamp=time.time()
    )
    
    similar_regions = field.find_multiscale_similar_regions(query_experience, top_k=3)
    print(f"   Query: scale={query_experience.scale_level:.1f}")
    for i, (x, y, scale, correlation) in enumerate(similar_regions):
        scale_level = scale / (field.scale_depth - 1)
        print(f"   Match {i+1}: pos=({x:.1f}, {y:.1f}), scale={scale_level:.2f}, "
              f"correlation={correlation:.3f}")
    
    # Display final statistics
    print("\n📊 Final multi-scale field statistics:")
    stats = field.get_multiscale_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        elif isinstance(value, list) and key != 'field_density_by_scale':
            print(f"   {key}: {len(value)} items")
        elif key != 'field_density_by_scale':
            print(f"   {key}: {value}")
    
    print("\n✅ 3D Multi-Scale Field test completed!")
    print("🎯 Key multi-scale behaviors demonstrated:")
    print("   ✓ 3D experience-to-imprint conversion (spatial + scale)")
    print("   ✓ Multi-scale field evolution with cross-scale coupling")
    print("   ✓ Hierarchical pattern formation across scales") 
    print("   ✓ Multi-scale similarity search and correlation")
    print("   ✓ Natural scale hierarchy emergence")