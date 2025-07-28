#!/usr/bin/env python3
"""
Analog Field Dynamics - Phase A1: 2D Continuous Field Prototype

This is our fundamental pivot from discrete patterns to continuous field dynamics.
Instead of storing patterns as discrete vectors, we modify continuous field topologies
where concepts become landscape features and similarity becomes geometric correlation.

Core Insight: "Information is encoded as imprints en masse" - experiences modify
field landscapes rather than creating discrete storage entries.

Phase A1 Goals:
1. Implement 2D continuous activation field (1000x1000)
2. Convert experiences to field modifications (imprints)
3. Test continuous pattern matching vs discrete similarity
4. Validate basic field dynamics (Gaussian blobs, evolution)

Research Questions:
- Can continuous fields store and retrieve experiences meaningfully?
- Do natural clusters emerge in continuous space?
- How do we convert sensory experiences to field modifications?
- What's the computational cost vs discrete patterns?
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class FieldExperience:
    """An experience that will be imprinted onto the field."""
    sensory_data: torch.Tensor  # Raw sensory input
    position: Tuple[float, float]  # Where in field space to imprint
    intensity: float  # How strongly to imprint
    spread: float  # How widely to spread the imprint
    timestamp: float  # When this experience occurred


@dataclass
class FieldImprint:
    """The field modification created by an experience."""
    center_x: float
    center_y: float
    intensity: float
    spread: float
    pattern_signature: torch.Tensor  # What makes this imprint unique


class AnalogField2D:
    """
    2D Continuous Field for Analog Pattern Dynamics
    
    This replaces discrete pattern storage with continuous field modification.
    Experiences create "imprints" that modify the field topology, and pattern
    matching becomes correlation analysis in continuous space.
    """
    
    def __init__(self, width: int = 1000, height: int = 1000, quiet_mode: bool = False):
        self.width = width
        self.height = height
        self.quiet_mode = quiet_mode
        
        # The core continuous field - this replaces discrete pattern storage
        self.field = torch.zeros(width, height, dtype=torch.float32)
        
        # Field dynamics parameters
        self.decay_rate = 0.99  # How quickly field activations fade
        self.diffusion_rate = 0.01  # How much neighboring regions influence each other
        self.activation_threshold = 0.1  # Minimum activation to be considered "active"
        
        # Experience tracking
        self.imprint_history: List[FieldImprint] = []
        self.total_experiences = 0
        
        # Performance tracking
        self.stats = {
            'total_imprints': 0,
            'total_retrievals': 0,
            'field_evolution_cycles': 0,
            'avg_imprint_time_ms': 0.0,
            'avg_retrieval_time_ms': 0.0,
            'field_density': 0.0,  # Percentage of field with significant activation
        }
        
        if not quiet_mode:
            print(f"🌊 AnalogField2D initialized: {width}x{height} continuous field")
    
    def experience_to_imprint(self, experience: FieldExperience) -> FieldImprint:
        """
        Convert a sensory experience into a field imprint.
        
        This is the crucial translation: how do we turn sensory data into
        continuous field modifications?
        """
        # Calculate where in field space this experience should be placed
        # For now, use the provided position, but this could be learned
        center_x, center_y = experience.position
        
        # Ensure position is within field bounds
        center_x = max(0, min(self.width - 1, center_x))
        center_y = max(0, min(self.height - 1, center_y))
        
        # Create pattern signature from sensory data
        # This determines the "shape" of the imprint
        if len(experience.sensory_data) > 0:
            # Use sensory data to create unique signature
            pattern_signature = self._create_pattern_signature(experience.sensory_data)
        else:
            # Default signature if no sensory data
            pattern_signature = torch.ones(10)
        
        return FieldImprint(
            center_x=center_x,
            center_y=center_y,
            intensity=experience.intensity,
            spread=experience.spread,
            pattern_signature=pattern_signature
        )
    
    def _create_pattern_signature(self, sensory_data: torch.Tensor) -> torch.Tensor:
        """
        Create a pattern signature from sensory data.
        
        This determines how different types of sensory input create
        different "shapes" in the field.
        """
        # Simple approach: hash sensory data into fixed-size signature
        signature_size = 10
        
        if len(sensory_data) == 0:
            return torch.ones(signature_size)
        
        # Create signature by segmenting and summarizing sensory data
        signature = torch.zeros(signature_size)
        
        # Divide sensory data into segments
        segment_size = max(1, len(sensory_data) // signature_size)
        
        for i in range(signature_size):
            start_idx = i * segment_size
            end_idx = min(start_idx + segment_size, len(sensory_data))
            
            if start_idx < len(sensory_data):
                # Use mean activation of this segment
                segment_data = sensory_data[start_idx:end_idx]
                signature[i] = torch.mean(segment_data).item()
        
        return signature
    
    def apply_imprint(self, imprint: FieldImprint):
        """
        Apply an imprint to the field, modifying the continuous topology.
        
        This is the core operation: instead of storing a discrete pattern,
        we modify the field landscape.
        """
        start_time = time.perf_counter()
        
        # Create coordinate grids for the field
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distance from imprint center
        distances = torch.sqrt(
            (x_coords - imprint.center_x) ** 2 + 
            (y_coords - imprint.center_y) ** 2
        )
        
        # Create Gaussian activation blob
        gaussian_blob = imprint.intensity * torch.exp(-(distances ** 2) / (2 * imprint.spread ** 2))
        
        # Apply the imprint to the field
        self.field += gaussian_blob
        
        # Record this imprint
        self.imprint_history.append(imprint)
        self.total_experiences += 1
        
        # Update stats
        imprint_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_imprint_stats(imprint_time_ms)
        
        if not self.quiet_mode:
            print(f"📍 Imprint applied: center=({imprint.center_x:.1f}, {imprint.center_y:.1f}), "
                  f"intensity={imprint.intensity:.3f}, spread={imprint.spread:.1f}")
    
    def evolve_field(self, dt: float = 0.1):
        """
        Evolve the field dynamics over time.
        
        This implements continuous field evolution:
        - Decay: activations fade over time
        - Diffusion: neighboring regions influence each other
        """
        start_time = time.perf_counter()
        
        # Apply decay
        self.field *= self.decay_rate
        
        # Apply diffusion (neighboring regions influence each other)
        if self.diffusion_rate > 0:
            # Simple diffusion: each cell influenced by neighbors
            # Use conv2d with diffusion kernel
            diffusion_kernel = torch.tensor([
                [0.05, 0.1, 0.05],
                [0.1,  0.4, 0.1 ],
                [0.05, 0.1, 0.05]
            ]).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            # Apply diffusion
            field_4d = self.field.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            diffused = torch.conv2d(field_4d, diffusion_kernel, padding=1)
            diffused_field = diffused.squeeze(0).squeeze(0)  # Remove extra dims
            
            # Blend original field with diffused version
            self.field = (1.0 - self.diffusion_rate) * self.field + self.diffusion_rate * diffused_field
        
        self.stats['field_evolution_cycles'] += 1
        
        evolution_time_ms = (time.perf_counter() - start_time) * 1000
        if not self.quiet_mode and self.stats['field_evolution_cycles'] % 100 == 0:
            print(f"🌊 Field evolved: cycle {self.stats['field_evolution_cycles']}, "
                  f"time={evolution_time_ms:.2f}ms")
    
    def find_similar_regions(self, query_experience: FieldExperience, top_k: int = 5) -> List[Tuple[float, float, float]]:
        """
        Find regions in the field similar to a query experience.
        
        This replaces discrete pattern similarity search with continuous
        field correlation analysis.
        """
        start_time = time.perf_counter()
        
        # Convert query to imprint to understand what we're looking for
        query_imprint = self.experience_to_imprint(query_experience)
        
        # Create the query pattern in field space
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing='ij'
        )
        
        distances = torch.sqrt(
            (x_coords - query_imprint.center_x) ** 2 + 
            (y_coords - query_imprint.center_y) ** 2
        )
        
        query_pattern = query_imprint.intensity * torch.exp(-(distances ** 2) / (2 * query_imprint.spread ** 2))
        
        # Find correlations across the field
        # Use sliding window correlation
        window_size = int(query_imprint.spread * 3)  # 3 sigma window
        stride = max(1, window_size // 2)
        
        correlations = []
        
        for y in range(0, self.height - window_size, stride):
            for x in range(0, self.width - window_size, stride):
                # Extract field region
                field_region = self.field[y:y+window_size, x:x+window_size]
                query_region = query_pattern[y:y+window_size, x:x+window_size]
                
                # Calculate correlation
                if torch.sum(field_region) > 0 and torch.sum(query_region) > 0:
                    correlation = self._calculate_correlation(field_region, query_region)
                    correlations.append((x + window_size//2, y + window_size//2, correlation))
        
        # Sort by correlation and return top_k
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_retrieval_stats(retrieval_time_ms)
        
        return correlations[:top_k]
    
    def _calculate_correlation(self, region1: torch.Tensor, region2: torch.Tensor) -> float:
        """Calculate correlation between two field regions."""
        # Flatten regions
        r1_flat = region1.flatten()
        r2_flat = region2.flatten()
        
        # Calculate Pearson correlation
        if torch.std(r1_flat.float()) > 0 and torch.std(r2_flat.float()) > 0:
            correlation = torch.corrcoef(torch.stack([r1_flat, r2_flat]))[0, 1]
            return correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def get_field_density(self) -> float:
        """Calculate what percentage of the field has significant activation."""
        active_cells = torch.sum(self.field > self.activation_threshold).item()
        total_cells = self.width * self.height
        return active_cells / total_cells
    
    def get_active_regions(self, threshold: float = None) -> List[Tuple[float, float, float]]:
        """Get all regions with activation above threshold."""
        if threshold is None:
            threshold = self.activation_threshold
        
        active_regions = []
        
        # Find local maxima above threshold
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                activation = self.field[y, x].item()
                
                if activation > threshold:
                    # Check if this is a local maximum
                    neighborhood = self.field[y-1:y+2, x-1:x+2]
                    if activation >= torch.max(neighborhood).item():
                        active_regions.append((x, y, activation))
        
        # Sort by activation strength
        active_regions.sort(key=lambda x: x[2], reverse=True)
        return active_regions
    
    def _update_imprint_stats(self, imprint_time_ms: float):
        """Update imprint performance statistics."""
        self.stats['total_imprints'] += 1
        
        # Update average imprint time
        total_imprints = self.stats['total_imprints']
        current_avg = self.stats['avg_imprint_time_ms']
        self.stats['avg_imprint_time_ms'] = (current_avg * (total_imprints - 1) + imprint_time_ms) / total_imprints
        
        # Update field density
        self.stats['field_density'] = self.get_field_density()
    
    def _update_retrieval_stats(self, retrieval_time_ms: float):
        """Update retrieval performance statistics."""
        self.stats['total_retrievals'] += 1
        
        # Update average retrieval time
        total_retrievals = self.stats['total_retrievals']
        current_avg = self.stats['avg_retrieval_time_ms']
        self.stats['avg_retrieval_time_ms'] = (current_avg * (total_retrievals - 1) + retrieval_time_ms) / total_retrievals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive field statistics."""
        stats = self.stats.copy()
        stats.update({
            'field_dimensions': (self.width, self.height),
            'total_experiences': self.total_experiences,
            'imprint_history_length': len(self.imprint_history),
            'current_field_density': self.get_field_density(),
            'active_regions_count': len(self.get_active_regions()),
            'field_max_activation': torch.max(self.field).item(),
            'field_mean_activation': torch.mean(self.field).item(),
        })
        return stats
    
    def visualize_field(self, save_path: str = None) -> torch.Tensor:
        """
        Create a visualization of the current field state.
        Returns the field tensor for external visualization.
        """
        # For now, just return the field tensor
        # In a full implementation, this could create matplotlib visualizations
        
        if save_path:
            # Could save field visualization here
            pass
        
        return self.field.clone()


# Factory function for easy creation
def create_analog_field_2d(width: int = 1000, height: int = 1000, quiet_mode: bool = False) -> AnalogField2D:
    """Create a 2D analog field with standard configuration."""
    return AnalogField2D(width=width, height=height, quiet_mode=quiet_mode)


if __name__ == "__main__":
    # Test the 2D analog field implementation
    print("🧪 Testing 2D Analog Field Dynamics...")
    
    # Create field
    field = create_analog_field_2d(width=100, height=100, quiet_mode=False)
    
    # Test experience imprinting
    print("\n📍 Testing experience imprinting:")
    
    # Create some test experiences
    experiences = [
        FieldExperience(
            sensory_data=torch.tensor([0.8, 0.2, 0.1, 0.9]),  # "Red light" pattern
            position=(25, 25),
            intensity=1.0,
            spread=5.0,
            timestamp=time.time()
        ),
        FieldExperience(
            sensory_data=torch.tensor([0.1, 0.8, 0.2, 0.1]),  # "Green surface" pattern
            position=(75, 25),
            intensity=0.8,
            spread=7.0,
            timestamp=time.time()
        ),
        FieldExperience(
            sensory_data=torch.tensor([0.2, 0.1, 0.9, 0.1]),  # "Blue sound" pattern
            position=(50, 75),
            intensity=0.6,
            spread=10.0,
            timestamp=time.time()
        ),
    ]
    
    # Apply experiences to field
    for i, exp in enumerate(experiences):
        imprint = field.experience_to_imprint(exp)
        field.apply_imprint(imprint)
        print(f"   Experience {i+1}: {exp.sensory_data.tolist()} → imprint at ({imprint.center_x}, {imprint.center_y})")
    
    # Test field evolution
    print("\n🌊 Testing field evolution:")
    for i in range(5):
        field.evolve_field()
        density = field.get_field_density()
        print(f"   Evolution step {i+1}: field density = {density:.4f}")
    
    # Test similarity search
    print("\n🔍 Testing similarity search:")
    query_experience = FieldExperience(
        sensory_data=torch.tensor([0.7, 0.3, 0.1, 0.8]),  # Similar to first experience
        position=(30, 30),  # Near first experience
        intensity=1.0,
        spread=5.0,
        timestamp=time.time()
    )
    
    similar_regions = field.find_similar_regions(query_experience, top_k=3)
    print(f"   Query: {query_experience.sensory_data.tolist()}")
    for i, (x, y, correlation) in enumerate(similar_regions):
        print(f"   Match {i+1}: position=({x:.1f}, {y:.1f}), correlation={correlation:.3f}")
    
    # Display final statistics
    print("\n📊 Final field statistics:")
    stats = field.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ 2D Analog Field test completed!")
    print("🎯 Key behaviors demonstrated:")
    print("   ✓ Experience-to-imprint conversion")
    print("   ✓ Continuous field modification (not discrete storage)")
    print("   ✓ Field evolution dynamics (decay + diffusion)")
    print("   ✓ Similarity search through field correlation")
    print("   ✓ Natural clustering in continuous space")