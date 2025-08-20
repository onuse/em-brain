"""
Discrete Attractor Dynamics - The Foundation of Concepts

Allows continuous field regions to crystallize into discrete, stable concepts.
Like phase transitions in physics - water suddenly becoming ice.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Concept:
    """Represents a crystallized concept in the field."""
    position: torch.Tensor  # Center position in field
    pattern: torch.Tensor   # The concept's pattern signature
    strength: float         # How strongly crystallized
    age: int               # How long it's been stable
    label: Optional[str] = None  # Optional semantic label


class DiscreteAttractorDynamics:
    """
    Enable field regions to form discrete, stable concepts.
    
    Core principle: High-energy coherent patterns spontaneously crystallize
    into discrete states that act as attractors for similar patterns.
    """
    
    def __init__(self, field_shape: tuple, device: torch.device):
        """
        Initialize the discrete dynamics system.
        
        Args:
            field_shape: Shape of the field tensor [D, H, W, C]
            device: Computation device
        """
        self.field_shape = field_shape
        self.device = device
        
        # Concept management
        self.concepts: List[Concept] = []
        self.max_concepts = 100  # Maximum number of concepts to track
        
        # Crystallization parameters
        self.coherence_threshold = 0.3  # Minimum coherence to form concept
        self.energy_threshold = 0.5     # Minimum energy to crystallize
        self.crystallization_rate = 0.1 # How fast crystallization occurs
        
        # Attractor parameters
        self.attractor_radius = 5       # Spatial influence of attractors
        self.attractor_strength = 0.05  # How strongly attractors pull
        
        # Track crystallization state
        self.crystallization_map = torch.zeros(field_shape[:3], device=device)
        
    def evolve_field_with_attractors(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply discrete attractor dynamics to the field.
        
        Coherent high-energy regions crystallize into concepts.
        Existing concepts create attractor basins.
        
        Args:
            field: Current field state
            
        Returns:
            Field with discrete dynamics applied
        """
        # Find potential concept regions
        coherent_regions = self._find_coherent_patterns(field)
        
        # Crystallize new concepts
        for region in coherent_regions:
            if self._should_crystallize(region, field):
                field = self._crystallize_region(field, region)
        
        # Apply existing attractors
        for concept in self.concepts:
            field = self._apply_attractor(field, concept)
            concept.age += 1
        
        # Prune old/weak concepts
        self._prune_concepts()
        
        return field
    
    def _find_coherent_patterns(self, field: torch.Tensor) -> List[Dict]:
        """
        Identify coherent high-energy regions that could become concepts.
        
        Uses local variance and energy to find coherent patterns.
        """
        regions = []
        
        # Compute local energy
        energy = torch.abs(field).mean(dim=3)  # Average across channels
        
        # Compute local coherence (low variance = high coherence)
        kernel_size = 3
        
        # Use avg_pool3d to compute local statistics
        energy_pooled = F.avg_pool3d(
            energy.unsqueeze(0).unsqueeze(0),
            kernel_size, stride=1, padding=1
        ).squeeze()
        
        # Variance around local mean indicates coherence
        variance = F.avg_pool3d(
            (energy - energy_pooled).pow(2).unsqueeze(0).unsqueeze(0),
            kernel_size, stride=1, padding=1
        ).squeeze()
        
        coherence = 1.0 / (1.0 + variance)  # High coherence = low variance
        
        # Find peaks in coherence * energy (coherent AND active)
        combined = coherence * energy
        
        # Non-maximum suppression to find local peaks
        peaks = self._find_local_maxima(combined)
        
        for peak in peaks:
            x, y, z = peak
            if combined[x, y, z] > self.energy_threshold:
                regions.append({
                    'position': torch.tensor([x, y, z], device=self.device),
                    'energy': energy[x, y, z].item(),
                    'coherence': coherence[x, y, z].item(),
                    'pattern': field[x, y, z, :].clone()
                })
        
        return regions
    
    def _find_local_maxima(self, tensor: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Find local maxima in a 3D tensor."""
        peaks = []
        
        # Simple local maxima detection
        padded = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        
        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                for z in range(tensor.shape[2]):
                    # Check if this point is maximum in its neighborhood
                    center = tensor[x, y, z]
                    neighborhood = padded[x:x+3, y:y+3, z:z+3]
                    
                    if center == neighborhood.max() and center > self.coherence_threshold:
                        peaks.append((x, y, z))
        
        return peaks
    
    def _should_crystallize(self, region: Dict, field: torch.Tensor) -> bool:
        """
        Determine if a region should crystallize into a concept.
        
        Checks:
        - Sufficient energy and coherence
        - Not too close to existing concepts
        - Not already crystallized
        """
        pos = region['position']
        
        # Check if already crystallized
        if self.crystallization_map[pos[0], pos[1], pos[2]] > 0.5:
            return False
        
        # Check distance to existing concepts
        for concept in self.concepts:
            distance = torch.norm(pos.float() - concept.position.float())
            if distance < self.attractor_radius / 2:
                return False  # Too close to existing concept
        
        # Check energy and coherence thresholds
        return (region['energy'] > self.energy_threshold and 
                region['coherence'] > self.coherence_threshold)
    
    def _crystallize_region(self, field: torch.Tensor, region: Dict) -> torch.Tensor:
        """
        Crystallize a region into a discrete concept.
        
        Forces the pattern into a more discrete, stable state.
        """
        pos = region['position']
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
        
        # Define crystallization region
        x_min = max(0, x - 2)
        x_max = min(field.shape[0], x + 3)
        y_min = max(0, y - 2)
        y_max = min(field.shape[1], y + 3)
        z_min = max(0, z - 2)
        z_max = min(field.shape[2], z + 3)
        
        # Extract region
        region_field = field[x_min:x_max, y_min:y_max, z_min:z_max, :].clone()
        
        # Discretize: push values toward ±1 or 0
        # This creates binary-like patterns
        discretized = torch.sign(region_field) * torch.clamp(
            torch.abs(region_field) * 2,  # Amplify
            min=0, max=1
        )
        
        # Smooth transition to avoid discontinuities
        alpha = self.crystallization_rate
        field[x_min:x_max, y_min:y_max, z_min:z_max, :] = (
            (1 - alpha) * field[x_min:x_max, y_min:y_max, z_min:z_max, :] +
            alpha * discretized
        )
        
        # Update crystallization map
        self.crystallization_map[x_min:x_max, y_min:y_max, z_min:z_max] += 0.3
        self.crystallization_map = torch.clamp(self.crystallization_map, 0, 1)
        
        # Create and store new concept
        new_concept = Concept(
            position=pos,
            pattern=region['pattern'],
            strength=region['energy'],
            age=0
        )
        
        if len(self.concepts) < self.max_concepts:
            self.concepts.append(new_concept)
        
        return field
    
    def _apply_attractor(self, field: torch.Tensor, concept: Concept) -> torch.Tensor:
        """
        Apply attractor dynamics from an existing concept.
        
        Nearby patterns are pulled toward the concept's pattern.
        """
        # Create distance map from concept center
        xx, yy, zz = torch.meshgrid(
            torch.arange(field.shape[0], device=self.device),
            torch.arange(field.shape[1], device=self.device),
            torch.arange(field.shape[2], device=self.device),
            indexing='ij'
        )
        
        distance = torch.sqrt(
            (xx - concept.position[0])**2 +
            (yy - concept.position[1])**2 +
            (zz - concept.position[2])**2
        )
        
        # Attractor influence falls off with distance
        influence = torch.exp(-distance / self.attractor_radius)
        influence = influence.unsqueeze(-1)  # Add channel dimension
        
        # Pull field toward concept pattern
        target_pattern = concept.pattern.view(1, 1, 1, -1)
        attraction = (target_pattern - field) * influence * self.attractor_strength
        
        # Stronger concepts have stronger attraction
        attraction *= concept.strength
        
        field = field + attraction
        
        return field
    
    def _prune_concepts(self):
        """Remove old or weak concepts."""
        # Remove concepts that are too old or too weak
        self.concepts = [
            c for c in self.concepts
            if c.age < 1000 and c.strength > 0.1
        ]
        
        # Keep only the strongest concepts if over limit
        if len(self.concepts) > self.max_concepts:
            self.concepts.sort(key=lambda c: c.strength, reverse=True)
            self.concepts = self.concepts[:self.max_concepts]
    
    def get_active_concepts(self) -> List[Concept]:
        """Return list of currently active concepts."""
        return [c for c in self.concepts if c.strength > 0.3]
    
    def find_similar_concept(self, pattern: torch.Tensor) -> Optional[Concept]:
        """
        Find a concept similar to the given pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Most similar concept or None
        """
        if not self.concepts:
            return None
        
        best_match = None
        best_similarity = 0
        
        for concept in self.concepts:
            # Cosine similarity
            similarity = F.cosine_similarity(
                pattern.flatten().unsqueeze(0),
                concept.pattern.flatten().unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = concept
        
        return best_match if best_similarity > 0.7 else None