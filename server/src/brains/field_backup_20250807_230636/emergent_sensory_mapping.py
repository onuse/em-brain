"""
Emergent Sensory Mapping

Instead of dumping all sensory input into the center of the field,
this system lets sensory patterns find their natural place through
resonance, similarity, and self-organization.

Key principles:
1. Similar patterns naturally cluster
2. Correlated inputs become neighbors
3. Important patterns claim more space
4. Organization emerges from dynamics
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque

from ...utils.tensor_ops import create_zeros, safe_normalize


class EmergentSensoryMapping:
    """
    Maps sensory inputs to field locations through emergent dynamics.
    
    Instead of fixed mappings, patterns find their place based on:
    - Resonance with existing patterns
    - Correlation with other inputs
    - Importance (frequency and reward association)
    - Available space in the field
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 device: torch.device,
                 resonance_threshold: float = 0.3,
                 spatial_decay: float = 0.95):
        """
        Initialize emergent sensory mapping.
        
        Args:
            field_shape: Shape of the 4D field
            device: Computation device
            resonance_threshold: Minimum resonance to reuse location
            spatial_decay: How quickly influence decays with distance
        """
        self.field_shape = field_shape
        self.device = device
        self.resonance_threshold = resonance_threshold
        self.spatial_decay = spatial_decay
        
        # Track recent sensory patterns and their locations
        self.pattern_memory = deque(maxlen=100)
        self.location_memory = deque(maxlen=100)
        
        # Track pattern correlations for clustering
        self.correlation_matrix = {}
        self.co_activation_counts = {}
        
        # Importance scores for patterns
        self.pattern_importance = {}
        
        # Field occupancy - which regions are "claimed"
        self.occupancy_map = create_zeros(field_shape[:3], device=device)
        
        # Evolution tracking
        self.mapping_events = 0
        self.reorganization_count = 0
        
    def find_imprint_location(self, 
                             sensory_pattern: torch.Tensor,
                             field_state: torch.Tensor,
                             reward: float = 0.0) -> Tuple[int, int, int]:
        """
        Find the best location in the field for this sensory pattern.
        
        This is where the magic happens - patterns find their natural home
        based on resonance, correlation, and field dynamics.
        
        Args:
            sensory_pattern: The sensory input to place
            field_state: Current field state
            reward: Associated reward (affects importance)
            
        Returns:
            Tuple of (x, y, z) coordinates for imprinting
        """
        # Normalize pattern for comparison
        pattern_norm = safe_normalize(sensory_pattern.flatten())
        pattern_hash = self._hash_pattern(pattern_norm)
        
        # Update importance based on reward
        self._update_pattern_importance(pattern_hash, reward)
        
        # 1. Check if this pattern has a established location
        existing_location = self._find_existing_location(pattern_norm, field_state)
        if existing_location is not None:
            self._update_pattern_memory(pattern_norm, existing_location)
            return existing_location
        
        # 2. Find location through resonance with field
        resonance_location = self._find_resonant_location(pattern_norm, field_state)
        if resonance_location is not None:
            self._update_pattern_memory(pattern_norm, resonance_location)
            return resonance_location
        
        # 3. Find location based on correlations with other patterns
        correlation_location = self._find_correlated_location(pattern_norm)
        if correlation_location is not None:
            self._update_pattern_memory(pattern_norm, correlation_location)
            return correlation_location
        
        # 4. Find empty location with good properties
        empty_location = self._find_empty_location(field_state, pattern_norm)
        self._update_pattern_memory(pattern_norm, empty_location)
        
        # Mark location as occupied
        x, y, z = empty_location
        self.occupancy_map[x, y, z] = 1.0
        
        self.mapping_events += 1
        return empty_location
    
    def _find_existing_location(self, 
                               pattern: torch.Tensor,
                               field_state: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """Check if this pattern already has an established location."""
        best_match = None
        best_similarity = 0.0
        
        for i, (mem_pattern, mem_location) in enumerate(zip(self.pattern_memory, self.location_memory)):
            similarity = F.cosine_similarity(
                pattern.unsqueeze(0),
                mem_pattern.unsqueeze(0)
            ).detach().item()
            
            if similarity > 0.9 and similarity > best_similarity:  # Very high threshold
                # Verify location is still responsive
                x, y, z = mem_location
                local_energy = torch.mean(torch.abs(
                    field_state[x-1:x+2, y-1:y+2, z-1:z+2, :]
                )).detach().item()
                
                if local_energy < 2.0:  # Not saturated
                    best_match = mem_location
                    best_similarity = similarity
        
        return best_match
    
    def _find_resonant_location(self,
                               pattern: torch.Tensor,
                               field_state: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """Find location where pattern resonates with existing field activity."""
        # Create pattern field - what would this pattern look like in field space
        pattern_field = self._pattern_to_field_projection(pattern)
        
        # Compute resonance with current field state
        # Use only first few channels for efficiency
        field_projection = field_state[:, :, :, :8].mean(dim=3)
        
        # Find peaks of resonance
        resonance_map = F.cosine_similarity(
            pattern_field.unsqueeze(3),
            field_projection.unsqueeze(3),
            dim=3
        )
        
        # Apply spatial smoothing to find regions, not points
        # Use manual convolution for MPS compatibility
        kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27.0
        resonance_padded = F.pad(
            resonance_map.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1, 1, 1),
            mode='constant',
            value=0
        )
        resonance_smooth = F.conv3d(
            resonance_padded,
            kernel,
            stride=1,
            padding=0
        ).squeeze()
        
        # Find maximum resonance location
        max_resonance = torch.max(resonance_smooth).detach().item()
        
        if max_resonance > self.resonance_threshold:
            # Find location of maximum
            max_idx = torch.argmax(resonance_smooth.flatten())
            z = max_idx % self.field_shape[2]
            y = (max_idx // self.field_shape[2]) % self.field_shape[1]
            x = max_idx // (self.field_shape[1] * self.field_shape[2])
            
            return (x.detach().item(), y.detach().item(), z.detach().item())
        
        return None
    
    def _find_correlated_location(self, pattern: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """Find location near correlated patterns."""
        pattern_hash = self._hash_pattern(pattern)
        
        # Find most correlated patterns
        best_correlation = 0.0
        best_location = None
        
        for other_hash, correlation in self.correlation_matrix.get(pattern_hash, {}).items():
            if correlation > 0.5:  # Significant correlation
                # Find this pattern's location
                for mem_pattern, mem_location in zip(self.pattern_memory, self.location_memory):
                    if self._hash_pattern(mem_pattern) == other_hash:
                        # Look for nearby empty location
                        nearby = self._find_nearby_empty(mem_location)
                        if nearby is not None:
                            return nearby
        
        return None
    
    def _find_empty_location(self, 
                            field_state: torch.Tensor,
                            pattern: torch.Tensor) -> Tuple[int, int, int]:
        """Find a good empty location for a new pattern."""
        # Compute field information to avoid crowded areas
        field_information = torch.mean(torch.abs(field_state), dim=3)
        
        # Compute available space (low information, low occupancy)
        availability = (1 - self.occupancy_map) * (1 - torch.sigmoid(field_information * 5))
        
        # Add noise to prevent all patterns going to same spot
        availability += torch.randn_like(availability) * 0.1
        
        # Prefer locations with some activity (not dead zones)
        min_activity = 0.01
        availability *= (field_information > min_activity).float()
        
        # Find best available location
        max_idx = torch.argmax(availability.flatten())
        z = max_idx % self.field_shape[2]
        y = (max_idx // self.field_shape[2]) % self.field_shape[1]
        x = max_idx // (self.field_shape[1] * self.field_shape[2])
        
        return (x.detach().item(), y.detach().item(), z.detach().item())
    
    def _find_nearby_empty(self, 
                          location: Tuple[int, int, int],
                          search_radius: int = 5) -> Optional[Tuple[int, int, int]]:
        """Find empty location near given location."""
        x, y, z = location
        
        # Search in expanding shells
        for radius in range(1, search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        # Check if on shell surface
                        if max(abs(dx), abs(dy), abs(dz)) == radius:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            
                            # Check bounds
                            if (0 <= nx < self.field_shape[0] and
                                0 <= ny < self.field_shape[1] and
                                0 <= nz < self.field_shape[2]):
                                
                                # Check occupancy
                                if self.occupancy_map[nx, ny, nz] < 0.5:
                                    return (nx, ny, nz)
        
        return None
    
    def _pattern_to_field_projection(self, pattern: torch.Tensor) -> torch.Tensor:
        """Project pattern into field space for resonance calculation."""
        # Simple projection - could be learned
        field_proj = create_zeros(self.field_shape[:3], device=self.device)
        
        # Create spatial structure from pattern
        # This is simplified - in reality would be more sophisticated
        pattern_len = len(pattern)
        grid_size = int(np.ceil(np.sqrt(pattern_len)))
        
        for i, value in enumerate(pattern):
            # Map linear index to 2D grid
            gx = i % grid_size
            gy = i // grid_size
            
            # Map to 3D field (with some spread)
            fx = int(gx * self.field_shape[0] / grid_size)
            fy = int(gy * self.field_shape[1] / grid_size)
            fz = int(i * self.field_shape[2] / pattern_len)
            
            # Add with some spatial spread
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        nx = min(max(fx + dx, 0), self.field_shape[0] - 1)
                        ny = min(max(fy + dy, 0), self.field_shape[1] - 1)
                        nz = min(max(fz + dz, 0), self.field_shape[2] - 1)
                        
                        weight = self.spatial_decay ** (abs(dx) + abs(dy) + abs(dz))
                        field_proj[nx, ny, nz] += value.detach().item() * weight
        
        return field_proj
    
    def _update_pattern_memory(self, pattern: torch.Tensor, location: Tuple[int, int, int]):
        """Update memory of patterns and their locations."""
        self.pattern_memory.append(pattern.detach().clone())
        self.location_memory.append(location)
        
        # Update correlations with recent patterns
        pattern_hash = self._hash_pattern(pattern)
        
        for other_pattern in list(self.pattern_memory)[-10:]:  # Last 10 patterns
            other_hash = self._hash_pattern(other_pattern)
            if other_hash != pattern_hash:
                correlation = F.cosine_similarity(
                    pattern.unsqueeze(0),
                    other_pattern.unsqueeze(0)
                ).detach().item()
                
                # Update correlation matrix
                if pattern_hash not in self.correlation_matrix:
                    self.correlation_matrix[pattern_hash] = {}
                self.correlation_matrix[pattern_hash][other_hash] = correlation
                
                # Track co-activations
                key = tuple(sorted([pattern_hash, other_hash]))
                self.co_activation_counts[key] = self.co_activation_counts.get(key, 0) + 1
    
    def _update_pattern_importance(self, pattern_hash: int, reward: float):
        """Update importance score for pattern based on reward."""
        current = self.pattern_importance.get(pattern_hash, 0.0)
        # Increase importance with positive reward, decrease with negative
        self.pattern_importance[pattern_hash] = current * 0.95 + reward * 0.5
    
    def _hash_pattern(self, pattern: torch.Tensor) -> int:
        """Create hash for pattern identification."""
        # Simple hash based on quantized values
        quantized = torch.round(pattern * 10) / 10
        return hash(tuple(quantized.detach().cpu().numpy()))
    
    def reorganize_mappings(self, field_state: torch.Tensor):
        """Periodically reorganize mappings for better organization."""
        # This could be called during consolidation
        # Identify patterns that should be neighbors but aren't
        # Move patterns to create better topological organization
        
        # For now, just decay occupancy to allow remapping
        self.occupancy_map *= 0.99
        self.reorganization_count += 1
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about sensory organization."""
        unique_patterns = len(set(self._hash_pattern(p) for p in self.pattern_memory))
        
        # Compute clustering coefficient
        clustering = 0.0
        if self.co_activation_counts:
            total_weight = sum(self.co_activation_counts.values())
            clustering = len(self.co_activation_counts) / (unique_patterns + 1)
        
        return {
            'unique_patterns': unique_patterns,
            'mapping_events': self.mapping_events,
            'reorganization_count': self.reorganization_count,
            'clustering_coefficient': clustering,
            'occupancy_ratio': torch.mean(self.occupancy_map).detach().item()
        }