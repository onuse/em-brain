"""
Unified Pattern System

Extracts patterns from field dynamics for both attention and motor generation.
Replaces duplicate pattern extraction logic with a single, efficient system.

Key insight: Both attention and motor systems need the same pattern features -
they just use them differently.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class FieldPattern:
    """Represents a pattern extracted from the field."""
    # Core features
    energy: float           # Mean absolute activation
    variance: float        # Spatial/temporal variance
    coherence: float       # Organization/structure
    
    # Dynamic features
    oscillation: float     # Rhythmic component
    flow_divergence: float # Expansion/contraction
    flow_curl: float       # Rotational component
    
    # Meta features
    novelty: float         # How unique vs history
    salience: float        # Overall importance
    location: Optional[Tuple[int, ...]] = None  # Where in field
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy use."""
        return {
            'energy': self.energy,
            'variance': self.variance,
            'coherence': self.coherence,
            'oscillation': self.oscillation,
            'flow_divergence': self.flow_divergence,
            'flow_curl': self.flow_curl,
            'novelty': self.novelty,
            'salience': self.salience
        }


class UnifiedPatternSystem:
    """
    Unified system for extracting patterns from field dynamics.
    
    Serves both attention (what to focus on) and motor (how to act).
    Efficient sampling and feature extraction for real-time operation.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...],
                 device: torch.device = torch.device('cpu'),
                 max_patterns: int = 50,
                 history_size: int = 100):
        """
        Initialize unified pattern system.
        
        Args:
            field_shape: Shape of the field tensor
            device: Computation device
            max_patterns: Maximum patterns to extract per cycle
            history_size: Number of patterns to remember
        """
        self.field_shape = field_shape
        self.device = device
        self.max_patterns = max_patterns
        
        # Pattern memory for novelty detection
        self.pattern_history = deque(maxlen=history_size)
        self.pattern_cache = {}  # Cache recent computations
        
        # Field history for temporal analysis
        self.field_history = deque(maxlen=3)
        
        # Sampling parameters
        self.spatial_samples = min(8, field_shape[0])  # Sample grid
        self.sample_size = min(8, field_shape[0])      # Region size
        
        # Feature normalization stats (will adapt)
        self.feature_stats = {
            'energy': {'mean': 0.5, 'std': 0.2},
            'variance': {'mean': 0.1, 'std': 0.05},
            'oscillation': {'mean': 0.3, 'std': 0.2}
        }
        
    def extract_patterns(self, 
                        field: torch.Tensor,
                        n_patterns: Optional[int] = None) -> List[FieldPattern]:
        """
        Extract patterns from field.
        
        Args:
            field: Current field state
            n_patterns: Number of patterns to extract (None = auto)
            
        Returns:
            List of extracted patterns sorted by salience
        """
        # Update field history
        self.field_history.append(field.clone())
        
        # Determine number of patterns
        if n_patterns is None:
            n_patterns = min(self.max_patterns, self._estimate_pattern_count(field))
        
        patterns = []
        
        # 1. Extract global pattern (whole field)
        global_pattern = self._extract_pattern_features(field)
        patterns.append(global_pattern)
        
        # 2. Extract local patterns (sampled regions)
        sample_positions = self._generate_sample_positions(n_patterns - 1)
        
        for pos in sample_positions:
            region = self._extract_region(field, pos, self.sample_size)
            if region is not None:
                pattern = self._extract_pattern_features(region, location=pos)
                patterns.append(pattern)
        
        # 3. Sort by salience
        patterns.sort(key=lambda p: p.salience, reverse=True)
        
        # 4. Update pattern history
        for pattern in patterns[:10]:  # Remember top patterns
            self.pattern_history.append(pattern)
        
        return patterns
    
    def _extract_pattern_features(self, 
                                 region: torch.Tensor,
                                 location: Optional[Tuple[int, ...]] = None) -> FieldPattern:
        """Extract all features from a field region."""
        # Ensure we have temporal history
        if len(self.field_history) < 2:
            # No history - create static pattern
            return self._create_static_pattern(region, location)
        
        # Get previous state for temporal analysis
        prev_region = self._get_previous_region(region, location)
        temporal_change = region - prev_region
        
        # Core features
        energy = float(torch.mean(torch.abs(region)))
        variance = float(torch.var(region))
        
        # Dynamic features
        oscillation = self._compute_oscillation(temporal_change)
        flow_div, flow_curl = self._compute_flow(temporal_change)
        
        # Coherence (spatial organization)
        coherence = self._compute_coherence(region)
        
        # Novelty (compared to history)
        novelty = self._compute_novelty(region)
        
        # Salience (combined importance)
        salience = self._compute_salience(
            energy, variance, oscillation, novelty, coherence
        )
        
        return FieldPattern(
            energy=energy,
            variance=variance,
            coherence=coherence,
            oscillation=oscillation,
            flow_divergence=flow_div,
            flow_curl=flow_curl,
            novelty=novelty,
            salience=salience,
            location=location,
            timestamp=time.time()
        )
    
    def _create_static_pattern(self, 
                              region: torch.Tensor,
                              location: Optional[Tuple[int, ...]] = None) -> FieldPattern:
        """Create pattern without temporal features."""
        energy = float(torch.mean(torch.abs(region)))
        variance = float(torch.var(region))
        coherence = self._compute_coherence(region)
        novelty = self._compute_novelty(region)
        salience = energy + variance * 0.5 + novelty * 0.3
        
        return FieldPattern(
            energy=energy,
            variance=variance,
            coherence=coherence,
            oscillation=0.0,
            flow_divergence=0.0,
            flow_curl=0.0,
            novelty=novelty,
            salience=salience,
            location=location,
            timestamp=time.time()
        )
    
    def _compute_oscillation(self, temporal_change: torch.Tensor) -> float:
        """Detect oscillatory patterns."""
        # Count sign changes
        flat = temporal_change.flatten()
        if len(flat) < 2:
            return 0.0
            
        sign_changes = torch.sum(flat[:-1] * flat[1:] < 0)
        oscillation = sign_changes.item() / len(flat)
        
        return min(1.0, oscillation * 10)  # Normalize to [0, 1]
    
    def _compute_flow(self, field: torch.Tensor) -> Tuple[float, float]:
        """Compute flow characteristics."""
        if field.numel() < 4:
            return 0.0, 0.0
        
        # Reshape to 2D for gradient analysis
        size = int(np.sqrt(field.numel()))
        if size < 2:
            return 0.0, 0.0
            
        try:
            field_2d = field.flatten()[:size*size].reshape(size, size)
            
            # Compute gradients
            dy = field_2d[1:, :] - field_2d[:-1, :]
            dx = field_2d[:, 1:] - field_2d[:, :-1]
            
            # Divergence (expansion/contraction)
            div = (torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))) / 2
            
            # Curl (rotation) - simplified
            if size > 2:
                curl = torch.mean(torch.abs(dx[:-1, :-1] - dy[:-1, :-1]))
            else:
                curl = torch.tensor(0.0)
                
            return float(div), float(curl)
            
        except:
            return 0.0, 0.0
    
    def _compute_coherence(self, region: torch.Tensor) -> float:
        """Measure spatial organization."""
        if region.numel() < 2:
            return 0.0
            
        # Local correlation
        flat = region.flatten()
        if len(flat) < 2:
            return 0.0
            
        # Auto-correlation at lag 1
        corr = torch.corrcoef(torch.stack([flat[:-1], flat[1:]]))[0, 1]
        return float(torch.abs(corr))
    
    def _compute_novelty(self, region: torch.Tensor) -> float:
        """Compare to pattern history."""
        if len(self.pattern_history) == 0:
            return 1.0
        
        # Downsample for efficient comparison
        # Flatten and sample to fixed size
        flat_region = region.flatten()
        if len(flat_region) > 64:
            # Sample evenly
            indices = torch.linspace(0, len(flat_region)-1, 64, dtype=torch.long, device=region.device)
            region_small = flat_region[indices]
        else:
            region_small = flat_region
        
        # Normalize
        region_norm = region_small / (torch.norm(region_small) + 1e-8)
        
        # Compare to recent patterns
        max_similarity = 0.0
        for past_pattern in list(self.pattern_history)[-20:]:
            # Create comparable representation
            similarity = self._pattern_similarity(region_norm, past_pattern)
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _pattern_similarity(self, 
                           region_norm: torch.Tensor, 
                           pattern: FieldPattern) -> float:
        """Compute similarity to a stored pattern."""
        # Simple feature-based similarity
        # In practice, could store actual tensor data
        return 0.5  # Placeholder - would use actual comparison
    
    def _compute_salience(self, 
                         energy: float,
                         variance: float,
                         oscillation: float,
                         novelty: float,
                         coherence: float) -> float:
        """Combine features into overall salience."""
        # Adaptive weighting based on context
        salience = (
            energy * 0.3 +
            variance * 0.2 +
            oscillation * 0.2 +
            novelty * 0.2 +
            coherence * 0.1
        )
        return min(1.0, salience)
    
    def _estimate_pattern_count(self, field: torch.Tensor) -> int:
        """Estimate how many patterns to extract based on field complexity."""
        # Simple heuristic: more variance = more patterns
        variance = torch.var(field).item()
        base_count = 10
        variance_factor = min(5.0, variance * 10)
        return int(base_count * (1 + variance_factor))
    
    def _generate_sample_positions(self, n: int) -> List[Tuple[int, ...]]:
        """Generate positions for local pattern extraction."""
        positions = []
        
        # Grid sampling
        step = max(1, self.field_shape[0] // self.spatial_samples)
        
        for i in range(0, self.field_shape[0], step):
            for j in range(0, min(self.field_shape[1], self.field_shape[0]), step):
                for k in range(0, min(self.field_shape[2], self.field_shape[0]), step):
                    if len(positions) < n:
                        positions.append((i, j, k))
                    else:
                        return positions
        
        return positions
    
    def _extract_region(self, 
                       field: torch.Tensor,
                       position: Tuple[int, ...],
                       size: int) -> Optional[torch.Tensor]:
        """Extract a local region from field."""
        try:
            slices = []
            for i, pos in enumerate(position):
                start = max(0, pos)
                end = min(field.shape[i], pos + size)
                slices.append(slice(start, end))
            
            region = field[tuple(slices)]
            
            if region.numel() > 0:
                return region
            else:
                return None
                
        except:
            return None
    
    def _get_previous_region(self, 
                            current: torch.Tensor,
                            location: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Get corresponding region from previous field state."""
        if len(self.field_history) < 2:
            return torch.zeros_like(current)
        
        prev_field = self.field_history[-2]
        
        if location is None:
            # Global pattern
            return prev_field
        else:
            # Extract same region from previous field
            region = self._extract_region(prev_field, location, self.sample_size)
            if region is not None and region.shape == current.shape:
                return region
            else:
                return torch.zeros_like(current)