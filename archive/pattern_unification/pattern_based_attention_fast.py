#!/usr/bin/env python3
"""
Fast Pattern-Based Attention System

Optimized version that samples the field intelligently rather than
exhaustively searching millions of patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import deque
import time

from .field_types import FieldDynamicsFamily
from .pattern_based_attention import PatternSignature, AttentionFocus


class FastPatternBasedAttention:
    """
    Optimized attention system that samples patterns intelligently.
    
    Key optimizations:
    1. Sample field regions instead of exhaustive search
    2. Use hierarchical pattern extraction
    3. Cache pattern computations
    4. Limit FFT to reasonable sizes
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...],
                 attention_capacity: int = 5,
                 device: torch.device = torch.device('cpu'),
                 quiet_mode: bool = False):
        """Initialize fast pattern-based attention."""
        self.field_shape = field_shape
        self.attention_capacity = attention_capacity
        self.device = device
        self.quiet_mode = quiet_mode
        
        # Calculate field statistics
        self.field_size = np.prod(field_shape)
        self.spatial_dims = min(3, len(field_shape))  # First 3 dims are spatial
        
        # Sampling parameters
        self.max_samples_per_dim = 8  # Sample at most 8 positions per dimension
        self.max_patterns_extracted = 100  # Limit total patterns per cycle
        self.fft_size_limit = 1024  # Maximum size for FFT
        
        # Pattern memory (reuse from original)
        self.known_patterns: Dict[str, PatternSignature] = {}
        self.pattern_history = deque(maxlen=100)
        
        # Attention state
        self.current_focus = AttentionFocus()
        self.attention_buffer: List[Tuple[str, float]] = []
        
        # Salience factors
        self.novelty_weight = 0.4
        self.surprise_weight = 0.3
        self.importance_weight = 0.3
        
        # Pattern statistics
        self.pattern_statistics = {
            'mean_energy': 0.5,
            'mean_variance': 0.1,
            'mean_frequency': 0.5
        }
        
        if not quiet_mode:
            print(f"👁️  Fast Pattern-Based Attention initialized")
            print(f"   Field size: {self.field_size:,} elements")
            print(f"   Sampling strategy: hierarchical with {self.max_samples_per_dim}³ spatial samples")
    
    def process_field_patterns(self, 
                             field: torch.Tensor,
                             sensory_patterns: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Process field patterns efficiently."""
        pattern_saliences = []
        
        # Process sensory patterns first (small, high priority)
        if sensory_patterns:
            for modality, pattern_tensor in sensory_patterns.items():
                if pattern_tensor is not None and len(pattern_tensor) > 0:
                    # These are small, so use original method
                    sensory_subpatterns = self._extract_sensory_patterns_simple(pattern_tensor, modality)
                    
                    for subpattern in sensory_subpatterns:
                        salience = self._calculate_pattern_salience_fast(subpattern, modality)
                        salience *= 1.5  # Boost sensory patterns
                        
                        pattern_id = self._find_or_create_pattern(subpattern, modality)
                        pattern_saliences.append((pattern_id, salience))
        
        # Extract field patterns using sampling
        field_patterns = self._extract_field_patterns_fast(field)
        
        for pattern_tensor, modality in field_patterns:
            salience = self._calculate_pattern_salience_fast(pattern_tensor, modality)
            pattern_id = self._find_or_create_pattern(pattern_tensor, modality)
            pattern_saliences.append((pattern_id, salience))
        
        # Sort by salience and select top patterns
        pattern_saliences.sort(key=lambda x: x[1], reverse=True)
        self.attention_buffer = pattern_saliences[:self.attention_capacity]
        
        # Update attention focus
        self._update_attention_focus()
        
        return {
            'focus': self.current_focus,
            'attended_patterns': [p[0] for p in self.attention_buffer],
            'salience_map': None,  # Skip expensive salience map creation
            'bindings': [],  # Skip binding detection for now
            'pattern_count': len(self.known_patterns)
        }
    
    def _extract_sensory_patterns_simple(self, sensory_input: torch.Tensor, modality: str) -> List[torch.Tensor]:
        """Simple pattern extraction for small sensory inputs."""
        patterns = []
        
        if not isinstance(sensory_input, torch.Tensor):
            sensory_input = torch.tensor(sensory_input, device=self.device)
        
        # Just take the whole input and a few statistics
        patterns.append(sensory_input)
        
        # Statistical summary
        if len(sensory_input) >= 4:
            stat_pattern = torch.tensor([
                torch.mean(sensory_input).item(),
                torch.std(sensory_input).item() if len(sensory_input) > 1 else 0.0,
                torch.max(sensory_input).item(),
                torch.min(sensory_input).item()
            ], device=self.device)
            patterns.append(stat_pattern)
        
        return patterns
    
    def _extract_field_patterns_fast(self, field: torch.Tensor) -> List[Tuple[torch.Tensor, str]]:
        """Extract patterns using intelligent sampling."""
        patterns = []
        
        # 1. Sample spatial regions (not exhaustive search)
        spatial_patterns = self._sample_spatial_regions(field)
        patterns.extend([(p, 'field') for p in spatial_patterns])
        
        # 2. Extract global statistics (cheap)
        stat_pattern = torch.tensor([
            torch.mean(field).item(),
            torch.std(field).item(),
            torch.min(field).item(),
            torch.max(field).item(),
            self._calculate_entropy_fast(field)
        ], device=self.device)
        patterns.append((stat_pattern, 'field_stats'))
        
        # 3. Sample along important dimensions
        if len(self.field_shape) > 3:
            dim_patterns = self._sample_dimension_slices(field)
            patterns.extend([(p, 'field_dim') for p in dim_patterns])
        
        # Limit total patterns
        return patterns[:self.max_patterns_extracted]
    
    def _sample_spatial_regions(self, field: torch.Tensor) -> List[torch.Tensor]:
        """Sample spatial regions instead of exhaustive search."""
        patterns = []
        
        # Get spatial dimensions
        spatial_shape = self.field_shape[:self.spatial_dims]
        
        # Calculate stride for sampling
        strides = [max(1, dim // self.max_samples_per_dim) for dim in spatial_shape]
        
        # Sample regions
        region_size = 2  # Small 2x2x2 regions
        sample_count = 0
        
        for x in range(0, spatial_shape[0] - region_size + 1, strides[0]):
            for y in range(0, spatial_shape[1] - region_size + 1, strides[1]):
                if self.spatial_dims > 2:
                    for z in range(0, spatial_shape[2] - region_size + 1, strides[2]):
                        if sample_count >= 20:  # Limit samples
                            return patterns
                        
                        region = field[x:x+region_size, y:y+region_size, z:z+region_size, ...]
                        pattern = region.flatten()[:32]  # Take first 32 elements
                        if self._is_meaningful_pattern_fast(pattern):
                            patterns.append(pattern)
                            sample_count += 1
                else:
                    if sample_count >= 20:
                        return patterns
                    
                    region = field[x:x+region_size, y:y+region_size, ...]
                    pattern = region.flatten()[:32]
                    if self._is_meaningful_pattern_fast(pattern):
                        patterns.append(pattern)
                        sample_count += 1
        
        return patterns
    
    def _sample_dimension_slices(self, field: torch.Tensor) -> List[torch.Tensor]:
        """Sample slices along non-spatial dimensions."""
        patterns = []
        
        # Sample middle slices of higher dimensions
        for dim_idx in range(self.spatial_dims, min(len(self.field_shape), self.spatial_dims + 3)):
            dim_size = self.field_shape[dim_idx]
            mid_idx = dim_size // 2
            
            # Create slice indices
            indices = [slice(None)] * len(self.field_shape)
            indices[dim_idx] = mid_idx
            
            # Extract slice and flatten
            slice_data = field[tuple(indices)].flatten()
            
            # Sample from slice
            if len(slice_data) > 64:
                # Take evenly spaced samples
                sample_indices = torch.linspace(0, len(slice_data)-1, 64, dtype=torch.long)
                pattern = slice_data[sample_indices]
            else:
                pattern = slice_data
            
            if self._is_meaningful_pattern_fast(pattern):
                patterns.append(pattern)
        
        return patterns
    
    def _is_meaningful_pattern_fast(self, pattern: torch.Tensor) -> bool:
        """Fast check if pattern is meaningful."""
        if len(pattern) < 2:
            return False
        
        # Quick variance check
        if torch.var(pattern).item() < 0.001:
            return False
        
        # Quick range check
        if torch.max(torch.abs(pattern)).item() < 0.01:
            return False
        
        return True
    
    def _calculate_pattern_salience_fast(self, pattern: torch.Tensor, modality: str) -> float:
        """Fast salience calculation."""
        # Simple energy-based salience
        energy = torch.mean(torch.abs(pattern)).item()
        variance = torch.var(pattern).item() if len(pattern) > 1 else 0
        
        # Base salience
        salience = energy + variance * 0.5
        
        # Modality boost
        modality_boost = {
            'visual': 1.2,
            'motor': 1.3,
            'field': 1.0,
            'field_stats': 0.8,
            'field_dim': 0.9
        }.get(modality, 1.0)
        
        return salience * modality_boost
    
    def _calculate_entropy_fast(self, field: torch.Tensor) -> float:
        """Fast entropy approximation."""
        # Sample random elements
        flat = field.flatten()
        if len(flat) > 1000:
            indices = torch.randint(0, len(flat), (1000,))
            sample = flat[indices]
        else:
            sample = flat
        
        # Simple entropy estimate
        sample_positive = sample - torch.min(sample) + 1e-8
        sample_probs = sample_positive / torch.sum(sample_positive)
        entropy = -torch.sum(sample_probs * torch.log2(sample_probs + 1e-8))
        
        return entropy.item()
    
    def _find_or_create_pattern(self, pattern: torch.Tensor, modality: str) -> str:
        """Find or create pattern (reuse from original)."""
        # For speed, just create new patterns without similarity check
        pattern_id = f"pat_{len(self.known_patterns)}_{modality[:3]}"
        
        new_pattern = PatternSignature(
            pattern_id=pattern_id,
            pattern_tensor=pattern.clone().detach(),
            modality=modality,
            frequency=1.0,
            last_seen=time.time(),
            importance=0.5,
            associations=set()
        )
        
        self.known_patterns[pattern_id] = new_pattern
        self.pattern_history.append(pattern_id)
        
        return pattern_id
    
    def _update_attention_focus(self):
        """Update attention focus."""
        if not self.attention_buffer:
            self.current_focus = AttentionFocus()
            return
        
        primary_id, primary_salience = self.attention_buffer[0]
        
        if primary_id != self.current_focus.primary_pattern:
            self.current_focus = AttentionFocus(
                primary_pattern=primary_id,
                supporting_patterns=[p[0] for p in self.attention_buffer[1:]],
                focus_strength=primary_salience,
                duration=0.0
            )
        else:
            self.current_focus.duration += 0.05
            self.current_focus.focus_strength = primary_salience
            self.current_focus.supporting_patterns = [p[0] for p in self.attention_buffer[1:]]
    
    def modulate_field_with_attention(self, field: torch.Tensor, attention_state: Dict[str, Any]) -> torch.Tensor:
        """Simple field modulation."""
        # For now, just return field unchanged
        # Full implementation would selectively boost attended regions
        return field
    
    def get_attention_metrics(self) -> Dict[str, Any]:
        """Get attention metrics."""
        return {
            'known_patterns': len(self.known_patterns),
            'current_focus': self.current_focus.primary_pattern,
            'focus_strength': self.current_focus.focus_strength,
            'focus_duration': self.current_focus.duration,
            'attended_count': len(self.attention_buffer),
            'pattern_statistics': self.pattern_statistics,
            'capacity_usage': len(self.attention_buffer) / self.attention_capacity
        }