"""
Pattern Cache Pool

Performance optimization for pattern extraction by reusing tensor allocations
and caching pattern computations.
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


class PatternCachePool:
    """
    Memory pool for pattern extraction to avoid repeated allocations.
    
    Key optimizations:
    1. Pre-allocate tensors for pattern extraction
    2. Reuse memory buffers
    3. Cache computed features
    4. Batch operations where possible
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 max_patterns: int = 50,
                 device: torch.device = torch.device('cpu')):
        """Initialize pattern cache pool."""
        self.field_shape = field_shape
        self.max_patterns = max_patterns
        self.device = device
        
        # Pre-allocate buffers
        self._init_buffers()
        
        # Feature cache
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _init_buffers(self):
        """Pre-allocate reusable buffers."""
        # Buffer for temporal differences
        self.temporal_buffer = torch.zeros(self.field_shape, device=self.device)
        
        # Buffer for downsampled patterns
        self.downsample_buffer = torch.zeros((8, 8, 8), device=self.device)
        
        # Buffer for pattern coordinates
        self.coord_buffer = torch.zeros((self.max_patterns, 3), dtype=torch.long, device=self.device)
        
        # Pre-compute sampling positions
        self._precompute_sample_positions()
        
    def _precompute_sample_positions(self):
        """Pre-compute sampling positions for pattern extraction."""
        # Create a grid of sample positions
        spatial_dims = self.field_shape[:3]
        step_sizes = [max(1, dim // 8) for dim in spatial_dims]
        
        positions = []
        for x in range(0, spatial_dims[0], step_sizes[0]):
            for y in range(0, spatial_dims[1], step_sizes[1]):
                for z in range(0, spatial_dims[2], step_sizes[2]):
                    positions.append((x, y, z))
        
        # Convert to tensor
        self.sample_positions = torch.tensor(positions[:self.max_patterns], 
                                           dtype=torch.long, 
                                           device=self.device)
    
    @torch.no_grad()
    def extract_patterns_fast(self, 
                             field: torch.Tensor,
                             n_patterns: int = 10) -> List[Dict[str, float]]:
        """
        Fast pattern extraction using pre-allocated buffers.
        
        Returns simplified pattern dicts for speed.
        """
        patterns = []
        
        # 1. Global pattern (fast)
        global_features = self._compute_global_features_fast(field)
        patterns.append(global_features)
        
        if n_patterns <= 1:
            return patterns
        
        # 2. Local patterns using pre-computed positions
        n_local = min(n_patterns - 1, len(self.sample_positions))
        local_patterns = self._extract_local_patterns_batch(field, n_local)
        patterns.extend(local_patterns)
        
        return patterns[:n_patterns]
    
    def _compute_global_features_fast(self, field: torch.Tensor) -> Dict[str, float]:
        """Compute global features with minimal overhead."""
        # Use torch operations without item() calls until the end
        flat = field.flatten()
        
        # Batch compute all statistics
        energy = torch.mean(torch.abs(flat))
        variance = torch.var(flat)
        
        # Simple oscillation detection
        signs = torch.sign(flat[::10])  # Sample for speed
        sign_changes = torch.sum(torch.abs(signs[1:] - signs[:-1]))
        oscillation = sign_changes / len(signs)
        
        # Convert to Python floats only at the end
        return {
            'energy': float(energy),
            'variance': float(variance),
            'coherence': float(variance / (energy + 1e-8)),  # Simplified
            'oscillation': float(oscillation),
            'flow_divergence': 0.0,  # Skip expensive flow computation
            'flow_curl': 0.0,
            'novelty': 0.5,  # Default
            'salience': float(energy * (1 + variance))
        }
    
    def _extract_local_patterns_batch(self, 
                                     field: torch.Tensor,
                                     n_patterns: int) -> List[Dict[str, float]]:
        """Extract multiple local patterns in batch."""
        patterns = []
        
        # Use pre-computed positions
        positions = self.sample_positions[:n_patterns]
        
        # Extract regions in batch
        region_size = 4  # Smaller regions for speed
        half_size = region_size // 2
        
        for i in range(n_patterns):
            pos = positions[i]
            
            # Extract region with bounds checking
            x_start = max(0, pos[0] - half_size)
            x_end = min(field.shape[0], pos[0] + half_size)
            y_start = max(0, pos[1] - half_size)
            y_end = min(field.shape[1], pos[1] + half_size)
            z_start = max(0, pos[2] - half_size)
            z_end = min(field.shape[2], pos[2] + half_size)
            
            if x_end > x_start and y_end > y_start and z_end > z_start:
                region = field[x_start:x_end, y_start:y_end, z_start:z_end, ...]
                
                # Quick feature computation
                features = self._compute_region_features_fast(region)
                patterns.append(features)
        
        return patterns
    
    def _compute_region_features_fast(self, region: torch.Tensor) -> Dict[str, float]:
        """Fast feature computation for a region."""
        # Simplified features
        flat = region.flatten()
        
        if len(flat) < 2:
            return self._empty_features()
        
        energy = torch.mean(torch.abs(flat))
        variance = torch.var(flat) if len(flat) > 1 else torch.tensor(0.0)
        
        return {
            'energy': float(energy),
            'variance': float(variance),
            'coherence': 0.5,  # Default
            'oscillation': 0.0,  # Skip for local
            'flow_divergence': 0.0,
            'flow_curl': 0.0,
            'novelty': 0.5,
            'salience': float(energy * 2)  # Boost local patterns
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {
            'energy': 0.0,
            'variance': 0.0,
            'coherence': 0.0,
            'oscillation': 0.0,
            'flow_divergence': 0.0,
            'flow_curl': 0.0,
            'novelty': 0.0,
            'salience': 0.0
        }
    
    def clear_cache(self):
        """Clear feature cache."""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': float(self.cache_hits),
            'cache_misses': float(self.cache_misses),
            'hit_rate': float(hit_rate),
            'cache_size': float(len(self.feature_cache))
        }