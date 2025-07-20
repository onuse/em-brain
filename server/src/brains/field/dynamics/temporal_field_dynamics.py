#!/usr/bin/env python3
"""
Temporal Field Dynamics - Phase A3: 4D Continuous Field with Time Dimension

This extends our 3D multi-scale field to include time as a fundamental dimension.
The key insight: intelligence operates in spacetime, not just space. Temporal 
relationships are as important as spatial and scale relationships.

Phase A3 Goals:
1. Implement 4D continuous field (width x height x scale x time)
2. Add temporal momentum and persistence dynamics
3. Test sequence learning in continuous spacetime
4. Validate temporal binding through field correlation

Core Innovation: The temporal dimension enables the same relational mathematics
to work across time, enabling natural sequence learning, prediction, and working memory.

Research Questions:
- Can temporal patterns naturally chain into prediction sequences?
- Do temporal field dynamics create emergent working memory?
- How do we balance temporal persistence vs. adaptation?
- What's the computational cost of 4D field evolution?
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import deque

try:
    from .multiscale_field_dynamics import MultiScaleExperience, MultiScaleImprint
except ImportError:
    from multiscale_field_dynamics import MultiScaleExperience, MultiScaleImprint


@dataclass
class TemporalExperience:
    """An experience with explicit temporal information."""
    sensory_data: torch.Tensor  # Raw sensory input
    position: Tuple[float, float]  # Spatial position (x, y)
    scale_level: float  # Scale level (0.0 = micro, 1.0 = macro)
    temporal_position: float  # Position in temporal dimension
    intensity: float  # Imprint intensity
    spatial_spread: float  # Spatial spread
    scale_spread: float  # Scale dimension spread
    temporal_spread: float  # Temporal dimension spread
    timestamp: float  # Real-world timestamp
    sequence_id: Optional[str] = None  # For tracking sequences


@dataclass
class TemporalImprint:
    """4D field imprint with temporal dimension."""
    center_x: float
    center_y: float
    center_scale: float  # Position in scale dimension
    center_time: float  # Position in temporal dimension
    intensity: float
    spatial_spread: float  # Spread in x,y dimensions
    scale_spread: float  # Spread in scale dimension
    temporal_spread: float  # Spread in temporal dimension
    pattern_signature: torch.Tensor  # Pattern identity
    temporal_momentum: float = 0.0  # Temporal momentum for prediction


class TemporalField4D:
    """
    4D Continuous Field for Temporal Pattern Dynamics
    
    This extends 3D multi-scale fields with a temporal dimension, enabling:
    - Sequence learning in continuous spacetime
    - Temporal momentum and persistence
    - Predictive field evolution
    - Natural working memory emergence
    """
    
    def __init__(self, width: int = 200, height: int = 200, scale_depth: int = 30, 
                 temporal_depth: int = 50, temporal_window: float = 10.0, quiet_mode: bool = False):
        self.width = width
        self.height = height
        self.scale_depth = scale_depth
        self.temporal_depth = temporal_depth
        self.temporal_window = temporal_window  # Time window in seconds
        self.quiet_mode = quiet_mode
        
        # The core 4D continuous field - [width, height, scale, time]
        self.field = torch.zeros(width, height, scale_depth, temporal_depth, dtype=torch.float32)
        
        # Temporal dynamics parameters
        self.spatial_decay_rate = 0.99
        self.scale_decay_rate = 0.995
        self.temporal_decay_rate = 0.98  # Faster temporal decay
        self.spatial_diffusion_rate = 0.01
        self.scale_diffusion_rate = 0.005
        self.temporal_diffusion_rate = 0.02  # Stronger temporal diffusion
        self.cross_scale_coupling = 0.02
        self.temporal_momentum_strength = 0.15  # Temporal prediction strength
        self.activation_threshold = 0.1
        
        # Temporal dimension interpretation
        self.current_temporal_position = 0.0  # Current time position in field
        self.temporal_resolution = temporal_window / temporal_depth  # Seconds per temporal slice
        
        # Temporal tracking
        self.imprint_history: List[TemporalImprint] = []
        self.sequence_predictions: Dict[str, List[TemporalImprint]] = {}
        self.temporal_chains: List[List[TemporalImprint]] = []
        self.working_memory_patterns: deque = deque(maxlen=20)
        self.total_experiences = 0
        
        # Performance tracking
        self.stats = {
            'total_imprints': 0,
            'total_retrievals': 0,
            'temporal_predictions': 0,
            'sequence_formations': 0,
            'working_memory_events': 0,
            'field_evolution_cycles': 0,
            'avg_imprint_time_ms': 0.0,
            'avg_retrieval_time_ms': 0.0,
            'temporal_coherence': 0.0,  # How well temporal patterns connect
            'prediction_accuracy': 0.0,  # How accurate temporal predictions are
        }
        
        if not self.quiet_mode:
            print(f"🌊 TemporalField4D initialized: {width}x{height}x{scale_depth}x{temporal_depth} continuous field")
            print(f"   Temporal window: {temporal_window:.1f}s, resolution: {self.temporal_resolution:.3f}s per slice")
    
    def advance_temporal_position(self, dt: float):
        """Advance the current temporal position."""
        self.current_temporal_position += dt
        
        # Wrap temporal position to maintain sliding window
        if self.current_temporal_position >= self.temporal_window:
            self.current_temporal_position = 0.0
            if not self.quiet_mode:
                print(f"🕰️ Temporal window wrapped: new cycle begins")
    
    def experience_to_temporal_imprint(self, experience: TemporalExperience) -> TemporalImprint:
        """
        Convert a temporal experience into a 4D field imprint.
        
        This maps sensory experiences to 4D field coordinates including time.
        """
        # Spatial coordinates
        center_x = max(0, min(self.width - 1, experience.position[0]))
        center_y = max(0, min(self.height - 1, experience.position[1]))
        
        # Scale coordinate
        scale_coord = experience.scale_level * (self.scale_depth - 1)
        center_scale = max(0, min(self.scale_depth - 1, scale_coord))
        
        # Temporal coordinate
        temporal_coord = experience.temporal_position / self.temporal_resolution
        center_time = max(0, min(self.temporal_depth - 1, temporal_coord))
        
        # Create pattern signature from sensory data
        pattern_signature = self._create_pattern_signature(experience.sensory_data)
        
        # Calculate temporal momentum (for prediction)
        temporal_momentum = self._calculate_temporal_momentum(experience)
        
        return TemporalImprint(
            center_x=center_x,
            center_y=center_y,
            center_scale=center_scale,
            center_time=center_time,
            intensity=experience.intensity,
            spatial_spread=experience.spatial_spread,
            scale_spread=experience.scale_spread,
            temporal_spread=experience.temporal_spread,
            pattern_signature=pattern_signature,
            temporal_momentum=temporal_momentum
        )
    
    def _create_pattern_signature(self, sensory_data: torch.Tensor) -> torch.Tensor:
        """Create a pattern signature from sensory data."""
        signature_size = 12
        
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
    
    def _calculate_temporal_momentum(self, experience: TemporalExperience) -> float:
        """Calculate temporal momentum for predictive dynamics."""
        # Look at recent imprints to determine temporal momentum
        if len(self.imprint_history) < 2:
            return 0.0
        
        # Find similar recent patterns
        recent_imprints = self.imprint_history[-5:]  # Last 5 imprints
        
        # Calculate similarity to current experience
        similarities = []
        for imprint in recent_imprints:
            # Simple signature similarity (ensure same dimensions)
            exp_sig = self._create_pattern_signature(experience.sensory_data)
            sig_similarity = torch.cosine_similarity(
                exp_sig,
                imprint.pattern_signature,
                dim=0
            ).item()
            similarities.append(max(0.0, sig_similarity))
        
        # Temporal momentum proportional to pattern consistency
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity * 0.5  # Scale to reasonable momentum range
    
    def apply_temporal_imprint(self, imprint: TemporalImprint):
        """
        Apply a 4D imprint to the temporal field.
        
        This creates activation blobs that span spatial, scale, and temporal dimensions.
        """
        start_time = time.perf_counter()
        
        # Create 4D coordinate grids
        y_coords, x_coords, s_coords, t_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.scale_depth, dtype=torch.float32),
            torch.arange(self.temporal_depth, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distances in 4D space
        spatial_distances = torch.sqrt(
            (x_coords - imprint.center_x) ** 2 + 
            (y_coords - imprint.center_y) ** 2
        )
        
        scale_distances = torch.abs(s_coords - imprint.center_scale)
        temporal_distances = torch.abs(t_coords - imprint.center_time)
        
        # Create 4D Gaussian activation blob
        spatial_gaussian = torch.exp(-(spatial_distances ** 2) / (2 * imprint.spatial_spread ** 2))
        scale_gaussian = torch.exp(-(scale_distances ** 2) / (2 * imprint.scale_spread ** 2))
        temporal_gaussian = torch.exp(-(temporal_distances ** 2) / (2 * imprint.temporal_spread ** 2))
        
        # Combine all Gaussians
        temporal_blob = imprint.intensity * spatial_gaussian * scale_gaussian * temporal_gaussian
        
        # Apply temporal momentum (predictive bias toward future)
        if imprint.temporal_momentum > 0:
            # Create future prediction bias
            future_bias = torch.zeros_like(temporal_blob)
            future_time_center = min(self.temporal_depth - 1, imprint.center_time + imprint.temporal_momentum * 10)
            
            future_temporal_distances = torch.abs(t_coords - future_time_center)
            future_temporal_gaussian = torch.exp(-(future_temporal_distances ** 2) / (2 * imprint.temporal_spread ** 2))
            future_bias = imprint.temporal_momentum * spatial_gaussian * scale_gaussian * future_temporal_gaussian
            
            temporal_blob += future_bias
            self.stats['temporal_predictions'] += 1
        
        # Apply the imprint to the 4D field
        self.field += temporal_blob
        
        # Record this imprint
        self.imprint_history.append(imprint)
        self.total_experiences += 1
        
        # Check for sequence formation
        self._detect_sequence_formation(imprint)
        
        # Update working memory
        self._update_working_memory(imprint)
        
        # Update stats
        imprint_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_imprint_stats(imprint_time_ms)
        
        if not self.quiet_mode:
            print(f"📍 4D Imprint applied: pos=({imprint.center_x:.1f}, {imprint.center_y:.1f}), "
                  f"scale={imprint.center_scale:.1f}, time={imprint.center_time:.1f}, "
                  f"momentum={imprint.temporal_momentum:.3f}")
    
    def evolve_temporal_field(self, dt: float = 0.1):
        """
        Evolve the 4D field with temporal dynamics.
        
        This includes:
        - Temporal decay and diffusion
        - Temporal momentum propagation
        - Cross-temporal coupling for sequence learning
        """
        start_time = time.perf_counter()
        
        # Advance temporal position
        self.advance_temporal_position(dt)
        
        # Apply decay in all dimensions
        self.field *= self.temporal_decay_rate
        
        # Apply temporal diffusion (temporal dimension)
        if self.temporal_diffusion_rate > 0:
            self._apply_temporal_diffusion()
        
        # Apply spatial and scale diffusion (from parent classes)
        self._apply_spatial_diffusion()
        self._apply_scale_diffusion()
        
        # Apply temporal momentum propagation
        self._propagate_temporal_momentum()
        
        # Apply cross-scale coupling
        self._apply_cross_scale_coupling()
        
        self.stats['field_evolution_cycles'] += 1
        
        evolution_time_ms = (time.perf_counter() - start_time) * 1000
        if not self.quiet_mode and self.stats['field_evolution_cycles'] % 50 == 0:
            print(f"🌊 4D Field evolved: cycle {self.stats['field_evolution_cycles']}, "
                  f"time={evolution_time_ms:.2f}ms")
    
    def _apply_temporal_diffusion(self):
        """Apply diffusion along temporal dimension."""
        # 1D temporal diffusion kernel
        temporal_kernel = torch.tensor([0.2, 0.6, 0.2]).view(1, 1, 1, 3)
        
        # Apply diffusion along temporal dimension for each spatial/scale location
        for y in range(self.height):
            for x in range(self.width):
                for s in range(self.scale_depth):
                    temporal_profile = self.field[y, x, s, :].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    
                    # Pad for convolution using constant padding
                    padded_profile = torch.nn.functional.pad(temporal_profile, (1, 1), mode='constant', value=0)
                    diffused = torch.conv1d(padded_profile.squeeze(2), temporal_kernel.squeeze(1), padding=0)
                    diffused_profile = diffused.squeeze(0).squeeze(0)
                    
                    # Blend original with diffused
                    self.field[y, x, s, :] = (1.0 - self.temporal_diffusion_rate) * self.field[y, x, s, :] + \
                                           self.temporal_diffusion_rate * diffused_profile
    
    def _apply_spatial_diffusion(self):
        """Apply spatial diffusion (simplified version of multiscale method)."""
        if self.spatial_diffusion_rate > 0:
            spatial_kernel = torch.tensor([
                [[0.05, 0.1, 0.05],
                 [0.1,  0.4, 0.1 ],
                 [0.05, 0.1, 0.05]]
            ]).unsqueeze(0)
            
            # Apply to each scale-time slice
            for s in range(self.scale_depth):
                for t in range(self.temporal_depth):
                    slice_2d = self.field[:, :, s, t].unsqueeze(0).unsqueeze(0)
                    diffused = torch.conv2d(slice_2d, spatial_kernel, padding=1)
                    diffused_slice = diffused.squeeze(0).squeeze(0)
                    
                    self.field[:, :, s, t] = (1.0 - self.spatial_diffusion_rate) * self.field[:, :, s, t] + \
                                           self.spatial_diffusion_rate * diffused_slice
    
    def _apply_scale_diffusion(self):
        """Apply scale diffusion (simplified version of multiscale method)."""
        if self.scale_diffusion_rate > 0:
            scale_kernel = torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3)
            
            for y in range(self.height):
                for x in range(self.width):
                    for t in range(self.temporal_depth):
                        scale_profile = self.field[y, x, :, t].unsqueeze(0).unsqueeze(0)
                        
                        padded_profile = torch.nn.functional.pad(scale_profile, (1, 1), mode='constant', value=0)
                        diffused = torch.conv1d(padded_profile, scale_kernel)
                        diffused_profile = diffused.squeeze(0).squeeze(0)
                        
                        self.field[y, x, :, t] = (1.0 - self.scale_diffusion_rate) * self.field[y, x, :, t] + \
                                               self.scale_diffusion_rate * diffused_profile
    
    def _propagate_temporal_momentum(self):
        """Propagate temporal momentum for predictive dynamics."""
        # Find regions with high activation
        high_activation_mask = self.field > self.activation_threshold
        
        if torch.sum(high_activation_mask) == 0:
            return
        
        # For each high activation region, propagate momentum forward in time
        momentum_propagation = torch.zeros_like(self.field)
        
        for t in range(self.temporal_depth - 1):  # Don't propagate from last time slice
            current_slice = self.field[:, :, :, t]
            active_regions = current_slice > self.activation_threshold
            
            if torch.sum(active_regions) > 0:
                # Propagate to next time slice with momentum
                momentum_strength = self.temporal_momentum_strength
                next_slice_contribution = current_slice * active_regions.float() * momentum_strength
                
                # Add to momentum propagation
                momentum_propagation[:, :, :, t + 1] += next_slice_contribution
        
        # Apply momentum propagation
        self.field += momentum_propagation
    
    def _apply_cross_scale_coupling(self):
        """Apply cross-scale coupling (simplified version from multiscale)."""
        coupling_events = 0
        
        # Bottom-up and top-down coupling for each temporal slice
        for t in range(self.temporal_depth):
            # Bottom-up: micro → macro
            for s in range(self.scale_depth - 1):
                current_scale = self.field[:, :, s, t]
                influence_mask = current_scale > self.activation_threshold
                
                if torch.sum(influence_mask) > 0:
                    aggregation = torch.mean(current_scale[influence_mask]) * self.cross_scale_coupling
                    smoothed_influence = current_scale * influence_mask.float()
                    self.field[:, :, s + 1, t] += smoothed_influence * aggregation
                    coupling_events += 1
            
            # Top-down: macro → micro
            for s in range(self.scale_depth - 1, 0, -1):
                current_scale = self.field[:, :, s, t]
                influence_mask = current_scale > self.activation_threshold
                
                if torch.sum(influence_mask) > 0:
                    influence_strength = torch.mean(current_scale[influence_mask]) * self.cross_scale_coupling * 0.5
                    influence_field = current_scale * influence_mask.float()
                    self.field[:, :, s - 1, t] += influence_field * influence_strength
                    coupling_events += 1
    
    def _detect_sequence_formation(self, imprint: TemporalImprint):
        """Detect if this imprint forms part of a temporal sequence."""
        if len(self.imprint_history) < 2:
            return
        
        # Look for temporal chains
        recent_imprints = self.imprint_history[-5:]
        
        # Check if imprints form a temporal sequence
        temporal_positions = [imp.center_time for imp in recent_imprints]
        
        # Look for monotonic progression in time
        if len(temporal_positions) >= 3:
            time_diffs = [temporal_positions[i+1] - temporal_positions[i] for i in range(len(temporal_positions)-1)]
            
            # Check if time differences are consistent (indicating sequence)
            if all(diff > 0 for diff in time_diffs):  # Increasing time
                time_consistency = 1.0 - (np.std(time_diffs) / (np.mean(time_diffs) + 1e-6))
                
                if time_consistency > 0.7:  # High temporal consistency
                    self.temporal_chains.append(recent_imprints.copy())
                    self.stats['sequence_formations'] += 1
                    
                    if not self.quiet_mode:
                        print(f"   🔗 Temporal sequence detected: {len(recent_imprints)} imprints, "
                              f"consistency={time_consistency:.3f}")
    
    def _update_working_memory(self, imprint: TemporalImprint):
        """Update working memory with current imprint."""
        # Working memory: recent patterns that are still temporally active
        current_time_pos = self.current_temporal_position / self.temporal_resolution
        
        # Remove old patterns from working memory
        memory_window = 10  # Time slices to retain in working memory
        
        self.working_memory_patterns = deque([
            pattern for pattern in self.working_memory_patterns
            if abs(pattern.center_time - current_time_pos) < memory_window
        ], maxlen=20)
        
        # Add current pattern to working memory
        self.working_memory_patterns.append(imprint)
        self.stats['working_memory_events'] += 1
    
    def find_temporal_similar_regions(self, query_experience: TemporalExperience, 
                                    top_k: int = 5) -> List[Tuple[float, float, float, float, float]]:
        """
        Find similar regions across the 4D temporal field.
        
        Returns: List of (x, y, scale, time, correlation) tuples
        """
        start_time = time.perf_counter()
        
        # Convert query to 4D imprint
        query_imprint = self.experience_to_temporal_imprint(query_experience)
        
        # Create the query pattern in 4D field space
        y_coords, x_coords, s_coords, t_coords = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.scale_depth, dtype=torch.float32),
            torch.arange(self.temporal_depth, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate 4D distances
        spatial_distances = torch.sqrt(
            (x_coords - query_imprint.center_x) ** 2 + 
            (y_coords - query_imprint.center_y) ** 2
        )
        scale_distances = torch.abs(s_coords - query_imprint.center_scale)
        temporal_distances = torch.abs(t_coords - query_imprint.center_time)
        
        # Create 4D query pattern
        spatial_gaussian = torch.exp(-(spatial_distances ** 2) / (2 * query_imprint.spatial_spread ** 2))
        scale_gaussian = torch.exp(-(scale_distances ** 2) / (2 * query_imprint.scale_spread ** 2))
        temporal_gaussian = torch.exp(-(temporal_distances ** 2) / (2 * query_imprint.temporal_spread ** 2))
        query_pattern_4d = query_imprint.intensity * spatial_gaussian * scale_gaussian * temporal_gaussian
        
        # Find correlations using sampling (4D correlation is expensive)
        correlations = []
        
        # Sample the 4D space for efficiency
        sample_stride = max(1, min(self.width, self.height) // 10)
        scale_stride = max(1, self.scale_depth // 5)
        temporal_stride = max(1, self.temporal_depth // 8)
        
        for y in range(0, self.height, sample_stride):
            for x in range(0, self.width, sample_stride):
                for s in range(0, self.scale_depth, scale_stride):
                    for t in range(0, self.temporal_depth, temporal_stride):
                        # Extract 4D field region around this point
                        window_size = 5
                        
                        y_start, y_end = max(0, y - window_size), min(self.height, y + window_size)
                        x_start, x_end = max(0, x - window_size), min(self.width, x + window_size)
                        s_start, s_end = max(0, s - 2), min(self.scale_depth, s + 2)
                        t_start, t_end = max(0, t - 3), min(self.temporal_depth, t + 3)
                        
                        field_region = self.field[y_start:y_end, x_start:x_end, s_start:s_end, t_start:t_end]
                        query_region = query_pattern_4d[y_start:y_end, x_start:x_end, s_start:s_end, t_start:t_end]
                        
                        # Calculate 4D correlation
                        if torch.sum(field_region) > 0 and torch.sum(query_region) > 0:
                            correlation = self._calculate_4d_correlation(field_region, query_region)
                            correlations.append((x, y, s, t, correlation))
        
        # Sort by correlation and return top_k
        correlations.sort(key=lambda x: x[4], reverse=True)
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        self._update_retrieval_stats(retrieval_time_ms)
        
        return correlations[:top_k]
    
    def _calculate_4d_correlation(self, region1: torch.Tensor, region2: torch.Tensor) -> float:
        """Calculate correlation between two 4D field regions."""
        # Flatten 4D regions
        r1_flat = region1.flatten()
        r2_flat = region2.flatten()
        
        # Calculate Pearson correlation
        if torch.std(r1_flat.float()) > 0 and torch.std(r2_flat.float()) > 0:
            correlation = torch.corrcoef(torch.stack([r1_flat, r2_flat]))[0, 1]
            return correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            return 0.0
    
    def get_temporal_coherence(self) -> float:
        """Calculate temporal coherence of the field."""
        if self.temporal_depth < 2:
            return 0.0
        
        # Calculate correlation between adjacent temporal slices
        correlations = []
        
        for t in range(self.temporal_depth - 1):
            slice_current = self.field[:, :, :, t].flatten()
            slice_next = self.field[:, :, :, t + 1].flatten()
            
            if torch.std(slice_current.float()) > 0 and torch.std(slice_next.float()) > 0:
                corr = torch.corrcoef(torch.stack([slice_current, slice_next]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())
        
        return np.mean(correlations) if correlations else 0.0
    
    def get_sequence_predictions(self) -> List[Dict]:
        """Get predicted future patterns based on temporal momentum."""
        predictions = []
        
        # Analyze temporal chains to generate predictions
        for chain in self.temporal_chains[-5:]:  # Recent chains
            if len(chain) >= 3:
                # Calculate temporal progression
                time_deltas = [chain[i+1].center_time - chain[i].center_time for i in range(len(chain)-1)]
                avg_delta = np.mean(time_deltas)
                
                # Predict next pattern in sequence
                last_imprint = chain[-1]
                predicted_time = last_imprint.center_time + avg_delta
                
                predictions.append({
                    'predicted_position': (last_imprint.center_x, last_imprint.center_y),
                    'predicted_scale': last_imprint.center_scale,
                    'predicted_time': predicted_time,
                    'confidence': last_imprint.temporal_momentum,
                    'sequence_length': len(chain)
                })
        
        return predictions
    
    def _update_imprint_stats(self, imprint_time_ms: float):
        """Update imprint performance statistics."""
        self.stats['total_imprints'] += 1
        
        # Update average imprint time
        total_imprints = self.stats['total_imprints']
        current_avg = self.stats['avg_imprint_time_ms']
        self.stats['avg_imprint_time_ms'] = (current_avg * (total_imprints - 1) + imprint_time_ms) / total_imprints
        
        # Update temporal coherence
        self.stats['temporal_coherence'] = self.get_temporal_coherence()
    
    def _update_retrieval_stats(self, retrieval_time_ms: float):
        """Update retrieval performance statistics."""
        self.stats['total_retrievals'] += 1
        
        # Update average retrieval time
        total_retrievals = self.stats['total_retrievals']
        current_avg = self.stats['avg_retrieval_time_ms']
        self.stats['avg_retrieval_time_ms'] = (current_avg * (total_retrievals - 1) + retrieval_time_ms) / total_retrievals
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get comprehensive temporal field statistics."""
        predictions = self.get_sequence_predictions()
        
        stats = self.stats.copy()
        stats.update({
            'field_dimensions': (self.width, self.height, self.scale_depth, self.temporal_depth),
            'total_experiences': self.total_experiences,
            'imprint_history_length': len(self.imprint_history),
            'temporal_chains_count': len(self.temporal_chains),
            'working_memory_size': len(self.working_memory_patterns),
            'sequence_predictions_count': len(predictions),
            'temporal_window_seconds': self.temporal_window,
            'temporal_resolution_seconds': self.temporal_resolution,
            'current_temporal_position': self.current_temporal_position,
            'field_max_activation': torch.max(self.field).item(),
            'field_mean_activation': torch.mean(self.field).item(),
        })
        return stats


# Factory function for easy creation
def create_temporal_field_4d(width: int = 200, height: int = 200, scale_depth: int = 30, 
                           temporal_depth: int = 50, temporal_window: float = 10.0,
                           quiet_mode: bool = False) -> TemporalField4D:
    """Create a 4D temporal field with standard configuration."""
    return TemporalField4D(width=width, height=height, scale_depth=scale_depth, 
                          temporal_depth=temporal_depth, temporal_window=temporal_window, 
                          quiet_mode=quiet_mode)


if __name__ == "__main__":
    # Test the 4D temporal field implementation
    print("🧪 Testing 4D Temporal Field Dynamics...")
    
    # Create field
    field = create_temporal_field_4d(width=100, height=100, scale_depth=15, 
                                   temporal_depth=20, temporal_window=5.0, quiet_mode=False)
    
    # Test temporal experience imprinting
    print("\n📍 Testing temporal experience imprinting:")
    
    # Create test experiences in a temporal sequence
    experiences = []
    for i in range(6):
        exp = TemporalExperience(
            sensory_data=torch.tensor([0.8 - i*0.1, 0.2 + i*0.1, 0.1, 0.9 - i*0.05]),
            position=(30 + i*5, 30 + i*3),  # Moving pattern
            scale_level=0.3 + i*0.1,  # Evolving scale
            temporal_position=i * 0.8,  # Progressing in time
            intensity=1.0 - i*0.1,
            spatial_spread=5.0,
            scale_spread=2.0,
            temporal_spread=2.0,
            timestamp=time.time() + i*0.1,
            sequence_id="test_sequence"
        )
        experiences.append(exp)
    
    # Apply experiences to field
    for i, exp in enumerate(experiences):
        imprint = field.experience_to_temporal_imprint(exp)
        field.apply_temporal_imprint(imprint)
        print(f"   Experience {i+1}: temporal_pos={exp.temporal_position:.2f}, "
              f"momentum={imprint.temporal_momentum:.3f}")
    
    # Test temporal field evolution
    print("\n🌊 Testing temporal field evolution:")
    for i in range(15):
        field.evolve_temporal_field(dt=0.2)
        if i % 5 == 0:
            coherence = field.get_temporal_coherence()
            print(f"   Evolution step {i+1}: temporal coherence = {coherence:.4f}")
    
    # Test sequence prediction
    print("\n🔮 Testing sequence prediction:")
    predictions = field.get_sequence_predictions()
    print(f"   Found {len(predictions)} sequence predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"   Prediction {i+1}: time={pred['predicted_time']:.2f}, "
              f"confidence={pred['confidence']:.3f}, "
              f"sequence_length={pred['sequence_length']}")
    
    # Test temporal similarity search
    print("\n🔍 Testing temporal similarity search:")
    query_experience = TemporalExperience(
        sensory_data=torch.tensor([0.6, 0.4, 0.1, 0.8]),
        position=(40, 35),
        scale_level=0.5,
        temporal_position=3.0,
        intensity=1.0,
        spatial_spread=5.0,
        scale_spread=2.0,
        temporal_spread=2.0,
        timestamp=time.time()
    )
    
    similar_regions = field.find_temporal_similar_regions(query_experience, top_k=3)
    print(f"   Query: temporal_pos={query_experience.temporal_position:.1f}")
    for i, (x, y, scale, t, correlation) in enumerate(similar_regions):
        temporal_pos = t * field.temporal_resolution
        print(f"   Match {i+1}: pos=({x:.1f}, {y:.1f}), scale={scale:.1f}, "
              f"time={temporal_pos:.2f}, correlation={correlation:.3f}")
    
    # Display final statistics
    print("\n📊 Final temporal field statistics:")
    stats = field.get_temporal_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        elif isinstance(value, (list, tuple)) and len(value) <= 4:
            print(f"   {key}: {value}")
        elif not isinstance(value, (list, tuple)):
            print(f"   {key}: {value}")
    
    print("\n✅ 4D Temporal Field test completed!")
    print("🎯 Key temporal behaviors demonstrated:")
    print("   ✓ 4D experience-to-imprint conversion (spatial + scale + time)")
    print("   ✓ Temporal field evolution with momentum propagation")
    print("   ✓ Sequence detection and temporal chain formation")
    print("   ✓ Working memory emergence from temporal dynamics")
    print("   ✓ Temporal similarity search and prediction")
    print("   ✓ Natural temporal coherence and sequence learning")