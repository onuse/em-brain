#!/usr/bin/env python3
"""
Attention-Guided Super-Resolution

Implements dynamic spatial resolution that increases resolution in attention regions
while maintaining efficient baseline resolution elsewhere. Mimics biological
attention/interest mechanisms.

Key Features:
1. Baseline resolution (e.g., 50³) for general spatial awareness
2. Super-resolution patches (e.g., 100³) for attention regions
3. Dynamic attention detection based on field activity and gradients
4. Efficient field coupling between resolutions
5. Interest-driven attention shifting
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque

try:
    from .adaptive_field_impl import FieldImplementation
    from .attention_guided import AttentionGuidedProcessing
except ImportError:
    from adaptive_field_impl import FieldImplementation
    from attention_guided import AttentionGuidedProcessing


@dataclass
class SuperResolutionConfig:
    """Configuration for attention-guided super-resolution."""
    enable_super_resolution: bool = True
    base_resolution: int = 50               # Baseline field resolution
    focus_resolution: int = 100             # Super-resolution for attention regions
    max_focus_regions: int = 3              # Maximum concurrent high-res regions
    attention_threshold: float = 0.3        # Minimum activity to trigger focus
    interest_decay_rate: float = 0.95       # How quickly interest fades
    focus_region_size: float = 0.3          # Size of focus regions (0.0-1.0)
    resolution_blend_rate: float = 0.1      # How fast resolution changes
    min_focus_duration: int = 5             # Minimum cycles to maintain focus


@dataclass
class AttentionRegion:
    """Represents a high-resolution attention region."""
    center_x: float
    center_y: float
    center_z: float
    size: float                 # Region size (0.0-1.0)
    intensity: float           # Attention intensity
    age: int                   # How long this region has been active
    interest_level: float      # Interest level (drives attention duration)
    high_res_field: Optional[torch.Tensor] = None  # Local high-res field


class AttentionSuperResolution:
    """
    Attention-Guided Super-Resolution System
    
    Provides dynamic spatial resolution that increases where attention is focused,
    mimicking biological vision and attention systems.
    """
    
    def __init__(self, base_field: FieldImplementation,
                 attention_processor: Optional[AttentionGuidedProcessing] = None,
                 config: Optional[SuperResolutionConfig] = None,
                 quiet_mode: bool = False):
        
        self.base_field = base_field
        self.attention_processor = attention_processor
        self.config = config or SuperResolutionConfig()
        self.quiet_mode = quiet_mode
        
        # Active attention regions with super-resolution
        self.attention_regions: List[AttentionRegion] = []
        
        # Interest tracking for attention guidance
        self.interest_map = torch.zeros(
            self.config.base_resolution, 
            self.config.base_resolution, 
            self.config.base_resolution
        )
        
        # Resolution blending state
        self.current_resolution_map = torch.full(
            (self.config.base_resolution,) * 3, 
            float(self.config.base_resolution)
        )
        
        # Performance tracking
        self.stats = {
            'total_focus_regions_created': 0,
            'total_super_resolution_cycles': 0,
            'avg_focus_regions_active': 0.0,
            'interest_peaks_detected': 0,
            'attention_shifts': 0,
            'super_resolution_overhead_ms': 0.0
        }
        
        if not quiet_mode:
            print(f"🔍 Attention Super-Resolution initialized")
            print(f"   Base resolution: {self.config.base_resolution}³")
            print(f"   Focus resolution: {self.config.focus_resolution}³")
            print(f"   Max focus regions: {self.config.max_focus_regions}")
    
    def detect_interest_regions(self, field_gradients: Dict[str, torch.Tensor]) -> List[Tuple[float, float, float, float]]:
        """
        Detect regions of high interest based on field activity and gradients.
        
        Returns: List of (x, y, z, intensity) tuples for interesting regions
        """
        if not field_gradients:
            return []
        
        interest_candidates = []
        
        # Analyze field gradients to find regions of high activity
        for grad_name, grad_tensor in field_gradients.items():
            if grad_tensor.numel() == 0 or grad_tensor.dim() < 3:
                continue
            
            # Downsample gradient to base resolution if needed
            if grad_tensor.shape[0] != self.config.base_resolution:
                # Simple downsampling - take every nth element
                stride = max(1, grad_tensor.shape[0] // self.config.base_resolution)
                grad_tensor = grad_tensor[::stride, ::stride, ::stride]
            
            # Ensure we don't exceed base resolution
            max_size = self.config.base_resolution
            grad_tensor = grad_tensor[:max_size, :max_size, :max_size]
            
            # Find local maxima in gradient magnitude
            grad_magnitude = torch.abs(grad_tensor)
            
            # Apply smoothing to avoid noise
            if grad_magnitude.dim() == 3 and grad_magnitude.numel() > 27:
                # Simple 3x3x3 averaging
                smoothed = torch.zeros_like(grad_magnitude)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            x_shift = max(0, min(grad_magnitude.shape[0]-1, slice(max(0, dx), grad_magnitude.shape[0] + min(0, dx))))
                            y_shift = max(0, min(grad_magnitude.shape[1]-1, slice(max(0, dy), grad_magnitude.shape[1] + min(0, dy))))
                            z_shift = max(0, min(grad_magnitude.shape[2]-1, slice(max(0, dz), grad_magnitude.shape[2] + min(0, dz))))
                            
                            if dx == 0 and dy == 0 and dz == 0:
                                smoothed += grad_magnitude * 0.4  # Center weight
                            else:
                                try:
                                    if dx == 0:
                                        x_slice = slice(None)
                                    else:
                                        x_slice = slice(max(0, -dx), grad_magnitude.shape[0] - max(0, dx))
                                    if dy == 0:
                                        y_slice = slice(None)
                                    else:
                                        y_slice = slice(max(0, -dy), grad_magnitude.shape[1] - max(0, dy))
                                    if dz == 0:
                                        z_slice = slice(None)
                                    else:
                                        z_slice = slice(max(0, -dz), grad_magnitude.shape[2] - max(0, dz))
                                    
                                    shifted = grad_magnitude[x_slice, y_slice, z_slice]
                                    
                                    # Pad to match original size
                                    pad_x = grad_magnitude.shape[0] - shifted.shape[0]
                                    pad_y = grad_magnitude.shape[1] - shifted.shape[1]
                                    pad_z = grad_magnitude.shape[2] - shifted.shape[2]
                                    
                                    if pad_x > 0 or pad_y > 0 or pad_z > 0:
                                        shifted = torch.nn.functional.pad(shifted, 
                                            (0, pad_z, 0, pad_y, 0, pad_x), mode='constant', value=0)
                                    
                                    smoothed += shifted * 0.6 / 26  # Neighbor weight
                                except:
                                    continue  # Skip if indexing fails
                
                grad_magnitude = smoothed
            
            # Find regions above interest threshold
            threshold = torch.quantile(grad_magnitude, 0.8)  # Top 20% of activity
            high_activity = grad_magnitude > max(threshold, self.config.attention_threshold)
            
            if torch.sum(high_activity) > 0:
                # Find connected components of high activity
                active_coords = torch.nonzero(high_activity)
                
                if len(active_coords) > 0:
                    # Group nearby coordinates
                    for coord in active_coords:
                        x, y, z = coord.tolist()
                        intensity = grad_magnitude[x, y, z].item()
                        
                        # Convert to normalized coordinates (0.0-1.0)
                        norm_x = x / max(1, grad_magnitude.shape[0] - 1)
                        norm_y = y / max(1, grad_magnitude.shape[1] - 1)
                        norm_z = z / max(1, grad_magnitude.shape[2] - 1)
                        
                        interest_candidates.append((norm_x, norm_y, norm_z, intensity))
        
        # Merge nearby candidates and sort by intensity
        merged_candidates = self._merge_nearby_candidates(interest_candidates)
        merged_candidates.sort(key=lambda x: x[3], reverse=True)  # Sort by intensity
        
        return merged_candidates[:self.config.max_focus_regions * 2]  # Return extra candidates
    
    def _merge_nearby_candidates(self, candidates: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Merge candidates that are close to each other."""
        if not candidates:
            return []
        
        merged = []
        merge_distance = 0.2  # Merge candidates within 20% of field size
        
        for x, y, z, intensity in candidates:
            # Check if this candidate is close to an existing merged candidate
            merged_with_existing = False
            
            for i, (mx, my, mz, mi) in enumerate(merged):
                distance = np.sqrt((x - mx)**2 + (y - my)**2 + (z - mz)**2)
                
                if distance < merge_distance:
                    # Merge by taking intensity-weighted average
                    total_intensity = intensity + mi
                    new_x = (x * intensity + mx * mi) / total_intensity
                    new_y = (y * intensity + my * mi) / total_intensity  
                    new_z = (z * intensity + mz * mi) / total_intensity
                    
                    merged[i] = (new_x, new_y, new_z, total_intensity)
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append((x, y, z, intensity))
        
        return merged
    
    def update_attention_regions(self, interest_regions: List[Tuple[float, float, float, float]]) -> None:
        """Update attention regions based on detected interest."""
        current_time = time.time()
        
        # Age existing regions and decay interest
        for region in self.attention_regions:
            region.age += 1
            region.interest_level *= self.config.interest_decay_rate
        
        # Remove regions that have lost interest or are too old
        min_duration = self.config.min_focus_duration
        self.attention_regions = [
            region for region in self.attention_regions
            if (region.interest_level > 0.1 and region.age < min_duration) or
               (region.interest_level > 0.05 and region.age >= min_duration)
        ]
        
        # Add new attention regions for high-interest areas
        for x, y, z, intensity in interest_regions:
            if len(self.attention_regions) >= self.config.max_focus_regions:
                break
            
            # Check if this region overlaps with existing attention regions
            overlaps_existing = False
            for existing in self.attention_regions:
                distance = np.sqrt(
                    (x - existing.center_x)**2 + 
                    (y - existing.center_y)**2 + 
                    (z - existing.center_z)**2
                )
                if distance < self.config.focus_region_size:
                    # Update existing region instead of creating new one
                    existing.intensity = max(existing.intensity, intensity)
                    existing.interest_level = min(1.0, existing.interest_level + 0.2)
                    overlaps_existing = True
                    break
            
            if not overlaps_existing and intensity > self.config.attention_threshold:
                # Create new attention region
                new_region = AttentionRegion(
                    center_x=x,
                    center_y=y, 
                    center_z=z,
                    size=self.config.focus_region_size,
                    intensity=intensity,
                    age=0,
                    interest_level=intensity
                )
                
                self.attention_regions.append(new_region)
                self.stats['total_focus_regions_created'] += 1
                
                if not self.quiet_mode:
                    print(f"🎯 New attention region: ({x:.2f}, {y:.2f}, {z:.2f}) intensity={intensity:.3f}")
    
    def process_super_resolution(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Main super-resolution processing cycle.
        
        Returns information about current attention regions and processing.
        """
        if not self.config.enable_super_resolution:
            return {'super_resolution_enabled': False}
        
        start_time = time.time()
        
        # Get field gradients from base field
        field_gradients = self.base_field.compute_field_gradients()
        
        # Detect regions of interest
        interest_regions = self.detect_interest_regions(field_gradients)
        
        # Update attention regions
        self.update_attention_regions(interest_regions)
        
        # Update statistics
        self.stats['total_super_resolution_cycles'] += 1
        active_regions = len(self.attention_regions)
        total_cycles = self.stats['total_super_resolution_cycles']
        
        self.stats['avg_focus_regions_active'] = (
            self.stats['avg_focus_regions_active'] * (total_cycles - 1) + active_regions
        ) / total_cycles
        
        processing_time_ms = (time.time() - start_time) * 1000
        self.stats['super_resolution_overhead_ms'] += processing_time_ms
        
        return {
            'super_resolution_enabled': True,
            'active_attention_regions': len(self.attention_regions),
            'interest_regions_detected': len(interest_regions),
            'attention_regions': [
                {
                    'center': (region.center_x, region.center_y, region.center_z),
                    'intensity': region.intensity,
                    'interest_level': region.interest_level,
                    'age': region.age
                }
                for region in self.attention_regions
            ],
            'processing_time_ms': processing_time_ms
        }
    
    def get_effective_resolution_at_point(self, x: float, y: float, z: float) -> float:
        """Get the effective resolution at a specific point in the field."""
        base_res = float(self.config.base_resolution)
        focus_res = float(self.config.focus_resolution)
        
        # Check if point is within any attention region
        max_resolution = base_res
        
        for region in self.attention_regions:
            distance = np.sqrt(
                (x - region.center_x)**2 + 
                (y - region.center_y)**2 + 
                (z - region.center_z)**2
            )
            
            if distance < region.size:
                # Inside attention region - blend resolution based on distance
                blend_factor = 1.0 - (distance / region.size)
                region_resolution = base_res + (focus_res - base_res) * blend_factor * region.intensity
                max_resolution = max(max_resolution, region_resolution)
        
        return max_resolution
    
    def get_super_resolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive super-resolution statistics."""
        avg_overhead = 0.0
        if self.stats['total_super_resolution_cycles'] > 0:
            avg_overhead = (self.stats['super_resolution_overhead_ms'] / 
                          self.stats['total_super_resolution_cycles'])
        
        return {
            'config': {
                'base_resolution': self.config.base_resolution,
                'focus_resolution': self.config.focus_resolution,
                'max_focus_regions': self.config.max_focus_regions
            },
            'statistics': self.stats,
            'average_overhead_ms': avg_overhead,
            'current_attention_regions': len(self.attention_regions),
            'attention_region_details': [
                {
                    'center': (r.center_x, r.center_y, r.center_z),
                    'size': r.size,
                    'intensity': r.intensity,
                    'interest_level': r.interest_level,
                    'age': r.age
                }
                for r in self.attention_regions
            ]
        }


def create_attention_super_resolution(base_field: FieldImplementation,
                                    attention_processor: Optional[AttentionGuidedProcessing] = None,
                                    base_resolution: int = 50,
                                    focus_resolution: int = 100,
                                    quiet_mode: bool = False) -> AttentionSuperResolution:
    """Factory function to create attention super-resolution system."""
    config = SuperResolutionConfig(
        base_resolution=base_resolution,
        focus_resolution=focus_resolution
    )
    
    return AttentionSuperResolution(
        base_field=base_field,
        attention_processor=attention_processor,
        config=config,
        quiet_mode=quiet_mode
    )