#!/usr/bin/env python3
"""
Attention-Guided Processing

Implementation-agnostic attention mechanisms that work with any field implementation
through the FieldImplementation interface.

Key Features:
1. Selective Field Processing - Focus computational resources on important regions
2. Top-Down Attention Modulation - Goal-directed attention control
3. Bottom-Up Saliency Detection - Data-driven attention capture
4. Attention Memory - Track and learn attention patterns
5. Multi-Scale Attention - Operate across different field scales
"""

import torch
import time
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from .field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience, 
        FieldNativeAction
    )
    from .adaptive_field_impl import FieldImplementation
except ImportError:
    from field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience,
        FieldNativeAction
    )
    from adaptive_field_impl import FieldImplementation


@dataclass
class AttentionConfig:
    """Configuration for attention-guided processing."""
    attention_threshold: float = 0.1         # Minimum activity for attention
    focus_radius: float = 3.0               # Spatial attention radius
    top_down_strength: float = 0.4          # Top-down attention influence
    bottom_up_strength: float = 0.6         # Bottom-up saliency influence
    attention_decay_rate: float = 0.95      # Attention decay over time
    max_attention_regions: int = 5          # Maximum simultaneous attention regions
    novelty_boost: float = 0.3              # Extra attention for novel patterns
    memory_window: int = 50                 # Attention pattern memory size


@dataclass
class AttentionRegion:
    """Represents an active attention region."""
    center_coordinates: torch.Tensor        # Field coordinates of attention center
    intensity: float                        # Attention intensity (0-1)
    radius: float                          # Spatial radius of attention
    creation_time: float                   # When this attention was created
    attention_type: str                    # "top_down", "bottom_up", "learned"
    source_gradient: str                   # Which gradient triggered this attention
    persistence_score: float              # How long this attention should persist
    
    def is_expired(self, current_time: float, max_age: float = 5.0) -> bool:
        """Check if this attention region has expired."""
        return (current_time - self.creation_time) > max_age


class AttentionGuidedProcessing:
    """
    Attention-guided processing that works with any field implementation.
    
    Provides sophisticated attention mechanisms while maintaining implementation independence.
    """
    
    def __init__(self, field_impl: FieldImplementation,
                 config: Optional[AttentionConfig] = None,
                 quiet_mode: bool = False):
        
        self.field_impl = field_impl
        self.quiet_mode = quiet_mode
        self.config = config or AttentionConfig()
        
        # Active attention regions
        self.attention_regions: List[AttentionRegion] = []
        
        # Attention memory and learning
        self.attention_history = deque(maxlen=self.config.memory_window)
        self.learned_attention_patterns = {}
        self.attention_statistics = defaultdict(float)
        
        # Saliency detection
        self.saliency_map = {}
        self.gradient_history = deque(maxlen=10)
        self.novelty_detector = {}
        
        # Top-down attention goals
        self.attention_goals = []  # List of goal-directed attention targets
        self.goal_priorities = {}
        
        # Performance tracking
        self.attention_effectiveness = {}
        self.processing_focus_metrics = {}
        
        if not quiet_mode:
            print(f"👁️ Attention-Guided Processing initialized for {field_impl.get_implementation_type()}")
    
    def process_with_attention(self, input_stream: List[float],
                             top_down_goals: Optional[List[str]] = None) -> torch.Tensor:
        """
        Process input with attention-guided field selection and enhancement.
        """
        # 1. Update attention goals if provided
        if top_down_goals:
            self._update_attention_goals(top_down_goals)
        
        # 2. Detect bottom-up saliency
        self._detect_bottom_up_saliency()
        
        # 3. Apply top-down attention modulation
        self._apply_top_down_attention()
        
        # 4. Update and prune attention regions
        self._update_attention_regions()
        
        # 5. Process input with attention enhancement
        enhanced_output = self._process_with_attention_enhancement(input_stream)
        
        # 6. Learn from attention patterns
        self._learn_attention_patterns()
        
        # 7. Update attention statistics
        self._update_attention_statistics()
        
        return enhanced_output
    
    def _detect_bottom_up_saliency(self) -> None:
        """Detect salient regions based on field gradients and activity."""
        gradients = self.field_impl.compute_field_gradients()
        
        # Store gradient history for novelty detection
        current_gradient_signature = {}
        for grad_name, grad_tensor in gradients.items():
            if grad_tensor.numel() > 0:
                signature = torch.mean(torch.abs(grad_tensor)).item()
                current_gradient_signature[grad_name] = signature
        
        self.gradient_history.append(current_gradient_signature)
        
        # Detect novel patterns (high current activity, low historical activity)
        if len(self.gradient_history) > 3:
            for grad_name, current_activity in current_gradient_signature.items():
                # Calculate historical average
                historical_avg = 0.0
                count = 0
                for hist_grad in list(self.gradient_history)[:-1]:  # Exclude current
                    if grad_name in hist_grad:
                        historical_avg += hist_grad[grad_name]
                        count += 1
                
                if count > 0:
                    historical_avg /= count
                    
                    # Check for novelty (current significantly higher than historical)
                    novelty_ratio = current_activity / (historical_avg + 0.01)
                    
                    if novelty_ratio > 2.0 and current_activity > self.config.attention_threshold:
                        self._create_attention_region(
                            gradient_name=grad_name,
                            intensity=min(1.0, current_activity * self.config.novelty_boost),
                            attention_type="bottom_up",
                            source="novelty_detection"
                        )
        
        # Detect high-activity regions
        for grad_name, grad_tensor in gradients.items():
            if grad_tensor.numel() > 0:
                activity_level = torch.mean(torch.abs(grad_tensor)).item()
                
                if activity_level > self.config.attention_threshold:
                    # Check if we don't already have attention on this region
                    if not self._has_attention_on_gradient(grad_name):
                        self._create_attention_region(
                            gradient_name=grad_name,
                            intensity=min(1.0, activity_level * self.config.bottom_up_strength),
                            attention_type="bottom_up",
                            source="high_activity"
                        )
    
    def _apply_top_down_attention(self) -> None:
        """Apply goal-directed top-down attention."""
        if not self.attention_goals:
            return
        
        for goal in self.attention_goals:
            goal_type = goal.get('type', 'general')
            goal_intensity = goal.get('intensity', 0.5)
            goal_coordinates = goal.get('coordinates')
            
            if goal_coordinates is not None:
                # Create top-down attention region
                attention_region = AttentionRegion(
                    center_coordinates=goal_coordinates,
                    intensity=goal_intensity * self.config.top_down_strength,
                    radius=self.config.focus_radius,
                    creation_time=time.time(),
                    attention_type="top_down",
                    source_gradient=f"goal_{goal_type}",
                    persistence_score=goal.get('persistence', 3.0)
                )
                
                self._add_attention_region(attention_region)
    
    def _create_attention_region(self, gradient_name: str, intensity: float,
                               attention_type: str, source: str) -> None:
        """Create attention region based on gradient activity."""
        # Generate coordinates based on gradient activity
        gradients = self.field_impl.compute_field_gradients()
        
        if gradient_name in gradients:
            grad_tensor = gradients[gradient_name]
            
            # Find peak activity location in gradient
            if grad_tensor.numel() > 0:
                peak_idx = torch.argmax(torch.abs(grad_tensor.flatten()))
                
                # Convert to field coordinates (simplified spatial mapping)
                coords = torch.randn(36, device=self.field_impl.field_device) * 0.2
                
                # Set spatial coordinates based on peak location
                if grad_tensor.dim() >= 3:  # Has spatial dimensions
                    spatial_shape = grad_tensor.shape[:3]
                    total_spatial = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
                    
                    if total_spatial > 0:
                        spatial_idx = peak_idx % total_spatial
                        z_idx = spatial_idx // (spatial_shape[0] * spatial_shape[1])
                        y_idx = (spatial_idx % (spatial_shape[0] * spatial_shape[1])) // spatial_shape[0]
                        x_idx = spatial_idx % spatial_shape[0]
                        
                        # Normalize to [-1, 1] range
                        coords[0] = (x_idx / max(1, spatial_shape[0] - 1)) * 2 - 1
                        coords[1] = (y_idx / max(1, spatial_shape[1] - 1)) * 2 - 1
                        coords[2] = (z_idx / max(1, spatial_shape[2] - 1)) * 2 - 1
                
                attention_region = AttentionRegion(
                    center_coordinates=coords,
                    intensity=intensity,
                    radius=self.config.focus_radius,
                    creation_time=time.time(),
                    attention_type=attention_type,
                    source_gradient=gradient_name,
                    persistence_score=intensity * 2.0
                )
                
                self._add_attention_region(attention_region)
    
    def _add_attention_region(self, attention_region: AttentionRegion) -> None:
        """Add attention region, managing capacity limits."""
        # Check if region is too similar to existing ones
        for existing in self.attention_regions:
            coord_distance = torch.norm(
                existing.center_coordinates - attention_region.center_coordinates
            ).item()
            
            if coord_distance < self.config.focus_radius * 0.5:
                # Merge with existing region (update intensity)
                if attention_region.intensity > existing.intensity:
                    existing.intensity = attention_region.intensity
                    existing.creation_time = attention_region.creation_time
                return
        
        # Add new region
        self.attention_regions.append(attention_region)
        
        # Prune if too many regions
        if len(self.attention_regions) > self.config.max_attention_regions:
            # Sort by intensity and keep the strongest
            self.attention_regions.sort(key=lambda r: r.intensity, reverse=True)
            self.attention_regions = self.attention_regions[:self.config.max_attention_regions]
    
    def _update_attention_regions(self) -> None:
        """Update and prune attention regions."""
        current_time = time.time()
        
        # Remove expired regions
        self.attention_regions = [
            region for region in self.attention_regions
            if not region.is_expired(current_time, region.persistence_score)
        ]
        
        # Apply attention decay
        for region in self.attention_regions:
            age = current_time - region.creation_time
            decay_factor = self.config.attention_decay_rate ** age
            region.intensity *= decay_factor
        
        # Remove regions that have decayed too much
        self.attention_regions = [
            region for region in self.attention_regions
            if region.intensity > 0.05
        ]
    
    def _process_with_attention_enhancement(self, input_stream: List[float]) -> torch.Tensor:
        """Process input with attention-enhanced field interactions."""
        if not self.attention_regions:
            # No attention - standard processing
            input_tensor = torch.tensor(input_stream, device=self.field_impl.field_device)
            field_coords = torch.randn(36, device=self.field_impl.field_device)
            return self.field_impl.generate_field_output(field_coords)
        
        # Apply attention-enhanced imprinting
        input_tensor = torch.tensor(input_stream, device=self.field_impl.field_device)
        
        for region in self.attention_regions:
            # Create attention-enhanced experience
            enhanced_intensity = 0.4 + region.intensity * 0.6  # Base + attention boost
            
            experience = UnifiedFieldExperience(
                timestamp=time.time(),
                field_coordinates=region.center_coordinates,
                raw_input_stream=input_tensor,
                field_intensity=enhanced_intensity,
                dynamics_family_activations={
                    FieldDynamicsFamily.FLOW: region.intensity,
                    FieldDynamicsFamily.COUPLING: region.intensity * 0.8,
                    FieldDynamicsFamily.EMERGENCE: region.intensity * 0.6
                }
            )
            
            self.field_impl.imprint_experience(experience)
        
        # Generate output from attention-weighted field state
        output_components = []
        
        for region in self.attention_regions:
            region_output = self.field_impl.generate_field_output(region.center_coordinates)
            weighted_output = region_output * region.intensity
            output_components.append(weighted_output)
        
        if output_components:
            # Combine attention-weighted outputs
            combined_output = torch.stack(output_components)
            final_output = torch.mean(combined_output, dim=0)
            
            # Record attention effectiveness
            output_magnitude = torch.norm(final_output).item()
            self.attention_effectiveness[len(self.attention_regions)] = output_magnitude
            
            return final_output
        else:
            # Fallback to standard output
            field_coords = torch.randn(36, device=self.field_impl.field_device)
            return self.field_impl.generate_field_output(field_coords)
    
    def _learn_attention_patterns(self) -> None:
        """Learn effective attention patterns for future use."""
        if not self.attention_regions:
            return
        
        # Create attention pattern signature
        pattern_signature = []
        for region in self.attention_regions:
            signature = {
                'type': region.attention_type,
                'source': region.source_gradient,
                'intensity': region.intensity,
                'spatial_coords': region.center_coordinates[:3].cpu().tolist()
            }
            pattern_signature.append(signature)
        
        # Store in attention history
        self.attention_history.append({
            'timestamp': time.time(),
            'regions': pattern_signature,
            'effectiveness': self.attention_effectiveness.get(len(self.attention_regions), 0.0)
        })
        
        # Learn from highly effective patterns
        if len(self.attention_history) > 10:
            recent_patterns = list(self.attention_history)[-10:]
            avg_effectiveness = sum(p['effectiveness'] for p in recent_patterns) / len(recent_patterns)
            
            for pattern in recent_patterns:
                if pattern['effectiveness'] > avg_effectiveness * 1.2:
                    # This is an effective pattern - remember it
                    pattern_key = self._create_pattern_key(pattern['regions'])
                    if pattern_key not in self.learned_attention_patterns:
                        self.learned_attention_patterns[pattern_key] = {
                            'effectiveness': pattern['effectiveness'],
                            'usage_count': 1,
                            'pattern': pattern['regions']
                        }
                    else:
                        # Update effectiveness with moving average
                        current = self.learned_attention_patterns[pattern_key]
                        current['effectiveness'] = (current['effectiveness'] + pattern['effectiveness']) / 2
                        current['usage_count'] += 1
    
    def _create_pattern_key(self, regions: List[Dict]) -> str:
        """Create a hashable key for attention patterns."""
        # Simplified pattern key based on attention types and sources
        types_and_sources = [(r['type'], r['source']) for r in regions]
        types_and_sources.sort()  # Ensure consistent ordering
        return str(types_and_sources)
    
    def _update_attention_statistics(self) -> None:
        """Update attention processing statistics."""
        self.attention_statistics['total_regions'] = len(self.attention_regions)
        self.attention_statistics['avg_intensity'] = (
            sum(r.intensity for r in self.attention_regions) / len(self.attention_regions)
            if self.attention_regions else 0.0
        )
        
        # Count attention types
        type_counts = defaultdict(int)
        for region in self.attention_regions:
            type_counts[region.attention_type] += 1
        
        for att_type, count in type_counts.items():
            self.attention_statistics[f'{att_type}_count'] = count
        
        self.attention_statistics['learned_patterns'] = len(self.learned_attention_patterns)
    
    def _update_attention_goals(self, goals: List[str]) -> None:
        """Update top-down attention goals."""
        self.attention_goals = []
        
        for goal in goals:
            if goal == "spatial_exploration":
                # Create exploration-focused attention
                coords = torch.rand(36, device=self.field_impl.field_device) * 2 - 1
                self.attention_goals.append({
                    'type': 'exploration',
                    'coordinates': coords,
                    'intensity': 0.6,
                    'persistence': 2.0
                })
            
            elif goal == "pattern_recognition":
                # Focus on high-gradient regions for pattern detection
                gradients = self.field_impl.compute_field_gradients()
                for grad_name, grad_tensor in gradients.items():
                    if grad_tensor.numel() > 0 and torch.mean(torch.abs(grad_tensor)).item() > 0.2:
                        coords = torch.randn(36, device=self.field_impl.field_device) * 0.3
                        self.attention_goals.append({
                            'type': 'pattern_recognition',
                            'coordinates': coords,
                            'intensity': 0.8,
                            'persistence': 3.0
                        })
                        break
            
            elif goal == "energy_conservation":
                # Focus attention on low-energy regions to boost them
                coords = torch.randn(36, device=self.field_impl.field_device) * 0.2
                self.attention_goals.append({
                    'type': 'energy_conservation',
                    'coordinates': coords,
                    'intensity': 0.4,
                    'persistence': 4.0
                })
    
    def _has_attention_on_gradient(self, gradient_name: str) -> bool:
        """Check if there's already attention on the specified gradient."""
        return any(
            region.source_gradient == gradient_name
            for region in self.attention_regions
        )
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attention processing statistics."""
        stats = dict(self.attention_statistics)
        stats.update({
            'active_attention_regions': len(self.attention_regions),
            'learned_attention_patterns': len(self.learned_attention_patterns),
            'attention_history_length': len(self.attention_history),
            'current_goals': len(self.attention_goals),
            'effectiveness_scores': dict(self.attention_effectiveness)
        })
        
        # Add region details
        if self.attention_regions:
            stats['attention_regions'] = [
                {
                    'type': region.attention_type,
                    'source': region.source_gradient,
                    'intensity': region.intensity,
                    'age': time.time() - region.creation_time
                }
                for region in self.attention_regions
            ]
        
        return stats
    
    def set_attention_goals(self, goals: List[str]) -> None:
        """Manually set attention goals."""
        self._update_attention_goals(goals)
    
    def clear_attention(self) -> None:
        """Clear all attention regions and goals."""
        self.attention_regions.clear()
        self.attention_goals.clear()
    
    def apply_learned_pattern(self, pattern_key: str) -> bool:
        """Apply a previously learned attention pattern."""
        if pattern_key in self.learned_attention_patterns:
            pattern = self.learned_attention_patterns[pattern_key]
            
            for region_spec in pattern['pattern']:
                coords = torch.tensor(
                    region_spec['spatial_coords'] + [0.0] * (36 - 3),
                    device=self.field_impl.field_device
                )
                
                attention_region = AttentionRegion(
                    center_coordinates=coords,
                    intensity=region_spec['intensity'],
                    radius=self.config.focus_radius,
                    creation_time=time.time(),
                    attention_type="learned",
                    source_gradient=region_spec['source'],
                    persistence_score=3.0
                )
                
                self._add_attention_region(attention_region)
            
            pattern['usage_count'] += 1
            return True
        
        return False