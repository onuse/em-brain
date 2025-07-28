#!/usr/bin/env python3
"""
Integrated Attention System for Dynamic Unified Field Brain

This module bridges the sophisticated attention systems (UniversalAttentionSystem
and CrossModalAttentionSystem) with the main brain architecture, providing:

1. Sensory saliency detection for input prioritization
2. Field-based attention regions for focused processing
3. Cross-modal object binding and persistence
4. Attention-based modulation of spontaneous dynamics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

from ...attention import UniversalAttentionSystem, ModalityType, CrossModalAttentionSystem
from ...parameters.cognitive_config import get_cognitive_config


@dataclass
class AttentionIntegrationConfig:
    """Configuration for attention system integration."""
    enable_sensory_attention: bool = True
    enable_cross_modal_binding: bool = True
    attention_influence_on_field: float = 0.3  # How much attention affects field dynamics
    attention_to_confidence_factor: float = 0.2  # How attention affects prediction confidence
    saliency_threshold: float = 0.3  # Minimum saliency to trigger attention
    max_attention_regions: int = 5  # Maximum concurrent attention regions
    attention_decay_rate: float = 0.95  # How quickly attention fades
    binding_persistence_cycles: int = 10  # How long cross-modal bindings persist


class IntegratedAttentionSystem:
    """
    Integrates sophisticated attention mechanisms into the unified field brain.
    
    This system provides:
    - Sensory saliency detection using UniversalAttentionSystem
    - Cross-modal object formation using CrossModalAttentionSystem
    - Field attention regions that modulate brain dynamics
    - Attention-based confidence adjustments
    """
    
    def __init__(self, 
                 field_shape: Tuple[int, ...],
                 sensory_dim: int,
                 device: torch.device,
                 config: Optional[AttentionIntegrationConfig] = None,
                 quiet_mode: bool = False):
        """
        Initialize integrated attention system.
        
        Args:
            field_shape: Shape of the unified field tensor
            sensory_dim: Dimension of sensory input
            device: Torch device for computation
            config: Attention integration configuration
            quiet_mode: Suppress debug output
        """
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.device = device
        self.config = config or AttentionIntegrationConfig()
        self.quiet_mode = quiet_mode
        
        # Initialize attention subsystems
        self.universal_attention = UniversalAttentionSystem()
        self.cross_modal_attention = CrossModalAttentionSystem()
        
        # Get cognitive configuration
        self.cognitive_config = get_cognitive_config()
        
        # Attention state tracking
        self.current_attention_regions = []
        self.attention_history = deque(maxlen=50)
        self.cross_modal_objects = {}
        self.saliency_maps = {}
        
        # Statistics
        self.attention_triggers = 0
        self.binding_events = 0
        
        if not quiet_mode:
            print(f"🎯 Integrated Attention System initialized")
            print(f"   Field shape: {field_shape}")
            print(f"   Sensory dimension: {sensory_dim}")
            print(f"   Device: {device}")
    
    def process_sensory_attention(self, sensory_input: List[float]) -> Dict[str, Any]:
        """
        Process sensory input through attention system to detect salient features.
        
        Args:
            sensory_input: Raw sensory input vector
            
        Returns:
            Dictionary containing:
            - saliency_map: Attention weights for sensory channels
            - attention_regions: Detected salient regions
            - mean_saliency: Overall saliency level
        """
        if not self.config.enable_sensory_attention:
            return {
                'saliency_map': np.ones(len(sensory_input)),
                'attention_regions': [],
                'mean_saliency': 0.5
            }
        
        # Convert sensory input to numpy array
        sensory_array = np.array(sensory_input)
        
        # Detect signal shape and modality
        modality = self._infer_modality(sensory_array)
        
        # Process through universal attention system
        if len(sensory_array) >= 16:  # Minimum size for meaningful attention
            # Reshape if needed for 2D processing
            if modality == ModalityType.VISUAL and len(sensory_array) >= 64:
                # Assume square visual input
                side_length = int(np.sqrt(len(sensory_array)))
                if side_length * side_length == len(sensory_array):
                    signal_2d = sensory_array.reshape(side_length, side_length)
                    attention_map, _ = self.universal_attention.calculate_attention_map(signal_2d, modality)
                    saliency_map = attention_map.flatten()
                else:
                    # Fallback to 1D processing
                    saliency_map, _ = self.universal_attention.calculate_attention_map(sensory_array, modality)
            else:
                # Process as 1D signal
                saliency_map, _ = self.universal_attention.calculate_attention_map(sensory_array, modality)
        else:
            # Too small for attention processing
            saliency_map = np.ones_like(sensory_array) * 0.5
        
        # Detect attention regions based on saliency peaks
        attention_regions = self._detect_attention_regions(saliency_map)
        
        # Store saliency map
        self.saliency_maps['current'] = saliency_map
        
        # Update statistics
        mean_saliency = float(np.mean(saliency_map))
        if mean_saliency > self.config.saliency_threshold:
            self.attention_triggers += 1
        
        return {
            'saliency_map': saliency_map,
            'attention_regions': attention_regions,
            'mean_saliency': mean_saliency
        }
    
    def modulate_field_with_attention(self, unified_field: torch.Tensor, 
                                    attention_data: Dict[str, Any]) -> torch.Tensor:
        """
        Modulate unified field based on attention regions.
        
        Args:
            unified_field: Current unified field tensor
            attention_data: Attention data from process_sensory_attention
            
        Returns:
            Modulated field tensor
        """
        if not attention_data['attention_regions']:
            return unified_field
        
        # Create attention modulation field
        attention_field = torch.ones_like(unified_field)
        
        # Apply attention boost to salient regions
        for region in attention_data['attention_regions']:
            # Map sensory attention region to field coordinates
            field_coords = self._map_sensory_to_field_coords(region)
            
            if field_coords is not None:
                # Create gaussian attention boost around the region
                boost = self._create_attention_boost(
                    unified_field.shape,
                    field_coords,
                    region['intensity']
                )
                attention_field = torch.maximum(attention_field, boost)
        
        # Apply attention modulation
        influence = self.config.attention_influence_on_field
        modulated_field = unified_field * (1.0 + attention_field * influence)
        
        # Ensure modulation has meaningful effect
        modulated_field = torch.where(
            attention_field > 1.0,
            modulated_field * 1.5,  # Extra boost for high attention areas
            modulated_field
        )
        
        return modulated_field
    
    def update_confidence_with_attention(self, base_confidence: float,
                                       attention_data: Dict[str, Any]) -> float:
        """
        Adjust prediction confidence based on attention state.
        
        High attention (novelty) should reduce confidence.
        Low attention (familiarity) should increase confidence.
        
        Args:
            base_confidence: Current prediction confidence
            attention_data: Attention data from process_sensory_attention
            
        Returns:
            Adjusted confidence value
        """
        mean_saliency = attention_data.get('mean_saliency', 0.5)
        
        # High saliency = novel/surprising = lower confidence
        # Low saliency = familiar/expected = higher confidence
        attention_factor = 1.0 - mean_saliency  # Invert saliency
        
        # Apply attention influence to confidence
        adjustment = (attention_factor - 0.5) * self.config.attention_to_confidence_factor
        adjusted_confidence = base_confidence + adjustment
        
        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_confidence))
    
    def process_cross_modal_binding(self, sensory_data: Dict[str, np.ndarray],
                                  motor_output: List[float]) -> Dict[str, Any]:
        """
        Process cross-modal binding to form unified object representations.
        
        Args:
            sensory_data: Dictionary of sensory modalities and their data
            motor_output: Current motor commands
            
        Returns:
            Dictionary of cross-modal objects and their properties
        """
        if not self.config.enable_cross_modal_binding:
            return {}
        
        # Update cross-modal attention system
        active_objects = self.cross_modal_attention.update(sensory_data)
        
        # Track object persistence
        for obj_id, obj in active_objects.items():
            if obj_id not in self.cross_modal_objects:
                self.binding_events += 1
                if not self.quiet_mode:
                    print(f"🔗 New cross-modal object: {obj_id}")
            
            self.cross_modal_objects[obj_id] = {
                'object': obj,
                'last_seen': 0,  # Cycles since last update
                'persistence': self.config.binding_persistence_cycles
            }
        
        # Update persistence counters
        for obj_id in list(self.cross_modal_objects.keys()):
            if obj_id in active_objects:
                self.cross_modal_objects[obj_id]['last_seen'] = 0
            else:
                self.cross_modal_objects[obj_id]['last_seen'] += 1
                self.cross_modal_objects[obj_id]['persistence'] -= 1
                
                # Remove expired objects
                if self.cross_modal_objects[obj_id]['persistence'] <= 0:
                    del self.cross_modal_objects[obj_id]
                    if not self.quiet_mode:
                        print(f"🔗 Cross-modal object expired: {obj_id}")
        
        return self.cross_modal_objects
    
    def get_attention_state(self) -> Dict[str, Any]:
        """Get current attention system state."""
        return {
            'current_regions': len(self.current_attention_regions),
            'mean_saliency': self.saliency_maps.get('current', np.array([0.5])).mean(),
            'active_objects': len(self.cross_modal_objects),
            'attention_triggers': self.attention_triggers,
            'binding_events': self.binding_events,
            'attention_history': list(self.attention_history)
        }
    
    # Private helper methods
    
    def _infer_modality(self, sensory_array: np.ndarray) -> ModalityType:
        """Infer modality type from sensory array characteristics."""
        # Simple heuristic - in practice, this would use metadata
        if len(sensory_array) >= 64 and np.sqrt(len(sensory_array)) % 1 == 0:
            return ModalityType.VISUAL  # Square arrays likely visual
        elif len(sensory_array) > 100:
            return ModalityType.AUDIO  # Long arrays likely audio
        else:
            return ModalityType.MOTOR  # Short arrays likely motor/proprioceptive
    
    def _detect_attention_regions(self, saliency_map: np.ndarray) -> List[Dict[str, Any]]:
        """Detect attention regions from saliency map."""
        regions = []
        
        # Find peaks above threshold
        threshold = self.config.saliency_threshold
        high_saliency_indices = np.where(saliency_map > threshold)[0]
        
        if len(high_saliency_indices) > 0:
            # Simple clustering of nearby high saliency points
            # In practice, use more sophisticated clustering
            for idx in high_saliency_indices[:self.config.max_attention_regions]:
                regions.append({
                    'center_idx': int(idx),
                    'intensity': float(saliency_map[idx]),
                    'size': 1  # Simple point attention for now
                })
        
        return regions
    
    def _map_sensory_to_field_coords(self, attention_region: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Map sensory attention region to field coordinates."""
        # This is a simplified mapping - in practice would use dimension mapping
        sensory_idx = attention_region['center_idx']
        
        # Map sensory index to field dimensions
        # For now, distribute sensory dimensions across field dimensions
        if len(self.field_shape) > 0:
            field_coords = torch.zeros(len(self.field_shape), device=self.device)
            
            # Simple mapping: spread sensory index across field dimensions
            dims_per_field = self.sensory_dim / len(self.field_shape)
            for i in range(len(self.field_shape)):
                dim_contribution = sensory_idx / self.sensory_dim
                field_coords[i] = dim_contribution * self.field_shape[i]
            
            return field_coords
        
        return None
    
    def _create_attention_boost(self, field_shape: Tuple[int, ...],
                               center_coords: torch.Tensor,
                               intensity: float) -> torch.Tensor:
        """Create gaussian attention boost field."""
        boost_field = torch.zeros(field_shape, device=self.device)
        
        # Create gaussian boost around center
        # Simplified for efficiency - just boost the nearest discrete point
        discrete_coords = tuple(int(c) for c in center_coords)
        
        # Ensure coordinates are within bounds
        valid_coords = tuple(
            max(0, min(c, s-1)) 
            for c, s in zip(discrete_coords, field_shape)
        )
        
        boost_field[valid_coords] = intensity
        
        # Apply gaussian blur for smooth attention
        # In practice, use proper n-dimensional gaussian
        # For now, just decay neighboring points
        
        return boost_field