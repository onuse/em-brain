#!/usr/bin/env python3
"""
Active Sensing Through Predictive Sampling - Generalized System

Core principle: Direct attention to areas of high uncertainty to maximize
information gain. This applies to any rich sensor modality - vision, audio,
tactile, chemical, etc.

The base system handles:
1. Uncertainty map generation from predictions
2. Information gain tracking
3. Attention value learning
4. Generic control signal generation

Specific modalities (vision, audio, etc) extend this with their own
movement patterns and control strategies.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from abc import ABC, abstractmethod

from ...utils.tensor_ops import create_zeros


@dataclass
class UncertaintyMap:
    """Spatial uncertainty derived from prediction confidence."""
    spatial_uncertainty: torch.Tensor  # 3D spatial map
    feature_uncertainty: torch.Tensor  # Per-feature uncertainty
    total_uncertainty: float  # Overall uncertainty metric
    peak_locations: List[Tuple[int, int, int]]  # High uncertainty areas


@dataclass
class AttentionRequest:
    """Generic request for focused sensory attention."""
    location: Tuple[float, ...]  # N-dimensional location
    focus_params: Dict[str, float]  # Modality-specific parameters
    priority: float  # How important is this request
    modality: str  # Which sensor system


class ActiveSensingSystem(ABC):
    """
    Base class for uncertainty-driven active sensing.
    
    This abstract system provides the core functionality for directing
    sensory attention based on predictive uncertainty. Specific sensor
    modalities extend this with their own control strategies.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 control_dims: int,
                 modality: str,
                 device: torch.device):
        """
        Initialize active sensing system.
        
        Args:
            field_shape: Shape of the 4D field
            control_dims: Number of control dimensions for this sensor
            modality: Name of the sensor modality
            device: Computation device
        """
        self.field_shape = field_shape
        self.control_dims = control_dims
        self.modality = modality
        self.device = device
        
        # Information gain tracking
        self.attention_history = deque(maxlen=100)
        self.information_gain_history = deque(maxlen=100)
        
        # Learned attention value
        self.attention_value_estimate = 0.5  # Start neutral
        
        # Current attention state
        self.current_attention_state = None
        
    def generate_uncertainty_map(self, 
                                topology_regions: List[Any],
                                field: torch.Tensor) -> UncertaintyMap:
        """
        Generate spatial uncertainty map from topology region confidences.
        
        This is generic across all modalities - uncertainty exists in space
        regardless of how we sense it.
        
        Args:
            topology_regions: List of topology regions with prediction confidence
            field: Current field state
            
        Returns:
            UncertaintyMap with spatial and feature uncertainty
        """
        # Initialize uncertainty map (inverse of confidence)
        spatial_uncertainty = create_zeros(self.field_shape[:3], device=self.device)
        feature_uncertainty = create_zeros((self.field_shape[3],), device=self.device)
        
        # Base uncertainty from field entropy
        field_entropy = self._compute_field_entropy(field)
        spatial_uncertainty += field_entropy * 0.2
        
        # Add uncertainty from topology regions
        for region in topology_regions:
            if hasattr(region, 'prediction_confidence') and hasattr(region, 'spatial_center'):
                # Low confidence = high uncertainty
                uncertainty = 1.0 - region.prediction_confidence
                
                # Spread uncertainty around region center
                x, y, z = region.spatial_center
                
                # Create gaussian blob of uncertainty
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        for dz in range(-3, 4):
                            px = x + dx
                            py = y + dy
                            pz = z + dz
                            
                            if (0 <= px < self.field_shape[0] and
                                0 <= py < self.field_shape[1] and
                                0 <= pz < self.field_shape[2]):
                                
                                # Gaussian falloff
                                dist_sq = dx*dx + dy*dy + dz*dz
                                weight = np.exp(-dist_sq / 4.0)
                                
                                spatial_uncertainty[px, py, pz] += uncertainty * weight * region.importance
        
        # Normalize uncertainty to [0, 1]
        if torch.max(spatial_uncertainty) > 0:
            spatial_uncertainty = spatial_uncertainty / torch.max(spatial_uncertainty)
        
        # Compute feature uncertainty from field variance
        feature_uncertainty = torch.std(field, dim=(0, 1, 2))
        feature_uncertainty = feature_uncertainty / (torch.max(feature_uncertainty) + 1e-6)
        
        # Find peak uncertainty locations
        flat_uncertainty = spatial_uncertainty.flatten()
        k = min(5, len(flat_uncertainty))
        _, indices = torch.topk(flat_uncertainty, k)
        
        peak_locations = []
        for idx in indices:
            z = idx % self.field_shape[2]
            y = (idx // self.field_shape[2]) % self.field_shape[1]
            x = idx // (self.field_shape[1] * self.field_shape[2])
            peak_locations.append((x.item(), y.item(), z.item()))
        
        # Compute total uncertainty
        total_uncertainty = torch.mean(spatial_uncertainty).item()
        
        return UncertaintyMap(
            spatial_uncertainty=spatial_uncertainty,
            feature_uncertainty=feature_uncertainty,
            total_uncertainty=total_uncertainty,
            peak_locations=peak_locations
        )
    
    @abstractmethod
    def generate_attention_control(self,
                                  uncertainty_map: UncertaintyMap,
                                  current_predictions: Optional[Dict[str, torch.Tensor]] = None,
                                  exploration_drive: float = 0.5) -> torch.Tensor:
        """
        Generate control signals to direct sensor attention.
        
        This is modality-specific and must be implemented by subclasses.
        
        Args:
            uncertainty_map: Current spatial uncertainty
            current_predictions: Current sensory predictions
            exploration_drive: Balance between exploration and exploitation
            
        Returns:
            Control signals for this sensor modality
        """
        pass
    
    @abstractmethod
    def classify_attention_pattern(self) -> str:
        """
        Classify the current attention pattern.
        
        Different modalities have different patterns:
        - Vision: saccades, smooth pursuit, fixation
        - Audio: sweeping, tracking, focusing
        - Touch: scanning, pressing, vibrating
        
        Returns:
            String describing current attention pattern
        """
        pass
    
    def process_attention_return(self,
                                attention_data: Dict[str, torch.Tensor],
                                uncertainty_before: float,
                                uncertainty_after: float):
        """
        Process the results of focused attention to learn its value.
        
        This is generic - all modalities benefit from tracking whether
        their attention improved predictions.
        
        Args:
            attention_data: The focused sensory data that was received
            uncertainty_before: Uncertainty before attention
            uncertainty_after: Uncertainty after processing attention
        """
        # Calculate information gain
        information_gain = uncertainty_before - uncertainty_after
        
        # Update attention value estimate
        self.attention_value_estimate = (
            0.9 * self.attention_value_estimate + 
            0.1 * information_gain
        )
        
        # Store in history
        self.information_gain_history.append(information_gain)
        
        # Track attention statistics
        self.attention_history.append({
            'time': torch.cuda.Event() if self.device.type == 'cuda' else None,
            'state': self.current_attention_state,
            'info_gain': information_gain,
            'data_size': sum(g.numel() for g in attention_data.values())
        })
    
    def _compute_field_entropy(self, field: torch.Tensor) -> torch.Tensor:
        """Compute spatial entropy of field activation."""
        # Sum across features
        spatial_field = torch.sum(torch.abs(field), dim=3)
        
        # Normalize to probability distribution
        spatial_field = spatial_field / (torch.sum(spatial_field) + 1e-6)
        
        # Compute entropy
        entropy = -torch.sum(
            spatial_field * torch.log(spatial_field + 1e-10),
            dim=None
        )
        
        # Convert to spatial uncertainty
        max_entropy = np.log(np.prod(self.field_shape[:3]))
        normalized_entropy = entropy / max_entropy
        
        # Create spatial map weighted by local variance
        uncertainty = torch.zeros_like(spatial_field)
        for x in range(1, self.field_shape[0] - 1):
            for y in range(1, self.field_shape[1] - 1):
                for z in range(1, self.field_shape[2] - 1):
                    local_var = torch.var(
                        field[x-1:x+2, y-1:y+2, z-1:z+2, :]
                    )
                    uncertainty[x, y, z] = local_var * normalized_entropy
        
        return uncertainty
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attention behavior."""
        if not self.attention_history:
            return {
                'modality': self.modality,
                'attention_count': 0,
                'avg_info_gain': 0.0,
                'attention_value': self.attention_value_estimate,
                'current_pattern': 'initializing'
            }
        
        recent_gains = list(self.information_gain_history)[-20:]
        
        return {
            'modality': self.modality,
            'attention_count': len(self.attention_history),
            'avg_info_gain': np.mean(recent_gains) if recent_gains else 0.0,
            'attention_value': self.attention_value_estimate,
            'current_pattern': self.classify_attention_pattern()
        }
    
    def reset(self):
        """Reset attention system to initial state."""
        self.attention_history.clear()
        self.information_gain_history.clear()
        self.attention_value_estimate = 0.5
        self.current_attention_state = None