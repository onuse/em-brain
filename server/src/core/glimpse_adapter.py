"""
Glimpse-based Sensory Adapter for Active Vision

Extends the basic sensory adapter to support predictive sampling,
where rich sensors (like cameras) provide focused glimpses based
on field uncertainty rather than full frames.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .adapters import SensoryAdapter
from .interfaces import Robot


@dataclass
class GlimpseRequest:
    """Request for a focused sensory glimpse."""
    center_x: float  # Normalized coordinates [-1, 1]
    center_y: float
    zoom: float = 1.0  # 1.0 = normal, >1 = zoomed in
    channel: str = "camera"  # Which sensor to sample


class GlimpseSensoryAdapter(SensoryAdapter):
    """
    Sensory adapter that supports active vision through glimpses.
    
    For rich sensors (cameras, microphones), this adapter:
    1. Generates glimpse requests based on uncertainty
    2. Receives focused patches instead of full data
    3. Maintains spatial context across glimpses
    """
    
    def __init__(self, robot: Robot, field_dimensions: int, 
                 glimpse_size: int = 32,
                 max_glimpses_per_cycle: int = 5):
        super().__init__(robot, field_dimensions)
        
        self.glimpse_size = glimpse_size
        self.max_glimpses_per_cycle = max_glimpses_per_cycle
        
        # Track which channels support glimpsing
        self.glimpse_channels = self._identify_glimpse_channels()
        
        # Spatial memory for context
        self.spatial_memory = {}
        self.glimpse_history = []
        
    def to_field_space_with_glimpses(self, 
                                   sensory: List[float],
                                   glimpse_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Convert sensory input to field space, incorporating glimpse data.
        
        Args:
            sensory: Regular sensor values (non-glimpse sensors)
            glimpse_data: Dict mapping channel names to glimpse tensors
            
        Returns:
            Field space representation combining all sensory data
        """
        # Process regular sensors
        base_field = super().to_field_space(sensory)
        
        if glimpse_data is None:
            return base_field
        
        # Integrate glimpse data
        glimpse_features = []
        for channel_name, glimpse_tensor in glimpse_data.items():
            # Extract features from glimpse (simplified for now)
            features = self._extract_glimpse_features(glimpse_tensor)
            glimpse_features.append(features)
        
        if glimpse_features:
            # Combine glimpse features
            combined_glimpses = torch.cat(glimpse_features, dim=0)
            
            # Project to field space
            glimpse_projection = self._create_glimpse_projection(len(combined_glimpses))
            glimpse_field = glimpse_projection(combined_glimpses)
            
            # Blend with base sensory field
            # Glimpses get higher weight as they're focused attention
            alpha = 0.7  # Glimpse influence
            field_coords = alpha * glimpse_field + (1 - alpha) * base_field
        else:
            field_coords = base_field
        
        return torch.tanh(field_coords)
    
    def generate_glimpse_requests(self, uncertainty_map: torch.Tensor) -> List[GlimpseRequest]:
        """
        Generate glimpse requests based on field uncertainty.
        
        Args:
            uncertainty_map: 3D tensor of spatial uncertainty
            
        Returns:
            List of glimpse requests for high-uncertainty regions
        """
        requests = []
        
        # Find peaks in uncertainty map
        flat_uncertainty = uncertainty_map.flatten()
        
        # Get top-k uncertain locations
        k = min(self.max_glimpses_per_cycle, len(flat_uncertainty))
        _, indices = torch.topk(flat_uncertainty, k)
        
        # Convert indices to spatial coordinates
        shape = uncertainty_map.shape
        for idx in indices:
            # Unravel index
            z = idx % shape[2]
            y = (idx // shape[2]) % shape[1]
            x = idx // (shape[1] * shape[2])
            
            # Convert to normalized coordinates [-1, 1]
            norm_x = 2.0 * (x.float() / (shape[0] - 1)) - 1.0
            norm_y = 2.0 * (y.float() / (shape[1] - 1)) - 1.0
            
            # Add small random offset for exploration
            norm_x += torch.randn(1).item() * 0.1
            norm_y += torch.randn(1).item() * 0.1
            
            # Clamp to valid range
            norm_x = torch.clamp(norm_x, -1.0, 1.0)
            norm_y = torch.clamp(norm_y, -1.0, 1.0)
            
            # Create request
            request = GlimpseRequest(
                center_x=norm_x.item(),
                center_y=norm_y.item(),
                zoom=1.0 + uncertainty_map[x, y, z].item()  # Higher uncertainty = more zoom
            )
            requests.append(request)
        
        # Update history
        self.glimpse_history.extend(requests)
        if len(self.glimpse_history) > 100:
            self.glimpse_history = self.glimpse_history[-100:]
        
        return requests
    
    def _identify_glimpse_channels(self) -> Dict[str, Dict[str, Any]]:
        """Identify which sensory channels support glimpsing."""
        glimpse_channels = {}
        
        for channel in self.robot.sensory_channels:
            # Heuristic: channels with "camera" or "image" support glimpsing
            # In practice, this would be part of the robot profile
            if any(term in channel.name.lower() for term in ["camera", "image", "visual"]):
                glimpse_channels[channel.name] = {
                    'index': channel.index,
                    'type': 'visual',
                    'supports_zoom': True
                }
            elif any(term in channel.name.lower() for term in ["audio", "microphone", "sound"]):
                glimpse_channels[channel.name] = {
                    'index': channel.index,
                    'type': 'audio',
                    'supports_zoom': False  # Frequency focusing instead
                }
        
        return glimpse_channels
    
    def _extract_glimpse_features(self, glimpse: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a glimpse.
        
        This is simplified - in practice would use learned features
        or biologically-inspired edge/motion detectors.
        """
        # Flatten if needed
        if glimpse.dim() > 1:
            glimpse_flat = glimpse.flatten()
        else:
            glimpse_flat = glimpse
        
        # Basic statistics as features
        features = torch.tensor([
            glimpse_flat.mean(),
            glimpse_flat.std(),
            glimpse_flat.max() - glimpse_flat.min(),  # Contrast
            torch.abs(glimpse_flat[1:] - glimpse_flat[:-1]).mean()  # Roughness
        ])
        
        return features
    
    def _create_glimpse_projection(self, feature_dim: int) -> torch.nn.Module:
        """Create projection for glimpse features to field space."""
        projection = torch.nn.Linear(feature_dim, self.field_dimensions)
        torch.nn.init.xavier_uniform_(projection.weight)
        torch.nn.init.constant_(projection.bias, 0.0)
        return projection
    
    def get_glimpse_statistics(self) -> Dict[str, Any]:
        """Get statistics about glimpsing behavior."""
        if not self.glimpse_history:
            return {
                'total_glimpses': 0,
                'unique_channels': 0,
                'spatial_coverage': 0.0
            }
        
        # Analyze glimpse patterns
        positions = [(g.center_x, g.center_y) for g in self.glimpse_history[-50:]]
        channels = set(g.channel for g in self.glimpse_history[-50:])
        
        # Estimate spatial coverage
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            coverage = (x_range * y_range) / 4.0  # Normalized by full range
        else:
            coverage = 0.0
        
        return {
            'total_glimpses': len(self.glimpse_history),
            'unique_channels': len(channels),
            'spatial_coverage': coverage,
            'avg_zoom': np.mean([g.zoom for g in self.glimpse_history[-10:]]) if self.glimpse_history else 1.0
        }