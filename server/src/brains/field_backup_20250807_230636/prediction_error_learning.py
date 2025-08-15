#!/usr/bin/env python3
"""
Prediction Error Learning System - Phase 2

This system makes prediction error the primary driver of all learning in the brain.
Instead of just modulating imprinting strength, prediction errors directly shape
field dynamics, resource allocation, and self-modification strength.

Core principle: The brain is a prediction error minimization machine.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque

from ...utils.tensor_ops import create_zeros, safe_normalize


class PredictionErrorLearning:
    """
    Transforms prediction errors into spatial learning signals.
    
    This system:
    1. Maps prediction errors to field-space representations
    2. Modulates self-modification based on error magnitude
    3. Allocates resources to high-error regions
    4. Consolidates low-error (well-predicted) regions
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 sensory_dim: int,
                 device: torch.device):
        """
        Initialize prediction error learning system.
        
        Args:
            field_shape: Shape of the 4D field
            sensory_dim: Number of sensory inputs
            device: Computation device
        """
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.device = device
        
        # Error history for tracking improvement
        self.error_history = deque(maxlen=100)
        self.error_by_region = {}  # region_id -> error history
        
        # Learning rate modulation
        self.base_learning_rate = 0.1
        self.error_amplification = 2.0  # How much errors boost learning
        
    def error_to_field(self, 
                      prediction_errors: torch.Tensor,
                      topology_regions: List[any],
                      current_field: torch.Tensor) -> torch.Tensor:
        """
        Convert prediction errors into a spatial field representation.
        
        This is the key innovation - errors become spatial patterns that
        directly modify field dynamics where they occur.
        
        Args:
            prediction_errors: Per-sensor prediction errors
            topology_regions: Active topology regions
            current_field: Current field state
            
        Returns:
            Error field with same shape as main field
        """
        error_field = create_zeros(self.field_shape, device=self.device)
        
        # Map errors through topology regions
        for region in topology_regions:
            if not hasattr(region, 'is_sensory_predictive') or not region.is_sensory_predictive:
                continue
                
            # Get this region's prediction errors
            region_errors = []
            error_size = prediction_errors.shape[0] if len(prediction_errors.shape) > 0 else 1
            for sensor_idx in region.sensor_indices:
                if sensor_idx < error_size:
                    region_errors.append(prediction_errors[sensor_idx].detach().item())
            
            if not region_errors:
                continue
                
            # Compute region's average error
            avg_error = np.mean(region_errors)
            
            # Track error history for this region
            if region.region_id not in self.error_by_region:
                self.error_by_region[region.region_id] = deque(maxlen=50)
            self.error_by_region[region.region_id].append(avg_error)
            
            # Create error signal at region's location
            if hasattr(region, 'spatial_center'):
                x, y, z = region.spatial_center
                
                # Error magnitude determines signal strength
                error_strength = min(1.0, avg_error * self.error_amplification)
                
                # Create spatial error pattern
                # High error = strong, sharp signal (need to change)
                # Low error = weak, diffuse signal (consolidate)
                if avg_error > 0.1:  # Significant error
                    # Sharp, strong signal for learning
                    spread = 1
                    error_field[
                        max(0, x-spread):min(self.field_shape[0], x+spread+1),
                        max(0, y-spread):min(self.field_shape[1], y+spread+1),
                        max(0, z-spread):min(self.field_shape[2], z+spread+1),
                        :
                    ] += error_strength
                else:  # Good prediction
                    # Weak, diffuse signal for consolidation
                    spread = 2
                    consolidation_strength = 0.1 * (1.0 - avg_error)
                    error_field[
                        max(0, x-spread):min(self.field_shape[0], x+spread+1),
                        max(0, y-spread):min(self.field_shape[1], y+spread+1),
                        max(0, z-spread):min(self.field_shape[2], z+spread+1),
                        :
                    ] -= consolidation_strength  # Negative = consolidation
        
        # Also create a global error signal based on overall prediction quality
        global_error = torch.mean(torch.abs(prediction_errors)).detach().item()
        self.error_history.append(global_error)
        
        # Add weak global signal
        error_field += global_error * 0.05
        
        return error_field
    
    def compute_learning_modulation(self, error_field: torch.Tensor) -> Dict[str, float]:
        """
        Compute how prediction errors should modulate learning.
        
        Args:
            error_field: Spatial error representation
            
        Returns:
            Dictionary of learning modulation parameters
        """
        # Compute error statistics
        mean_error = torch.mean(torch.abs(error_field)).detach().item()
        max_error = torch.max(torch.abs(error_field)).detach().item()
        error_variance = torch.var(error_field).detach().item()
        
        # High errors should increase self-modification
        # Low errors should decrease it (system is predicting well)
        # More sensitive scaling: 0.25 error -> 2x boost, 0.5 error -> 3x boost
        self_mod_boost = min(3.0, 1.0 + mean_error * 4.0)
        
        # Learning rate adapts to error magnitude
        adaptive_learning_rate = self.base_learning_rate * self_mod_boost
        
        # Resource allocation - more to high-error regions
        resource_bias = min(0.8, max_error)  # 0-80% bias toward errors
        
        # Exploration pressure based on learning progress
        if len(self.error_history) > 20:
            recent_errors = list(self.error_history)[-10:]
            older_errors = list(self.error_history)[-20:-10]
            
            improvement = np.mean(older_errors) - np.mean(recent_errors)
            if improvement < 0.001:  # Not improving
                exploration_boost = 1.5
            else:
                exploration_boost = 0.8  # Reduce exploration when learning
        else:
            exploration_boost = 1.0
        
        return {
            'self_modification_boost': self_mod_boost,
            'learning_rate': adaptive_learning_rate,
            'resource_bias': resource_bias,
            'exploration_boost': exploration_boost,
            'mean_error': mean_error,
            'max_error': max_error,
            'error_variance': error_variance
        }
    
    def allocate_field_resources(self,
                               error_field: torch.Tensor,
                               current_dynamics: torch.Tensor) -> torch.Tensor:
        """
        Allocate field resources (dynamics parameters) based on prediction quality.
        
        High-error regions get:
        - Faster dynamics (lower decay)
        - More diffusion (spread learning)
        - Higher plasticity
        
        Low-error regions get:
        - Slower dynamics (higher decay)
        - Less diffusion (preserve patterns)
        - Lower plasticity (consolidation)
        
        Args:
            error_field: Spatial error representation
            current_dynamics: Current dynamics features
            
        Returns:
            Updated dynamics features
        """
        # Normalize error field to [0, 1]
        error_norm = torch.abs(error_field)
        error_norm = error_norm / (torch.max(error_norm) + 1e-8)
        
        # Create resource allocation mask
        # High error = 1 (need resources), Low error = 0 (consolidate)
        resource_mask = error_norm
        
        # Update dynamics based on errors
        new_dynamics = current_dynamics.clone()
        
        # Decay rates (features 0-3): High error → low decay (keep active)
        new_dynamics[:, :, :, 0:4] = (
            0.9 * current_dynamics[:, :, :, 0:4] +
            0.1 * (0.5 + 0.5 * (1.0 - resource_mask[:, :, :, :4]))
        )
        
        # Diffusion (features 4-7): High error → high diffusion (spread learning)
        new_dynamics[:, :, :, 4:8] = (
            0.9 * current_dynamics[:, :, :, 4:8] +
            0.1 * (resource_mask[:, :, :, :4] * 0.2 - 0.1)
        )
        
        # Plasticity (features 12-15): High error → high plasticity
        new_dynamics[:, :, :, 12:16] = (
            0.9 * current_dynamics[:, :, :, 12:16] +
            0.1 * torch.sigmoid(resource_mask[:, :, :, :4] * 2.0 - 0.5)
        )
        
        return new_dynamics
    
    def get_learning_statistics(self) -> Dict[str, any]:
        """Get statistics about prediction error learning."""
        stats = {
            'global_error_trend': self._compute_error_trend(),
            'current_error': self.error_history[-1] if self.error_history else 0.5,
            'improving_regions': self._count_improving_regions(),
            'total_regions_tracked': len(self.error_by_region)
        }
        
        return stats
    
    def _compute_error_trend(self) -> float:
        """Compute trend in global prediction error."""
        if len(self.error_history) < 10:
            return 0.0
            
        errors = list(self.error_history)
        x = np.arange(len(errors))
        
        # Linear regression
        slope = np.polyfit(x, errors, 1)[0]
        return -slope  # Negative slope = improvement
    
    def _count_improving_regions(self) -> int:
        """Count how many regions are improving their predictions."""
        improving = 0
        
        for region_id, errors in self.error_by_region.items():
            if len(errors) > 10:
                recent = np.mean(list(errors)[-5:])
                older = np.mean(list(errors)[-10:-5])
                if recent < older:
                    improving += 1
        
        return improving