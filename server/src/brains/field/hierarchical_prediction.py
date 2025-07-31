#!/usr/bin/env python3
"""
Hierarchical Prediction System - Phase 3

This system learns temporal patterns at different timescales by:
1. Encoding sensory history into field features at each timescale
2. Learning mappings from those features to predictions
3. Updating both encodings and mappings based on errors

Key insight: Each timescale maintains its own "temporal receptive field"
- Immediate: 1 timestep (raw sensory)
- Short-term: ~10 timesteps (patterns)
- Long-term: ~100 timesteps (sequences)
- Abstract: ~1000 timesteps (invariants)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from ...utils.tensor_ops import create_zeros, safe_normalize


@dataclass
class HierarchicalPrediction:
    """Multi-timescale prediction with confidence at each level."""
    immediate: torch.Tensor      # Next cycle prediction
    short_term: torch.Tensor     # Next ~10 cycles
    long_term: torch.Tensor      # Next ~100 cycles
    abstract: torch.Tensor       # Invariant patterns
    
    immediate_confidence: float
    short_term_confidence: float
    long_term_confidence: float
    abstract_confidence: float


class HierarchicalPredictionSystem:
    """
    Proper hierarchical prediction that actually learns patterns.
    
    Each timescale maintains:
    1. A temporal buffer of appropriate length
    2. Learned weights mapping from buffer to predictions
    3. Confidence based on prediction accuracy
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 sensory_dim: int,
                 device: torch.device):
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.device = device
        
        # Feature allocation remains the same
        self.immediate_features = slice(0, 16)
        self.short_term_features = slice(16, 32)
        self.long_term_features = slice(32, 48)
        self.abstract_features = slice(48, 64)
        
        # Temporal windows
        self.immediate_window = 1
        self.short_term_window = 10
        self.long_term_window = 100
        self.abstract_window = 1000
        
        # Sensory history buffers
        self.sensory_history = deque(maxlen=self.abstract_window)
        
        # Learned prediction weights for each timescale
        # These map from temporal features to sensory predictions
        self.immediate_weights = create_zeros((16, sensory_dim), device=device)
        self.short_term_weights = create_zeros((16, sensory_dim), device=device)
        self.long_term_weights = create_zeros((16, sensory_dim), device=device)
        self.abstract_weights = create_zeros((16, sensory_dim), device=device)
        
        # Initialize with reasonable random weights
        torch.nn.init.xavier_uniform_(self.immediate_weights, gain=1.0)
        torch.nn.init.xavier_uniform_(self.short_term_weights, gain=0.5)
        torch.nn.init.xavier_uniform_(self.long_term_weights, gain=0.3)
        torch.nn.init.xavier_uniform_(self.abstract_weights, gain=0.2)
        
        # Learning rates (more aggressive for dev testing)
        self.immediate_lr = 0.3
        self.short_term_lr = 0.1
        self.long_term_lr = 0.03
        self.abstract_lr = 0.01
        
        # Error tracking
        self.immediate_errors = deque(maxlen=50)
        self.short_term_errors = deque(maxlen=50)
        self.long_term_errors = deque(maxlen=50)
        self.abstract_errors = deque(maxlen=50)
        
    def extract_hierarchical_predictions(self, field: torch.Tensor) -> HierarchicalPrediction:
        """
        Extract predictions at all timescales from the field.
        
        This version actually uses learned weights to predict.
        """
        # Extract features for each timescale
        immediate_field = field[:, :, :, self.immediate_features]
        short_term_field = field[:, :, :, self.short_term_features]
        long_term_field = field[:, :, :, self.long_term_features]
        abstract_field = field[:, :, :, self.abstract_features]
        
        # Get field encodings (spatial average for now)
        immediate_encoding = torch.mean(immediate_field, dim=(0, 1, 2))
        short_term_encoding = torch.mean(short_term_field, dim=(0, 1, 2))
        long_term_encoding = torch.mean(long_term_field, dim=(0, 1, 2))
        abstract_encoding = torch.mean(abstract_field, dim=(0, 1, 2))
        
        # Generate predictions using learned weights
        immediate_pred = torch.matmul(immediate_encoding, self.immediate_weights)
        short_term_pred = torch.matmul(short_term_encoding, self.short_term_weights)
        long_term_pred = torch.matmul(long_term_encoding, self.long_term_weights)
        abstract_pred = torch.matmul(abstract_encoding, self.abstract_weights)
        
        # Apply activation to predictions
        immediate_pred = torch.tanh(immediate_pred)
        short_term_pred = torch.tanh(short_term_pred)
        long_term_pred = torch.tanh(long_term_pred)
        abstract_pred = torch.tanh(abstract_pred)
        
        # Calculate confidence
        immediate_conf = self._calculate_confidence(self.immediate_errors)
        short_term_conf = self._calculate_confidence(self.short_term_errors)
        long_term_conf = self._calculate_confidence(self.long_term_errors)
        abstract_conf = self._calculate_confidence(self.abstract_errors)
        
        return HierarchicalPrediction(
            immediate=immediate_pred,
            short_term=short_term_pred,
            long_term=long_term_pred,
            abstract=abstract_pred,
            immediate_confidence=immediate_conf,
            short_term_confidence=short_term_conf,
            long_term_confidence=long_term_conf,
            abstract_confidence=abstract_conf
        )
    
    def process_hierarchical_errors(self,
                                   predicted: HierarchicalPrediction,
                                   actual_sensory: torch.Tensor,
                                   current_field: torch.Tensor) -> torch.Tensor:
        """
        Process prediction errors and update both field and weights.
        """
        field_update = create_zeros(self.field_shape, device=self.device)
        
        # Add current sensory to history
        self.sensory_history.append(actual_sensory.detach().clone())
        
        # 1. Immediate prediction error
        min_dim = min(actual_sensory.shape[0], predicted.immediate.shape[0])
        immediate_error = actual_sensory[:min_dim] - predicted.immediate[:min_dim]
        self.immediate_errors.append(torch.mean(torch.abs(immediate_error)).item())
        
        # Update immediate weights using gradient descent
        immediate_encoding = torch.mean(current_field[:, :, :, self.immediate_features], dim=(0, 1, 2))
        weight_grad = torch.outer(immediate_encoding, immediate_error[:min_dim])
        self.immediate_weights[:, :min_dim] += weight_grad * self.immediate_lr
        
        # Update immediate field features to better encode current pattern
        immediate_update = self._encode_pattern_to_field(
            actual_sensory, self.immediate_features, immediate_error
        )
        field_update[:, :, :, self.immediate_features] += immediate_update
        
        # 2. Short-term prediction error (if we have enough history)
        if len(self.sensory_history) >= self.short_term_window:
            # Get average pattern over short term window
            recent_history = list(self.sensory_history)[-self.short_term_window:]
            short_term_pattern = torch.mean(torch.stack(recent_history), dim=0)
            
            short_term_error = short_term_pattern[:min_dim] - predicted.short_term[:min_dim]
            self.short_term_errors.append(torch.mean(torch.abs(short_term_error)).item())
            
            # Update short-term weights
            short_encoding = torch.mean(current_field[:, :, :, self.short_term_features], dim=(0, 1, 2))
            weight_grad = torch.outer(short_encoding, short_term_error[:min_dim])
            self.short_term_weights[:, :min_dim] += weight_grad * self.short_term_lr
            
            # Update field
            short_update = self._encode_pattern_to_field(
                short_term_pattern, self.short_term_features, short_term_error
            )
            field_update[:, :, :, self.short_term_features] += short_update
        
        # 3. Long-term and abstract (similar pattern)
        if len(self.sensory_history) >= self.long_term_window:
            # Long-term pattern
            long_history = list(self.sensory_history)[-self.long_term_window:]
            long_term_pattern = torch.mean(torch.stack(long_history), dim=0)
            
            long_term_error = long_term_pattern[:min_dim] - predicted.long_term[:min_dim]
            self.long_term_errors.append(torch.mean(torch.abs(long_term_error)).item())
            
            # Update weights and field
            long_encoding = torch.mean(current_field[:, :, :, self.long_term_features], dim=(0, 1, 2))
            weight_grad = torch.outer(long_encoding, long_term_error[:min_dim])
            self.long_term_weights[:, :min_dim] += weight_grad * self.long_term_lr
            
            long_update = self._encode_pattern_to_field(
                long_term_pattern, self.long_term_features, long_term_error
            )
            field_update[:, :, :, self.long_term_features] += long_update
        
        # 4. Abstract (very long-term average)
        if len(self.sensory_history) >= self.abstract_window // 10:
            # Abstract pattern (very smoothed)
            abstract_pattern = torch.mean(torch.stack(list(self.sensory_history)), dim=0)
            
            abstract_error = abstract_pattern[:min_dim] - predicted.abstract[:min_dim]
            self.abstract_errors.append(torch.mean(torch.abs(abstract_error)).item())
            
            # Update weights and field
            abstract_encoding = torch.mean(current_field[:, :, :, self.abstract_features], dim=(0, 1, 2))
            weight_grad = torch.outer(abstract_encoding, abstract_error[:min_dim])
            self.abstract_weights[:, :min_dim] += weight_grad * self.abstract_lr
            
            abstract_update = self._encode_pattern_to_field(
                abstract_pattern, self.abstract_features, abstract_error
            )
            field_update[:, :, :, self.abstract_features] += abstract_update
        
        return field_update
    
    def _encode_pattern_to_field(self, 
                                pattern: torch.Tensor, 
                                feature_slice: slice,
                                error: torch.Tensor) -> torch.Tensor:
        """
        Encode a sensory pattern into field features.
        
        Uses error magnitude to determine update strength.
        """
        update = create_zeros((self.field_shape[0], self.field_shape[1],
                              self.field_shape[2], 16), device=self.device)
        
        error_magnitude = torch.mean(torch.abs(error)).item()
        
        if error_magnitude > 0.01:
            # Encode pattern spatially
            # Use different spatial locations for different sensors
            for i in range(min(len(pattern), 16)):
                # Each sensor gets a region in the field
                x_center = int((i % 4) * self.field_shape[0] / 4 + self.field_shape[0] / 8)
                y_center = int((i // 4) * self.field_shape[1] / 4 + self.field_shape[1] / 8)
                z_center = self.field_shape[2] // 2
                
                # Spread based on error - high error = broader update
                spread = max(1, min(3, int(error_magnitude * 10)))
                
                # Update field with pattern value, scaled by learning rate
                # Use actual pattern value, not just error-scaled
                value = pattern[i].item() * (0.5 + error_magnitude)
                
                x_start = max(0, x_center - spread)
                x_end = min(self.field_shape[0], x_center + spread + 1)
                y_start = max(0, y_center - spread)
                y_end = min(self.field_shape[1], y_center + spread + 1)
                z_start = max(0, z_center - spread)
                z_end = min(self.field_shape[2], z_center + spread + 1)
                
                update[x_start:x_end, y_start:y_end, z_start:z_end, i] = value
        
        return update
    
    def _calculate_confidence(self, error_history: deque) -> float:
        """Calculate confidence based on recent prediction errors."""
        if len(error_history) < 5:
            return 0.5
        
        recent_errors = list(error_history)[-10:]
        mean_error = sum(recent_errors) / len(recent_errors)
        
        # Low error = high confidence
        confidence = 1.0 - min(1.0, mean_error * 2.0)
        
        # Boost confidence if improving
        if len(error_history) > 20:
            old_errors = list(error_history)[-20:-10]
            if sum(recent_errors)/len(recent_errors) < sum(old_errors)/len(old_errors):
                confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def combine_hierarchical_predictions(self, predictions: HierarchicalPrediction) -> torch.Tensor:
        """Combine predictions from all timescales weighted by confidence."""
        total_weight = (predictions.immediate_confidence + 
                       predictions.short_term_confidence * 0.5 +
                       predictions.long_term_confidence * 0.25 +
                       predictions.abstract_confidence * 0.125)
        
        if total_weight < 0.01:
            return predictions.immediate
        
        combined = (
            predictions.immediate * predictions.immediate_confidence +
            predictions.short_term * predictions.short_term_confidence * 0.5 +
            predictions.long_term * predictions.long_term_confidence * 0.25 +
            predictions.abstract * predictions.abstract_confidence * 0.125
        ) / total_weight
        
        return combined
    
    def get_temporal_state(self) -> Dict[str, any]:
        """Get current state of hierarchical predictions."""
        return {
            'immediate_confidence': self._calculate_confidence(self.immediate_errors),
            'short_term_confidence': self._calculate_confidence(self.short_term_errors),
            'long_term_confidence': self._calculate_confidence(self.long_term_errors),
            'abstract_confidence': self._calculate_confidence(self.abstract_errors),
            'immediate_error': list(self.immediate_errors)[-1] if self.immediate_errors else 0.5,
            'short_term_error': list(self.short_term_errors)[-1] if self.short_term_errors else 0.5,
            'long_term_error': list(self.long_term_errors)[-1] if self.long_term_errors else 0.5,
            'abstract_error': list(self.abstract_errors)[-1] if self.abstract_errors else 0.5,
            'history_length': len(self.sensory_history)
        }