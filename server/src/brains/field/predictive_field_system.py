#!/usr/bin/env python3
"""
Predictive Field System - The Brain IS Prediction

This system makes prediction explicit in the field dynamics. Rather than adding
prediction as a feature, we're revealing that the entire field operation is
fundamentally predictive.

Key principles:
1. Every field state embodies predictions about the future
2. Topology regions are predictive models for specific sensors
3. Temporal features encode prediction trajectories
4. Prediction errors drive all learning
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from ...utils.tensor_ops import create_zeros, safe_normalize
from .hierarchical_prediction import HierarchicalPredictionSystem, HierarchicalPrediction


@dataclass
class SensoryPrediction:
    """Prediction for sensory input with confidence."""
    values: torch.Tensor  # Predicted sensor values
    confidence: torch.Tensor  # Per-sensor confidence (0-1)
    source_regions: List[int]  # Which topology regions contributed
    temporal_basis: str  # "immediate", "short_term", "long_term"


class PredictiveFieldSystem:
    """
    Makes the field's predictive nature explicit.
    
    This system:
    1. Extracts predictions from field state
    2. Tracks which regions predict which sensors
    3. Learns from prediction errors
    4. Enables emergent sensory specialization
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 sensory_dim: int,
                 device: torch.device):
        """
        Initialize predictive field system.
        
        Args:
            field_shape: Shape of the 4D field
            sensory_dim: Number of sensory inputs to predict
            device: Computation device
        """
        self.field_shape = field_shape
        self.sensory_dim = sensory_dim
        self.device = device
        
        # Feature organization (matching field structure)
        self.spatial_features = field_shape[-1] - 32  # First 32 for content
        self.temporal_features = 16  # Next 16 for temporal
        self.dynamics_features = 16  # Last 16 for dynamics
        
        # Prediction tracking
        self.prediction_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        
        # Region-sensor associations (learned through use)
        # Start with no associations - they emerge
        self.region_sensor_affinity = {}  # region_id -> sensor_indices
        self.sensor_region_affinity = {}  # sensor_idx -> region_ids
        
        # Temporal prediction bases
        self.immediate_window = 1  # Next cycle
        self.short_term_window = 10  # Next 10 cycles
        self.long_term_window = 100  # Next 100 cycles
        
        # Hierarchical prediction system (Phase 3)
        self.hierarchical_system = HierarchicalPredictionSystem(
            field_shape=field_shape,
            sensory_dim=sensory_dim,
            device=device
        )
        self.use_hierarchical = False  # Will be enabled when ready
        
    def enable_hierarchical_prediction(self, enable: bool = True):
        """Enable or disable hierarchical prediction (Phase 3)."""
        self.use_hierarchical = enable
        
    def generate_sensory_prediction(self, 
                                   field: torch.Tensor,
                                   topology_regions: List[any],
                                   recent_sensory: Optional[deque] = None) -> SensoryPrediction:
        """
        Generate sensory predictions from current field state.
        
        This is where the magic happens - the field reveals its predictions.
        
        Args:
            field: Current field state
            topology_regions: Active topology regions
            recent_sensory: Recent sensory history for momentum
            
        Returns:
            SensoryPrediction with values and confidence
        """
        # Initialize predictions
        predictions = create_zeros((self.sensory_dim,), device=self.device)
        confidences = create_zeros((self.sensory_dim,), device=self.device)
        source_regions = []
        temporal_basis = "immediate"  # Default
        
        # Phase 3: Use hierarchical predictions if enabled
        if self.use_hierarchical:
            hierarchical_pred = self.hierarchical_system.extract_hierarchical_predictions(field)
            self._last_hierarchical_prediction = hierarchical_pred  # Store for error processing
            
            # Combine predictions from all timescales
            hierarchical_combined = self.hierarchical_system.combine_hierarchical_predictions(hierarchical_pred)
            # Handle dimension mismatch
            min_dim = min(predictions.shape[0], hierarchical_combined.shape[0])
            predictions[:min_dim] += hierarchical_combined[:min_dim] * 0.4  # Weight hierarchical contribution
            
            # Use best confidence from any timescale
            best_confidence = max(
                hierarchical_pred.immediate_confidence,
                hierarchical_pred.short_term_confidence,
                hierarchical_pred.long_term_confidence,
                hierarchical_pred.abstract_confidence
            )
            confidences += best_confidence * 0.3
            
            # Determine primary temporal basis
            if hierarchical_pred.abstract_confidence > 0.7:
                temporal_basis = "abstract"
            elif hierarchical_pred.long_term_confidence > 0.6:
                temporal_basis = "long_term"
            elif hierarchical_pred.short_term_confidence > 0.5:
                temporal_basis = "short_term"
        
        # 1. Extract temporal predictive features
        temporal_field = field[:, :, :, self.spatial_features:self.spatial_features + self.temporal_features]
        
        # 2. Get momentum-based predictions from recent history
        if recent_sensory and len(recent_sensory) >= 2:
            # Simple momentum: continue recent trends
            recent_vals = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in list(recent_sensory)[-3:]])
            momentum = self._compute_momentum_prediction(recent_vals)
            predictions += momentum * 0.3  # Weight momentum contribution
            confidences += 0.2  # Base confidence from momentum
        
        # 3. Let topology regions make predictions using their enhanced capabilities
        for region in topology_regions:
            # Skip non-predictive regions
            if not hasattr(region, 'is_sensory_predictive') or not region.is_sensory_predictive:
                continue
            
            # Use the region's own prediction method
            if hasattr(region, 'predict_from_field'):
                region_predictions = region.predict_from_field(field, temporal_field)
                
                # Apply predictions to the appropriate sensors
                for i, sensor_idx in enumerate(region.sensor_indices):
                    if sensor_idx < self.sensory_dim and i < len(region_predictions):
                        # Weight by region's confidence
                        weight = region.prediction_confidence * region.stability
                        predictions[sensor_idx] += region_predictions[i] * weight
                        confidences[sensor_idx] = max(confidences[sensor_idx], region.prediction_confidence)
                
                source_regions.append(region.region_id)
        
        # 4. Normalize predictions that had multiple contributors
        contributor_count = create_zeros((self.sensory_dim,), device=self.device)
        for i in source_regions:
            if i in self.region_sensor_affinity:
                for idx in self.region_sensor_affinity[i]:
                    contributor_count[idx] += 1
        
        mask = contributor_count > 1
        predictions[mask] /= contributor_count[mask]
        
        # 5. Add field-wide prediction bias
        # Low field activity predicts low sensory activity
        field_activity = torch.mean(torch.abs(field[:, :, :, :self.spatial_features]))
        predictions += field_activity * 0.1
        
        # 6. Ensure predictions are in reasonable range
        predictions = torch.clamp(predictions, -1.0, 1.0)
        confidences = torch.clamp(confidences, 0.0, 1.0)
        
        # For sensors with no prediction, use low confidence
        no_prediction_mask = confidences < 0.01
        confidences[no_prediction_mask] = 0.1  # Low but not zero
        
        return SensoryPrediction(
            values=predictions,
            confidence=confidences,
            source_regions=source_regions,
            temporal_basis=temporal_basis
        )
    
    def _compute_momentum_prediction(self, recent_vals: torch.Tensor) -> torch.Tensor:
        """Compute momentum-based prediction from recent values."""
        if recent_vals.shape[0] < 2:
            return create_zeros((self.sensory_dim,), device=self.device)
        
        # Simple linear extrapolation
        if recent_vals.shape[0] == 2:
            momentum = recent_vals[-1] - recent_vals[-2]
        else:
            # Weight recent changes more
            momentum = 0.5 * (recent_vals[-1] - recent_vals[-2]) + \
                      0.3 * (recent_vals[-2] - recent_vals[-3])
        
        # Predict next value
        prediction = recent_vals[-1] + momentum
        
        return prediction
    
    def _extract_region_prediction(self,
                                  region: any,
                                  temporal_field: torch.Tensor,
                                  sensor_indices: List[int]) -> Dict[int, Tuple[float, float]]:
        """Extract prediction from a topology region."""
        predictions = {}
        
        # Get region's location in field
        if hasattr(region, 'location'):
            x, y, z = region.location
        else:
            # Use region's pattern location
            pattern = region.pattern
            if hasattr(pattern, 'location'):
                x, y, z = pattern.location
            else:
                return predictions
        
        # Extract temporal features at this location
        local_temporal = temporal_field[
            max(0, x-1):min(temporal_field.shape[0], x+2),
            max(0, y-1):min(temporal_field.shape[1], y+2),
            max(0, z-1):min(temporal_field.shape[2], z+2),
            :
        ]
        
        # Average temporal activity
        temporal_mean = torch.mean(local_temporal, dim=(0, 1, 2))
        
        # Map temporal features to sensor predictions
        # This is where regions learn to predict specific sensors
        for sensor_idx in sensor_indices:
            # Use specific temporal features for this sensor
            # This mapping will improve through learning
            feature_idx = sensor_idx % self.temporal_features
            
            # Prediction is based on temporal feature activation
            prediction_val = temporal_mean[feature_idx].detach().item()
            
            # Confidence based on feature strength
            confidence = min(1.0, abs(prediction_val) * 2.0)
            
            predictions[sensor_idx] = (prediction_val, confidence)
        
        return predictions
    
    def process_prediction_error(self,
                                predicted: torch.Tensor,
                                actual: torch.Tensor,
                                topology_regions: List[any],
                                current_field: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Process prediction error and update region-sensor associations.
        
        This is where learning happens - regions that predict well
        strengthen their sensor associations.
        
        Args:
            predicted: Predicted sensor values
            actual: Actual sensor values
            topology_regions: Current topology regions
            
        Returns:
            Error analysis dict
        """
        # Compute per-sensor errors
        errors = actual - predicted
        abs_errors = torch.abs(errors)
        
        # Track error history
        self.error_history.append(abs_errors.mean().detach().item())
        
        # Phase 3: Process hierarchical errors if enabled
        if self.use_hierarchical and hasattr(self, '_last_hierarchical_prediction'):
            # Create field update from hierarchical error processing
            hierarchical_update = self.hierarchical_system.process_hierarchical_errors(
                predicted=self._last_hierarchical_prediction,
                actual_sensory=actual,
                current_field=current_field if current_field is not None else actual
            )
            # Store update to be applied in the next field evolution
            self._pending_hierarchical_update = hierarchical_update
        else:
            self._pending_hierarchical_update = None
        
        # Update region prediction confidence based on their performance
        for region in topology_regions:
            if hasattr(region, 'is_sensory_predictive') and region.is_sensory_predictive:
                if hasattr(region, 'update_prediction_success'):
                    # Get predicted values for this region's sensors
                    region_predictions = torch.zeros(len(region.sensor_indices), device=self.device)
                    for i, sensor_idx in enumerate(region.sensor_indices):
                        if sensor_idx < len(predicted):
                            region_predictions[i] = predicted[sensor_idx]
                    
                    # Update the region's prediction tracking
                    region.update_prediction_success(actual, region_predictions)
        
        # Compute error statistics
        error_stats = {
            'mean_error': abs_errors.mean().detach().item(),
            'max_error': abs_errors.max().detach().item(),
            'per_sensor_error': abs_errors.detach().cpu().numpy(),
            'improving': self._is_prediction_improving(),
            'specialized_sensors': sum(1 for r in topology_regions if getattr(r, 'is_sensory_predictive', False))
        }
        
        return error_stats
    
    def _update_region_sensor_affinity(self,
                                      errors: torch.Tensor,
                                      topology_regions: List[any]):
        """Update which regions predict which sensors based on errors."""
        # Only update every 10 cycles to allow patterns to stabilize
        if len(self.error_history) % 10 != 0:
            return
        
        # For each region, track its prediction success
        for i, region in enumerate(topology_regions):
            if not hasattr(region, 'pattern') or region.pattern is None:
                continue
            
            # Get current sensor associations
            if i not in self.region_sensor_affinity:
                # New region starts by trying to predict all sensors
                self.region_sensor_affinity[i] = list(range(self.sensory_dim))
            
            current_sensors = self.region_sensor_affinity[i]
            if not current_sensors:
                continue
            
            # Compute average error for this region's sensors
            region_errors = errors[current_sensors]
            avg_error = region_errors.mean().detach().item()
            
            # If prediction is good, strengthen association
            # If bad, weaken and potentially remove
            if avg_error < 0.2:  # Good prediction threshold
                # This region is good at these sensors
                for sensor_idx in current_sensors:
                    if sensor_idx not in self.sensor_region_affinity:
                        self.sensor_region_affinity[sensor_idx] = []
                    if i not in self.sensor_region_affinity[sensor_idx]:
                        self.sensor_region_affinity[sensor_idx].append(i)
            
            elif avg_error > 0.5:  # Poor prediction threshold
                # Remove sensors this region can't predict well
                poor_sensors = []
                for j, sensor_idx in enumerate(current_sensors):
                    if errors[sensor_idx] > 0.5:
                        poor_sensors.append(sensor_idx)
                
                # Remove poor sensors from this region
                self.region_sensor_affinity[i] = [
                    s for s in current_sensors if s not in poor_sensors
                ]
                
                # Remove this region from poor sensors
                for sensor_idx in poor_sensors:
                    if sensor_idx in self.sensor_region_affinity:
                        self.sensor_region_affinity[sensor_idx] = [
                            r for r in self.sensor_region_affinity[sensor_idx] if r != i
                        ]
    
    def _is_prediction_improving(self) -> bool:
        """Check if predictions are improving over time."""
        if len(self.error_history) < 20:
            return False
        
        recent_errors = list(self.error_history)[-10:]
        older_errors = list(self.error_history)[-20:-10]
        
        return np.mean(recent_errors) < np.mean(older_errors)
    
    def get_predictive_statistics(self) -> Dict[str, any]:
        """Get statistics about predictive performance."""
        stats = {
            'prediction_count': len(self.prediction_history),
            'mean_error': np.mean(list(self.error_history)) if self.error_history else 0.5,
            'error_trend': self._compute_error_trend(),
            'specialized_regions': len(self.region_sensor_affinity),
            'sensor_coverage': self._compute_sensor_coverage(),
            'improving': self._is_prediction_improving()
        }
        
        return stats
    
    def _compute_error_trend(self) -> float:
        """Compute trend in prediction error (-1 to 1, negative is improving)."""
        if len(self.error_history) < 10:
            return 0.0
        
        errors = list(self.error_history)
        x = np.arange(len(errors))
        
        # Simple linear regression
        slope = np.polyfit(x, errors, 1)[0]
        
        # Normalize to [-1, 1]
        return np.tanh(slope * 10)
    
    def _compute_sensor_coverage(self) -> float:
        """Compute fraction of sensors with dedicated predictors."""
        sensors_with_predictors = len(self.sensor_region_affinity)
        return sensors_with_predictors / self.sensory_dim if self.sensory_dim > 0 else 0.0
    
    def get_pending_hierarchical_update(self) -> Optional[torch.Tensor]:
        """Get pending hierarchical field update if available."""
        update = getattr(self, '_pending_hierarchical_update', None)
        # Clear it after retrieval
        self._pending_hierarchical_update = None
        return update