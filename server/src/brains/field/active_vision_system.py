#!/usr/bin/env python3
"""
Active Vision System - Vision-specific implementation

Extends the base ActiveSensingSystem with vision-specific behaviors:
- Saccades: rapid jumps to uncertain areas
- Smooth pursuit: tracking predictable motion
- Fixation: dwelling on complex patterns
- Scanning: exploratory figure-8 patterns
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque

from .active_sensing_system import ActiveSensingSystem, UncertaintyMap


class ActiveVisionSystem(ActiveSensingSystem):
    """
    Vision-specific active sensing through eye movements.
    
    Implements natural eye movement patterns that emerge from
    uncertainty-driven attention.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, int, int, int],
                 motor_dim: int,
                 device: torch.device,
                 sensor_control_start_idx: Optional[int] = None):
        """
        Initialize active vision system.
        
        Args:
            field_shape: Shape of the 4D field
            motor_dim: Total motor dimensions
            device: Computation device
            sensor_control_start_idx: Motor index where sensor control starts
                                    (if None, uses last 2 motors for pan/tilt)
        """
        # Sensor control motors (default: last 2 for pan/tilt)
        if sensor_control_start_idx is None:
            sensor_control_start = max(0, motor_dim - 2)
        else:
            sensor_control_start = sensor_control_start_idx
            
        control_dims = motor_dim - sensor_control_start
        
        # Initialize base class
        super().__init__(
            field_shape=field_shape,
            control_dims=control_dims,
            modality="vision",
            device=device
        )
        
        # Vision-specific state
        self.sensor_control_start = sensor_control_start
        
        # Current eye position (normalized -1 to 1)
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        # Smooth pursuit state
        self.pursuit_target = None
        self.pursuit_velocity = torch.zeros(2, device=device)
        
        # Movement history for pattern classification
        self.position_history = deque(maxlen=10)
        
        # Store for motion detection
        self._last_prediction = None
        self._motion_history = deque(maxlen=5)
    
    def generate_attention_control(self,
                                  uncertainty_map: UncertaintyMap,
                                  current_predictions: Optional[Dict[str, torch.Tensor]] = None,
                                  exploration_drive: float = 0.5) -> torch.Tensor:
        """
        Generate pan/tilt control based on visual uncertainty.
        
        Movement selection:
        - High uncertainty -> Saccades
        - Predictable motion -> Smooth pursuit
        - Mixed -> Scanning with uncertainty bias
        """
        sensor_control = torch.zeros(self.control_dims, device=self.device)
        
        if self.control_dims >= 2:
            # Determine movement type based on uncertainty and predictions
            if uncertainty_map.total_uncertainty > 0.7:
                # High uncertainty: rapid saccades to uncertain areas
                pan, tilt = self._generate_saccade(uncertainty_map)
                
            elif self._has_predictable_motion(current_predictions):
                # Predictable motion: smooth pursuit
                pan, tilt = self._generate_smooth_pursuit(current_predictions)
                
            else:
                # Mixed: uncertainty-driven scanning with some randomness
                pan, tilt = self._generate_scanning_motion(uncertainty_map, exploration_drive)
            
            # Apply movement with momentum for biological realism
            sensor_control[0] = self._apply_momentum(pan, self.current_pan, 0.7)
            if self.control_dims > 1:
                sensor_control[1] = self._apply_momentum(tilt, self.current_tilt, 0.7)
            
            # Update current position
            pan_update = sensor_control[0].item() if torch.is_tensor(sensor_control[0]) else sensor_control[0]
            self.current_pan = np.clip(self.current_pan + pan_update * 0.1, -1.0, 1.0)
            if self.control_dims > 1:
                tilt_update = sensor_control[1].item() if torch.is_tensor(sensor_control[1]) else sensor_control[1]
                self.current_tilt = np.clip(self.current_tilt + tilt_update * 0.1, -1.0, 1.0)
            
            # Track position history
            self.position_history.append((self.current_pan, self.current_tilt))
            
            # Update attention state
            self.current_attention_state = {
                'pan': self.current_pan,
                'tilt': self.current_tilt,
                'pattern': self.classify_attention_pattern()
            }
        
        return sensor_control
    
    def classify_attention_pattern(self) -> str:
        """Classify the current eye movement pattern."""
        if len(self.position_history) < 3:
            return "initializing"
        
        # Analyze recent movements
        recent_positions = list(self.position_history)
        
        # Calculate movement magnitudes
        movements = []
        for i in range(1, len(recent_positions)):
            pan_change = abs(recent_positions[i][0] - recent_positions[i-1][0])
            tilt_change = abs(recent_positions[i][1] - recent_positions[i-1][1])
            movements.append(max(pan_change, tilt_change))
        
        avg_movement = np.mean(movements)
        movement_std = np.std(movements)
        
        # Classify based on movement characteristics
        if avg_movement > 0.5:
            return "saccadic"
        elif avg_movement < 0.1:
            return "fixation"
        elif movement_std < 0.05:
            return "smooth_pursuit"
        else:
            return "scanning"
    
    def _generate_saccade(self, uncertainty_map: UncertaintyMap) -> Tuple[float, float]:
        """Generate rapid saccade to highest uncertainty area."""
        if not uncertainty_map.peak_locations:
            return 0.0, 0.0
        
        # Target highest uncertainty peak
        target_x, target_y, target_z = uncertainty_map.peak_locations[0]
        
        # Convert field coordinates to pan/tilt
        # Assuming pan maps to x-y plane, tilt to z
        norm_x = 2.0 * (target_x / (self.field_shape[0] - 1)) - 1.0
        norm_y = 2.0 * (target_y / (self.field_shape[1] - 1)) - 1.0
        norm_z = 2.0 * (target_z / (self.field_shape[2] - 1)) - 1.0
        
        # Combine x-y for pan
        target_pan = (norm_x + norm_y) / 2.0
        target_tilt = norm_z * 0.5  # Less tilt range
        
        # Generate saccade (fast movement)
        pan_error = target_pan - self.current_pan
        tilt_error = target_tilt - self.current_tilt
        
        # Saccadic suppression: move quickly
        pan = np.clip(pan_error * 2.0, -1.0, 1.0)
        tilt = np.clip(tilt_error * 1.5, -1.0, 1.0)
        
        return pan, tilt
    
    def _generate_smooth_pursuit(self, 
                               predictions: Optional[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
        """Generate smooth pursuit based on predictable motion."""
        if predictions is None or 'immediate' not in predictions:
            return 0.0, 0.0
        
        # Extract motion from predictions
        immediate_pred = predictions['immediate']
        
        # Estimate motion vector
        if self._last_prediction is not None:
            motion = immediate_pred - self._last_prediction
            motion_magnitude = torch.norm(motion)
            
            if motion_magnitude > 0.1:  # Significant motion
                # Update pursuit velocity
                self.pursuit_velocity = 0.8 * self.pursuit_velocity + 0.2 * motion[:2]
                
                # Generate smooth pursuit command
                pan = self.pursuit_velocity[0].item() * 0.5
                tilt = self.pursuit_velocity[1].item() * 0.3
                
                self._last_prediction = immediate_pred.clone()
                return pan, tilt
        
        self._last_prediction = immediate_pred.clone()
        return 0.0, 0.0
    
    def _generate_scanning_motion(self,
                                uncertainty_map: UncertaintyMap,
                                exploration_drive: float) -> Tuple[float, float]:
        """Generate scanning motion weighted by uncertainty."""
        # Base scanning pattern (figure-8)
        t = (len(self.attention_history) * 0.1) % (2 * np.pi)
        
        # Figure-8 pattern
        base_pan = np.sin(t) * 0.3
        base_tilt = np.sin(2 * t) * 0.2
        
        # Weight by uncertainty gradient
        if uncertainty_map.peak_locations:
            # Bias toward uncertain areas
            target_x, target_y, target_z = uncertainty_map.peak_locations[0]
            norm_x = 2.0 * (target_x / (self.field_shape[0] - 1)) - 1.0
            norm_y = 2.0 * (target_y / (self.field_shape[1] - 1)) - 1.0
            
            uncertainty_pan = (norm_x + norm_y) / 2.0 - self.current_pan
            base_pan += uncertainty_pan * exploration_drive
        
        # Add exploration noise
        pan = base_pan + np.random.normal(0, 0.1 * exploration_drive)
        tilt = base_tilt + np.random.normal(0, 0.05 * exploration_drive)
        
        return pan, tilt
    
    def _has_predictable_motion(self, predictions: Optional[Dict[str, torch.Tensor]]) -> bool:
        """Check if there's predictable motion to track."""
        if predictions is None:
            return False
        
        # Update motion history
        if 'immediate' in predictions and self._last_prediction is not None:
            motion_mag = torch.norm(predictions['immediate'] - self._last_prediction).item()
            self._motion_history.append(motion_mag)
        
        # Check if we have consistent motion
        if len(self._motion_history) >= 3:
            recent_motion = list(self._motion_history)[-3:]
            # Low variance in motion = predictable
            return np.std(recent_motion) < 0.1 and np.mean(recent_motion) > 0.05
        
        return False
    
    def _apply_momentum(self, target: float, current: float, momentum: float) -> float:
        """Apply momentum to movement for biological realism."""
        return momentum * current + (1 - momentum) * target