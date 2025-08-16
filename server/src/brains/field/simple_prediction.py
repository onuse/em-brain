"""
Simple Prediction System

The field's current state implies its future state.
No hierarchies, no timescales - just next-state prediction.
"""

import torch
from typing import List, Optional
from collections import deque


class SimplePrediction:
    """
    Simple next-state prediction from field dynamics.
    
    Core idea: The field's current pattern implies what sensors will see next.
    """
    
    def __init__(self, sensory_dim: int, device: torch.device, channels: int = 32):
        """
        Initialize prediction system.
        
        Args:
            sensory_dim: Number of sensors to predict
            device: Computation device
            channels: Number of field channels
        """
        self.sensory_dim = sensory_dim
        self.device = device
        self.channels = channels
        
        # Simple linear projection from field to sensors
        # This learns which field regions predict which sensors
        self.projection = torch.randn(sensory_dim, channels, device=device) * 0.1
        
        # History for momentum-based prediction
        self.sensor_history = deque(maxlen=5)
        
    def predict_next_sensors(self, field: torch.Tensor) -> torch.Tensor:
        """
        Predict next sensory input from current field state.
        
        Args:
            field: Current field state [D, H, W, C]
            
        Returns:
            Predicted sensor values [sensory_dim]
        """
        # Take a "slice" of the field (middle layer)
        mid_z = field.shape[2] // 2
        field_slice = field[:, :, mid_z, :]  # [D, H, C]
        
        # Global pooling to get summary
        field_summary = field_slice.mean(dim=[0, 1])  # [C]
        
        # Project to sensor space
        predictions = torch.matmul(self.projection, field_summary)
        
        # Add momentum from history if available
        if len(self.sensor_history) >= 2:
            # Simple momentum: current trend continues
            recent = torch.tensor(self.sensor_history[-1], device=self.device)
            older = torch.tensor(self.sensor_history[-2], device=self.device)
            momentum = recent - older
            predictions = predictions + momentum * 0.3
        
        return predictions
    
    def update_history(self, actual_sensors: torch.Tensor):
        """Update sensor history for momentum prediction."""
        self.sensor_history.append(actual_sensors.cpu().numpy())
    
    def compute_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error.
        
        Args:
            predicted: What we predicted
            actual: What actually happened
            
        Returns:
            Error per sensor
        """
        return actual - predicted
    
    def learn_from_error(self, error: torch.Tensor, field: torch.Tensor, learning_rate: float = 0.01):
        """
        Update projection based on prediction error.
        Simple supervised learning: adjust weights to reduce error.
        
        Args:
            error: Prediction error [sensory_dim]
            field: Field state that made the prediction
            learning_rate: How fast to learn
        """
        # Get field summary that was used for prediction
        mid_z = field.shape[2] // 2
        field_slice = field[:, :, mid_z, :]
        field_summary = field_slice.mean(dim=[0, 1])  # [C]
        
        # Gradient descent on projection weights
        # If we under-predicted sensor i, increase weights from active field channels
        for i in range(self.sensory_dim):
            self.projection[i] += learning_rate * error[i] * field_summary
        
        # Keep weights bounded
        self.projection = torch.clamp(self.projection, -1, 1)