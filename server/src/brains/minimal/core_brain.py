#!/usr/bin/env python3
"""
Minimal Vector Stream Brain

This is a prototype implementation of vector-stream processing that replaces 
experience nodes with continuous vector streams and temporal patterns.

Key concepts:
- Modular streams (sensory, motor, temporal)
- Time as a data stream (organic metronome)
- Cross-stream pattern learning
- Continuous prediction from vector flow
"""

import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..brain_maintenance_interface import BrainMaintenanceInterface


@dataclass
class VectorPattern:
    """A learned pattern in a vector stream."""
    activation_pattern: torch.Tensor
    temporal_context: float  # Relative timing when this pattern typically occurs
    frequency: int  # How often this pattern has been seen
    last_seen: float  # When this pattern was last activated


class VectorStream:
    """
    A continuous stream of vector activations representing one modality.
    
    This is like a brain region (visual cortex, motor cortex, etc.) that:
    - Maintains a rolling buffer of recent activations
    - Learns patterns within the stream
    - Can predict future activations based on current patterns
    """
    
    def __init__(self, dim: int, buffer_size: int = 100, name: str = "stream"):
        self.dim = dim
        self.buffer_size = buffer_size
        self.name = name
        
        # Rolling buffer of recent activations
        self.activation_buffer = torch.zeros(buffer_size, dim)
        self.time_buffer = torch.zeros(buffer_size)
        self.buffer_index = 0
        self.buffer_full = False
        
        # Learned patterns in this stream
        self.patterns: List[VectorPattern] = []
        self.pattern_similarity_threshold = 0.8
        
        # Prediction state
        self.current_activation = torch.zeros(dim)
        self.predicted_next_activation = torch.zeros(dim)
        
        print(f"🧠 VectorStream '{name}' initialized: {dim}D, buffer={buffer_size}")
    
    def update(self, new_activation: torch.Tensor, timestamp: float) -> torch.Tensor:
        """
        Update the stream with a new activation vector.
        
        This is like neural firing in a brain region.
        """
        # Input validation
        if not torch.isfinite(new_activation).all():
            # Replace NaN/Inf with zeros for stability
            new_activation = torch.where(torch.isfinite(new_activation), new_activation, torch.zeros_like(new_activation))
        
        if new_activation.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim}D activation, got {new_activation.shape[0]}D")
        
        # Store activation in rolling buffer
        self.activation_buffer[self.buffer_index] = new_activation
        self.time_buffer[self.buffer_index] = timestamp
        
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.buffer_full = True
        
        # Update current state
        self.current_activation = new_activation
        
        # Learn patterns from this activation
        self._learn_patterns(new_activation, timestamp)
        
        # Generate prediction for next activation
        self.predicted_next_activation = self._predict_next_activation(timestamp)
        
        return self.current_activation
    
    def _learn_patterns(self, activation: torch.Tensor, timestamp: float):
        """Learn recurring patterns in the activation stream."""
        # Check if this activation matches any existing patterns
        for pattern in self.patterns:
            # Handle zero vectors in cosine similarity
            if torch.norm(activation) < 1e-8 or torch.norm(pattern.activation_pattern) < 1e-8:
                similarity = 0.0
            else:
                similarity = torch.cosine_similarity(activation, pattern.activation_pattern, dim=0).item()
            
            if similarity > self.pattern_similarity_threshold:
                # Reinforce existing pattern
                pattern.frequency += 1
                pattern.last_seen = timestamp
                # Update pattern with weighted average
                alpha = 0.1  # Learning rate
                pattern.activation_pattern = (1 - alpha) * pattern.activation_pattern + alpha * activation
                return
        
        # Create new pattern if none matched
        if len(self.patterns) < 50:  # Limit pattern memory
            new_pattern = VectorPattern(
                activation_pattern=activation.clone(),
                temporal_context=self._get_temporal_context(timestamp),
                frequency=1,
                last_seen=timestamp
            )
            self.patterns.append(new_pattern)
    
    def _get_temporal_context(self, timestamp: float) -> float:
        """Get temporal context relative to recent activations."""
        if not self.buffer_full and self.buffer_index < 2:
            return 0.0
        
        # Look at timing pattern of recent activations
        recent_times = self.time_buffer[:self.buffer_index] if not self.buffer_full else self.time_buffer
        if len(recent_times) > 1:
            # Calculate average interval
            intervals = torch.diff(recent_times)
            avg_interval = torch.mean(intervals).item()
            return avg_interval
        
        return 0.0
    
    def _predict_next_activation(self, current_time: float) -> torch.Tensor:
        """Predict the next activation based on current patterns and timing."""
        if len(self.patterns) == 0:
            return torch.zeros_like(self.current_activation)
        
        # Find patterns that might predict what comes next
        prediction = torch.zeros_like(self.current_activation)
        total_weight = 0.0
        
        for pattern in self.patterns:
            # Weight by frequency and recency
            frequency_weight = pattern.frequency / max(1, len(self.patterns))
            recency_weight = 1.0 / (1.0 + abs(current_time - pattern.last_seen))
            
            # Weight by temporal context similarity
            expected_interval = pattern.temporal_context
            current_interval = self._get_temporal_context(current_time)
            temporal_weight = 1.0 / (1.0 + abs(expected_interval - current_interval))
            
            total_pattern_weight = frequency_weight * recency_weight * temporal_weight
            
            prediction += total_pattern_weight * pattern.activation_pattern
            total_weight += total_pattern_weight
        
        if total_weight > 1e-8:  # Avoid division by zero
            prediction = prediction / total_weight
        else:
            prediction = torch.zeros_like(self.current_activation)
        
        return prediction
    
    def get_stream_state(self) -> Dict[str, Any]:
        """Get current stream state for debugging."""
        return {
            'name': self.name,
            'current_activation': self.current_activation.tolist(),
            'predicted_next': self.predicted_next_activation.tolist(),
            'pattern_count': len(self.patterns),
            'buffer_utilization': self.buffer_index / self.buffer_size if not self.buffer_full else 1.0
        }


class MinimalVectorStreamBrain(BrainMaintenanceInterface):
    """
    Minimal vector stream brain with 3 modular streams:
    - Sensory stream (external world)
    - Motor stream (actions)  
    - Temporal stream (timing/metronome)
    """
    
    def __init__(self, sensory_dim: int = 16, motor_dim: int = 8, temporal_dim: int = 4):
        # Initialize maintenance interface
        super().__init__()
        
        # Create modular streams
        self.sensory_stream = VectorStream(sensory_dim, name="sensory")
        self.motor_stream = VectorStream(motor_dim, name="motor")
        self.temporal_stream = VectorStream(temporal_dim, name="temporal")
        
        # Cross-stream learning weights (like synaptic connections between brain regions)
        self.sensory_to_motor = torch.randn(sensory_dim, motor_dim) * 0.1
        self.temporal_to_motor = torch.randn(temporal_dim, motor_dim) * 0.1
        self.sensory_to_temporal = torch.randn(sensory_dim, temporal_dim) * 0.1
        
        # Temporal metronome state
        self.start_time = time.time()
        self.last_process_time = self.start_time
        
        # Statistics
        self.total_cycles = 0
        self.prediction_accuracy_history = []
        
        print("🧠 MinimalVectorStreamBrain initialized")
        print(f"   Sensory stream: {sensory_dim}D")
        print(f"   Motor stream: {motor_dim}D") 
        print(f"   Temporal stream: {temporal_dim}D")
        print("   Cross-stream learning enabled")
    
    def process_sensory_input(self, sensory_vector: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input through vector streams to generate motor output.
        
        This is the core prediction loop - biological style!
        """
        current_time = time.time()
        cycle_start = current_time
        
        # Input validation and conversion
        if not sensory_vector:
            raise ValueError("Sensory vector cannot be empty")
        
        try:
            sensory_tensor = torch.tensor(sensory_vector, dtype=torch.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid sensory vector format: {e}")
        
        # Validate tensor dimensions
        if sensory_tensor.dim() != 1:
            raise ValueError(f"Expected 1D sensory vector, got {sensory_tensor.dim()}D")
        
        # Ensure finite values
        if not torch.isfinite(sensory_tensor).all():
            sensory_tensor = torch.where(torch.isfinite(sensory_tensor), sensory_tensor, torch.zeros_like(sensory_tensor))
        
        # Generate temporal context (organic metronome)
        temporal_vector = self._generate_temporal_context(current_time)
        
        # Update streams with current inputs
        sensory_activation = self.sensory_stream.update(sensory_tensor, current_time)
        temporal_activation = self.temporal_stream.update(temporal_vector, current_time)
        
        # Cross-stream prediction: sensory + temporal -> motor
        motor_prediction = self._predict_motor_output(sensory_activation, temporal_activation)
        
        # Update motor stream with prediction
        motor_activation = self.motor_stream.update(motor_prediction, current_time)
        
        # Learn cross-stream associations
        self._update_cross_stream_weights(sensory_activation, temporal_activation, motor_activation)
        
        # Track performance
        self.total_cycles += 1
        self.last_process_time = current_time
        
        # Return motor output and brain state
        brain_state = {
            'cycle_time_ms': (time.time() - cycle_start) * 1000,
            'sensory_stream': self.sensory_stream.get_stream_state(),
            'motor_stream': self.motor_stream.get_stream_state(),
            'temporal_stream': self.temporal_stream.get_stream_state(),
            'total_cycles': self.total_cycles,
            'prediction_confidence': self._estimate_prediction_confidence()
        }
        
        return motor_activation.tolist(), brain_state
    
    def _generate_temporal_context(self, current_time: float) -> torch.Tensor:
        """
        Generate temporal context vector (organic metronome).
        
        This represents stable biological rhythms that provide temporal reference.
        """
        relative_time = current_time - self.start_time
        time_since_last = current_time - self.last_process_time
        
        # Multiple temporal frequencies (like different brain oscillations)
        temporal_vector = torch.tensor([
            np.sin(relative_time * 2 * np.pi * 1.0),    # 1 Hz oscillation (like breathing)
            np.sin(relative_time * 2 * np.pi * 10.0),   # 10 Hz oscillation (like alpha waves)
            time_since_last,                             # Delta timing
            relative_time % 1.0                          # Cyclic component
        ], dtype=torch.float32)
        
        return temporal_vector
    
    def _predict_motor_output(self, sensory_activation: torch.Tensor, temporal_activation: torch.Tensor) -> torch.Tensor:
        """Predict motor output from sensory and temporal activations."""
        # Linear combination (like neural network layer)
        sensory_contribution = torch.matmul(sensory_activation, self.sensory_to_motor)
        temporal_contribution = torch.matmul(temporal_activation, self.temporal_to_motor)
        
        # Combine contributions
        motor_prediction = sensory_contribution + temporal_contribution
        
        # Apply activation function (like neuron firing threshold)
        motor_prediction = torch.tanh(motor_prediction)
        
        return motor_prediction
    
    def _update_cross_stream_weights(self, sensory: torch.Tensor, temporal: torch.Tensor, motor: torch.Tensor):
        """Update cross-stream connection weights based on successful predictions."""
        learning_rate = 0.001
        
        # Hebbian learning: strengthen connections that fire together
        # sensory -> motor
        sensory_motor_update = torch.outer(sensory, motor) * learning_rate
        self.sensory_to_motor += sensory_motor_update
        
        # temporal -> motor  
        temporal_motor_update = torch.outer(temporal, motor) * learning_rate
        self.temporal_to_motor += temporal_motor_update
        
        # Keep weights bounded
        self.sensory_to_motor = torch.clamp(self.sensory_to_motor, -1.0, 1.0)
        self.temporal_to_motor = torch.clamp(self.temporal_to_motor, -1.0, 1.0)
    
    def _estimate_prediction_confidence(self) -> float:
        """Estimate how confident the brain is in its predictions."""
        # Simple heuristic based on pattern consistency
        total_patterns = (len(self.sensory_stream.patterns) + 
                         len(self.motor_stream.patterns) + 
                         len(self.temporal_stream.patterns))
        
        if total_patterns == 0:
            return 0.0
        
        # More patterns = more confidence, but with diminishing returns
        confidence = min(0.9, total_patterns / 50.0)
        return confidence
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics."""
        return {
            'total_cycles': self.total_cycles,
            'streams': {
                'sensory': self.sensory_stream.get_stream_state(),
                'motor': self.motor_stream.get_stream_state(),
                'temporal': self.temporal_stream.get_stream_state()
            },
            'cross_stream_weights': {
                'sensory_to_motor_norm': torch.norm(self.sensory_to_motor).item(),
                'temporal_to_motor_norm': torch.norm(self.temporal_to_motor).item()
            },
            'prediction_confidence': self._estimate_prediction_confidence()
        }
    
    def light_maintenance(self) -> None:
        """Quick cleanup operations for minimal vector stream brain."""
        current_time = time.time()
        
        # Clean up old patterns from each stream (light cleanup)
        for stream in [self.sensory_stream, self.motor_stream, self.temporal_stream]:
            if len(stream.patterns) > 40:  # Only if we have many patterns
                # Remove least recently used patterns
                stream.patterns.sort(key=lambda p: p.last_seen)
                stream.patterns = stream.patterns[-35:]  # Keep most recent 35
        
        # Clear old prediction accuracy history
        if len(self.prediction_accuracy_history) > 100:
            self.prediction_accuracy_history = self.prediction_accuracy_history[-50:]
    
    def heavy_maintenance(self) -> None:
        """Moderate maintenance operations for minimal vector stream brain."""
        # Normalize cross-stream weights to prevent drift
        weight_scale = 0.95  # Slight decay to prevent unbounded growth
        self.sensory_to_motor *= weight_scale
        self.temporal_to_motor *= weight_scale
        self.sensory_to_temporal *= weight_scale
        
        # Consolidate similar patterns in each stream
        for stream in [self.sensory_stream, self.motor_stream, self.temporal_stream]:
            if len(stream.patterns) > 20:
                self._consolidate_similar_patterns(stream)
    
    def deep_consolidation(self) -> None:
        """Intensive consolidation operations for minimal vector stream brain."""
        # Reset buffer indices to defragment memory
        for stream in [self.sensory_stream, self.motor_stream, self.temporal_stream]:
            if stream.buffer_full:
                # Rotate buffer to bring most recent data to the front
                recent_data = stream.activation_buffer[stream.buffer_index:].clone()
                older_data = stream.activation_buffer[:stream.buffer_index].clone()
                stream.activation_buffer = torch.cat([recent_data, older_data])
                
                recent_times = stream.time_buffer[stream.buffer_index:].clone()
                older_times = stream.time_buffer[:stream.buffer_index].clone()
                stream.time_buffer = torch.cat([recent_times, older_times])
                
                stream.buffer_index = 0
                stream.buffer_full = False
        
        # Aggressive pattern consolidation and cleanup
        for stream in [self.sensory_stream, self.motor_stream, self.temporal_stream]:
            # Remove low-frequency patterns
            stream.patterns = [p for p in stream.patterns if p.frequency > 1]
            
            # Consolidate very similar patterns
            self._consolidate_similar_patterns(stream, threshold=0.9)  # Higher threshold
            
            # Limit pattern count
            if len(stream.patterns) > 30:
                stream.patterns.sort(key=lambda p: p.frequency * (1.0 / (time.time() - p.last_seen + 1)), reverse=True)
                stream.patterns = stream.patterns[:25]  # Keep top 25
        
        # Reset cross-stream weights if they've become too large
        max_weight = 2.0
        if torch.max(torch.abs(self.sensory_to_motor)) > max_weight:
            self.sensory_to_motor = torch.clamp(self.sensory_to_motor, -max_weight, max_weight)
        if torch.max(torch.abs(self.temporal_to_motor)) > max_weight:
            self.temporal_to_motor = torch.clamp(self.temporal_to_motor, -max_weight, max_weight)
        if torch.max(torch.abs(self.sensory_to_temporal)) > max_weight:
            self.sensory_to_temporal = torch.clamp(self.sensory_to_temporal, -max_weight, max_weight)
    
    def _consolidate_similar_patterns(self, stream: VectorStream, threshold: float = 0.85):
        """Consolidate patterns that are very similar to reduce memory usage."""
        if len(stream.patterns) < 2:
            return
        
        consolidated_patterns = []
        used_indices = set()
        
        for i, pattern1 in enumerate(stream.patterns):
            if i in used_indices:
                continue
            
            # Find similar patterns
            similar_patterns = [pattern1]
            used_indices.add(i)
            
            for j, pattern2 in enumerate(stream.patterns):
                if j <= i or j in used_indices:
                    continue
                
                # Calculate similarity
                if (torch.norm(pattern1.activation_pattern) > 1e-8 and 
                    torch.norm(pattern2.activation_pattern) > 1e-8):
                    similarity = torch.cosine_similarity(
                        pattern1.activation_pattern, 
                        pattern2.activation_pattern, 
                        dim=0
                    ).item()
                    
                    if similarity > threshold:
                        similar_patterns.append(pattern2)
                        used_indices.add(j)
            
            # Create consolidated pattern from similar patterns
            if len(similar_patterns) > 1:
                total_frequency = sum(p.frequency for p in similar_patterns)
                total_weight = sum(p.frequency for p in similar_patterns)
                
                # Weighted average of activation patterns
                consolidated_activation = torch.zeros_like(similar_patterns[0].activation_pattern)
                for pattern in similar_patterns:
                    weight = pattern.frequency / total_weight
                    consolidated_activation += weight * pattern.activation_pattern
                
                # Use most recent temporal context and last_seen
                most_recent = max(similar_patterns, key=lambda p: p.last_seen)
                
                consolidated_pattern = VectorPattern(
                    activation_pattern=consolidated_activation,
                    temporal_context=most_recent.temporal_context,
                    frequency=total_frequency,
                    last_seen=most_recent.last_seen
                )
                consolidated_patterns.append(consolidated_pattern)
            else:
                consolidated_patterns.append(pattern1)
        
        stream.patterns = consolidated_patterns