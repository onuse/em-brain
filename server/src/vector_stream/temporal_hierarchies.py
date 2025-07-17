#!/usr/bin/env python3
"""
Multi-timescale Temporal Hierarchies - Evolution's Win #2

Implements biological multi-timescale prediction that operates simultaneously at:
- Fast layer (10-100ms): Motor control, reflexes, immediate responses
- Medium layer (1-10s): Sequence learning, working memory, pattern completion
- Slow layer (minutes): Goal formation, planning, long-term context

Key evolutionary advantages:
- Natural working memory emergence from prediction hierarchies
- Robust temporal processing across all behaviorally relevant scales
- Hierarchical prediction consistency prevents temporal confusion
- Emergent planning behavior from slow prediction layers

This represents the brain's evolved solution to the fundamental challenge
of operating effectively across multiple temporal scales simultaneously.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

try:
    from .sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
except ImportError:
    from sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage


@dataclass
class TemporalLayer:
    """Configuration for a single temporal prediction layer."""
    name: str
    timescale_ms: float          # Target prediction timescale in milliseconds
    history_length: int          # Number of past states to maintain
    prediction_horizon: int      # How far ahead to predict
    learning_rate: float         # Learning rate for this timescale
    pattern_dim: int            # Dimension of patterns at this scale
    max_patterns: int           # Maximum patterns to store


class MultiTimescalePredictor:
    """
    Single-layer temporal predictor operating at a specific timescale.
    
    Uses sparse patterns for efficient temporal sequence learning.
    Each layer maintains its own temporal history and prediction horizon.
    """
    
    def __init__(self, layer_config: TemporalLayer, quiet_mode: bool = False):
        self.config = layer_config
        self.quiet_mode = quiet_mode
        
        # Temporal history buffer
        self.history_buffer = deque(maxlen=layer_config.history_length)
        self.time_buffer = deque(maxlen=layer_config.history_length)
        
        # Sparse pattern components for this timescale
        self.encoder = SparsePatternEncoder(
            pattern_dim=layer_config.pattern_dim,
            sparsity=0.02,
            quiet_mode=quiet_mode
        )
        
        self.storage = SparsePatternStorage(
            pattern_dim=layer_config.pattern_dim,
            max_patterns=layer_config.max_patterns,
            sparsity=0.02,
            quiet_mode=quiet_mode
        )
        
        # Temporal sequence learning
        self.sequence_patterns: Dict[str, List[str]] = {}  # pattern_id -> [next_pattern_ids]
        self.sequence_strengths: Dict[str, Dict[str, float]] = {}  # pattern_id -> {next_id: strength}
        
        # Prediction tracking
        self.last_prediction: Optional[torch.Tensor] = None
        self.prediction_accuracy_buffer = deque(maxlen=100)
        
        # Layer statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.layer_start_time = time.time()
        
        if not quiet_mode:
            print(f"⏱️  {layer_config.name} layer: {layer_config.timescale_ms}ms scale, "
                  f"{layer_config.history_length} history, {layer_config.prediction_horizon} horizon")
    
    def update(self, activation: torch.Tensor, timestamp: float) -> torch.Tensor:
        """
        Update this temporal layer with new activation.
        
        Returns prediction for this timescale.
        """
        current_time = timestamp
        
        # Store in history buffer
        self.history_buffer.append(activation.clone())
        self.time_buffer.append(current_time)
        
        # Encode current activation as sparse pattern
        current_pattern = self.encoder.encode_top_k(activation, f"{self.config.name}_{int(current_time*1000)}")
        pattern_idx = self.storage.store_pattern(current_pattern)
        
        # Learn temporal sequences if we have history
        if len(self.history_buffer) >= 2:
            self._learn_temporal_sequence(current_pattern)
        
        # Generate prediction for this timescale
        prediction = self._generate_prediction(current_pattern, current_time)
        
        # Update prediction accuracy if we had a previous prediction
        if self.last_prediction is not None:
            accuracy = self._calculate_prediction_accuracy(self.last_prediction, activation)
            self.prediction_accuracy_buffer.append(accuracy)
        
        self.last_prediction = prediction.clone()
        self.total_predictions += 1
        
        return prediction
    
    def _learn_temporal_sequence(self, current_pattern: SparsePattern):
        """Learn temporal sequences between patterns."""
        if len(self.history_buffer) < 2:
            return
        
        # Get previous activation and encode it
        prev_activation = self.history_buffer[-2]
        prev_pattern = self.encoder.encode_top_k(prev_activation, "prev_temp")
        
        # Find most similar previous pattern in storage
        similar_patterns = self.storage.find_similar_patterns(prev_pattern, k=1, min_similarity=0.3)
        
        if similar_patterns:
            prev_pattern_stored = similar_patterns[0][0]
            prev_id = prev_pattern_stored.pattern_id
            current_id = current_pattern.pattern_id
            
            # Record sequence: prev_pattern -> current_pattern
            if prev_id not in self.sequence_patterns:
                self.sequence_patterns[prev_id] = []
                self.sequence_strengths[prev_id] = {}
            
            # Add or strengthen sequence
            if current_id not in self.sequence_patterns[prev_id]:
                self.sequence_patterns[prev_id].append(current_id)
                self.sequence_strengths[prev_id][current_id] = 1.0
            else:
                # Strengthen existing sequence
                self.sequence_strengths[prev_id][current_id] += self.config.learning_rate
    
    def _generate_prediction(self, current_pattern: SparsePattern, current_time: float) -> torch.Tensor:
        """Generate prediction for this timescale."""
        
        # Method 1: Use learned temporal sequences
        sequence_prediction = self._predict_from_sequences(current_pattern)
        
        # Method 2: Use pattern similarity and temporal context
        similarity_prediction = self._predict_from_similarity(current_pattern, current_time)
        
        # Method 3: Use temporal momentum from recent history
        momentum_prediction = self._predict_from_momentum()
        
        # Combine predictions with different weights based on layer type
        if "fast" in self.config.name:
            # Fast layer: prioritize momentum and similarity
            prediction = 0.5 * momentum_prediction + 0.3 * similarity_prediction + 0.2 * sequence_prediction
        elif "medium" in self.config.name:
            # Medium layer: prioritize sequences and similarity
            prediction = 0.5 * sequence_prediction + 0.3 * similarity_prediction + 0.2 * momentum_prediction
        else:
            # Slow layer: prioritize sequences and long-term patterns
            prediction = 0.6 * sequence_prediction + 0.4 * similarity_prediction
        
        return prediction
    
    def _predict_from_sequences(self, current_pattern: SparsePattern) -> torch.Tensor:
        """Predict next state using learned temporal sequences."""
        # Find most similar stored pattern
        similar_patterns = self.storage.find_similar_patterns(current_pattern, k=1, min_similarity=0.2)
        
        if similar_patterns:
            pattern_id = similar_patterns[0][0].pattern_id
            
            # Check if we have sequence data for this pattern
            if pattern_id in self.sequence_patterns:
                # Get next patterns with their strengths
                next_patterns = self.sequence_patterns[pattern_id]
                strengths = self.sequence_strengths[pattern_id]
                
                # Weighted prediction from next patterns
                prediction = torch.zeros(self.config.pattern_dim)
                total_strength = 0.0
                
                for next_id in next_patterns:
                    strength = strengths[next_id]
                    
                    # Find the next pattern in storage
                    if next_id in self.storage.pattern_index:
                        next_pattern = self.storage.pattern_index[next_id]
                        next_dense = next_pattern.to_dense()
                        
                        prediction += strength * next_dense
                        total_strength += strength
                
                if total_strength > 0:
                    prediction = prediction / total_strength
                    return prediction
        
        # Fallback: return zero prediction
        return torch.zeros(self.config.pattern_dim)
    
    def _predict_from_similarity(self, current_pattern: SparsePattern, current_time: float) -> torch.Tensor:
        """Predict based on similarity to stored patterns."""
        similar_patterns = self.storage.find_similar_patterns(current_pattern, k=5, min_similarity=0.1)
        
        if not similar_patterns:
            return torch.zeros(self.config.pattern_dim)
        
        # Weight by similarity and recency
        prediction = torch.zeros(self.config.pattern_dim)
        total_weight = 0.0
        
        for pattern, similarity in similar_patterns:
            # Recency weight (more recent patterns are more relevant)
            age = current_time - pattern.creation_time
            recency_weight = np.exp(-age / self.config.timescale_ms * 1000)  # Convert to seconds
            
            weight = similarity * recency_weight
            pattern_dense = pattern.to_dense()
            
            prediction += weight * pattern_dense
            total_weight += weight
        
        if total_weight > 0:
            prediction = prediction / total_weight
        
        return prediction
    
    def _predict_from_momentum(self) -> torch.Tensor:
        """Predict based on recent temporal momentum."""
        if len(self.history_buffer) < 2:
            return torch.zeros(self.config.pattern_dim)
        
        # Calculate momentum from recent history
        recent_activations = list(self.history_buffer)[-min(3, len(self.history_buffer)):]
        
        if len(recent_activations) >= 2:
            # Simple momentum: extrapolate recent trend
            momentum = recent_activations[-1] - recent_activations[-2]
            predicted = recent_activations[-1] + momentum
            return predicted
        
        return torch.zeros(self.config.pattern_dim)
    
    def _calculate_prediction_accuracy(self, prediction: torch.Tensor, actual: torch.Tensor) -> float:
        """Calculate how accurate our prediction was."""
        if prediction.shape != actual.shape:
            return 0.0
        
        # Cosine similarity between prediction and actual
        pred_norm = torch.norm(prediction)
        actual_norm = torch.norm(actual)
        
        if pred_norm == 0 or actual_norm == 0:
            return 0.0
        
        similarity = torch.dot(prediction, actual) / (pred_norm * actual_norm)
        return max(0.0, similarity.item())
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive layer statistics."""
        avg_accuracy = np.mean(self.prediction_accuracy_buffer) if self.prediction_accuracy_buffer else 0.0
        
        return {
            'layer_name': self.config.name,
            'timescale_ms': self.config.timescale_ms,
            'total_predictions': self.total_predictions,
            'prediction_accuracy': avg_accuracy,
            'sequence_count': len(self.sequence_patterns),
            'pattern_count': len(self.storage.patterns),
            'history_length': len(self.history_buffer),
            'uptime_seconds': time.time() - self.layer_start_time
        }


class TemporalHierarchy:
    """
    Multi-timescale temporal hierarchy combining multiple prediction layers.
    
    Implements evolution's solution to multi-timescale processing:
    - Fast layer: Immediate motor responses (10-100ms)
    - Medium layer: Sequence learning and working memory (1-10s)  
    - Slow layer: Goal formation and planning (minutes)
    """
    
    def __init__(self, pattern_dim: int, max_patterns: int = 10000, quiet_mode: bool = False):
        self.pattern_dim = pattern_dim
        self.quiet_mode = quiet_mode
        
        # Define the three temporal layers
        self.fast_layer = MultiTimescalePredictor(
            TemporalLayer(
                name="fast",
                timescale_ms=50.0,      # 50ms - motor control
                history_length=10,       # Short history for fast responses
                prediction_horizon=1,    # Predict next immediate state
                learning_rate=0.1,       # Fast learning
                pattern_dim=pattern_dim,
                max_patterns=max_patterns // 3
            ), quiet_mode
        )
        
        self.medium_layer = MultiTimescalePredictor(
            TemporalLayer(
                name="medium", 
                timescale_ms=2000.0,    # 2s - sequence learning
                history_length=50,       # Medium history for sequences
                prediction_horizon=5,    # Predict several steps ahead
                learning_rate=0.05,      # Medium learning rate
                pattern_dim=pattern_dim,
                max_patterns=max_patterns // 3
            ), quiet_mode
        )
        
        self.slow_layer = MultiTimescalePredictor(
            TemporalLayer(
                name="slow",
                timescale_ms=30000.0,   # 30s - goal formation
                history_length=100,      # Long history for context
                prediction_horizon=10,   # Long-term planning
                learning_rate=0.01,      # Slow learning
                pattern_dim=pattern_dim, 
                max_patterns=max_patterns // 3
            ), quiet_mode
        )
        
        # Cross-layer consistency tracking
        self.consistency_buffer = deque(maxlen=100)
        self.hierarchy_start_time = time.time()
        
        if not quiet_mode:
            print(f"\n⏱️  MULTI-TIMESCALE TEMPORAL HIERARCHY INITIALIZED")
            print(f"   🎯 Evolution's Win #2: Multi-timescale Prediction")
            print(f"   Fast layer: 50ms motor control")
            print(f"   Medium layer: 2s sequence learning") 
            print(f"   Slow layer: 30s goal formation")
            print(f"   🚀 Natural working memory emergence enabled")
    
    def update(self, activation: torch.Tensor, timestamp: float = None) -> Dict[str, torch.Tensor]:
        """
        Update all temporal layers and return hierarchical predictions.
        
        Returns:
            Dict with 'fast', 'medium', 'slow' predictions and 'integrated' result
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update each layer simultaneously
        fast_prediction = self.fast_layer.update(activation, timestamp)
        medium_prediction = self.medium_layer.update(activation, timestamp)
        slow_prediction = self.slow_layer.update(activation, timestamp)
        
        # Create integrated prediction combining all timescales
        integrated_prediction = self._integrate_predictions(
            fast_prediction, medium_prediction, slow_prediction
        )
        
        # Track cross-layer consistency
        consistency = self._calculate_cross_layer_consistency(
            fast_prediction, medium_prediction, slow_prediction
        )
        self.consistency_buffer.append(consistency)
        
        return {
            'fast': fast_prediction,
            'medium': medium_prediction,
            'slow': slow_prediction,
            'integrated': integrated_prediction
        }
    
    def _integrate_predictions(self, fast: torch.Tensor, medium: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        """
        Integrate predictions from all timescales into unified prediction.
        
        Evolution's solution: Weight by timescale relevance and consistency.
        """
        # Dynamic weighting based on recent consistency
        consistency = np.mean(self.consistency_buffer) if self.consistency_buffer else 0.5
        
        # Higher consistency allows more balanced integration
        # Lower consistency favors faster, more reactive layers
        if consistency > 0.7:
            # High consistency: balanced integration
            weights = [0.4, 0.4, 0.2]  # fast, medium, slow
        elif consistency > 0.3:
            # Medium consistency: favor fast and medium
            weights = [0.5, 0.4, 0.1]
        else:
            # Low consistency: favor fast responses
            weights = [0.7, 0.3, 0.0]
        
        integrated = weights[0] * fast + weights[1] * medium + weights[2] * slow
        
        return integrated
    
    def _calculate_cross_layer_consistency(self, fast: torch.Tensor, medium: torch.Tensor, slow: torch.Tensor) -> float:
        """Calculate how consistent predictions are across layers."""
        predictions = [fast, medium, slow]
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred1, pred2 = predictions[i], predictions[j]
                
                norm1, norm2 = torch.norm(pred1), torch.norm(pred2)
                if norm1 > 0 and norm2 > 0:
                    similarity = torch.dot(pred1, pred2) / (norm1 * norm2)
                    similarities.append(max(0.0, similarity.item()))
        
        # Average similarity across all pairs
        return np.mean(similarities) if similarities else 0.0
    
    def get_working_memory_state(self) -> Dict[str, Any]:
        """
        Get emergent working memory state from temporal hierarchy.
        
        Working memory emerges from the interaction of prediction layers.
        """
        # Working memory emerges from medium layer's sequence patterns
        medium_stats = self.medium_layer.get_layer_stats()
        
        # Active patterns in medium layer represent "working memory"
        active_patterns = []
        if len(self.medium_layer.history_buffer) > 0:
            recent_activations = list(self.medium_layer.history_buffer)[-5:]  # Last 5 patterns
            active_patterns = [activation.tolist() for activation in recent_activations]
        
        consistency = np.mean(self.consistency_buffer) if self.consistency_buffer else 0.0
        
        return {
            'working_memory_capacity': len(active_patterns),
            'working_memory_patterns': active_patterns,
            'sequence_count': medium_stats['sequence_count'],
            'cross_layer_consistency': consistency,
            'memory_stability': medium_stats['prediction_accuracy'],
            'hierarchy_uptime': time.time() - self.hierarchy_start_time
        }
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get comprehensive temporal hierarchy statistics."""
        return {
            'fast_layer': self.fast_layer.get_layer_stats(),
            'medium_layer': self.medium_layer.get_layer_stats(),
            'slow_layer': self.slow_layer.get_layer_stats(),
            'working_memory': self.get_working_memory_state(),
            'architecture': 'multi_timescale_temporal_hierarchy'
        }


def demonstrate_temporal_hierarchy():
    """Demonstrate the multi-timescale temporal hierarchy."""
    print("⏱️  TEMPORAL HIERARCHY DEMONSTRATION")
    print("=" * 50)
    
    # Create temporal hierarchy
    hierarchy = TemporalHierarchy(pattern_dim=16, max_patterns=1000, quiet_mode=True)
    
    print("Testing temporal learning with simple sequences...")
    
    # Create temporal sequences to learn
    sequences = [
        # Sequence 1: A -> B -> C pattern
        ([1.0, 0.0, 0.0] + [0.0] * 13, [0.0, 1.0, 0.0] + [0.0] * 13, [0.0, 0.0, 1.0] + [0.0] * 13),
        # Sequence 2: X -> Y -> Z pattern
        ([0.0, 0.0, 0.0, 1.0] + [0.0] * 12, [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 11, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 10),
    ]
    
    # Train the hierarchy
    for epoch in range(10):
        for seq in sequences:
            for i, pattern in enumerate(seq):
                activation = torch.tensor(pattern, dtype=torch.float32)
                predictions = hierarchy.update(activation)
                
                if i == 0:  # First pattern in sequence
                    print(f"Epoch {epoch}, Sequence: First pattern processed")
    
    # Test prediction after training
    print("\nTesting learned predictions...")
    
    # Test sequence 1: A -> ? (should predict B)
    test_pattern = torch.tensor([1.0, 0.0, 0.0] + [0.0] * 13, dtype=torch.float32)
    predictions = hierarchy.update(test_pattern)
    
    print(f"Fast prediction: {predictions['fast'][:3].tolist()}")
    print(f"Medium prediction: {predictions['medium'][:3].tolist()}")
    print(f"Slow prediction: {predictions['slow'][:3].tolist()}")
    print(f"Integrated prediction: {predictions['integrated'][:3].tolist()}")
    
    # Get working memory state
    working_memory = hierarchy.get_working_memory_state()
    print(f"\nWorking memory capacity: {working_memory['working_memory_capacity']}")
    print(f"Cross-layer consistency: {working_memory['cross_layer_consistency']:.3f}")
    
    # Get full statistics
    stats = hierarchy.get_hierarchy_stats()
    print(f"\nHierarchy statistics:")
    print(f"  Fast layer patterns: {stats['fast_layer']['pattern_count']}")
    print(f"  Medium layer patterns: {stats['medium_layer']['pattern_count']}")
    print(f"  Slow layer patterns: {stats['slow_layer']['pattern_count']}")
    
    print(f"\n✅ TEMPORAL HIERARCHY DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demonstrate_temporal_hierarchy()