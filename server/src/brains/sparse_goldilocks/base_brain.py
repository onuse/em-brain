#!/usr/bin/env python3
"""
Goldilocks Brain - Massively Parallel Vector Stream Processing

The "just right" implementation: simple enough for emergence, sophisticated enough 
for intelligence. Built around 5 core primitives that scale to millions of patterns.

Core Philosophy:
- Embarrassingly simple operations
- Massive GPU parallelism 
- Emergent intelligence from scale + speed + time
- Minimal biological-inspired features where needed

Core Primitives:
1. Massively parallel pattern storage (1M+ patterns in GPU memory)
2. GPU-parallel similarity search across ALL patterns
3. Simple replacement strategy (frequency + recency)
4. Temporal recency weighting
5. Cross-stream co-activation tracking

Expected Emergence:
- Spatial intelligence from pattern clustering
- Motor skills from cross-stream co-activation
- Temporal learning from recency weighting
- Working memory from activation dynamics
- Planning from sequence patterns
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StreamConfig:
    """Configuration for a vector stream."""
    dim: int
    max_patterns: int = 1_000_000
    similarity_threshold: float = 0.7
    decay_rate: float = 0.1  # How fast old patterns fade
    replacement_rate: float = 0.01  # Fraction to replace when full


class MassivePatternStorage:
    """
    Massively parallel pattern storage using GPU tensors.
    
    This is the "abacus" - the fundamental computational primitive
    for storing and searching millions of patterns in parallel.
    """
    
    def __init__(self, config: StreamConfig, stream_name: str):
        self.config = config
        self.stream_name = stream_name
        self.max_patterns = config.max_patterns
        self.pattern_dim = config.dim
        
        # GPU device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 GPU acceleration enabled for {stream_name}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"🚀 MPS acceleration enabled for {stream_name}")
        else:
            self.device = torch.device('cpu')
            print(f"⚠️  CPU fallback for {stream_name}")
        
        # Core storage tensors - the fundamental data structures
        self.patterns = torch.zeros(self.max_patterns, self.pattern_dim, device=self.device)
        self.timestamps = torch.zeros(self.max_patterns, device=self.device)
        self.frequencies = torch.zeros(self.max_patterns, device=self.device)
        self.last_activated = torch.zeros(self.max_patterns, device=self.device)
        
        # Pattern tracking
        self.pattern_count = 0
        self.current_time = 0.0
        
        # Performance statistics
        self.total_searches = 0
        self.total_stores = 0
        
        print(f"🧠 MassivePatternStorage '{stream_name}' initialized")
        print(f"   Capacity: {self.max_patterns:,} patterns")
        print(f"   Dimensions: {self.pattern_dim}D")
        print(f"   Device: {self.device}")
        print(f"   Memory allocated: {self._get_memory_usage():.1f}MB")
    
    def store_pattern(self, pattern: torch.Tensor, timestamp: float = None) -> int:
        """
        Store a pattern using simple replacement strategy.
        
        Core Primitive #1: Massively parallel pattern storage
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.current_time = timestamp
        pattern = pattern.to(self.device)
        
        if self.pattern_count < self.max_patterns:
            # Still have space - simple append
            idx = self.pattern_count
            self.patterns[idx] = pattern
            self.timestamps[idx] = timestamp
            self.frequencies[idx] = 1.0
            self.last_activated[idx] = timestamp
            self.pattern_count += 1
        else:
            # Storage full - replace based on simple heuristic
            idx = self._find_replacement_index(timestamp)
            self.patterns[idx] = pattern
            self.timestamps[idx] = timestamp
            self.frequencies[idx] = 1.0
            self.last_activated[idx] = timestamp
        
        self.total_stores += 1
        return idx
    
    def find_similar_patterns(self, query: torch.Tensor, k: int = 100, 
                            threshold: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-parallel similarity search across ALL stored patterns.
        
        Core Primitive #2: Massively parallel similarity search
        This is where the magic happens - comparing against millions of patterns in parallel.
        """
        if threshold is None:
            threshold = self.config.similarity_threshold
        
        query = query.to(self.device)
        self.total_searches += 1
        
        if self.pattern_count == 0:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Core operation: parallel similarity across ALL patterns
        active_patterns = self.patterns[:self.pattern_count]
        
        # Handle zero vectors gracefully
        query_norm = torch.norm(query)
        pattern_norms = torch.norm(active_patterns, dim=1)
        
        if query_norm < 1e-8:
            similarities = torch.zeros(self.pattern_count, device=self.device)
        else:
            # Cosine similarity with broadcasting
            similarities = torch.zeros(self.pattern_count, device=self.device)
            valid_patterns = pattern_norms > 1e-8
            
            if valid_patterns.any():
                valid_indices = torch.where(valid_patterns)[0]
                valid_pattern_data = active_patterns[valid_indices]
                
                # Batch cosine similarity computation
                dot_products = torch.matmul(valid_pattern_data, query)
                valid_norms = pattern_norms[valid_indices]
                cos_sims = dot_products / (valid_norms * query_norm)
                
                similarities[valid_indices] = cos_sims
        
        # Apply temporal recency weighting (Core Primitive #4)
        recency_weights = self._calculate_recency_weights()
        weighted_similarities = similarities * recency_weights[:self.pattern_count]
        
        # Find top-k matches above threshold
        above_threshold = weighted_similarities >= threshold
        if not above_threshold.any():
            # No matches above threshold - return top matches anyway
            top_k = min(k, self.pattern_count)
            values, indices = torch.topk(weighted_similarities, top_k)
            return indices, values
        
        # Get matches above threshold
        valid_indices = torch.where(above_threshold)[0]
        valid_similarities = weighted_similarities[valid_indices]
        
        # Sort by similarity and take top-k
        sorted_indices = torch.argsort(valid_similarities, descending=True)
        top_k = min(k, len(sorted_indices))
        
        result_indices = valid_indices[sorted_indices[:top_k]]
        result_similarities = valid_similarities[sorted_indices[:top_k]]
        
        # Update activation times for accessed patterns
        self.last_activated[result_indices] = self.current_time
        
        return result_indices, result_similarities
    
    def _find_replacement_index(self, current_time: float) -> int:
        """
        Find pattern to replace using simple heuristic.
        
        Core Primitive #3: Simple replacement strategy
        Combines frequency and recency for biological realism.
        """
        # Calculate replacement scores (lower = more likely to replace)
        recency_weights = self._calculate_recency_weights()
        frequency_weights = torch.log1p(self.frequencies)  # Log to reduce frequency dominance
        
        # Combined score: patterns that are old AND infrequent get replaced
        replacement_scores = recency_weights * frequency_weights
        
        # Find minimum score pattern to replace
        return torch.argmin(replacement_scores).item()
    
    def _calculate_recency_weights(self) -> torch.Tensor:
        """
        Calculate temporal recency weights.
        
        Core Primitive #4: Temporal recency weighting
        Recent patterns get higher weights in similarity search.
        """
        if self.pattern_count == 0:
            return torch.tensor([], device=self.device)
        
        # Time since last activation
        time_deltas = self.current_time - self.last_activated[:self.pattern_count]
        
        # Exponential decay: recent patterns get higher weights
        decay_rate = self.config.decay_rate
        recency_weights = torch.exp(-time_deltas * decay_rate)
        
        return recency_weights
    
    def reinforce_pattern(self, pattern_idx: int, strength: float = 1.0):
        """Reinforce a pattern by increasing its frequency."""
        if 0 <= pattern_idx < self.pattern_count:
            self.frequencies[pattern_idx] += strength
            self.last_activated[pattern_idx] = self.current_time
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern storage."""
        if self.pattern_count == 0:
            return {
                'pattern_count': 0,
                'utilization': 0.0,
                'avg_frequency': 0.0,
                'total_searches': self.total_searches,
                'total_stores': self.total_stores
            }
        
        active_frequencies = self.frequencies[:self.pattern_count]
        
        return {
            'pattern_count': self.pattern_count,
            'utilization': self.pattern_count / self.max_patterns,
            'avg_frequency': torch.mean(active_frequencies).item(),
            'max_frequency': torch.max(active_frequencies).item(),
            'total_searches': self.total_searches,
            'total_stores': self.total_stores,
            'memory_usage_mb': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        bytes_per_element = 4  # float32
        total_elements = (
            self.max_patterns * self.pattern_dim +  # patterns
            self.max_patterns * 3  # timestamps, frequencies, last_activated
        )
        return (total_elements * bytes_per_element) / (1024 * 1024)


class CrossStreamCoactivation:
    """
    Tracks co-activation patterns between streams.
    
    Core Primitive #5: Cross-stream co-activation tracking
    When patterns are active simultaneously across streams, they get linked.
    """
    
    def __init__(self, stream_names: List[str], device: torch.device):
        self.stream_names = stream_names
        self.device = device
        self.num_streams = len(stream_names)
        
        # Co-activation matrix: [stream_a_pattern, stream_b_pattern] -> strength
        # Start small and grow as needed
        self.max_patterns_per_stream = 10000
        self.coactivation_strength = {}
        
        for i, stream_a in enumerate(stream_names):
            for j, stream_b in enumerate(stream_names):
                if i != j:
                    key = f"{stream_a}->{stream_b}"
                    self.coactivation_strength[key] = torch.zeros(
                        self.max_patterns_per_stream, 
                        self.max_patterns_per_stream,
                        device=device
                    )
        
        print(f"🔗 CrossStreamCoactivation initialized for {stream_names}")
    
    def record_coactivation(self, activations: Dict[str, List[int]], strength: float = 1.0):
        """
        Vectorized co-activation recording for GPU acceleration.
        
        This is how cross-modal learning emerges naturally.
        """
        # Vectorized co-activation recording (GPU optimized)
        for stream_a, patterns_a in activations.items():
            for stream_b, patterns_b in activations.items():
                if stream_a != stream_b:
                    key = f"{stream_a}->{stream_b}"
                    
                    if key in self.coactivation_strength and patterns_a and patterns_b:
                        # Filter valid pattern indices
                        valid_patterns_a = [p for p in patterns_a if p < self.max_patterns_per_stream]
                        valid_patterns_b = [p for p in patterns_b if p < self.max_patterns_per_stream]
                        
                        if valid_patterns_a and valid_patterns_b:
                            # Vectorized update using advanced indexing (GPU accelerated)
                            device = self.coactivation_strength[key].device
                            
                            # Create index tensors
                            patterns_a_tensor = torch.tensor(valid_patterns_a, device=device)
                            patterns_b_tensor = torch.tensor(valid_patterns_b, device=device)
                            
                            # Create meshgrid for all combinations
                            a_indices, b_indices = torch.meshgrid(patterns_a_tensor, patterns_b_tensor, indexing='ij')
                            
                            # Vectorized update (single GPU operation instead of nested loops)
                            self.coactivation_strength[key][a_indices, b_indices] += strength
    
    def get_cross_predictions(self, from_stream: str, pattern_indices: List[int], 
                            to_stream: str, k: int = 10) -> List[Tuple[int, float]]:
        """Get predicted patterns in target stream based on source stream patterns."""
        key = f"{from_stream}->{to_stream}"
        
        if key not in self.coactivation_strength:
            return []
        
        # Sum co-activation strengths for all source patterns
        total_strength = torch.zeros(self.max_patterns_per_stream, device=self.device)
        
        for pattern_idx in pattern_indices:
            if pattern_idx < self.max_patterns_per_stream:
                total_strength += self.coactivation_strength[key][pattern_idx]
        
        # Get top-k predictions
        if total_strength.sum() > 0:
            values, indices = torch.topk(total_strength, k)
            # Filter out zero values
            non_zero = values > 0
            return [(indices[i].item(), values[i].item()) 
                   for i in range(len(values)) if non_zero[i]]
        
        return []
    
    def get_coactivation_stats(self) -> Dict[str, Any]:
        """Get statistics about cross-stream co-activations."""
        stats = {}
        total_links = 0
        
        for key, matrix in self.coactivation_strength.items():
            non_zero = (matrix > 0).sum().item()
            max_strength = matrix.max().item()
            avg_strength = matrix[matrix > 0].mean().item() if non_zero > 0 else 0.0
            
            stats[key] = {
                'total_links': non_zero,
                'max_strength': max_strength,
                'avg_strength': avg_strength
            }
            total_links += non_zero
        
        stats['total_cross_stream_links'] = total_links
        return stats


class GoldilocksVectorStream:
    """
    A "just right" vector stream using massively parallel primitives.
    
    Simple enough for emergence, sophisticated enough for intelligence.
    """
    
    def __init__(self, config: StreamConfig, stream_name: str):
        self.config = config
        self.stream_name = stream_name
        
        # Core storage using massively parallel primitives
        self.storage = MassivePatternStorage(config, stream_name)
        
        # Current state
        self.current_activation = torch.zeros(config.dim, device=self.storage.device)
        self.predicted_next = torch.zeros(config.dim, device=self.storage.device)
        
        # Rolling buffer for immediate temporal context (biological realism)
        self.buffer_size = 100
        self.activation_buffer = torch.zeros(self.buffer_size, config.dim, device=self.storage.device)
        self.time_buffer = torch.zeros(self.buffer_size, device=self.storage.device)
        self.buffer_index = 0
        self.buffer_full = False
        
        # Performance tracking
        self.cycle_count = 0
        self.prediction_attempts = 0
        self.prediction_successes = 0
        
        print(f"🧠 GoldilocksVectorStream '{stream_name}' ready")
        print(f"   Scalable to {config.max_patterns:,} patterns")
        print(f"   GPU-parallel similarity search enabled")
    
    def update(self, new_activation: torch.Tensor, timestamp: float = None) -> torch.Tensor:
        """
        Update stream with new activation.
        
        This is the core cycle: store pattern, find similar, predict next.
        """
        if timestamp is None:
            timestamp = time.time()
        
        new_activation = new_activation.to(self.storage.device)
        self.current_activation = new_activation
        self.cycle_count += 1
        
        # Store in rolling buffer (immediate temporal context)
        self.activation_buffer[self.buffer_index] = new_activation
        self.time_buffer[self.buffer_index] = timestamp
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.buffer_full = True
        
        # Find similar patterns using massively parallel search
        similar_indices, similarities = self.storage.find_similar_patterns(
            new_activation, k=50, threshold=self.config.similarity_threshold
        )
        
        # Reinforce matched patterns (simple Hebbian learning)
        for idx, similarity in zip(similar_indices, similarities):
            self.storage.reinforce_pattern(idx.item(), strength=similarity.item())
        
        # Store pattern if sufficiently novel
        if len(similar_indices) == 0 or similarities[0] < 0.9:
            self.storage.store_pattern(new_activation, timestamp)
        
        # Generate prediction from similar patterns
        self.predicted_next = self._predict_next_activation(similar_indices, similarities)
        
        return self.current_activation
    
    def _predict_next_activation(self, similar_indices: torch.Tensor, 
                               similarities: torch.Tensor) -> torch.Tensor:
        """Generate prediction based on similar patterns."""
        if len(similar_indices) == 0:
            return torch.zeros_like(self.current_activation)
        
        # Weighted average of similar patterns
        weights = similarities / similarities.sum()
        prediction = torch.zeros_like(self.current_activation)
        
        for idx, weight in zip(similar_indices, weights):
            pattern = self.storage.patterns[idx]
            prediction += weight * pattern
        
        return prediction
    
    def get_active_pattern_indices(self, k: int = 10) -> List[int]:
        """Get indices of most recently active patterns."""
        if self.storage.pattern_count == 0:
            return []
        
        # Return most recently activated patterns
        recent_activations = self.storage.last_activated[:self.storage.pattern_count]
        _, indices = torch.topk(recent_activations, min(k, self.storage.pattern_count))
        return indices.tolist()
    
    def update_prediction_accuracy(self, actual_next: torch.Tensor):
        """Update prediction accuracy statistics."""
        if self.prediction_attempts == 0:
            return
        
        error = torch.norm(self.predicted_next - actual_next).item()
        max_error = torch.norm(actual_next).item() + torch.norm(self.predicted_next).item()
        
        if max_error > 0:
            accuracy = 1.0 - (error / max_error)
            if accuracy > 0.7:  # Good prediction
                self.prediction_successes += 1
        
        self.prediction_attempts += 1
    
    def get_stream_state(self) -> Dict[str, Any]:
        """Get comprehensive stream state."""
        storage_stats = self.storage.get_pattern_stats()
        
        prediction_accuracy = 0.0
        if self.prediction_attempts > 0:
            prediction_accuracy = self.prediction_successes / self.prediction_attempts
        
        return {
            'stream_name': self.stream_name,
            'cycle_count': self.cycle_count,
            'current_activation': self.current_activation.cpu().tolist(),
            'predicted_next': self.predicted_next.cpu().tolist(),
            'prediction_accuracy': prediction_accuracy,
            'buffer_utilization': self.buffer_index / self.buffer_size if not self.buffer_full else 1.0,
            'storage_stats': storage_stats
        }


class GoldilocksBrain:
    """
    The complete Goldilocks brain: just right for emergent intelligence.
    
    Combines multiple vector streams with cross-stream co-activation tracking.
    Built for massive scale and GPU parallelism.
    """
    
    def __init__(self, sensory_dim: int = 16, motor_dim: int = 8, temporal_dim: int = 4,
                 max_patterns: int = 1_000_000, quiet_mode: bool = False):
        
        # Stream configurations
        self.sensory_config = StreamConfig(dim=sensory_dim, max_patterns=max_patterns)
        self.motor_config = StreamConfig(dim=motor_dim, max_patterns=max_patterns)
        self.temporal_config = StreamConfig(dim=temporal_dim, max_patterns=max_patterns)
        
        # Create streams
        self.sensory_stream = GoldilocksVectorStream(self.sensory_config, "sensory")
        self.motor_stream = GoldilocksVectorStream(self.motor_config, "motor")
        self.temporal_stream = GoldilocksVectorStream(self.temporal_config, "temporal")
        
        # Cross-stream co-activation tracking
        stream_names = ["sensory", "motor", "temporal"]
        device = self.sensory_stream.storage.device
        self.coactivation = CrossStreamCoactivation(stream_names, device)
        
        # Timing and statistics
        self.start_time = time.time()
        self.total_cycles = 0
        
        if not quiet_mode:
            print(f"\n🎯 GOLDILOCKS BRAIN INITIALIZED")
            print(f"   Sensory stream: {sensory_dim}D, {max_patterns:,} patterns")
            print(f"   Motor stream: {motor_dim}D, {max_patterns:,} patterns") 
            print(f"   Temporal stream: {temporal_dim}D, {max_patterns:,} patterns")
            print(f"   Device: {device}")
            print(f"   Total capacity: {3 * max_patterns:,} patterns")
            print(f"   Cross-stream co-activation enabled")
            print(f"   🚀 Ready for emergent intelligence at scale!")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        Core brain cycle: process sensory input and generate motor output.
        
        This is where emergence happens through massive parallel processing.
        """
        cycle_start = time.time()
        current_time = cycle_start
        
        # Convert to tensors
        sensory_tensor = torch.tensor(sensory_input, dtype=torch.float32)
        
        # Generate temporal context
        temporal_vector = self._generate_temporal_context(current_time)
        
        # Update all streams
        sensory_activation = self.sensory_stream.update(sensory_tensor, current_time)
        temporal_activation = self.temporal_stream.update(temporal_vector, current_time)
        
        # Generate motor prediction from cross-stream patterns
        motor_prediction = self._predict_motor_output(sensory_activation, temporal_activation)
        motor_activation = self.motor_stream.update(motor_prediction, current_time)
        
        # Record cross-stream co-activation (Core Primitive #5)
        active_patterns = {
            'sensory': self.sensory_stream.get_active_pattern_indices(k=5),
            'motor': self.motor_stream.get_active_pattern_indices(k=5),
            'temporal': self.temporal_stream.get_active_pattern_indices(k=5)
        }
        self.coactivation.record_coactivation(active_patterns)
        
        # Compile brain state
        cycle_time = time.time() - cycle_start
        self.total_cycles += 1
        
        brain_state = {
            'total_cycles': self.total_cycles,
            'cycle_time_ms': cycle_time * 1000,
            'architecture': 'goldilocks_massive_parallel',
            'sensory_stream': self.sensory_stream.get_stream_state(),
            'motor_stream': self.motor_stream.get_stream_state(),
            'temporal_stream': self.temporal_stream.get_stream_state(),
            'coactivation_stats': self.coactivation.get_coactivation_stats(),
            'prediction_confidence': self._estimate_prediction_confidence()
        }
        
        # Adjust action dimensions if needed
        motor_output = motor_activation.cpu().tolist()
        if action_dimensions and action_dimensions != len(motor_output):
            if action_dimensions < len(motor_output):
                motor_output = motor_output[:action_dimensions]
            else:
                motor_output = motor_output + [0.0] * (action_dimensions - len(motor_output))
        
        return motor_output, brain_state
    
    def _generate_temporal_context(self, current_time: float) -> torch.Tensor:
        """Generate temporal context vector (biological rhythms)."""
        relative_time = current_time - self.start_time
        
        # Multiple biological frequencies
        temporal_vector = torch.tensor([
            np.sin(relative_time * 2 * np.pi * 1.0),    # 1 Hz (breathing-like)
            np.sin(relative_time * 2 * np.pi * 10.0),   # 10 Hz (alpha waves)
            (relative_time % 1.0),                       # Cyclic component
            relative_time / 3600.0                       # Hour-scale component
        ], dtype=torch.float32)
        
        return temporal_vector
    
    def _predict_motor_output(self, sensory_activation: torch.Tensor, 
                            temporal_activation: torch.Tensor) -> torch.Tensor:
        """Predict motor output using cross-stream co-activation patterns."""
        
        # Get predictions from cross-stream patterns
        sensory_indices = self.sensory_stream.get_active_pattern_indices(k=5)
        temporal_indices = self.temporal_stream.get_active_pattern_indices(k=5)
        
        motor_predictions = []
        
        # Sensory -> Motor predictions
        sensory_motor_preds = self.coactivation.get_cross_predictions(
            'sensory', sensory_indices, 'motor', k=10
        )
        motor_predictions.extend(sensory_motor_preds)
        
        # Temporal -> Motor predictions  
        temporal_motor_preds = self.coactivation.get_cross_predictions(
            'temporal', temporal_indices, 'motor', k=10
        )
        motor_predictions.extend(temporal_motor_preds)
        
        if not motor_predictions:
            # No cross-stream predictions - use simple transformation
            combined = torch.cat([sensory_activation, temporal_activation])
            # Simple linear transformation to motor dimensions
            motor_pred = torch.randn(self.motor_config.dim) * 0.1
            return motor_pred
        
        # Combine predictions weighted by strength
        prediction = torch.zeros(self.motor_config.dim, device=self.sensory_stream.storage.device)
        total_weight = 0.0
        
        for pattern_idx, strength in motor_predictions:
            if pattern_idx < self.motor_stream.storage.pattern_count:
                pattern = self.motor_stream.storage.patterns[pattern_idx]
                prediction += strength * pattern
                total_weight += strength
        
        if total_weight > 0:
            prediction = prediction / total_weight
        
        return prediction
    
    def _estimate_prediction_confidence(self) -> float:
        """Estimate prediction confidence based on pattern density."""
        # Simple heuristic: more patterns = more confidence
        total_patterns = (self.sensory_stream.storage.pattern_count +
                         self.motor_stream.storage.pattern_count +
                         self.temporal_stream.storage.pattern_count)
        
        # Confidence grows logarithmically with pattern count
        confidence = min(0.95, np.log10(max(1, total_patterns)) / 6.0)
        return confidence
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics."""
        return {
            'total_cycles': self.total_cycles,
            'uptime_seconds': time.time() - self.start_time,
            'architecture': 'goldilocks_massive_parallel',
            'streams': {
                'sensory': self.sensory_stream.get_stream_state(),
                'motor': self.motor_stream.get_stream_state(),
                'temporal': self.temporal_stream.get_stream_state()
            },
            'cross_stream': self.coactivation.get_coactivation_stats(),
            'total_patterns': (
                self.sensory_stream.storage.pattern_count +
                self.motor_stream.storage.pattern_count +
                self.temporal_stream.storage.pattern_count
            ),
            'gpu_memory_usage_mb': self._calculate_total_memory_usage()
        }
    
    def _calculate_total_memory_usage(self) -> float:
        """Calculate total GPU memory usage."""
        return (self.sensory_stream.storage._get_memory_usage() +
                self.motor_stream.storage._get_memory_usage() +
                self.temporal_stream.storage._get_memory_usage())
    
    def __str__(self) -> str:
        total_patterns = (self.sensory_stream.storage.pattern_count +
                         self.motor_stream.storage.pattern_count +
                         self.temporal_stream.storage.pattern_count)
        return (f"GoldilocksBrain({self.total_cycles} cycles, "
                f"{total_patterns:,} patterns, "
                f"massive_parallel_architecture)")