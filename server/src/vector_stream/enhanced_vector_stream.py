#!/usr/bin/env python3
"""
Enhanced Vector Stream with Hierarchical Pattern Memory

This replaces the simple VectorStream with sophisticated pattern memory
that can scale to thousands of patterns while maintaining real-time performance.

Key improvements over minimal_brain.py:
- Hierarchical pattern storage (active/working/consolidated tiers)
- Cross-stream pattern linking for episodic capabilities  
- Dynamic pattern promotion/demotion based on importance
- Intelligent memory management and forgetting
- Pattern importance scoring and utility tracking
"""

import time
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple

from .enhanced_pattern_memory import (
    HierarchicalPatternMemory, 
    EnhancedVectorPattern, 
    EpisodeManager,
    Episode
)


class EnhancedVectorStream:
    """
    Enhanced vector stream with sophisticated pattern memory capabilities.
    
    This maintains the biological continuous processing approach while adding
    the intelligence mechanisms needed for complex learning and memory.
    """
    
    def __init__(self, dim: int, buffer_size: int = 100, name: str = "stream",
                 max_active_patterns: int = 200, max_working_patterns: int = 1000):
        self.dim = dim
        self.buffer_size = buffer_size
        self.name = name
        
        # Rolling buffer of recent activations (unchanged from original)
        self.activation_buffer = torch.zeros(buffer_size, dim)
        self.time_buffer = torch.zeros(buffer_size)
        self.buffer_index = 0
        self.buffer_full = False
        
        # Enhanced hierarchical pattern memory
        self.pattern_memory = HierarchicalPatternMemory(
            stream_name=name,
            max_active=max_active_patterns,
            max_working=max_working_patterns
        )
        
        # Prediction state
        self.current_activation = torch.zeros(dim)
        self.predicted_next_activation = torch.zeros(dim)
        
        # Learning parameters (more sophisticated than fixed threshold)
        self.base_similarity_threshold = 0.8
        self.adaptive_threshold = 0.8
        self.learning_rate = 0.1
        
        # Cross-stream integration 
        self.cross_stream_manager = None  # Set by parent brain
        self.last_episode_id = None
        
        # Performance tracking
        self.pattern_creation_count = 0
        self.pattern_match_count = 0
        self.prediction_attempts = 0
        self.prediction_successes = 0
        
        print(f"🧠 EnhancedVectorStream '{name}' initialized: {dim}D")
        print(f"   Rolling buffer: {buffer_size} activations")
        print(f"   Pattern memory: {max_active_patterns} active, {max_working_patterns} working")
    
    def set_cross_stream_manager(self, manager):
        """Set the cross-stream manager for episode creation."""
        self.cross_stream_manager = manager
    
    def update(self, new_activation: torch.Tensor, timestamp: float) -> torch.Tensor:
        """
        Update the stream with a new activation vector.
        
        Enhanced with sophisticated pattern learning and cross-stream integration.
        """
        # Input validation (same as original)
        if not torch.isfinite(new_activation).all():
            new_activation = torch.where(torch.isfinite(new_activation), new_activation, torch.zeros_like(new_activation))
        
        if new_activation.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim}D activation, got {new_activation.shape[0]}D")
        
        # Store activation in rolling buffer (unchanged)
        self.activation_buffer[self.buffer_index] = new_activation
        self.time_buffer[self.buffer_index] = timestamp
        
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.buffer_full = True
        
        # Update current state
        self.current_activation = new_activation
        
        # Enhanced pattern learning with hierarchical memory
        matched_pattern_id = self._learn_patterns_enhanced(new_activation, timestamp)
        
        # Generate prediction for next activation
        self.predicted_next_activation = self._predict_next_activation_enhanced(timestamp)
        
        # Cross-stream episode creation (if manager available)
        if self.cross_stream_manager and matched_pattern_id:
            self.last_episode_id = self.cross_stream_manager.create_cross_stream_episode(
                self.name, matched_pattern_id, timestamp
            )
        
        return self.current_activation
    
    def _learn_patterns_enhanced(self, activation: torch.Tensor, timestamp: float) -> Optional[str]:
        """
        Enhanced pattern learning with hierarchical memory and importance scoring.
        
        Returns the pattern ID that was matched or created.
        """
        # Find similar patterns using hierarchical search
        similar_patterns = self.pattern_memory.find_similar_patterns(
            activation, 
            threshold=self.adaptive_threshold,
            max_results=5
        )
        
        if similar_patterns:
            # Match found - reinforce the most similar pattern
            best_pattern, similarity = similar_patterns[0]
            
            # Update pattern with weighted average (learning)
            alpha = self.learning_rate * (1.0 - similarity)  # Learn more from dissimilar inputs
            best_pattern.activation_pattern = (
                (1 - alpha) * best_pattern.activation_pattern + 
                alpha * activation
            )
            
            # Update usage statistics
            best_pattern.frequency += 1
            best_pattern.last_seen = timestamp
            best_pattern.recency_score = 1.0
            
            self.pattern_match_count += 1
            return best_pattern.pattern_id
        
        else:
            # No match found - create new pattern
            new_pattern = EnhancedVectorPattern(
                activation_pattern=activation.clone(),
                temporal_context=self._get_temporal_context_enhanced(timestamp),
                creation_time=timestamp,
                last_seen=timestamp,
                novelty_score=self._calculate_novelty_score(activation)
            )
            
            pattern_id = self.pattern_memory.store_pattern(new_pattern)
            self.pattern_creation_count += 1
            
            return pattern_id
    
    def _predict_next_activation_enhanced(self, current_time: float) -> torch.Tensor:
        """
        Enhanced prediction using hierarchical pattern memory and importance weighting.
        """
        # Get relevant patterns for prediction
        current_activation = self.current_activation
        relevant_patterns = self.pattern_memory.find_similar_patterns(
            current_activation,
            threshold=max(0.5, self.adaptive_threshold - 0.2),  # Lower threshold for prediction
            max_results=10
        )
        
        if not relevant_patterns:
            return torch.zeros_like(current_activation)
        
        # Weighted prediction based on pattern importance and similarity
        prediction = torch.zeros_like(current_activation)
        total_weight = 0.0
        
        for pattern, similarity in relevant_patterns:
            # Weight by similarity, frequency, recency, and prediction accuracy
            pattern.update_importance_score()
            
            similarity_weight = similarity
            importance_weight = pattern.importance_score
            temporal_weight = self._get_temporal_prediction_weight(pattern, current_time)
            
            total_pattern_weight = similarity_weight * importance_weight * temporal_weight
            
            # Use the pattern itself as prediction (could be enhanced with sequence learning)
            prediction += total_pattern_weight * pattern.activation_pattern
            total_weight += total_pattern_weight
            
            # Track prediction usage for pattern utility scoring
            pattern.prediction_attempts += 1
        
        if total_weight > 1e-8:
            prediction = prediction / total_weight
        else:
            prediction = torch.zeros_like(current_activation)
        
        self.prediction_attempts += 1
        return prediction
    
    def _get_temporal_context_enhanced(self, timestamp: float) -> float:
        """Enhanced temporal context calculation."""
        if not self.buffer_full and self.buffer_index < 2:
            return 0.0
        
        # Calculate multiple temporal features
        recent_times = self.time_buffer[:self.buffer_index] if not self.buffer_full else self.time_buffer
        
        if len(recent_times) > 1:
            intervals = torch.diff(recent_times)
            avg_interval = torch.mean(intervals).item()
            interval_variance = torch.var(intervals).item()
            
            # Combine average interval and regularity
            regularity = 1.0 / (1.0 + interval_variance)
            return avg_interval * regularity
        
        return 0.0
    
    def _calculate_novelty_score(self, activation: torch.Tensor) -> float:
        """Calculate how novel this activation is compared to existing patterns."""
        if self.pattern_memory.total_patterns == 0:
            return 1.0  # First pattern is completely novel
        
        # Find closest existing pattern
        similar_patterns = self.pattern_memory.find_similar_patterns(
            activation, threshold=0.0, max_results=1
        )
        
        if similar_patterns:
            _, max_similarity = similar_patterns[0]
            novelty = 1.0 - max_similarity
        else:
            novelty = 1.0  # No similar patterns found
        
        return novelty
    
    def _get_temporal_prediction_weight(self, pattern: EnhancedVectorPattern, current_time: float) -> float:
        """Calculate temporal weighting for prediction based on when pattern typically occurs."""
        if pattern.temporal_context <= 0:
            return 1.0
        
        # Calculate expected time for pattern occurrence
        time_since_last = current_time - pattern.last_seen
        expected_interval = pattern.temporal_context
        
        # Weight higher if we're near the expected time
        if expected_interval > 0:
            timing_factor = abs(time_since_last - expected_interval) / expected_interval
            temporal_weight = np.exp(-timing_factor)  # Gaussian-like weighting
        else:
            temporal_weight = 1.0
        
        return temporal_weight
    
    def update_prediction_accuracy(self, actual_activation: torch.Tensor):
        """Update prediction accuracy for pattern utility scoring."""
        if self.prediction_attempts == 0:
            return
        
        # Calculate prediction error
        prediction_error = torch.norm(self.predicted_next_activation - actual_activation).item()
        max_possible_error = torch.norm(actual_activation).item() + torch.norm(self.predicted_next_activation).item()
        
        if max_possible_error > 0:
            accuracy = 1.0 - (prediction_error / max_possible_error)
            
            if accuracy > 0.7:  # Good prediction
                self.prediction_successes += 1
                
                # Reward patterns that contributed to good prediction
                contributing_patterns = self.pattern_memory.find_similar_patterns(
                    self.current_activation, threshold=0.5, max_results=5
                )
                
                for pattern, _ in contributing_patterns:
                    pattern.prediction_successes += 1
                    pattern.utility_score = pattern.get_prediction_accuracy()
    
    def link_to_other_stream(self, other_stream_name: str, other_pattern_id: str, 
                           my_pattern_id: str = None):
        """Create a link between patterns in this stream and another stream."""
        if my_pattern_id is None:
            # Use most recently activated pattern
            if self.pattern_memory.active_patterns:
                my_pattern = max(self.pattern_memory.active_patterns, 
                               key=lambda p: p.last_activated)
                my_pattern_id = my_pattern.pattern_id
            else:
                return  # No patterns to link
        
        self.pattern_memory.link_patterns_across_streams(
            my_pattern_id, other_stream_name, other_pattern_id
        )
    
    def get_cross_stream_predictions(self, other_stream_name: str) -> List[Tuple[str, float]]:
        """Get predicted patterns in another stream based on current activation."""
        # Find patterns similar to current activation
        similar_patterns = self.pattern_memory.find_similar_patterns(
            self.current_activation, threshold=0.6, max_results=5
        )
        
        predictions = []
        for pattern, similarity in similar_patterns:
            linked_patterns = self.pattern_memory.get_linked_patterns(
                pattern.pattern_id, other_stream_name
            )
            
            for linked_pattern_id in linked_patterns:
                predictions.append((linked_pattern_id, similarity))
        
        # Sort by confidence (similarity)
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def get_stream_state(self) -> Dict[str, Any]:
        """Get comprehensive stream state for debugging and monitoring."""
        memory_stats = self.pattern_memory.get_memory_stats()
        
        prediction_accuracy = 0.0
        if self.prediction_attempts > 0:
            prediction_accuracy = self.prediction_successes / self.prediction_attempts
        
        return {
            'name': self.name,
            'current_activation': self.current_activation.tolist(),
            'predicted_next': self.predicted_next_activation.tolist(),
            'buffer_utilization': self.buffer_index / self.buffer_size if not self.buffer_full else 1.0,
            'memory_stats': memory_stats,
            'learning_stats': {
                'pattern_creation_count': self.pattern_creation_count,
                'pattern_match_count': self.pattern_match_count,
                'prediction_accuracy': prediction_accuracy,
                'adaptive_threshold': self.adaptive_threshold
            },
            'cross_stream_links': len(self.pattern_memory.cross_stream_links)
        }
    
    def cleanup_memory(self, max_age_hours: float = 168):
        """Clean up old, unused patterns to manage memory usage."""
        removed = self.pattern_memory.cleanup_old_patterns(max_age_hours)
        return removed
    
    def adapt_learning_threshold(self):
        """Adapt the similarity threshold based on pattern creation rate."""
        # If we're creating too many patterns, raise threshold
        # If we're creating too few, lower threshold
        
        if self.pattern_creation_count > 0 and self.pattern_match_count > 0:
            creation_rate = self.pattern_creation_count / (self.pattern_creation_count + self.pattern_match_count)
            
            target_creation_rate = 0.1  # Aim for 10% new patterns, 90% matches
            
            if creation_rate > target_creation_rate:
                # Creating too many patterns - raise threshold
                self.adaptive_threshold = min(0.95, self.adaptive_threshold + 0.01)
            elif creation_rate < target_creation_rate:
                # Creating too few patterns - lower threshold
                self.adaptive_threshold = max(0.5, self.adaptive_threshold - 0.01)


class CrossStreamManager:
    """
    Manages cross-stream pattern relationships and episode creation.
    
    This enables the system to learn associations between patterns 
    across different modalities (sensory, motor, temporal).
    """
    
    def __init__(self, streams: Dict[str, EnhancedVectorStream]):
        self.streams = streams
        self.episode_manager = EpisodeManager()
        
        # Set this manager in all streams
        for stream in streams.values():
            stream.set_cross_stream_manager(self)
        
        print(f"🔗 CrossStreamManager initialized with {len(streams)} streams")
    
    def create_cross_stream_episode(self, triggering_stream: str, pattern_id: str, 
                                  timestamp: float) -> str:
        """
        Create an episode linking patterns across all streams at this timestamp.
        
        This enables episodic memory capabilities.
        """
        # Collect active patterns from all streams
        stream_patterns = {}
        
        for stream_name, stream in self.streams.items():
            if stream_name == triggering_stream:
                stream_patterns[stream_name] = pattern_id
            else:
                # Find most recently activated pattern in other streams
                if stream.pattern_memory.active_patterns:
                    recent_pattern = max(stream.pattern_memory.active_patterns,
                                       key=lambda p: p.last_activated)
                    stream_patterns[stream_name] = recent_pattern.pattern_id
        
        # Create episode
        episode_id = self.episode_manager.create_episode(
            stream_patterns, 
            context=f"Cross-stream episode triggered by {triggering_stream}"
        )
        
        # Create cross-stream links between all patterns in this episode
        for stream_a, pattern_a in stream_patterns.items():
            for stream_b, pattern_b in stream_patterns.items():
                if stream_a != stream_b:
                    self.streams[stream_a].pattern_memory.link_patterns_across_streams(
                        pattern_a, stream_b, pattern_b
                    )
        
        return episode_id
    
    def get_cross_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about cross-stream relationships."""
        total_links = 0
        stream_link_counts = {}
        
        for stream_name, stream in self.streams.items():
            link_count = sum(len(links) for links in stream.pattern_memory.cross_stream_links.values())
            stream_link_counts[stream_name] = link_count
            total_links += link_count
        
        episode_stats = self.episode_manager.get_episode_stats()
        
        return {
            'total_cross_stream_links': total_links,
            'links_per_stream': stream_link_counts,
            'episode_stats': episode_stats
        }