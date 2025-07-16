"""
Adaptive Thresholds System (Phase 1)

Replaces hardcoded similarity and working memory thresholds with dynamic values
that adapt based on current system state and performance.

This is the first step toward fully emergent parameter management - replacing
magic numbers with context-sensitive, performance-driven thresholds.

Scientific Foundation:
- Percentile-based thresholds adapt to actual data distributions
- Working memory capacity adapts to cognitive load and performance
- All adaptations stay within biologically-plausible bounds
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import time

from ..cognitive_constants import CognitiveCapacityConstants, PredictionErrorConstants


class AdaptiveThresholds:
    """
    Manages dynamic thresholds that replace hardcoded magic numbers.
    
    Core Principle: Thresholds should be relative to current system state
    rather than absolute hardcoded values.
    """
    
    def __init__(self):
        """Initialize adaptive threshold system."""
        
        # Similarity threshold adaptation
        self.similarity_percentile = 70  # Target: top 30% of similarities
        self.min_similarity_samples = 10  # Need data before adapting
        self.similarity_history = deque(maxlen=100)  # Recent similarity values
        
        # Working memory adaptation (now hardware-adaptive)
        self.target_working_memory_size = CognitiveCapacityConstants.get_working_memory_limit()
        self.performance_history = deque(maxlen=20)  # Recent performance scores
        self.cognitive_load_history = deque(maxlen=10)  # Recent cognitive load measures
        
        # Adaptation tracking
        self.adaptation_count = 0
        self.last_adaptation_time = time.time()
        self.adaptation_effectiveness = {}  # Track how well adaptations work
        
        print("🎯 AdaptiveThresholds initialized - replacing magic numbers with dynamic values")
    
    def get_similarity_threshold(self, 
                               current_similarities: List[float], 
                               context: str = "general") -> float:
        """
        Get adaptive similarity threshold based on current similarity distribution.
        
        Args:
            current_similarities: All similarity scores for current search
            context: Context for threshold selection ("general", "working_memory", "pattern_recognition")
            
        Returns:
            Adaptive similarity threshold (0.0-1.0)
        """
        # Store similarities for learning
        self.similarity_history.extend(current_similarities)
        
        # Bootstrap with safe default if insufficient data
        if len(current_similarities) < self.min_similarity_samples:
            return self._get_bootstrap_similarity_threshold(context)
        
        # Context-specific percentile adjustment
        percentile = self._get_context_percentile(context)
        
        # Calculate adaptive threshold from current distribution
        threshold = np.percentile(current_similarities, percentile)
        
        # Apply safety bounds from cognitive constants
        min_threshold = 0.05  # Always allow some matches
        max_threshold = 0.95  # Don't require perfect matches
        threshold = np.clip(threshold, min_threshold, max_threshold)
        
        # Track adaptation effectiveness
        self._track_similarity_adaptation(threshold, context)
        
        return float(threshold)
    
    def get_working_memory_threshold(self, 
                                   current_activations: Dict[str, float],
                                   recent_performance: float,
                                   current_prediction_error: float) -> float:
        """
        Get adaptive working memory threshold based on cognitive load and performance.
        
        Args:
            current_activations: All current activation levels
            recent_performance: Recent prediction/consensus success rate (0.0-1.0)
            current_prediction_error: Current prediction error (0.0-1.0)
            
        Returns:
            Adaptive activation threshold for working memory inclusion
        """
        # Track performance and cognitive load
        self.performance_history.append(recent_performance)
        self.cognitive_load_history.append(len(current_activations))
        
        # Calculate target working memory size based on performance
        target_size = self._calculate_adaptive_working_memory_size(
            recent_performance, current_prediction_error
        )
        
        # Get threshold that achieves target size
        threshold = self._calculate_size_based_threshold(current_activations, target_size)
        
        # Track adaptation effectiveness
        self._track_working_memory_adaptation(threshold, target_size, recent_performance)
        
        return threshold
    
    def _get_bootstrap_similarity_threshold(self, context: str) -> float:
        """Get safe default similarity threshold during system bootstrap."""
        bootstrap_thresholds = {
            "general": 0.3,
            "working_memory": 0.4,  # Slightly higher for working memory selection
            "pattern_recognition": 0.5,  # Higher for pattern recognition
            "utility_assessment": 0.2   # Lower for utility-based activation
        }
        return bootstrap_thresholds.get(context, 0.3)
    
    def _get_context_percentile(self, context: str) -> float:
        """Get percentile target based on context."""
        # Different contexts need different selectivity
        context_percentiles = {
            "general": 70,           # Top 30% of similarities
            "working_memory": 75,    # Top 25% for working memory (more selective)
            "pattern_recognition": 80, # Top 20% for patterns (most selective) 
            "utility_assessment": 60  # Top 40% for utility (less selective)
        }
        return context_percentiles.get(context, 70)
    
    def _calculate_adaptive_working_memory_size(self, 
                                              performance: float, 
                                              prediction_error: float) -> int:
        """Calculate optimal working memory size based on current state."""
        
        base_size = self.target_working_memory_size
        
        # Performance-based adjustment
        if performance > 0.8:
            # High performance: can handle larger working memory
            performance_adjustment = 5
        elif performance < 0.3:
            # Poor performance: reduce cognitive load
            performance_adjustment = -5
        else:
            # Medium performance: no adjustment
            performance_adjustment = 0
        
        # Prediction error adjustment (following optimal challenge principle)
        optimal_error = PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET
        error_distance = abs(prediction_error - optimal_error)
        
        if error_distance > 0.2:
            # Far from optimal: simplify by reducing working memory
            error_adjustment = -3
        else:
            # Near optimal: can handle complexity
            error_adjustment = 2
        
        # Calculate final target size
        target_size = base_size + performance_adjustment + error_adjustment
        
        # Apply cognitive capacity bounds
        target_size = np.clip(target_size, 
                            CognitiveCapacityConstants.MIN_WORKING_MEMORY_SIZE,
                            CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE)
        
        return int(target_size)
    
    def _calculate_size_based_threshold(self, 
                                      activations: Dict[str, float], 
                                      target_size: int) -> float:
        """Calculate activation threshold that achieves target working memory size."""
        
        if not activations:
            return 0.1  # Default minimum threshold
        
        activation_values = list(activations.values())
        activation_values.sort(reverse=True)  # Highest first
        
        if len(activation_values) <= target_size:
            # Fewer activations than target: use minimum threshold
            return 0.05
        
        # Use activation value at target position as threshold
        threshold = activation_values[target_size - 1]
        
        # Ensure reasonable bounds
        threshold = np.clip(threshold, 0.05, 0.8)
        
        return float(threshold)
    
    def _track_similarity_adaptation(self, threshold: float, context: str):
        """Track effectiveness of similarity threshold adaptations."""
        adaptation_key = f"similarity_{context}"
        current_time = time.time()
        
        if adaptation_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[adaptation_key] = {
                'threshold_history': deque(maxlen=20),
                'performance_correlation': 0.0,
                'last_update': current_time
            }
        
        self.adaptation_effectiveness[adaptation_key]['threshold_history'].append({
            'threshold': threshold,
            'timestamp': current_time
        })
    
    def _track_working_memory_adaptation(self, threshold: float, target_size: int, performance: float):
        """Track effectiveness of working memory threshold adaptations."""
        adaptation_key = "working_memory"
        current_time = time.time()
        
        if adaptation_key not in self.adaptation_effectiveness:
            self.adaptation_effectiveness[adaptation_key] = {
                'adaptations': deque(maxlen=20),
                'size_accuracy': deque(maxlen=20),
                'performance_correlation': 0.0
            }
        
        self.adaptation_effectiveness[adaptation_key]['adaptations'].append({
            'threshold': threshold,
            'target_size': target_size,
            'performance': performance,
            'timestamp': current_time
        })
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about threshold adaptation effectiveness."""
        
        stats = {
            'total_adaptations': self.adaptation_count,
            'similarity_adaptations': {},
            'working_memory_adaptations': {},
            'adaptation_rate': 0.0
        }
        
        # Calculate adaptation rate
        time_elapsed = time.time() - self.last_adaptation_time
        if time_elapsed > 0:
            stats['adaptation_rate'] = self.adaptation_count / time_elapsed
        
        # Similarity adaptation stats
        if self.similarity_history:
            similarity_values = list(self.similarity_history)
            stats['similarity_adaptations'] = {
                'distribution_mean': np.mean(similarity_values),
                'distribution_std': np.std(similarity_values),
                'distribution_range': (np.min(similarity_values), np.max(similarity_values)),
                'samples_collected': len(similarity_values)
            }
        
        # Working memory adaptation stats
        if self.performance_history and self.cognitive_load_history:
            stats['working_memory_adaptations'] = {
                'average_performance': np.mean(self.performance_history),
                'performance_trend': self._calculate_trend(self.performance_history),
                'average_cognitive_load': np.mean(self.cognitive_load_history),
                'load_stability': np.std(self.cognitive_load_history)
            }
        
        # Adaptation effectiveness summary
        stats['effectiveness_summary'] = {}
        for adaptation_type, data in self.adaptation_effectiveness.items():
            if adaptation_type.startswith('similarity'):
                stats['effectiveness_summary'][adaptation_type] = {
                    'adaptations_tracked': len(data['threshold_history']),
                    'threshold_stability': self._calculate_threshold_stability(data['threshold_history'])
                }
            elif adaptation_type == 'working_memory':
                stats['effectiveness_summary'][adaptation_type] = {
                    'adaptations_tracked': len(data['adaptations']),
                    'performance_correlation': self._calculate_performance_correlation(data['adaptations'])
                }
        
        return stats
    
    def _calculate_trend(self, values: deque) -> float:
        """Calculate trend direction in recent values (+/- indicates improvement/decline)."""
        if len(values) < 5:
            return 0.0
        
        recent_values = list(values)
        first_half = np.mean(recent_values[:len(recent_values)//2])
        second_half = np.mean(recent_values[len(recent_values)//2:])
        
        return second_half - first_half
    
    def _calculate_threshold_stability(self, threshold_history: deque) -> float:
        """Calculate how stable threshold values have been (lower = more stable)."""
        if len(threshold_history) < 3:
            return 0.0
        
        thresholds = [entry['threshold'] for entry in threshold_history]
        return float(np.std(thresholds))
    
    def _calculate_performance_correlation(self, adaptations: deque) -> float:
        """Calculate correlation between adaptations and performance."""
        if len(adaptations) < 5:
            return 0.0
        
        thresholds = [entry['threshold'] for entry in adaptations]
        performances = [entry['performance'] for entry in adaptations]
        
        try:
            correlation_matrix = np.corrcoef(thresholds, performances)
            return float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
        except:
            return 0.0
    
    def reset_adaptation_history(self):
        """Reset adaptation tracking (useful for testing)."""
        self.similarity_history.clear()
        self.performance_history.clear()
        self.cognitive_load_history.clear()
        self.adaptation_effectiveness.clear()
        self.adaptation_count = 0
        self.last_adaptation_time = time.time()
        print("🔄 Adaptive thresholds reset - starting fresh adaptation tracking")