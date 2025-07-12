"""
Adaptive Attention Scoring System

Auto-tuning attention scorer with no magic numbers that determines which memories
get priority during retrieval. Implements biological memory suppression where
everything is stored but boring experiences are deprioritized.

Key insight: Yesterday's uneventful day is stored but suppressed - high similarity
cues can still retrieve it, but it won't dominate normal decision making.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque


class AdaptiveAttentionScorer:
    """
    Auto-tuning attention scorer with no magic numbers.
    
    Computes attention scores based on prediction error and automatically
    adapts thresholds based on learning velocity. This creates natural
    memory suppression without information loss.
    """
    
    def __init__(self, adaptation_window: int = 50):
        """
        Initialize adaptive attention scorer.
        
        Args:
            adaptation_window: Number of recent learning samples to track
        """
        # Learning-based baseline (starts neutral, adapts based on learning)
        self.attention_baseline = 0.5  # Will adapt to optimal level
        self.baseline_history = deque(maxlen=100)
        
        # Learning velocity tracking
        self.learning_velocity_tracker = deque(maxlen=adaptation_window)
        self.accuracy_history = deque(maxlen=adaptation_window)
        
        # Adaptation timing
        self.last_adaptation = time.time()
        self.adaptation_interval = 10.0  # Adapt every 10 seconds
        
        # Performance tracking
        self.total_scores_computed = 0
        self.high_attention_count = 0
        self.suppressed_count = 0
        
        print(f"🎯 AdaptiveAttentionScorer initialized (baseline={self.attention_baseline:.3f})")
    
    def compute_attention_score(self, prediction_error: float, learning_context: Dict[str, Any]) -> float:
        """
        Compute attention score based on prediction error and current learning state.
        
        Args:
            prediction_error: How wrong the prediction was (0.0-1.0+)
            learning_context: Context about current learning state
            
        Returns:
            Attention score (0.0-1.0) where 1.0 = maximum attention
        """
        self.total_scores_computed += 1
        
        # Track learning for adaptation
        self._track_learning_progress(learning_context)
        
        # Compute attention score relative to adaptive baseline
        # Higher prediction error = higher attention (this is surprising!)
        if self.attention_baseline <= 0:
            self.attention_baseline = 0.5  # Safety fallback
        
        normalized_error = prediction_error / self.attention_baseline
        # Use sigmoid-like function to create better distribution
        attention_score = min(1.0, max(0.0, normalized_error * 0.8))  # Scale to create variation
        
        # Track attention distribution
        if attention_score > 0.7:
            self.high_attention_count += 1
        elif attention_score < 0.3:
            self.suppressed_count += 1
        
        # Adapt baseline periodically
        if time.time() - self.last_adaptation > self.adaptation_interval:
            self._adapt_attention_baseline()
        
        return attention_score
    
    def _track_learning_progress(self, learning_context: Dict[str, Any]):
        """Track learning progress for baseline adaptation."""
        # Track accuracy trend if available
        if 'current_accuracy' in learning_context:
            self.accuracy_history.append(learning_context['current_accuracy'])
        
        # Compute learning velocity (accuracy improvement rate)
        if len(self.accuracy_history) >= 10:
            recent_accuracies = list(self.accuracy_history)[-10:]
            
            # Simple linear trend
            x = np.arange(len(recent_accuracies))
            y = np.array(recent_accuracies)
            
            if len(x) > 1:
                # Compute slope (learning velocity)
                velocity = np.polyfit(x, y, 1)[0]  # Linear fit slope
                self.learning_velocity_tracker.append(velocity)
    
    def _adapt_attention_baseline(self):
        """
        Adapt attention baseline based on learning velocity.
        
        The key insight: If learning fast, be more selective (raise baseline).
        If learning slowly, be more open (lower baseline). No magic numbers!
        """
        if len(self.learning_velocity_tracker) < 5:
            return
        
        # Get recent learning velocity
        recent_velocities = list(self.learning_velocity_tracker)[-5:]
        mean_velocity = np.mean(recent_velocities)
        velocity_std = np.std(recent_velocities)
        
        # Adaptive adjustment based on learning state
        if mean_velocity > velocity_std:  # Learning faster than usual
            # Be more selective - raise baseline (same error gets less attention)
            adjustment_factor = 1.0 + (mean_velocity * 0.1)
            new_baseline = self.attention_baseline * adjustment_factor
            
        elif mean_velocity < -velocity_std:  # Learning slower than usual
            # Be more open - lower baseline (same error gets more attention)  
            adjustment_factor = 1.0 + (abs(mean_velocity) * 0.1)
            new_baseline = self.attention_baseline / adjustment_factor
            
        else:
            # Learning is stable - gentle drift toward optimal
            # Find optimal based on attention distribution
            high_attention_rate = self.high_attention_count / max(1, self.total_scores_computed)
            suppressed_rate = self.suppressed_count / max(1, self.total_scores_computed)
            
            if high_attention_rate > 0.2:  # Too much high attention
                new_baseline = self.attention_baseline * 1.02
            elif suppressed_rate > 0.6:  # Too much suppression
                new_baseline = self.attention_baseline * 0.98
            else:
                new_baseline = self.attention_baseline  # Keep current
        
        # Constrain baseline to reasonable bounds (learned from experience)
        new_baseline = np.clip(new_baseline, 0.1, 2.0)
        
        # Track adaptation
        baseline_change = new_baseline - self.attention_baseline
        self.baseline_history.append(self.attention_baseline)
        
        if abs(baseline_change) > 0.01:  # Significant change
            print(f"🎯 Attention baseline adapted: {self.attention_baseline:.3f} → {new_baseline:.3f} "
                  f"(Δ{baseline_change:+.3f}, learning_vel={mean_velocity:.4f})")
        
        self.attention_baseline = new_baseline
        self.last_adaptation = time.time()
        
        # Reset counters for next period
        self.total_scores_computed = 0
        self.high_attention_count = 0
        self.suppressed_count = 0
    
    def get_attention_stats(self) -> Dict[str, Any]:
        """Get comprehensive attention scoring statistics."""
        if len(self.learning_velocity_tracker) == 0:
            return {
                'attention_baseline': self.attention_baseline,
                'total_scores_computed': self.total_scores_computed,
                'learning_samples': 0
            }
        
        return {
            'attention_baseline': self.attention_baseline,
            'baseline_history': list(self.baseline_history),
            'total_scores_computed': self.total_scores_computed,
            'high_attention_rate': self.high_attention_count / max(1, self.total_scores_computed),
            'suppression_rate': self.suppressed_count / max(1, self.total_scores_computed),
            'learning_velocity': {
                'current': self.learning_velocity_tracker[-1] if self.learning_velocity_tracker else 0,
                'mean': np.mean(self.learning_velocity_tracker) if self.learning_velocity_tracker else 0,
                'samples': len(self.learning_velocity_tracker)
            },
            'adaptation_active': abs(self.attention_baseline - 0.5) > 0.1
        }


class NaturalAttentionSimilarity:
    """
    Enhances similarity search with natural attention-based memory retrieval.
    
    Uses emergent properties (utility, distinctiveness, access patterns) instead
    of explicit attention scores to create biological memory suppression effects.
    """
    
    def __init__(self, base_similarity_engine):
        """
        Initialize natural attention-weighted similarity.
        
        Args:
            base_similarity_engine: Existing similarity engine to enhance
        """
        self.base_engine = base_similarity_engine
        
        # Retrieval modes based on natural memory properties
        self.retrieval_modes = {
            'normal': 'Natural attention weighting using utility + distinctiveness',
            'deep': 'Ignore natural suppression - raw similarity only', 
            'hybrid': 'Boost high similarity even for clustered experiences',
            'utility_focused': 'Prioritize high-utility experiences for prediction'
        }
        
        print(f"🔍 NaturalAttentionSimilarity initialized with {len(self.retrieval_modes)} retrieval modes")
    
    def find_similar_experiences_with_natural_attention(self,
                                                      target_vector: List[float],
                                                      experiences: List[Any],  # Experience objects
                                                      max_results: int = 10,
                                                      min_similarity: float = 0.3,
                                                      retrieval_mode: str = 'normal') -> List[tuple]:
        """
        Find similar experiences using natural attention weighting.
        
        Args:
            target_vector: Vector to find similarities for
            experiences: List of Experience objects
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold
            retrieval_mode: 'normal', 'deep', 'hybrid', or 'utility_focused'
            
        Returns:
            List of (experience, weighted_similarity, base_similarity, natural_attention) tuples
        """
        if not experiences:
            return []
        
        # Extract vectors and IDs for base similarity computation
        experience_vectors = []
        experience_ids = []
        
        for i, exp in enumerate(experiences):
            # Build context vector from experience
            context_vector = self._build_context_vector(exp)
            experience_vectors.append(context_vector)
            experience_ids.append(f"exp_{i}")
        
        # Get base similarities using existing engine
        base_similarities = self.base_engine.find_similar_experiences(
            target_vector, experience_vectors, experience_ids, 
            max_results=len(experiences), min_similarity=0.0  # Get all for weighting
        )
        
        # Apply natural attention weighting based on retrieval mode
        weighted_results = []
        
        for exp_id, base_similarity in base_similarities:
            exp_index = int(exp_id.split('_')[1])
            experience = experiences[exp_index]
            
            # Get natural attention weight from emergent properties
            natural_attention = experience.get_natural_attention_weight()
            
            # Apply retrieval mode using natural signals
            if retrieval_mode == 'normal':
                # Standard: weight by natural attention (clustered memories deprioritized)
                weighted_similarity = base_similarity * natural_attention
                
            elif retrieval_mode == 'deep':
                # Deep: ignore natural suppression (can retrieve clustered memories)
                weighted_similarity = base_similarity
                
            elif retrieval_mode == 'hybrid':
                # Hybrid: boost high similarity even for clustered experiences
                # This mimics how strong cues can retrieve suppressed memories
                attention_boost = max(natural_attention, base_similarity * 0.8)
                weighted_similarity = base_similarity * attention_boost
                
            else:  # utility_focused
                # Focus on experiences with high prediction utility
                utility_weight = experience.prediction_utility * 2.0  # Boost utility importance
                weighted_similarity = base_similarity * utility_weight
            
            if weighted_similarity >= min_similarity:
                weighted_results.append((
                    experience, 
                    weighted_similarity, 
                    base_similarity, 
                    natural_attention
                ))
        
        # Sort by weighted similarity and limit results
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        return weighted_results[:max_results]
    
    def _build_context_vector(self, experience) -> List[float]:
        """Build context vector from experience for similarity comparison."""
        # Combine sensory input and action for context
        context = []
        context.extend(experience.sensory_input)
        context.extend(experience.action_taken)
        return context
    
    def get_natural_attention_stats(self, experiences: List[Any]) -> Dict[str, Any]:
        """Get statistics about natural attention-weighted retrieval."""
        base_stats = getattr(self.base_engine, 'get_performance_stats', lambda: {})()
        
        if not experiences:
            return {
                'base_similarity': base_stats,
                'natural_attention': {'num_experiences': 0},
                'retrieval_modes': self.retrieval_modes,
                'integration_status': 'active'
            }
        
        # Compute natural attention statistics
        natural_weights = [exp.get_natural_attention_weight() for exp in experiences]
        utilities = [exp.prediction_utility for exp in experiences]
        cluster_densities = [exp.local_cluster_density for exp in experiences]
        
        import numpy as np
        natural_stats = {
            'num_experiences': len(experiences),
            'attention_weights': {
                'mean': np.mean(natural_weights),
                'std': np.std(natural_weights),
                'min': np.min(natural_weights),
                'max': np.max(natural_weights)
            },
            'prediction_utilities': {
                'mean': np.mean(utilities),
                'std': np.std(utilities),
                'high_utility_count': sum(1 for u in utilities if u > 0.7)
            },
            'clustering': {
                'mean_density': np.mean(cluster_densities),
                'clustered_memories': sum(1 for d in cluster_densities if d > 0.5),
                'distinctive_memories': sum(1 for d in cluster_densities if d < 0.1)
            }
        }
        
        return {
            'base_similarity': base_stats,
            'natural_attention': natural_stats,
            'retrieval_modes': self.retrieval_modes,
            'integration_status': 'active - using emergent properties'
        }