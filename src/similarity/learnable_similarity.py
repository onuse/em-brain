"""
Learnable Similarity Function

Instead of hardcoded cosine similarity, this learns what similarity means
based on prediction success. The similarity function evolves to optimize
predictive accuracy - if two experiences are considered "similar" but don't
help predict each other, the similarity function adapts.

This is a fundamental step toward true emergence - letting the system discover
what similarity means rather than engineering it.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from collections import defaultdict


class LearnableSimilarity:
    """
    Similarity function that learns from prediction success.
    
    Core principle: If experiences labeled as "similar" help predict each other,
    the similarity function is working. If not, it needs to adapt.
    """
    
    def __init__(self, vector_dimensions: int = None, learning_rate: float = 0.01):
        """
        Initialize learnable similarity function.
        
        Args:
            vector_dimensions: Dimensionality of experience vectors (learned if None)
            learning_rate: Initial learning rate (will become adaptive)
        """
        self.vector_dimensions = vector_dimensions
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate  # This will adapt based on learning success
        
        # Meta-learning parameters (Strategy 5)
        self.learning_rate_adaptation_rate = 0.1  # How fast learning rate itself adapts
        self.min_learning_rate = 0.001
        self.max_learning_rate = 0.1
        self.adaptation_success_history = []  # Track how well adaptations work
        
        # Learnable similarity parameters
        # Start with identity transformation (like cosine similarity)
        self.feature_weights = None  # Will be initialized when we see first vector
        self.interaction_matrix = None  # Learns feature interactions
        
        # Prediction success tracking
        self.similarity_predictions = defaultdict(list)  # similarity_score -> [prediction_success]
        self.prediction_outcomes = []  # Track recent prediction utilities
        
        # Adaptation tracking  
        self.adaptations_performed = 0
        self.similarity_evolution = []  # Track how similarity function changes
        
        print("LearnableSimilarity initialized - similarity will emerge from prediction success")
    
    def _initialize_parameters(self, vector_dim: int):
        """Initialize learnable parameters when we see first vector."""
        if self.feature_weights is not None:
            return  # Already initialized
            
        self.vector_dimensions = vector_dim
        
        # Feature weights - learns which dimensions matter for prediction
        # Start close to uniform (slight randomization to break symmetry)
        self.feature_weights = np.ones(vector_dim) + np.random.normal(0, 0.1, vector_dim)
        
        # Interaction matrix - learns how features interact
        # Start as identity (no interactions initially)
        self.interaction_matrix = np.eye(vector_dim) * 0.1
        
        print(f"Similarity parameters initialized for {vector_dim}D vectors")
    
    def compute_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """
        Compute learned similarity between two vectors.
        
        Args:
            vector_a: First experience vector
            vector_b: Second experience vector
            
        Returns:
            Similarity score (0.0-1.0)
        """
        vec_a = np.array(vector_a)
        vec_b = np.array(vector_b)
        
        # Initialize parameters if needed
        if self.feature_weights is None:
            self._initialize_parameters(len(vec_a))
        
        # Apply learned feature weighting
        weighted_a = vec_a * self.feature_weights
        weighted_b = vec_b * self.feature_weights
        
        # Apply learned feature interactions
        # This lets the system discover which feature combinations matter
        transformed_a = weighted_a + np.dot(self.interaction_matrix, vec_a)
        transformed_b = weighted_b + np.dot(self.interaction_matrix, vec_b)
        
        # Compute similarity in learned space
        # Start with cosine-like similarity but in transformed space
        norm_a = np.linalg.norm(transformed_a)
        norm_b = np.linalg.norm(transformed_b)
        
        if norm_a == 0 or norm_b == 0:
            # Handle zero vectors with distance-based similarity
            distance = np.linalg.norm(transformed_a - transformed_b)
            max_distance = np.sqrt(2 * len(vector_a))  # Rough upper bound
            similarity = max(0.0, 1.0 - (distance / max_distance))
        else:
            # Cosine similarity in learned space
            dot_product = np.dot(transformed_a, transformed_b)
            cosine_sim = dot_product / (norm_a * norm_b)
            # Convert from [-1, 1] to [0, 1]
            similarity = (cosine_sim + 1.0) / 2.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def record_prediction_outcome(self, 
                                query_vector: List[float],
                                similar_vector: List[float], 
                                prediction_success: float):
        """
        Record how well similar experiences helped with prediction.
        
        Args:
            query_vector: The experience we wanted to predict from
            similar_vector: The "similar" experience we used for prediction
            prediction_success: How well it worked (1.0 = perfect, 0.0 = useless)
        """
        similarity_score = self.compute_similarity(query_vector, similar_vector)
        
        # Record that this similarity score led to this prediction success
        self.similarity_predictions[round(similarity_score, 2)].append(prediction_success)
        
        # Track recent outcomes for adaptation
        self.prediction_outcomes.append({
            'query': np.array(query_vector),
            'similar': np.array(similar_vector), 
            'similarity': similarity_score,
            'success': prediction_success,
            'timestamp': time.time()
        })
        
        # Limit history to prevent memory bloat
        if len(self.prediction_outcomes) > 1000:
            self.prediction_outcomes = self.prediction_outcomes[-500:]
    
    def adapt_similarity_function(self):
        """
        Adapt the similarity function based on prediction success patterns.
        
        Core idea: If high similarity scores don't lead to good predictions,
        or if low similarity scores actually do help, adjust the function.
        """
        if len(self.prediction_outcomes) < 20:
            return  # Need sufficient data
        
        recent_outcomes = self.prediction_outcomes[-50:]  # Focus on recent performance
        
        # Analyze correlation between similarity scores and prediction success
        similarities = np.array([outcome['similarity'] for outcome in recent_outcomes])
        successes = np.array([outcome['success'] for outcome in recent_outcomes])
        
        if len(similarities) < 10:
            return
        
        # Track performance before adaptation for meta-learning
        correlation_before = np.corrcoef(similarities, successes)[0, 1]
        if np.isnan(correlation_before):
            correlation_before = 0.0
        
        # If high similarity isn't correlating with high prediction success, adapt
        if correlation_before < 0.3:  # Poor correlation
            print(f"Adapting similarity function (correlation: {correlation_before:.3f}, lr: {self.learning_rate:.4f})")
            
            # Store performance before adaptation
            pre_adaptation_performance = np.mean(successes[-10:])
            
            self._gradient_adapt(recent_outcomes)
            self.adaptations_performed += 1
            
            # Meta-learning: adapt the learning rate based on adaptation success
            self._adapt_learning_rate(pre_adaptation_performance)
    
    def _gradient_adapt(self, recent_outcomes: List[Dict]):
        """Perform gradient-based adaptation of similarity parameters."""
        
        # Simple gradient approach: adjust weights to improve similarity-success correlation
        for outcome in recent_outcomes[-10:]:  # Use recent examples
            query = outcome['query']
            similar = outcome['similar']
            similarity = outcome['similarity']
            success = outcome['success']
            
            # If success is high but similarity is low, strengthen features that differ
            # If success is low but similarity is high, weaken features that are similar
            
            feature_diff = np.abs(query - similar)
            
            if success > 0.7 and similarity < 0.5:
                # Good prediction from "dissimilar" experiences - maybe these features matter more
                gradient = feature_diff * self.learning_rate * (success - similarity)
                self.feature_weights += gradient
                
            elif success < 0.3 and similarity > 0.7:
                # Bad prediction from "similar" experiences - maybe these features matter less
                gradient = feature_diff * self.learning_rate * (similarity - success)
                self.feature_weights -= gradient
        
        # Keep weights positive and normalized
        self.feature_weights = np.maximum(0.1, self.feature_weights)
        self.feature_weights = self.feature_weights / np.mean(self.feature_weights)
        
        # Adapt interaction matrix more conservatively
        if len(recent_outcomes) >= 20:
            self._adapt_interaction_matrix(recent_outcomes[-20:])
    
    def _adapt_interaction_matrix(self, outcomes: List[Dict]):
        """Adapt feature interaction learning."""
        # Simple approach: strengthen interactions between features that co-predict
        
        for outcome in outcomes:
            if outcome['success'] > 0.6:  # Only learn from successful predictions
                query = outcome['query']
                similar = outcome['similar']
                
                # Feature activation patterns that led to successful prediction
                feature_activation = query * similar  # Element-wise interaction
                
                # Strengthen these interaction patterns slightly
                update = np.outer(feature_activation, feature_activation) * self.learning_rate * 0.1
                self.interaction_matrix += update
        
        # Keep interaction matrix bounded
        self.interaction_matrix = np.clip(self.interaction_matrix, -1.0, 1.0)
    
    def _adapt_learning_rate(self, pre_adaptation_performance: float):
        """
        Meta-learning: adapt the learning rate based on adaptation success.
        
        This is Strategy 5 in action - the parameter that controls adaptation
        is itself adaptive.
        
        Args:
            pre_adaptation_performance: Performance before the adaptation
        """
        # Wait a bit to see if the adaptation helped
        if len(self.prediction_outcomes) < 10:
            return
        
        # Compare performance after adaptation
        recent_performance = np.mean([outcome['success'] for outcome in self.prediction_outcomes[-5:]])
        
        # Calculate adaptation success
        adaptation_improvement = recent_performance - pre_adaptation_performance
        
        # Record this adaptation outcome for meta-meta-learning
        self.adaptation_success_history.append({
            'learning_rate': self.learning_rate,
            'improvement': adaptation_improvement,
            'timestamp': time.time()
        })
        
        # Limit history
        if len(self.adaptation_success_history) > 50:
            self.adaptation_success_history = self.adaptation_success_history[-25:]
        
        # Adapt learning rate based on whether adaptations are helping
        if len(self.adaptation_success_history) >= 3:
            recent_improvements = [entry['improvement'] for entry in self.adaptation_success_history[-3:]]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement > 0.05:
                # Adaptations are helping - maybe we can learn faster
                new_learning_rate = self.learning_rate * (1 + self.learning_rate_adaptation_rate)
                print(f"Meta-learning: Increasing learning rate {self.learning_rate:.4f} → {new_learning_rate:.4f} (improvement: {avg_improvement:.3f})")
            elif avg_improvement < -0.05:
                # Adaptations are hurting - learn more slowly
                new_learning_rate = self.learning_rate * (1 - self.learning_rate_adaptation_rate)
                print(f"Meta-learning: Decreasing learning rate {self.learning_rate:.4f} → {new_learning_rate:.4f} (improvement: {avg_improvement:.3f})")
            else:
                # No clear trend - slight adjustment toward optimal
                new_learning_rate = self.learning_rate
            
            # Apply bounds and update
            self.learning_rate = np.clip(new_learning_rate, self.min_learning_rate, self.max_learning_rate)
    
    def get_similarity_statistics(self) -> Dict:
        """Get statistics about similarity learning progress."""
        
        # Analyze similarity-success correlation
        if len(self.prediction_outcomes) >= 10:
            recent = self.prediction_outcomes[-50:]
            similarities = [o['similarity'] for o in recent]
            successes = [o['success'] for o in recent]
            correlation = np.corrcoef(similarities, successes)[0, 1] if len(similarities) > 1 else 0.0
        else:
            correlation = 0.0
        
        # Feature weight statistics
        if self.feature_weights is not None:
            weight_variance = np.var(self.feature_weights)
            dominant_features = np.argsort(self.feature_weights)[-3:]  # Top 3 features
        else:
            weight_variance = 0.0
            dominant_features = []
        
        return {
            'adaptations_performed': self.adaptations_performed,
            'prediction_outcomes_tracked': len(self.prediction_outcomes),
            'similarity_success_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'feature_weight_variance': float(weight_variance),
            'dominant_feature_indices': dominant_features.tolist() if len(dominant_features) > 0 else [],
            'similarity_function_type': 'learned_adaptive',
            'learning_rate': self.learning_rate,
            'meta_learning': self.get_meta_learning_stats()
        }
    
    def get_meta_learning_stats(self) -> Dict:
        """Get statistics about meta-learning (learning rate adaptation)."""
        if not self.adaptation_success_history:
            return {
                'meta_learning_active': False,
                'current_learning_rate': self.learning_rate,
                'initial_learning_rate': self.initial_learning_rate
            }
        
        recent_improvements = [entry['improvement'] for entry in self.adaptation_success_history[-10:]]
        learning_rates = [entry['learning_rate'] for entry in self.adaptation_success_history[-10:]]
        
        return {
            'meta_learning_active': True,
            'current_learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'learning_rate_adaptations': len(self.adaptation_success_history),
            'avg_adaptation_improvement': np.mean(recent_improvements) if recent_improvements else 0.0,
            'learning_rate_trend': {
                'min': np.min(learning_rates) if learning_rates else self.learning_rate,
                'max': np.max(learning_rates) if learning_rates else self.learning_rate,
                'current': self.learning_rate
            },
            'adaptation_success_rate': sum(1 for imp in recent_improvements if imp > 0) / len(recent_improvements) if recent_improvements else 0.0
        }
    
    def reset_learning(self):
        """Reset the similarity function to start learning fresh."""
        self.feature_weights = None
        self.interaction_matrix = None
        self.similarity_predictions.clear()
        self.prediction_outcomes.clear()
        self.adaptations_performed = 0
        self.similarity_evolution.clear()
        
        # Reset meta-learning parameters (Strategy 5)
        self.learning_rate = self.initial_learning_rate
        self.adaptation_success_history.clear()
        
        print("Similarity learning reset - starting fresh emergence process (including meta-learning)")