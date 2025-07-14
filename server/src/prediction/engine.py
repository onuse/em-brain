"""
Prediction Engine

Generates actions by finding consensus patterns in similar past experiences.
This is where stored experience becomes intelligent behavior.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import random
import time

from ..experience import Experience
from ..similarity import SimilarityEngine
from ..activation import ActivationDynamics
from .pattern_analyzer import GPUPatternAnalyzer


class PredictionEngine:
    """
    Generates action predictions by consensus from similar past experiences.
    
    Core logic: Find similar past situations, see what actions worked,
    weight by activation and success, return consensus action.
    """
    
    def __init__(self, 
                 min_similar_experiences: int = 3,
                 prediction_confidence_threshold: float = 0.3,
                 success_error_threshold: float = 0.3,
                 bootstrap_randomness: float = 0.8,
                 use_pattern_analysis: bool = True):
        """
        Initialize prediction engine.
        
        Args:
            min_similar_experiences: Minimum similar experiences needed for prediction
            prediction_confidence_threshold: Minimum confidence to trust prediction  
            success_error_threshold: Prediction error threshold for considering success
            bootstrap_randomness: Random action probability when no patterns found
            use_pattern_analysis: Whether to use GPU pattern analysis for predictions
        """
        self.min_similar_experiences = min_similar_experiences
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.success_error_threshold = success_error_threshold
        self.bootstrap_randomness = bootstrap_randomness
        
        # Pattern analysis for intelligent sequence prediction
        self.use_pattern_analysis = use_pattern_analysis
        if use_pattern_analysis:
            self.pattern_analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        else:
            self.pattern_analyzer = None
        
        # Performance tracking
        self.total_predictions = 0
        self.consensus_predictions = 0
        self.pattern_predictions = 0
        self.random_predictions = 0
        self.prediction_accuracies = []
        
        # Experience stream for pattern analysis
        self.recent_experiences = []
        
        pattern_info = "with pattern analysis" if use_pattern_analysis else "similarity-based only"
        print(f"🔮 PredictionEngine initialized ({pattern_info})")
    
    def predict_action(self, 
                      current_context: List[float],
                      similarity_engine: SimilarityEngine,
                      activation_dynamics: ActivationDynamics,
                      all_experiences: Dict[str, Experience],
                      action_dimensions: int) -> Tuple[List[float], float, Dict[str, Any]]:
        """
        Predict the best action for the current context.
        
        Args:
            current_context: Current sensory input vector
            similarity_engine: Similarity search engine
            activation_dynamics: Activation dynamics system
            all_experiences: All stored experiences
            action_dimensions: Number of dimensions in action vector
            
        Returns:
            Tuple of (predicted_action, confidence, prediction_details)
        """
        self.total_predictions += 1
        start_time = time.time()
        
        # Get all experience vectors for similarity search
        experience_vectors = []
        experience_ids = []
        for exp_id, experience in all_experiences.items():
            experience_vectors.append(experience.get_context_vector())
            experience_ids.append(exp_id)
        
        if not experience_vectors:
            # No experiences - bootstrap with random action
            return self._bootstrap_random_action(action_dimensions)
        
        # Find similar experiences
        similar_experiences = similarity_engine.find_similar_experiences(
            current_context, experience_vectors, experience_ids,
            max_results=20, min_similarity=0.4
        )
        
        # Try pattern-based prediction first (if enabled)
        if self.pattern_analyzer and len(self.recent_experiences) >= 2:
            # Convert recent experiences to pattern analyzer format
            converted_sequence = []
            for exp_data in self.recent_experiences[-3:]:
                converted_exp = {
                    'context': np.array(exp_data.get('sensory_input', exp_data.get('context', []))),
                    'action': np.array(exp_data.get('action_taken', exp_data.get('action', []))),
                    'outcome': np.array(exp_data.get('outcome', [])),
                    'timestamp': exp_data.get('timestamp', time.time()),
                    'experience_id': exp_data.get('experience_id', 'unknown')
                }
                converted_sequence.append(converted_exp)
            
            pattern_prediction = self.pattern_analyzer.predict_next_experience(
                np.array(current_context), converted_sequence
            )
            
            if pattern_prediction and pattern_prediction['confidence'] > self.prediction_confidence_threshold:
                self.pattern_predictions += 1
                prediction_time = time.time() - start_time
                
                details = {
                    'method': 'pattern_analysis',
                    'pattern_id': pattern_prediction['pattern_id'],
                    'pattern_frequency': pattern_prediction['pattern_frequency'],
                    'num_similar': len(similar_experiences),
                    'prediction_time': prediction_time
                }
                
                return pattern_prediction['predicted_action'], pattern_prediction['confidence'], details
        
        if len(similar_experiences) < self.min_similar_experiences:
            # Not enough similar experiences - bootstrap with random action
            return self._bootstrap_random_action(action_dimensions, 
                                               similar_experiences, all_experiences)
        
        # Generate consensus prediction from similar experiences
        action, confidence, details = self._generate_consensus_prediction(
            similar_experiences, all_experiences, activation_dynamics, action_dimensions
        )
        
        # Update performance tracking
        prediction_time = time.time() - start_time
        details['prediction_time'] = prediction_time
        
        if confidence >= self.prediction_confidence_threshold:
            self.consensus_predictions += 1
            return action, confidence, details
        else:
            # Low confidence - blend with random action
            return self._blend_with_random(action, confidence, action_dimensions, details)
    
    def add_experience_to_stream(self, experience_data: Dict[str, Any]):
        """
        Add experience to pattern analysis stream.
        
        Args:
            experience_data: Dictionary with sensory_input, action_taken, outcome, etc.
        """
        if self.pattern_analyzer:
            self.pattern_analyzer.add_experience_to_stream(experience_data)
        
        # Keep recent experiences for pattern prediction
        self.recent_experiences.append(experience_data)
        if len(self.recent_experiences) > 50:  # Keep last 50 experiences
            self.recent_experiences = self.recent_experiences[-25:]
    
    def record_prediction_outcome(self, prediction_details: Dict[str, Any], 
                                prediction_success: float):
        """
        Record how well a prediction worked for learning.
        
        Args:
            prediction_details: Details from the prediction
            prediction_success: How successful the prediction was (0-1)
        """
        # Record for pattern analyzer if it was a pattern prediction
        if (self.pattern_analyzer and 
            prediction_details.get('method') == 'pattern_analysis' and
            'pattern_id' in prediction_details):
            
            self.pattern_analyzer.record_prediction_outcome(
                prediction_details['pattern_id'], prediction_success
            )
    
    def _generate_consensus_prediction(self, 
                                     similar_experiences: List[Tuple[str, float]],
                                     all_experiences: Dict[str, Experience],
                                     activation_dynamics: ActivationDynamics,
                                     action_dimensions: int) -> Tuple[List[float], float, Dict[str, Any]]:
        """Generate action by consensus from similar experiences."""
        
        # Collect actions and weights from similar experiences
        actions = []
        weights = []
        prediction_errors = []
        
        for exp_id, similarity in similar_experiences:
            experience = all_experiences[exp_id]
            
            # Weight by similarity, activation, and inverse prediction error (success)
            activation_weight = experience.activation_level
            success_weight = max(0.1, 1.0 - experience.prediction_error)
            
            total_weight = similarity * (1.0 + activation_weight) * success_weight
            
            actions.append(experience.get_action_vector())
            weights.append(total_weight)
            prediction_errors.append(experience.prediction_error)
        
        # Compute weighted consensus action
        actions_array = np.array(actions)
        weights_array = np.array(weights)
        
        if np.sum(weights_array) == 0:
            # No valid weights - return first action with low confidence
            consensus_action = actions[0]
            confidence = 0.1
        else:
            # Weighted average of actions
            weights_normalized = weights_array / np.sum(weights_array)
            consensus_action = np.average(actions_array, axis=0, weights=weights_normalized)
            
            # Confidence based on weight concentration and similarity spread
            weight_concentration = np.max(weights_normalized) / np.mean(weights_normalized)
            avg_similarity = np.mean([sim for _, sim in similar_experiences])
            confidence = min(0.95, avg_similarity * (1.0 + weight_concentration * 0.1))
        
        details = {
            'method': 'consensus',
            'num_similar': len(similar_experiences),
            'avg_similarity': avg_similarity,
            'weight_concentration': weight_concentration,
            'avg_prediction_error': np.mean(prediction_errors),
            'similar_experience_ids': [exp_id for exp_id, _ in similar_experiences[:5]]
        }
        
        return consensus_action.tolist(), confidence, details
    
    def _bootstrap_random_action(self, action_dimensions: int, 
                               similar_experiences: List[Tuple[str, float]] = None,
                               all_experiences: Dict[str, Experience] = None) -> Tuple[List[float], float, Dict[str, Any]]:
        """Generate a random action for exploration/bootstrapping."""
        
        self.random_predictions += 1
        
        # If we have some similar experiences but not enough, blend with their actions
        if similar_experiences and all_experiences:
            # Use the best similar experience as a starting point
            best_exp_id = similar_experiences[0][0]
            best_experience = all_experiences[best_exp_id]
            base_action = np.array(best_experience.get_action_vector())
            
            # Add noise for exploration
            noise = np.random.normal(0, 0.3, action_dimensions)
            random_action = base_action + noise
            
            confidence = 0.2
            method = 'bootstrap_from_similar'
        else:
            # Pure random action
            random_action = np.random.normal(0, 1.0, action_dimensions)
            confidence = 0.1
            method = 'bootstrap_random'
        
        details = {
            'method': method,
            'num_similar': len(similar_experiences) if similar_experiences else 0,
            'bootstrap_randomness': self.bootstrap_randomness
        }
        
        return random_action.tolist(), confidence, details
    
    def _blend_with_random(self, predicted_action: List[float], confidence: float,
                          action_dimensions: int, details: Dict[str, Any]) -> Tuple[List[float], float, Dict[str, Any]]:
        """Blend low-confidence prediction with random action for exploration."""
        
        predicted_array = np.array(predicted_action)
        random_array = np.random.normal(0, 0.5, action_dimensions)
        
        # Blend based on confidence (low confidence = more randomness)
        blend_factor = confidence
        blended_action = blend_factor * predicted_array + (1 - blend_factor) * random_array
        
        # Adjust confidence slightly down due to blending
        blended_confidence = confidence * 0.8
        
        details['method'] = 'blended_with_random'
        details['blend_factor'] = blend_factor
        details['original_confidence'] = confidence
        
        return blended_action.tolist(), blended_confidence, details
    
    def store_prediction_outcome(self, predicted_action: List[float], 
                               actual_outcome: List[float],
                               prediction_confidence: float) -> float:
        """Store the outcome of a prediction for learning."""
        
        # Compute prediction error
        predicted_array = np.array(predicted_action)
        actual_array = np.array(actual_outcome)
        
        # Normalized prediction error (0.0 = perfect, 1.0 = worst possible)
        error = np.linalg.norm(predicted_array - actual_array)
        max_possible_error = np.linalg.norm(predicted_array) + np.linalg.norm(actual_array)
        
        if max_possible_error == 0:
            prediction_error = 0.0
        else:
            prediction_error = min(1.0, error / max_possible_error)
        
        # Track prediction accuracy
        self.prediction_accuracies.append(1.0 - prediction_error)
        
        # Limit history size
        if len(self.prediction_accuracies) > 1000:
            self.prediction_accuracies = self.prediction_accuracies[-500:]
        
        return prediction_error
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive prediction performance statistics."""
        
        if not self.prediction_accuracies:
            avg_accuracy = 0.0
            recent_accuracy = 0.0
        else:
            avg_accuracy = np.mean(self.prediction_accuracies)
            recent_accuracy = np.mean(self.prediction_accuracies[-100:]) if len(self.prediction_accuracies) >= 100 else avg_accuracy
        
        consensus_rate = self.consensus_predictions / max(1, self.total_predictions)
        pattern_rate = self.pattern_predictions / max(1, self.total_predictions)
        random_rate = self.random_predictions / max(1, self.total_predictions)
        
        base_stats = {
            'total_predictions': self.total_predictions,
            'consensus_predictions': self.consensus_predictions,
            'pattern_predictions': self.pattern_predictions,
            'random_predictions': self.random_predictions,
            'consensus_rate': consensus_rate,
            'pattern_rate': pattern_rate,
            'random_rate': random_rate,
            'avg_prediction_accuracy': avg_accuracy,
            'recent_prediction_accuracy': recent_accuracy,
            'prediction_improvement': recent_accuracy - avg_accuracy if len(self.prediction_accuracies) >= 100 else 0.0,
            'learning_curve_length': len(self.prediction_accuracies)
        }
        
        # Add pattern analysis statistics if enabled
        if self.pattern_analyzer:
            pattern_stats = self.pattern_analyzer.get_pattern_statistics()
            base_stats['pattern_analysis'] = pattern_stats
        
        return base_stats
    
    def reset_statistics(self):
        """Reset all performance statistics (for testing)."""
        self.total_predictions = 0
        self.consensus_predictions = 0
        self.pattern_predictions = 0
        self.random_predictions = 0
        self.prediction_accuracies.clear()
        self.recent_experiences = []
        
        # Reset pattern analyzer
        if self.pattern_analyzer:
            self.pattern_analyzer.reset_patterns()
        
        print("🧹 PredictionEngine statistics reset")
    
    def __str__(self) -> str:
        return f"PredictionEngine(predictions={self.total_predictions}, consensus_rate={self.consensus_predictions/max(1,self.total_predictions):.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()