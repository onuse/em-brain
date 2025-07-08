"""
Sensory Prediction System.

Implements sophisticated sensory prediction that allows the robot to "imagine" 
the consequences of actions before taking them. Instead of evaluating actions 
by their immediate properties, the system can now evaluate them based on their 
predicted future outcomes.

This transforms the robot from reactive to truly predictive intelligence.
"""

import numpy as np
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode


@dataclass
class SensoryPrediction:
    """Prediction of what the world will look and feel like after an action."""
    predicted_sensors: Dict[str, float]  # Predicted sensory values
    confidence: float                    # How confident we are in this prediction
    prediction_basis: List[ExperienceNode]  # Experiences used to make prediction
    uncertainty_map: Dict[str, float]   # Per-sensor uncertainty levels
    prediction_method: str              # How this prediction was generated
    
    def get_prediction_quality(self) -> float:
        """Get overall quality score for this prediction."""
        # Higher confidence and more basis experiences = higher quality
        basis_quality = min(1.0, len(self.prediction_basis) / 10.0)  # Up to 10 experiences ideal
        return (self.confidence * 0.7) + (basis_quality * 0.3)
    
    def get_sensor_prediction(self, sensor_name: str, default: float = 0.0) -> float:
        """Get prediction for a specific sensor with default fallback."""
        return self.predicted_sensors.get(sensor_name, default)


@dataclass
class PredictionEvaluation:
    """Evaluation of how accurate a prediction was after the fact."""
    predicted_sensors: Dict[str, float]
    actual_sensors: Dict[str, float]
    prediction_errors: Dict[str, float]
    overall_accuracy: float
    sensor_accuracies: Dict[str, float]
    
    def was_accurate(self, threshold: float = 0.8) -> bool:
        """Check if prediction was accurate enough."""
        return self.overall_accuracy >= threshold


class SensoryPredictor:
    """
    Predicts sensory outcomes of actions based on past experiences.
    
    This allows the robot to "imagine" the future and evaluate actions based
    on their predicted consequences rather than just their immediate properties.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Prediction tracking
        self.prediction_history = []
        self.accuracy_history = []
        self.prediction_count = 0
        
        # Adaptive parameters
        self.similarity_threshold = 0.6  # How similar experiences must be
        self.min_basis_experiences = 3   # Minimum experiences needed for prediction
        self.max_basis_experiences = 15  # Maximum to consider (performance)
        self.confidence_decay = 0.95     # Confidence decay for distant experiences
        
        # Sensor prediction weights (learned over time)
        self.sensor_weights = defaultdict(lambda: 1.0)
        self.sensor_prediction_accuracy = defaultdict(list)
        
        # Prediction method statistics
        self.method_statistics = {
            'direct_match': {'used': 0, 'accuracy': []},
            'similar_context': {'used': 0, 'accuracy': []},
            'action_pattern': {'used': 0, 'accuracy': []},
            'temporal_sequence': {'used': 0, 'accuracy': []},
            'fallback': {'used': 0, 'accuracy': []}
        }
    
    def predict_sensory_outcome(self, action: Dict[str, float], 
                               current_context: List[float],
                               current_sensors: Dict[str, float]) -> SensoryPrediction:
        """
        Predict what the world will look and feel like after taking this action.
        
        Args:
            action: Motor action to predict consequences of
            current_context: Current mental context 
            current_sensors: Current sensory readings
            
        Returns:
            SensoryPrediction with imagined future sensory state
        """
        self.prediction_count += 1
        
        # Try multiple prediction strategies in order of preference
        prediction_strategies = [
            self._predict_from_direct_matches,
            self._predict_from_similar_context,
            self._predict_from_action_patterns,
            self._predict_from_temporal_sequences,
            self._predict_fallback
        ]
        
        for strategy in prediction_strategies:
            prediction = strategy(action, current_context, current_sensors)
            if prediction and prediction.confidence > 0.3:  # Minimum confidence threshold
                self._log_prediction_method(prediction.prediction_method)
                return prediction
        
        # If all strategies fail, return fallback prediction
        fallback = self._predict_fallback(action, current_context, current_sensors)
        self._log_prediction_method('fallback')
        return fallback
    
    def _predict_from_direct_matches(self, action: Dict[str, float], 
                                   current_context: List[float],
                                   current_sensors: Dict[str, float]) -> Optional[SensoryPrediction]:
        """Find experiences with very similar context and action."""
        if not self.world_graph.has_nodes():
            return None
        
        similar_experiences = []
        all_nodes = self.world_graph.all_nodes()
        
        for node in all_nodes:
            # Check context similarity
            context_similarity = self._calculate_context_similarity(current_context, node.mental_context)
            if context_similarity < 0.8:  # Very high threshold for direct matches
                continue
            
            # Check action similarity
            action_similarity = self._calculate_action_similarity(action, node.action_taken)
            if action_similarity < 0.8:  # Very high threshold for direct matches
                continue
            
            # This is a direct match
            combined_similarity = (context_similarity + action_similarity) / 2
            similar_experiences.append((node, combined_similarity))
        
        if len(similar_experiences) < self.min_basis_experiences:
            return None
        
        # Sort by similarity and take the best matches
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        best_matches = similar_experiences[:self.max_basis_experiences]
        
        # Aggregate predicted sensory outcomes
        predicted_sensors, confidence = self._aggregate_sensory_predictions(best_matches)
        
        return SensoryPrediction(
            predicted_sensors=predicted_sensors,
            confidence=confidence * 0.95,  # High confidence for direct matches
            prediction_basis=[exp[0] for exp in best_matches],
            uncertainty_map=self._calculate_uncertainty_map(best_matches),
            prediction_method='direct_match'
        )
    
    def _predict_from_similar_context(self, action: Dict[str, float], 
                                    current_context: List[float],
                                    current_sensors: Dict[str, float]) -> Optional[SensoryPrediction]:
        """Find experiences with similar context, regardless of action."""
        if not self.world_graph.has_nodes():
            return None
        
        similar_experiences = []
        all_nodes = self.world_graph.all_nodes()
        
        for node in all_nodes:
            context_similarity = self._calculate_context_similarity(current_context, node.mental_context)
            if context_similarity >= self.similarity_threshold:
                similar_experiences.append((node, context_similarity))
        
        if len(similar_experiences) < self.min_basis_experiences:
            return None
        
        # Sort by similarity and take the best matches
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        best_matches = similar_experiences[:self.max_basis_experiences]
        
        # Aggregate predicted sensory outcomes with action adjustment
        predicted_sensors, base_confidence = self._aggregate_sensory_predictions(best_matches)
        
        # Adjust prediction based on action differences
        adjusted_sensors = self._adjust_prediction_for_action(predicted_sensors, action, best_matches)
        
        # Lower confidence since we're extrapolating across different actions
        confidence = base_confidence * 0.7
        
        return SensoryPrediction(
            predicted_sensors=adjusted_sensors,
            confidence=confidence,
            prediction_basis=[exp[0] for exp in best_matches],
            uncertainty_map=self._calculate_uncertainty_map(best_matches),
            prediction_method='similar_context'
        )
    
    def _predict_from_action_patterns(self, action: Dict[str, float], 
                                    current_context: List[float],
                                    current_sensors: Dict[str, float]) -> Optional[SensoryPrediction]:
        """Find experiences with similar actions, regardless of context."""
        if not self.world_graph.has_nodes():
            return None
        
        similar_experiences = []
        all_nodes = self.world_graph.all_nodes()
        
        for node in all_nodes:
            action_similarity = self._calculate_action_similarity(action, node.action_taken)
            if action_similarity >= self.similarity_threshold:
                similar_experiences.append((node, action_similarity))
        
        if len(similar_experiences) < self.min_basis_experiences:
            return None
        
        # Sort by similarity and take the best matches
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        best_matches = similar_experiences[:self.max_basis_experiences]
        
        # Aggregate predicted sensory outcomes
        predicted_sensors, base_confidence = self._aggregate_sensory_predictions(best_matches)
        
        # Lower confidence since we're ignoring context
        confidence = base_confidence * 0.5
        
        return SensoryPrediction(
            predicted_sensors=predicted_sensors,
            confidence=confidence,
            prediction_basis=[exp[0] for exp in best_matches],
            uncertainty_map=self._calculate_uncertainty_map(best_matches),
            prediction_method='action_pattern'
        )
    
    def _predict_from_temporal_sequences(self, action: Dict[str, float], 
                                       current_context: List[float],
                                       current_sensors: Dict[str, float]) -> Optional[SensoryPrediction]:
        """Predict based on recent temporal patterns."""
        if not self.world_graph.has_nodes():
            return None
        
        # Get recent experiences (temporal chain)
        recent_nodes = []
        if hasattr(self.world_graph, 'get_recent_experiences'):
            recent_nodes = self.world_graph.get_recent_experiences(limit=20)
        else:
            # Fallback: get nodes with high access counts (recently used)
            all_nodes = self.world_graph.all_nodes()
            sorted_nodes = sorted(all_nodes, key=lambda n: n.access_count, reverse=True)
            recent_nodes = sorted_nodes[:20]
        
        if len(recent_nodes) < 5:
            return None
        
        # Look for patterns in recent experiences
        pattern_predictions = []
        for i in range(len(recent_nodes) - 2):
            # Find sequences: context A + action B → outcome C
            if hasattr(recent_nodes[i], 'next_experiences'):
                for next_exp in recent_nodes[i].next_experiences:
                    if next_exp in recent_nodes:
                        # This is a sequence we can learn from
                        context_sim = self._calculate_context_similarity(current_context, recent_nodes[i].mental_context)
                        action_sim = self._calculate_action_similarity(action, recent_nodes[i].action_taken)
                        
                        if context_sim > 0.5 and action_sim > 0.5:
                            pattern_predictions.append((next_exp, context_sim * action_sim))
        
        if not pattern_predictions:
            return None
        
        # Take the best pattern match
        pattern_predictions.sort(key=lambda x: x[1], reverse=True)
        best_pattern = pattern_predictions[0]
        
        return SensoryPrediction(
            predicted_sensors=dict(enumerate(best_pattern[0].actual_sensory)),
            confidence=best_pattern[1] * 0.6,  # Moderate confidence for patterns
            prediction_basis=[best_pattern[0]],
            uncertainty_map={str(i): 0.3 for i in range(len(best_pattern[0].actual_sensory))},
            prediction_method='temporal_sequence'
        )
    
    def _predict_fallback(self, action: Dict[str, float], 
                         current_context: List[float],
                         current_sensors: Dict[str, float]) -> SensoryPrediction:
        """Fallback prediction when no good matches are found."""
        # Simple heuristic-based prediction
        predicted_sensors = current_sensors.copy()
        
        # Apply simple action-based adjustments
        if 'forward_motor' in action and action['forward_motor'] > 0.3:
            # Moving forward might increase wall sensor
            if 'wall_sensor' in predicted_sensors:
                predicted_sensors['wall_sensor'] = min(1.0, predicted_sensors['wall_sensor'] + 0.2)
        
        if 'brake_motor' in action and action['brake_motor'] > 0.5:
            # Braking might reduce speed sensors
            for sensor_name in predicted_sensors:
                if 'speed' in sensor_name.lower():
                    predicted_sensors[sensor_name] *= 0.5
        
        return SensoryPrediction(
            predicted_sensors=predicted_sensors,
            confidence=0.2,  # Low confidence for fallback
            prediction_basis=[],
            uncertainty_map={sensor: 0.8 for sensor in predicted_sensors},
            prediction_method='fallback'
        )
    
    def _calculate_context_similarity(self, context1: List[float], context2: List[float]) -> float:
        """Calculate similarity between two mental contexts."""
        if not context1 or not context2:
            return 0.0
        
        # Pad to same length
        max_len = max(len(context1), len(context2))
        padded1 = context1 + [0.0] * (max_len - len(context1))
        padded2 = context2 + [0.0] * (max_len - len(context2))
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(padded1, padded2))
        norm1 = sum(a * a for a in padded1) ** 0.5
        norm2 = sum(b * b for b in padded2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_action_similarity(self, action1: Dict[str, float], action2: Dict[str, float]) -> float:
        """Calculate similarity between two actions."""
        if not action1 or not action2:
            return 0.0
        
        # Get all action components
        all_keys = set(action1.keys()) | set(action2.keys())
        
        # Calculate component-wise similarity
        similarities = []
        for key in all_keys:
            val1 = action1.get(key, 0.0)
            val2 = action2.get(key, 0.0)
            
            # Euclidean distance converted to similarity
            distance = abs(val1 - val2)
            similarity = max(0.0, 1.0 - distance)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _aggregate_sensory_predictions(self, experiences: List[Tuple[ExperienceNode, float]]) -> Tuple[Dict[str, float], float]:
        """Aggregate sensory predictions from multiple experiences."""
        if not experiences:
            return {}, 0.0
        
        # Collect all sensory outcomes with weights
        sensor_values = defaultdict(list)
        weights = []
        
        for node, similarity in experiences:
            weight = similarity * (node.strength / 100.0)  # Incorporate node strength
            weights.append(weight)
            
            # Convert actual sensory to dictionary if needed
            if hasattr(node, 'actual_sensory') and node.actual_sensory:
                if isinstance(node.actual_sensory, list):
                    for i, value in enumerate(node.actual_sensory):
                        sensor_values[str(i)].append((value, weight))
                elif isinstance(node.actual_sensory, dict):
                    for sensor, value in node.actual_sensory.items():
                        sensor_values[sensor].append((value, weight))
        
        # Calculate weighted averages
        predicted_sensors = {}
        total_weight = sum(weights)
        
        for sensor, value_weight_pairs in sensor_values.items():
            weighted_sum = sum(value * weight for value, weight in value_weight_pairs)
            predicted_sensors[sensor] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on agreement and weight
        confidence = min(1.0, total_weight / len(experiences)) if experiences else 0.0
        
        return predicted_sensors, confidence
    
    def _adjust_prediction_for_action(self, base_prediction: Dict[str, float], 
                                    target_action: Dict[str, float],
                                    basis_experiences: List[Tuple[ExperienceNode, float]]) -> Dict[str, float]:
        """Adjust base prediction based on action differences."""
        adjusted_prediction = base_prediction.copy()
        
        # Simple heuristic adjustments based on action differences
        # This could be made much more sophisticated with learned action models
        
        for sensor in adjusted_prediction:
            adjustment = 0.0
            
            # Example: forward motion affects distance sensors
            if 'distance' in sensor.lower() or 'wall' in sensor.lower():
                forward_diff = target_action.get('forward_motor', 0.0)
                if forward_diff > 0.1:
                    adjustment += 0.1  # Moving forward increases distance sensor readings
                elif forward_diff < -0.1:
                    adjustment -= 0.1  # Moving backward decreases them
            
            # Apply adjustment with dampening
            adjusted_prediction[sensor] = max(0.0, min(1.0, adjusted_prediction[sensor] + adjustment * 0.5))
        
        return adjusted_prediction
    
    def _calculate_uncertainty_map(self, experiences: List[Tuple[ExperienceNode, float]]) -> Dict[str, float]:
        """Calculate per-sensor uncertainty levels."""
        if not experiences:
            return {}
        
        # Calculate variance for each sensor across experiences
        sensor_values = defaultdict(list)
        
        for node, similarity in experiences:
            if hasattr(node, 'actual_sensory') and node.actual_sensory:
                if isinstance(node.actual_sensory, list):
                    for i, value in enumerate(node.actual_sensory):
                        sensor_values[str(i)].append(value)
                elif isinstance(node.actual_sensory, dict):
                    for sensor, value in node.actual_sensory.items():
                        sensor_values[sensor].append(value)
        
        uncertainty_map = {}
        for sensor, values in sensor_values.items():
            if len(values) > 1:
                variance = statistics.variance(values)
                uncertainty = min(1.0, variance)  # Higher variance = higher uncertainty
            else:
                uncertainty = 0.5  # Default uncertainty for single data point
            
            uncertainty_map[sensor] = uncertainty
        
        return uncertainty_map
    
    def _log_prediction_method(self, method: str):
        """Log which prediction method was used."""
        if method in self.method_statistics:
            self.method_statistics[method]['used'] += 1
    
    def evaluate_prediction_accuracy(self, prediction: SensoryPrediction, 
                                   actual_sensors: Dict[str, float]) -> PredictionEvaluation:
        """Evaluate how accurate a prediction was after the fact."""
        prediction_errors = {}
        sensor_accuracies = {}
        
        # Calculate per-sensor accuracy
        for sensor in prediction.predicted_sensors:
            predicted = prediction.predicted_sensors[sensor]
            actual = actual_sensors.get(sensor, 0.0)
            
            error = abs(predicted - actual)
            accuracy = max(0.0, 1.0 - error)  # Convert error to accuracy
            
            prediction_errors[sensor] = error
            sensor_accuracies[sensor] = accuracy
        
        # Calculate overall accuracy
        overall_accuracy = sum(sensor_accuracies.values()) / len(sensor_accuracies) if sensor_accuracies else 0.0
        
        # Update learning statistics
        self.accuracy_history.append(overall_accuracy)
        self.method_statistics[prediction.prediction_method]['accuracy'].append(overall_accuracy)
        
        # Keep history manageable
        if len(self.accuracy_history) > 1000:
            self.accuracy_history = self.accuracy_history[-500:]
        
        evaluation = PredictionEvaluation(
            predicted_sensors=prediction.predicted_sensors,
            actual_sensors=actual_sensors,
            prediction_errors=prediction_errors,
            overall_accuracy=overall_accuracy,
            sensor_accuracies=sensor_accuracies
        )
        
        self.prediction_history.append(evaluation)
        return evaluation
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about prediction performance."""
        stats = {
            'total_predictions': self.prediction_count,
            'average_accuracy': sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else 0.0,
            'recent_accuracy': sum(self.accuracy_history[-20:]) / min(20, len(self.accuracy_history)) if self.accuracy_history else 0.0,
            'prediction_methods': {}
        }
        
        # Method-specific statistics
        for method, data in self.method_statistics.items():
            if data['used'] > 0:
                avg_accuracy = sum(data['accuracy']) / len(data['accuracy']) if data['accuracy'] else 0.0
                stats['prediction_methods'][method] = {
                    'times_used': data['used'],
                    'usage_percentage': (data['used'] / self.prediction_count * 100) if self.prediction_count > 0 else 0.0,
                    'average_accuracy': avg_accuracy
                }
        
        return stats
    
    def adapt_parameters(self):
        """Adapt prediction parameters based on performance."""
        if len(self.accuracy_history) < 10:
            return
        
        recent_accuracy = sum(self.accuracy_history[-10:]) / 10
        
        # If accuracy is low, be less strict about similarity
        if recent_accuracy < 0.6:
            self.similarity_threshold = max(0.4, self.similarity_threshold - 0.05)
            self.min_basis_experiences = max(2, self.min_basis_experiences - 1)
        
        # If accuracy is high, be more strict
        elif recent_accuracy > 0.8:
            self.similarity_threshold = min(0.8, self.similarity_threshold + 0.02)
            self.min_basis_experiences = min(5, self.min_basis_experiences + 1)