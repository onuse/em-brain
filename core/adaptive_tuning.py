"""
Adaptive Parameter Tuning System - The brain's ability to learn and optimize its own parameters.
This is the foundation for sensory adaptation, bandwidth handling, and emergent meaning discovery.
"""

import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class ParameterPerformance:
    """Tracks performance of a specific parameter value."""
    parameter_name: str
    parameter_value: float
    recent_prediction_errors: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    avg_prediction_error: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def add_performance_sample(self, prediction_error: float, success_threshold: float = 0.3):
        """Add a performance sample for this parameter value."""
        self.recent_prediction_errors.append(prediction_error)
        
        # Keep only recent samples (sliding window)
        if len(self.recent_prediction_errors) > 20:
            self.recent_prediction_errors.pop(0)
        
        # Update success/failure counts
        if prediction_error <= success_threshold:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update average
        self.avg_prediction_error = statistics.mean(self.recent_prediction_errors)
        self.last_updated = time.time()
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (higher is better)."""
        if not self.recent_prediction_errors:
            return 0.0
        
        # Base score from inverse of prediction error
        base_score = 1.0 / (1.0 + self.avg_prediction_error)
        
        # Bonus for consistent success
        total_attempts = self.success_count + self.failure_count
        if total_attempts > 0:
            success_rate = self.success_count / total_attempts
            consistency_bonus = success_rate * 0.5
        else:
            consistency_bonus = 0.0
        
        return base_score + consistency_bonus


@dataclass
class SensoryDimensionProfile:
    """Profile of a sensory dimension to understand its characteristics."""
    dimension_index: int
    recent_values: List[float] = field(default_factory=list)
    variance: float = 0.0
    dynamic_range: float = 0.0
    change_frequency: float = 0.0  # How often this dimension changes significantly
    predictability: float = 0.0    # How well we can predict this dimension
    importance: float = 0.0        # How much this dimension affects prediction accuracy
    
    def update_profile(self, new_value: float):
        """Update the profile with a new sensory value."""
        self.recent_values.append(new_value)
        
        # Keep sliding window
        if len(self.recent_values) > 100:
            self.recent_values.pop(0)
        
        if len(self.recent_values) >= 2:
            # Calculate variance (measure of "activity")
            self.variance = statistics.variance(self.recent_values)
            
            # Calculate dynamic range
            self.dynamic_range = max(self.recent_values) - min(self.recent_values)
            
            # Calculate change frequency
            changes = sum(1 for i in range(1, len(self.recent_values)) 
                         if abs(self.recent_values[i] - self.recent_values[i-1]) > 0.1)
            self.change_frequency = changes / len(self.recent_values)


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning system that learns optimal parameters through experience.
    This enables the brain to adapt to different sensory modalities and improve performance.
    """
    
    def __init__(self):
        # Parameter tracking
        self.parameter_history: Dict[str, List[ParameterPerformance]] = {}
        self.current_parameters: Dict[str, float] = {}
        
        # Sensory adaptation
        self.sensory_profiles: Dict[int, SensoryDimensionProfile] = {}
        self.bandwidth_adaptation_enabled = True
        self.detected_bandwidth_tier: str = "low"  # low, medium, high
        
        # Learning configuration
        self.learning_rate = 0.1
        self.exploration_probability = 0.2
        self.performance_window = 50  # How many samples to consider for performance
        
        # Adaptation statistics
        self.total_adaptations = 0
        self.successful_adaptations = 0
        self.last_adaptation_time = time.time()
        
        # Initialize default parameters
        self._initialize_default_parameters()
    
    def _initialize_default_parameters(self):
        """Initialize default parameter values."""
        self.current_parameters = {
            # Memory/graph parameters
            'similarity_threshold': 0.7,
            'activation_spread_iterations': 3,
            'consolidation_frequency': 50,
            'forgetting_threshold': 0.1,
            'connection_learning_rate': 0.1,
            
            # Prediction parameters
            'exploration_rate': 0.3,
            'time_budget_base': 0.1,
            'prediction_confidence_threshold': 0.5,
            
            # Sensory processing parameters
            'sensory_importance_threshold': 0.3,
            'dimension_clustering_threshold': 0.5,
            'bandwidth_adaptation_rate': 0.05,
        }
        
        # Initialize parameter history
        for param_name, value in self.current_parameters.items():
            self.parameter_history[param_name] = [
                ParameterPerformance(param_name, value)
            ]
    
    def update_sensory_profiles(self, sensory_vector: List[float]):
        """Update sensory dimension profiles and detect bandwidth characteristics."""
        for i, value in enumerate(sensory_vector):
            if i not in self.sensory_profiles:
                self.sensory_profiles[i] = SensoryDimensionProfile(i)
            
            self.sensory_profiles[i].update_profile(value)
        
        # Detect bandwidth tier
        if len(sensory_vector) > 50:  # High bandwidth (e.g., camera)
            self.detected_bandwidth_tier = "high"
        elif len(sensory_vector) > 10:  # Medium bandwidth (e.g., multiple sensors)
            self.detected_bandwidth_tier = "medium"
        else:  # Low bandwidth (e.g., simple sensors)
            self.detected_bandwidth_tier = "low"
    
    def adapt_parameters_from_prediction_error(self, prediction_error: float, 
                                             sensory_vector: List[float],
                                             context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Adapt parameters based on prediction error and sensory characteristics.
        Returns the updated parameters.
        """
        # Update sensory profiles
        self.update_sensory_profiles(sensory_vector)
        
        # Record performance for current parameters
        self._record_parameter_performance(prediction_error)
        
        # Decide whether to adapt
        if self._should_adapt_parameters(prediction_error):
            adapted_params = self._adapt_parameters(prediction_error, sensory_vector, context)
            self.total_adaptations += 1
            
            # Check if adaptation was successful (will be validated later)
            if prediction_error < 0.3:  # Reasonable threshold
                self.successful_adaptations += 1
            
            return adapted_params
        
        return self.current_parameters.copy()
    
    def _record_parameter_performance(self, prediction_error: float):
        """Record performance for current parameter values."""
        for param_name, param_value in self.current_parameters.items():
            if param_name in self.parameter_history:
                # Find or create performance record for this value
                performance_record = None
                for record in self.parameter_history[param_name]:
                    if abs(record.parameter_value - param_value) < 0.01:  # Close enough
                        performance_record = record
                        break
                
                if performance_record is None:
                    performance_record = ParameterPerformance(param_name, param_value)
                    self.parameter_history[param_name].append(performance_record)
                
                performance_record.add_performance_sample(prediction_error)
    
    def _should_adapt_parameters(self, prediction_error: float) -> bool:
        """Determine if parameters should be adapted."""
        # Always adapt if error is very high
        if prediction_error > 0.8:
            return True
        
        # Adapt probabilistically based on exploration rate
        if time.time() - self.last_adaptation_time < 5.0:  # Don't adapt too frequently
            return False
        
        # Explore occasionally even when doing well
        import random
        return random.random() < self.exploration_probability
    
    def _adapt_parameters(self, prediction_error: float, sensory_vector: List[float], 
                         context: Dict[str, Any] = None) -> Dict[str, float]:
        """Adapt parameters based on current performance and context."""
        adapted_params = self.current_parameters.copy()
        
        # 1. Bandwidth-based adaptation
        if self.bandwidth_adaptation_enabled:
            adapted_params = self._adapt_for_bandwidth(adapted_params, sensory_vector)
        
        # 2. Error-based adaptation
        adapted_params = self._adapt_for_prediction_error(adapted_params, prediction_error)
        
        # 3. Sensory dimension importance adaptation
        adapted_params = self._adapt_for_sensory_characteristics(adapted_params)
        
        # 4. Apply learning rate damping
        adapted_params = self._apply_learning_rate_damping(adapted_params)
        
        # Update current parameters
        self.current_parameters = adapted_params
        self.last_adaptation_time = time.time()
        
        return adapted_params
    
    def _adapt_for_bandwidth(self, params: Dict[str, float], sensory_vector: List[float]) -> Dict[str, float]:
        """Adapt parameters based on detected sensory bandwidth."""
        if self.detected_bandwidth_tier == "high":
            # High bandwidth: need more processing power
            params['activation_spread_iterations'] = min(5, params['activation_spread_iterations'] + 1)
            params['time_budget_base'] = min(0.2, params['time_budget_base'] + 0.02)
            # MEMORY FIX: Don't adapt similarity_threshold - keep it low for memory diversity
            # params['similarity_threshold'] = max(0.6, params['similarity_threshold'] - 0.05)
            
        elif self.detected_bandwidth_tier == "low":
            # Low bandwidth: can be more efficient
            params['activation_spread_iterations'] = max(2, params['activation_spread_iterations'] - 1)
            params['time_budget_base'] = max(0.05, params['time_budget_base'] - 0.01)
            # MEMORY FIX: Don't adapt similarity_threshold - keep it low for memory diversity
            # params['similarity_threshold'] = min(0.8, params['similarity_threshold'] + 0.02)
        
        return params
    
    def _adapt_for_prediction_error(self, params: Dict[str, float], prediction_error: float) -> Dict[str, float]:
        """Adapt parameters based on prediction error magnitude."""
        if prediction_error > 0.6:  # High error
            # Increase exploration and learning
            params['exploration_rate'] = min(0.7, params['exploration_rate'] + 0.1)
            params['connection_learning_rate'] = min(0.2, params['connection_learning_rate'] + 0.02)
            # MEMORY FIX: Don't adapt similarity_threshold - keep it low for memory diversity
            # params['similarity_threshold'] = max(0.5, params['similarity_threshold'] - 0.05)
            
        elif prediction_error < 0.2:  # Low error
            # Decrease exploration, increase exploitation
            params['exploration_rate'] = max(0.1, params['exploration_rate'] - 0.05)
            params['connection_learning_rate'] = max(0.05, params['connection_learning_rate'] - 0.01)
            # MEMORY FIX: Don't adapt similarity_threshold - keep it low for memory diversity
            # params['similarity_threshold'] = min(0.8, params['similarity_threshold'] + 0.02)
        
        return params
    
    def _adapt_for_sensory_characteristics(self, params: Dict[str, float]) -> Dict[str, float]:
        """Adapt parameters based on sensory dimension characteristics."""
        if not self.sensory_profiles:
            return params
        
        # Calculate average sensory activity
        avg_variance = statistics.mean([profile.variance for profile in self.sensory_profiles.values()])
        avg_change_freq = statistics.mean([profile.change_frequency for profile in self.sensory_profiles.values()])
        
        # Adapt based on sensory activity
        if avg_variance > 1.0:  # High variance sensors
            params['consolidation_frequency'] = min(100, params['consolidation_frequency'] + 10)
            params['forgetting_threshold'] = max(0.05, params['forgetting_threshold'] - 0.02)
            
        if avg_change_freq > 0.3:  # Rapidly changing sensors
            params['activation_spread_iterations'] = min(5, params['activation_spread_iterations'] + 1)
            params['time_budget_base'] = min(0.15, params['time_budget_base'] + 0.01)
        
        return params
    
    def _apply_learning_rate_damping(self, params: Dict[str, float]) -> Dict[str, float]:
        """Apply learning rate to smooth parameter changes."""
        for param_name, new_value in params.items():
            if param_name in self.current_parameters:
                current_value = self.current_parameters[param_name]
                # Smooth the change
                damped_value = current_value + (new_value - current_value) * self.learning_rate
                params[param_name] = damped_value
        
        return params
    
    def get_best_parameter_values(self) -> Dict[str, float]:
        """Get the best performing parameter values discovered so far."""
        best_params = {}
        
        for param_name, performance_list in self.parameter_history.items():
            if performance_list:
                best_performance = max(performance_list, key=lambda p: p.get_performance_score())
                best_params[param_name] = best_performance.parameter_value
        
        return best_params
    
    def get_sensory_insights(self) -> Dict[str, Any]:
        """Get insights about the sensory dimensions."""
        if not self.sensory_profiles:
            return {}
        
        insights = {
            'bandwidth_tier': self.detected_bandwidth_tier,
            'total_dimensions': len(self.sensory_profiles),
            'high_variance_dimensions': [],
            'high_change_dimensions': [],
            'stable_dimensions': [],
            'avg_variance': 0.0,
            'avg_change_frequency': 0.0
        }
        
        variances = []
        change_frequencies = []
        
        for dim_idx, profile in self.sensory_profiles.items():
            variances.append(profile.variance)
            change_frequencies.append(profile.change_frequency)
            
            if profile.variance > 1.0:
                insights['high_variance_dimensions'].append(dim_idx)
            if profile.change_frequency > 0.3:
                insights['high_change_dimensions'].append(dim_idx)
            if profile.variance < 0.1 and profile.change_frequency < 0.1:
                insights['stable_dimensions'].append(dim_idx)
        
        if variances:
            insights['avg_variance'] = statistics.mean(variances)
        if change_frequencies:
            insights['avg_change_frequency'] = statistics.mean(change_frequencies)
        
        return insights
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptation process."""
        adaptation_success_rate = 0.0
        if self.total_adaptations > 0:
            adaptation_success_rate = self.successful_adaptations / self.total_adaptations
        
        return {
            'total_adaptations': self.total_adaptations,
            'successful_adaptations': self.successful_adaptations,
            'adaptation_success_rate': adaptation_success_rate,
            'current_parameters': self.current_parameters.copy(),
            'sensory_insights': self.get_sensory_insights(),
            'parameter_performance_summary': self._get_parameter_performance_summary()
        }
    
    def _get_parameter_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of parameter performance."""
        summary = {}
        
        for param_name, performance_list in self.parameter_history.items():
            if performance_list:
                scores = [p.get_performance_score() for p in performance_list]
                summary[param_name] = {
                    'best_score': max(scores),
                    'avg_score': statistics.mean(scores),
                    'values_tested': len(performance_list),
                    'best_value': max(performance_list, key=lambda p: p.get_performance_score()).parameter_value
                }
        
        return summary
    
    def reset_adaptation_history(self):
        """Reset adaptation history (keeping current parameters)."""
        self.parameter_history.clear()
        self.sensory_profiles.clear()
        self.total_adaptations = 0
        self.successful_adaptations = 0
        self.last_adaptation_time = time.time()
        
        # Re-initialize parameter history with current values
        for param_name, value in self.current_parameters.items():
            self.parameter_history[param_name] = [
                ParameterPerformance(param_name, value)
            ]