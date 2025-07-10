"""
Auto-Tuned Energy-Based Traversal with Cognitive Parameter Optimization.
Integrates the energy traversal system with the cognitive energy tuner to automatically
find the goldilocks zone of parameters for optimal speed vs quality trade-offs.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket
from .energy_based_traversal import EnergyBasedTraversal, EnergyTraversalResult
from .cognitive_energy_tuner import CognitiveEnergyTuner, EnergyPerformanceMetrics
from core.adaptive_tuning import AdaptiveParameterTuner


@dataclass
class AutoTunedTraversalResult:
    """Result of auto-tuned energy traversal with performance metrics."""
    base_result: EnergyTraversalResult
    performance_metrics: EnergyPerformanceMetrics
    parameter_adjustments: Dict[str, float]
    tuning_triggered: bool
    goldilocks_score: float  # Overall performance score (higher is better)


class AutoTunedEnergyTraversal:
    """
    Energy-based traversal with automatic parameter tuning.
    
    This implements the "brain tuning itself" cognitive function - the system
    learns to optimize its own attention and exploration parameters based on
    performance feedback.
    """
    
    def __init__(self, 
                 # Base energy parameters (will be auto-tuned)
                 initial_energy: float = 100.0,
                 time_budget: float = 0.1,
                 interest_threshold: float = 0.6,
                 energy_boost_rate: float = 25.0,
                 energy_drain_rate: float = 15.0,
                 branching_threshold: float = 80.0,
                 minimum_energy: float = 5.0,
                 curiosity_momentum: float = 0.2,
                 randomness_factor: float = 0.3,
                 exploration_rate: float = 0.3,
                 # Auto-tuning parameters
                 enable_auto_tuning: bool = True,
                 tuning_frequency: int = 10,  # Tune every N traversals
                 performance_history_size: int = 20,
                 goldilocks_target_speed: float = 3.0,  # Target speed ratio
                 goldilocks_target_quality: float = 0.95):  # Target quality ratio
        """
        Initialize auto-tuned energy traversal system.
        
        Args:
            enable_auto_tuning: Whether to enable automatic parameter tuning
            tuning_frequency: Number of traversals between tuning attempts
            performance_history_size: Number of recent performances to consider
            goldilocks_target_speed: Target speed improvement ratio
            goldilocks_target_quality: Target quality ratio
        """
        # Initialize base energy traversal system
        self.energy_traversal = EnergyBasedTraversal(
            initial_energy=initial_energy,
            time_budget=time_budget,
            interest_threshold=interest_threshold,
            energy_boost_rate=energy_boost_rate,
            energy_drain_rate=energy_drain_rate,
            branching_threshold=branching_threshold,
            minimum_energy=minimum_energy,
            curiosity_momentum=curiosity_momentum,
            randomness_factor=randomness_factor,
            exploration_rate=exploration_rate
        )
        
        # Auto-tuning configuration
        self.enable_auto_tuning = enable_auto_tuning
        self.tuning_frequency = tuning_frequency
        self.performance_history_size = performance_history_size
        self.goldilocks_target_speed = goldilocks_target_speed
        self.goldilocks_target_quality = goldilocks_target_quality
        
        # Initialize cognitive tuner (if auto-tuning enabled)
        if self.enable_auto_tuning:
            # Create base adaptive tuner for the cognitive energy tuner
            base_adaptive_tuner = AdaptiveParameterTuner()
            self.cognitive_tuner = CognitiveEnergyTuner(base_adaptive_tuner)
        else:
            self.cognitive_tuner = None
        
        # Performance tracking
        self.performance_history: List[EnergyPerformanceMetrics] = []
        self.traversal_count = 0
        self.baseline_speed = 68.0  # From depth-limited results
        self.baseline_quality = 0.485
        
        # Goldilocks zone tracking
        self.best_goldilocks_score = 0.0
        self.best_parameters = {}
        self.goldilocks_achieved = False
    
    def traverse(self, start_context: List[float], world_graph: WorldGraph,
                random_seed: int = None, context: Dict[str, Any] = None) -> AutoTunedTraversalResult:
        """
        Perform auto-tuned energy traversal with parameter optimization.
        
        Args:
            start_context: Current mental context to start traversal from
            world_graph: The experience graph to traverse  
            random_seed: Random seed for reproducible traversals
            context: Additional context for tuning decisions
            
        Returns:
            AutoTunedTraversalResult with performance metrics and tuning info
        """
        self.traversal_count += 1
        tuning_triggered = False
        parameter_adjustments = {}
        
        # Perform base energy traversal
        start_time = time.time()
        base_result = self.energy_traversal.traverse(start_context, world_graph, random_seed)
        traversal_duration = time.time() - start_time
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            base_result, traversal_duration, context
        )
        
        # Add to performance history
        self.performance_history.append(performance_metrics)
        if len(self.performance_history) > self.performance_history_size:
            self.performance_history.pop(0)
        
        # Check if we should tune parameters
        if (self.enable_auto_tuning and 
            self.cognitive_tuner and
            self.traversal_count % self.tuning_frequency == 0):
            
            tuning_triggered = self._should_tune_parameters(performance_metrics, context)
            
            if tuning_triggered:
                parameter_adjustments = self._tune_parameters(context)
                self._apply_parameter_adjustments(parameter_adjustments)
        
        # Calculate goldilocks score
        goldilocks_score = self._calculate_goldilocks_score(performance_metrics)
        
        # Update best parameters if this is the best performance
        if goldilocks_score > self.best_goldilocks_score:
            self.best_goldilocks_score = goldilocks_score
            self.best_parameters = self._get_current_parameters()
            self.goldilocks_achieved = (
                performance_metrics.speed_ratio >= self.goldilocks_target_speed and
                performance_metrics.quality_ratio >= self.goldilocks_target_quality
            )
        
        return AutoTunedTraversalResult(
            base_result=base_result,
            performance_metrics=performance_metrics,
            parameter_adjustments=parameter_adjustments,
            tuning_triggered=tuning_triggered,
            goldilocks_score=goldilocks_score
        )
    
    def _calculate_performance_metrics(self, result: EnergyTraversalResult,
                                     duration: float, context: Dict[str, Any] = None) -> EnergyPerformanceMetrics:
        """Calculate comprehensive performance metrics for the traversal."""
        # Speed ratio compared to baseline
        searches_per_second = 1.0 / duration if duration > 0 else 0.0
        speed_ratio = searches_per_second / self.baseline_speed
        
        # Quality ratio (prediction quality compared to baseline)
        prediction_quality = self._evaluate_prediction_quality(result.prediction)
        quality_ratio = prediction_quality / self.baseline_quality
        
        # Exploration efficiency (quality per step taken)
        exploration_efficiency = prediction_quality / max(1, result.steps_taken)
        
        # Termination naturalness (natural stopping is better)
        termination_naturalness = 1.0 if result.termination_reason == "natural_stopping" else 0.5
        
        # Resource utilization (how much energy was used productively)
        resource_utilization = 1.0 - (result.final_energy / self.energy_traversal.initial_energy)
        
        return EnergyPerformanceMetrics(
            speed_ratio=speed_ratio,
            quality_ratio=quality_ratio,
            exploration_efficiency=exploration_efficiency,
            termination_naturalness=termination_naturalness,
            resource_utilization=resource_utilization
        )
    
    def _evaluate_prediction_quality(self, prediction) -> float:
        """Evaluate the quality of a prediction (0.0 to 1.0)."""
        if not prediction:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Confidence level
        confidence = getattr(prediction, 'confidence_level', 0.5)
        quality_factors.append(confidence * 0.4)
        
        # Factor 2: Thinking depth
        thinking_depth = getattr(prediction, 'thinking_depth', 1)
        depth_factor = min(1.0, thinking_depth / 10.0)
        quality_factors.append(depth_factor * 0.3)
        
        # Factor 3: Motor action coherence
        motor_action = prediction.motor_action
        action_magnitude = (abs(motor_action.get('forward_motor', 0.0)) + 
                          abs(motor_action.get('turn_motor', 0.0)) +
                          abs(motor_action.get('brake_motor', 0.0)))
        action_factor = min(1.0, action_magnitude)
        quality_factors.append(action_factor * 0.3)
        
        return sum(quality_factors)
    
    def _should_tune_parameters(self, current_performance: EnergyPerformanceMetrics,
                              context: Dict[str, Any] = None) -> bool:
        """Determine if parameters should be tuned."""
        if not self.cognitive_tuner:
            return False
        
        return self.cognitive_tuner.should_tune_parameters(current_performance, context)
    
    def _tune_parameters(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Perform parameter tuning and return adjustments made."""
        if not self.cognitive_tuner:
            return {}
        
        # Get recent performance for tuning
        recent_performance = self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history
        
        # Perform cognitive tuning
        tuned_energy_params = self.cognitive_tuner.tune_energy_parameters(recent_performance, context)
        
        # Calculate adjustments made
        current_params = self._get_current_parameters()
        adjustments = {}
        for param_name, new_value in tuned_energy_params.items():
            if param_name in current_params:
                old_value = current_params[param_name]
                if abs(new_value - old_value) > 0.001:  # Only track significant changes
                    adjustments[param_name] = new_value - old_value
        
        return adjustments
    
    def _apply_parameter_adjustments(self, adjustments: Dict[str, float]):
        """Apply parameter adjustments to the energy traversal system."""
        if not adjustments:
            return
        
        # Update energy traversal parameters
        for param_name, adjustment in adjustments.items():
            if hasattr(self.energy_traversal, param_name):
                current_value = getattr(self.energy_traversal, param_name)
                new_value = current_value + adjustment
                # Apply bounds checking
                new_value = self._apply_parameter_bounds(param_name, new_value)
                setattr(self.energy_traversal, param_name, new_value)
        
        # Also update cognitive tuner's parameter tracking
        if self.cognitive_tuner:
            for param_name, adjustment in adjustments.items():
                if param_name in self.cognitive_tuner.energy_parameters:
                    self.cognitive_tuner.energy_parameters[param_name] += adjustment
                    self.cognitive_tuner.energy_parameters[param_name] = self._apply_parameter_bounds(
                        param_name, self.cognitive_tuner.energy_parameters[param_name]
                    )
    
    def _apply_parameter_bounds(self, param_name: str, value: float) -> float:
        """Apply reasonable bounds to parameter values."""
        bounds = {
            'initial_energy': (10.0, 500.0),
            'energy_boost_rate': (5.0, 100.0),
            'energy_drain_rate': (1.0, 50.0),
            'interest_threshold': (0.1, 0.9),
            'branching_threshold': (20.0, 200.0),
            'minimum_energy': (1.0, 20.0),
            'curiosity_momentum': (0.0, 1.0),
            'randomness_factor': (0.0, 1.0)
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            return max(min_val, min(max_val, value))
        
        return value
    
    def _calculate_goldilocks_score(self, performance: EnergyPerformanceMetrics) -> float:
        """
        Calculate how close we are to the goldilocks zone.
        Score of 1.0 means perfect goldilocks performance.
        """
        # Speed component (target: goldilocks_target_speed, max benefit at 2x target)
        speed_score = min(1.0, performance.speed_ratio / self.goldilocks_target_speed)
        
        # Quality component (target: goldilocks_target_quality)  
        quality_score = min(1.0, performance.quality_ratio / self.goldilocks_target_quality)
        
        # Efficiency component
        efficiency_score = min(1.0, performance.exploration_efficiency)
        
        # Naturalness component (natural termination is good)
        naturalness_score = performance.termination_naturalness
        
        # Weighted composite score
        composite_score = (
            speed_score * 0.3 +
            quality_score * 0.4 +
            efficiency_score * 0.2 +
            naturalness_score * 0.1
        )
        
        # Penalty if we're way over target speed (diminishing returns)
        if performance.speed_ratio > self.goldilocks_target_speed * 2:
            speed_penalty = min(0.2, (performance.speed_ratio - self.goldilocks_target_speed * 2) * 0.05)
            composite_score -= speed_penalty
        
        return max(0.0, min(1.0, composite_score))
    
    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {
            'initial_energy': self.energy_traversal.initial_energy,
            'energy_boost_rate': self.energy_traversal.energy_boost_rate,
            'energy_drain_rate': self.energy_traversal.energy_drain_rate,
            'interest_threshold': self.energy_traversal.interest_threshold,
            'branching_threshold': self.energy_traversal.branching_threshold,
            'minimum_energy': self.energy_traversal.minimum_energy,
            'curiosity_momentum': self.energy_traversal.curiosity_momentum,
            'randomness_factor': self.energy_traversal.randomness_factor
        }
    
    def get_tuning_status(self) -> Dict[str, Any]:
        """Get comprehensive tuning status and performance summary."""
        status = {
            'traversal_count': self.traversal_count,
            'auto_tuning_enabled': self.enable_auto_tuning,
            'performance_samples': len(self.performance_history),
            'goldilocks_achieved': self.goldilocks_achieved,
            'best_goldilocks_score': self.best_goldilocks_score,
            'current_parameters': self._get_current_parameters(),
            'best_parameters': self.best_parameters
        }
        
        if self.cognitive_tuner:
            tuner_status = self.cognitive_tuner.get_tuning_status()
            status.update({'cognitive_tuner_status': tuner_status})
        
        if self.performance_history:
            recent_performance = self.performance_history[-1]
            status.update({
                'recent_performance': {
                    'speed_ratio': recent_performance.speed_ratio,
                    'quality_ratio': recent_performance.quality_ratio,
                    'exploration_efficiency': recent_performance.exploration_efficiency,
                    'termination_naturalness': recent_performance.termination_naturalness,
                    'composite_score': recent_performance.get_composite_score()
                }
            })
        
        return status
    
    def force_parameter_reset_to_best(self):
        """Reset parameters to the best performing configuration found so far."""
        if self.best_parameters:
            for param_name, value in self.best_parameters.items():
                if hasattr(self.energy_traversal, param_name):
                    setattr(self.energy_traversal, param_name, value)
            
            if self.cognitive_tuner:
                self.cognitive_tuner.energy_parameters.update(self.best_parameters)
    
    def get_goldilocks_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of goldilocks zone progress."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_performances = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        # Calculate trends
        speed_trend = self._calculate_trend([p.speed_ratio for p in recent_performances])
        quality_trend = self._calculate_trend([p.quality_ratio for p in recent_performances])
        efficiency_trend = self._calculate_trend([p.exploration_efficiency for p in recent_performances])
        
        # Current averages
        avg_speed = sum(p.speed_ratio for p in recent_performances) / len(recent_performances)
        avg_quality = sum(p.quality_ratio for p in recent_performances) / len(recent_performances)
        avg_efficiency = sum(p.exploration_efficiency for p in recent_performances) / len(recent_performances)
        
        # Distance to goldilocks zone
        speed_distance = abs(avg_speed - self.goldilocks_target_speed)
        quality_distance = abs(avg_quality - self.goldilocks_target_quality)
        
        return {
            "status": "goldilocks_achieved" if self.goldilocks_achieved else "searching",
            "current_averages": {
                "speed_ratio": avg_speed,
                "quality_ratio": avg_quality,
                "exploration_efficiency": avg_efficiency
            },
            "targets": {
                "speed_ratio": self.goldilocks_target_speed,
                "quality_ratio": self.goldilocks_target_quality
            },
            "distances_to_target": {
                "speed_distance": speed_distance,
                "quality_distance": quality_distance
            },
            "trends": {
                "speed_trend": speed_trend,
                "quality_trend": quality_trend,
                "efficiency_trend": efficiency_trend
            },
            "best_score_achieved": self.best_goldilocks_score,
            "performance_samples": len(self.performance_history)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, negative = declining, positive = improving)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        trend = numerator / denominator
        return max(-1.0, min(1.0, trend))  # Normalize to -1 to 1 range