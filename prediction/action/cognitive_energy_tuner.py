"""
Cognitive Energy Parameter Tuner - Meta-cognitive system that auto-tunes energy traversal parameters.
Extends the existing adaptive tuning system to optimize the energy-based traversal goldilocks zone.
"""

import time
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from core.adaptive_tuning import AdaptiveParameterTuner, ParameterPerformance


@dataclass
class EnergyPerformanceMetrics:
    """Performance metrics specific to energy-based traversal."""
    speed_ratio: float          # Searches per second compared to baseline
    quality_ratio: float        # Prediction quality compared to baseline  
    exploration_efficiency: float  # Quality per step taken
    termination_naturalness: float # Rate of natural vs forced termination
    resource_utilization: float    # Final energy / initial energy
    
    def get_composite_score(self, speed_weight=0.3, quality_weight=0.5, 
                          efficiency_weight=0.2) -> float:
        """Calculate weighted composite performance score."""
        return (speed_weight * min(self.speed_ratio / 5.0, 1.0) +  # Cap speed benefit
                quality_weight * self.quality_ratio +
                efficiency_weight * self.exploration_efficiency)


class CognitiveEnergyTuner:
    """
    Meta-cognitive system that learns to optimize energy traversal parameters.
    
    Key insight: The brain tuning its own attention and exploration parameters
    is itself a cognitive function - this is "thinking about thinking."
    """
    
    def __init__(self, base_adaptive_tuner: AdaptiveParameterTuner):
        self.base_tuner = base_adaptive_tuner
        
        # Energy-specific parameters to tune
        self.energy_parameters = {
            'initial_energy': 100.0,
            'energy_boost_rate': 25.0,
            'energy_drain_rate': 15.0,
            'interest_threshold': 0.6,
            'branching_threshold': 80.0,
            'minimum_energy': 5.0,
            'curiosity_momentum': 0.2
        }
        
        # Performance tracking
        self.performance_history: List[EnergyPerformanceMetrics] = []
        self.parameter_experiments: Dict[str, List[Tuple[float, float]]] = {}  # param -> [(value, score)]
        
        # Learning state
        self.exploration_phase = True
        self.exploitation_countdown = 0
        self.last_tuning_time = 0.0
        self.tuning_interval = 10.0  # Seconds between tuning attempts
        
        # Baseline performance (depth-limited search)
        self.baseline_speed = 65.0  # searches/second
        self.baseline_quality = 0.485
        
        # Cognitive load management
        self.cognitive_load_threshold = 0.8  # When to prioritize speed over quality
        self.current_cognitive_load = 0.5
    
    def should_tune_parameters(self, current_performance: EnergyPerformanceMetrics,
                             context: Dict[str, any] = None) -> bool:
        """
        Determine if energy parameters should be tuned.
        This is the meta-cognitive decision: "Should I think about how I think?"
        """
        # Don't tune too frequently (cognitive resource management)
        if time.time() - self.last_tuning_time < self.tuning_interval:
            return False
        
        # Always tune if performance is very poor
        composite_score = current_performance.get_composite_score()
        if composite_score < 0.6:
            return True
        
        # Context-aware tuning decisions
        if context:
            # High threat situations: prioritize speed
            threat_level = context.get('threat_level', 'normal')
            if threat_level in ['danger', 'critical']:
                return current_performance.speed_ratio < 3.0
            
            # Safe situations: optimize for quality
            elif threat_level == 'safe':
                return current_performance.quality_ratio < 0.95
            
            # Cognitive load considerations
            self.current_cognitive_load = context.get('cognitive_load', 0.5)
            if self.current_cognitive_load > self.cognitive_load_threshold:
                return current_performance.speed_ratio < 2.0
        
        # Periodic exploration (curiosity about our own cognition)
        import random
        exploration_probability = 0.1 if self.exploitation_countdown > 0 else 0.3
        return random.random() < exploration_probability
    
    def tune_energy_parameters(self, recent_performance: List[EnergyPerformanceMetrics],
                             context: Dict[str, any] = None) -> Dict[str, float]:
        """
        Adapt energy parameters based on recent performance.
        This implements the "thinking about thinking" cognitive function.
        """
        if not recent_performance:
            return self.energy_parameters.copy()
        
        # Calculate performance trends
        avg_performance = self._calculate_average_performance(recent_performance)
        performance_trend = self._calculate_performance_trend(recent_performance)
        
        # Multi-objective optimization strategy
        tuned_params = self.energy_parameters.copy()
        
        # 1. Speed vs Quality Trade-off Optimization
        if avg_performance.quality_ratio < 0.90 and avg_performance.speed_ratio > 5.0:
            # Too fast, not quality enough - slow down exploration
            tuned_params = self._adjust_for_quality_improvement(tuned_params)
        
        elif avg_performance.quality_ratio > 0.95 and avg_performance.speed_ratio < 2.0:
            # High quality but too slow - speed up exploration
            tuned_params = self._adjust_for_speed_improvement(tuned_params)
        
        # 2. Context-Aware Adaptations
        if context:
            tuned_params = self._adapt_for_context(tuned_params, context, avg_performance)
        
        # 3. Exploration vs Exploitation Meta-Strategy
        if self.exploration_phase:
            tuned_params = self._apply_exploration_strategy(tuned_params, avg_performance)
        else:
            tuned_params = self._apply_exploitation_strategy(tuned_params, avg_performance)
        
        # 4. Curiosity-Driven Parameter Discovery
        tuned_params = self._apply_curiosity_driven_tuning(tuned_params, recent_performance)
        
        # Update internal state
        self._update_tuning_state(avg_performance)
        self.energy_parameters = tuned_params
        self.last_tuning_time = time.time()
        
        return tuned_params
    
    def _adjust_for_quality_improvement(self, params: Dict[str, float]) -> Dict[str, float]:
        """Adjust parameters to improve prediction quality (allow more exploration)."""
        # Reduce energy drain to allow deeper exploration
        params['energy_drain_rate'] = max(5.0, params['energy_drain_rate'] - 2.0)
        
        # Lower interest threshold to find more interesting content
        params['interest_threshold'] = max(0.3, params['interest_threshold'] - 0.1)
        
        # Increase energy boost to reward good finds more
        params['energy_boost_rate'] = min(50.0, params['energy_boost_rate'] + 5.0)
        
        # Increase initial energy budget
        params['initial_energy'] = min(150.0, params['initial_energy'] + 10.0)
        
        return params
    
    def _adjust_for_speed_improvement(self, params: Dict[str, float]) -> Dict[str, float]:
        """Adjust parameters to improve search speed (terminate earlier)."""
        # Increase energy drain for faster termination
        params['energy_drain_rate'] = min(25.0, params['energy_drain_rate'] + 2.0)
        
        # Raise interest threshold to be more selective
        params['interest_threshold'] = min(0.8, params['interest_threshold'] + 0.1)
        
        # Reduce initial energy budget
        params['initial_energy'] = max(50.0, params['initial_energy'] - 10.0)
        
        return params
    
    def _adapt_for_context(self, params: Dict[str, float], context: Dict[str, any],
                          performance: EnergyPerformanceMetrics) -> Dict[str, float]:
        """Context-aware parameter adaptation."""
        threat_level = context.get('threat_level', 'normal')
        cognitive_load = context.get('cognitive_load', 0.5)
        
        # High threat: prioritize speed and decisiveness
        if threat_level in ['danger', 'critical']:
            params['energy_drain_rate'] = min(30.0, params['energy_drain_rate'] + 5.0)
            params['minimum_energy'] = max(10.0, params['minimum_energy'] + 2.0)
            params['interest_threshold'] = min(0.9, params['interest_threshold'] + 0.2)
        
        # Safe environment: optimize for learning and quality
        elif threat_level == 'safe':
            params['energy_drain_rate'] = max(3.0, params['energy_drain_rate'] - 3.0)
            params['interest_threshold'] = max(0.2, params['interest_threshold'] - 0.2)
            params['branching_threshold'] = max(40.0, params['branching_threshold'] - 10.0)
        
        # High cognitive load: reduce computational complexity
        if cognitive_load > self.cognitive_load_threshold:
            params['initial_energy'] = max(30.0, params['initial_energy'] - 20.0)
            params['branching_threshold'] = min(100.0, params['branching_threshold'] + 20.0)
        
        return params
    
    def _apply_curiosity_driven_tuning(self, params: Dict[str, float],
                                     recent_performance: List[EnergyPerformanceMetrics]) -> Dict[str, float]:
        """
        Apply curiosity-driven parameter exploration.
        The brain being curious about its own cognitive parameters!
        """
        if len(recent_performance) < 3:
            return params
        
        # Detect if we're in a performance plateau
        recent_scores = [p.get_composite_score() for p in recent_performance[-5:]]
        if len(recent_scores) >= 3:
            score_variance = statistics.variance(recent_scores)
            if score_variance < 0.01:  # Very stable performance
                # Try a random parameter perturbation (cognitive curiosity)
                import random
                param_to_explore = random.choice(list(params.keys()))
                current_value = params[param_to_explore]
                
                # Small random perturbation (±10%)
                perturbation = current_value * (random.random() - 0.5) * 0.2
                params[param_to_explore] = max(0.1, current_value + perturbation)
        
        return params
    
    def _calculate_average_performance(self, recent_performance: List[EnergyPerformanceMetrics]) -> EnergyPerformanceMetrics:
        """Calculate average performance metrics."""
        if not recent_performance:
            return EnergyPerformanceMetrics(1.0, 1.0, 1.0, 1.0, 1.0)
        
        return EnergyPerformanceMetrics(
            speed_ratio=statistics.mean([p.speed_ratio for p in recent_performance]),
            quality_ratio=statistics.mean([p.quality_ratio for p in recent_performance]),
            exploration_efficiency=statistics.mean([p.exploration_efficiency for p in recent_performance]),
            termination_naturalness=statistics.mean([p.termination_naturalness for p in recent_performance]),
            resource_utilization=statistics.mean([p.resource_utilization for p in recent_performance])
        )
    
    def _calculate_performance_trend(self, recent_performance: List[EnergyPerformanceMetrics]) -> float:
        """Calculate if performance is improving or declining."""
        if len(recent_performance) < 2:
            return 0.0
        
        scores = [p.get_composite_score() for p in recent_performance]
        
        # Simple linear trend
        n = len(scores)
        if n < 2:
            return 0.0
        
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(scores)
        
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _apply_exploration_strategy(self, params: Dict[str, float],
                                  performance: EnergyPerformanceMetrics) -> Dict[str, float]:
        """Apply exploration strategy - try different parameter combinations."""
        # During exploration, make bolder parameter adjustments
        import random
        
        if performance.get_composite_score() > 0.8:
            # Good performance, but explore nearby space
            for param_name in params:
                if random.random() < 0.3:  # 30% chance to adjust each parameter
                    current_value = params[param_name]
                    adjustment = current_value * (random.random() - 0.5) * 0.1  # ±5%
                    params[param_name] = max(0.1, current_value + adjustment)
        
        return params
    
    def _apply_exploitation_strategy(self, params: Dict[str, float],
                                   performance: EnergyPerformanceMetrics) -> Dict[str, float]:
        """Apply exploitation strategy - refine known good parameters."""
        # During exploitation, make smaller, more conservative adjustments
        # based on known performance patterns
        
        # Look for the best performing parameter combinations from history
        if len(self.performance_history) > 10:
            best_performances = sorted(self.performance_history, 
                                     key=lambda p: p.get_composite_score(), 
                                     reverse=True)[:3]
            
            # Gently move toward best performing configurations
            # (This would require storing parameter values with performance)
        
        return params
    
    def _update_tuning_state(self, performance: EnergyPerformanceMetrics):
        """Update internal tuning state and strategy."""
        self.performance_history.append(performance)
        
        # Keep history manageable
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # Switch between exploration and exploitation phases
        if self.exploration_phase:
            if len(self.performance_history) > 10:
                recent_improvement = self._calculate_performance_trend(self.performance_history[-10:])
                if recent_improvement < 0.001:  # Very little improvement
                    self.exploration_phase = False
                    self.exploitation_countdown = 20  # Exploit for 20 cycles
        else:
            self.exploitation_countdown -= 1
            if self.exploitation_countdown <= 0:
                self.exploration_phase = True
    
    def get_tuning_status(self) -> Dict[str, any]:
        """Get current tuning status for monitoring."""
        return {
            "exploration_phase": self.exploration_phase,
            "exploitation_countdown": self.exploitation_countdown,
            "performance_samples": len(self.performance_history),
            "current_cognitive_load": self.current_cognitive_load,
            "last_tuning_time": self.last_tuning_time,
            "avg_recent_performance": self._calculate_average_performance(
                self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history
            ).get_composite_score() if self.performance_history else 0.0
        }