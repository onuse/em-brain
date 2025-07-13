"""
Adaptive Bounds System (Phase 2 Lite)

Makes parameter bounds adaptive rather than the parameters themselves.
This preserves system stability while eliminating hardcoded limits.

Strategy: Instead of fixed bounds like (0.001, 0.5), use bounds that adapt
based on system performance and learning context.

Safety: Parameters still have bounds, but the bounds themselves are smart.
"""

import numpy as np
from typing import Dict, Any, Tuple
from collections import deque
import time

from ..cognitive_constants import StabilityConstants


class AdaptiveBounds:
    """
    Manages adaptive parameter bounds that respond to system performance.
    
    Key insight: We don't need to make parameters completely unbounded,
    we just need the bounds to be intelligent rather than arbitrary.
    """
    
    def __init__(self):
        """Initialize adaptive bounds system."""
        
        # Performance tracking for bound adaptation
        self.performance_history = deque(maxlen=50)
        self.stability_history = deque(maxlen=20)
        self.learning_effectiveness = deque(maxlen=30)
        
        # Current adaptive bounds (start with safe defaults)
        self.current_bounds = {
            'learning_rate': (0.001, 0.1),
            'decay_rate': (0.005, 0.05),
            'activation_threshold': (0.05, 0.8),
            'similarity_learning_rate': (0.005, 0.05)
        }
        
        # Bound adaptation rates (how fast bounds can change)
        self.bound_adaptation_rate = 0.002  # Biologically plausible (was 0.02)
        self.min_bound_separation = 0.001  # Bounds can't get too close
        
        # Performance metrics for adaptation
        self.last_performance_check = time.time()
        self.bound_adaptations = 0
        
        print("📊 AdaptiveBounds initialized - intelligent parameter limits")
    
    def get_learning_rate_bounds(self, 
                                learning_context: Dict[str, Any],
                                recent_performance: float) -> Tuple[float, float]:
        """
        Get adaptive bounds for learning rates based on learning context.
        
        Args:
            learning_context: Context about current learning (success rate, stability, etc.)
            recent_performance: Recent system performance (0.0-1.0)
            
        Returns:
            Tuple of (min_learning_rate, max_learning_rate)
        """
        # Track performance for bound adaptation
        self.performance_history.append(recent_performance)
        
        current_min, current_max = self.current_bounds['learning_rate']
        
        # Adapt bounds based on performance patterns
        if len(self.performance_history) >= 10:
            performance_trend = self._calculate_performance_trend()
            stability_score = self._calculate_stability_score(learning_context)
            
            # High performance + stable = can afford wider bounds (more exploration)
            if recent_performance > 0.8 and stability_score > 0.7:
                new_min = max(current_min * 0.95, StabilityConstants.MIN_LEARNING_RATE)
                new_max = min(current_max * 1.05, StabilityConstants.MAX_LEARNING_RATE)
            
            # Poor performance + unstable = narrow bounds (more conservative)
            elif recent_performance < 0.3 or stability_score < 0.3:
                new_min = min(current_min * 1.05, 0.01)  # Raise minimum (less aggressive)
                new_max = max(current_max * 0.95, 0.02)  # Lower maximum (more conservative)
            
            # Medium performance = gradual adaptation toward optimal
            else:
                target_center = 0.02  # Empirically good learning rate
                center_adjustment = (target_center - (current_min + current_max) / 2) * 0.1
                new_min = max(current_min + center_adjustment * 0.5, StabilityConstants.MIN_LEARNING_RATE)
                new_max = min(current_max + center_adjustment * 0.5, StabilityConstants.MAX_LEARNING_RATE)
            
            # Ensure minimum separation
            if new_max - new_min < self.min_bound_separation:
                center = (new_min + new_max) / 2
                new_min = center - self.min_bound_separation / 2
                new_max = center + self.min_bound_separation / 2
            
            # Update bounds gradually (stability)
            adaptation_rate = self.bound_adaptation_rate
            self.current_bounds['learning_rate'] = (
                current_min + (new_min - current_min) * adaptation_rate,
                current_max + (new_max - current_max) * adaptation_rate
            )
            
            self.bound_adaptations += 1
        
        return self.current_bounds['learning_rate']
    
    def get_decay_rate_bounds(self,
                             memory_pressure: float,
                             working_memory_effectiveness: float) -> Tuple[float, float]:
        """
        Get adaptive bounds for decay rates based on memory dynamics.
        
        Args:
            memory_pressure: How full memory is (0.0-1.0)
            working_memory_effectiveness: How well working memory is performing (0.0-1.0)
            
        Returns:
            Tuple of (min_decay_rate, max_decay_rate)
        """
        current_min, current_max = self.current_bounds['decay_rate']
        
        # High memory pressure = allow faster decay (need to free memory)
        if memory_pressure > 0.8:
            new_min = current_min  # Keep minimum unchanged
            new_max = min(current_max * 1.1, 0.1)  # Allow faster decay
        
        # Low memory pressure + effective working memory = allow slower decay
        elif memory_pressure < 0.3 and working_memory_effectiveness > 0.7:
            new_min = max(current_min * 0.9, 0.001)  # Allow slower decay
            new_max = current_max  # Keep maximum unchanged
        
        # Medium conditions = gradual return to center
        else:
            target_center = 0.02  # Good default decay rate
            current_center = (current_min + current_max) / 2
            center_adjustment = (target_center - current_center) * 0.1
            new_min = max(current_min + center_adjustment * 0.5, 0.001)
            new_max = min(current_max + center_adjustment * 0.5, 0.1)
        
        # Apply gradual adaptation
        adaptation_rate = self.bound_adaptation_rate
        self.current_bounds['decay_rate'] = (
            current_min + (new_min - current_min) * adaptation_rate,
            current_max + (new_max - current_max) * adaptation_rate
        )
        
        return self.current_bounds['decay_rate']
    
    def get_similarity_learning_bounds(self,
                                     similarity_effectiveness: float,
                                     adaptation_frequency: float) -> Tuple[float, float]:
        """
        Get adaptive bounds for similarity learning rates.
        
        Args:
            similarity_effectiveness: How well similarity adaptations are working
            adaptation_frequency: How often similarity is being adapted
            
        Returns:
            Tuple of (min_similarity_learning_rate, max_similarity_learning_rate)
        """
        current_min, current_max = self.current_bounds['similarity_learning_rate']
        
        # High effectiveness = allow faster learning
        if similarity_effectiveness > 0.7:
            new_max = min(current_max * 1.05, 0.1)
            new_min = current_min
        
        # Low effectiveness = be more conservative
        elif similarity_effectiveness < 0.3:
            new_max = max(current_max * 0.95, 0.01)
            new_min = max(current_min * 1.02, 0.001)
        
        # Too frequent adaptation = slow down
        elif adaptation_frequency > 0.5:  # More than every other cycle
            new_max = max(current_max * 0.98, 0.01)
            new_min = current_min
        
        else:
            new_min = current_min
            new_max = current_max
        
        # Apply gradual adaptation
        adaptation_rate = self.bound_adaptation_rate
        self.current_bounds['similarity_learning_rate'] = (
            current_min + (new_min - current_min) * adaptation_rate,
            current_max + (new_max - current_max) * adaptation_rate
        )
        
        return self.current_bounds['similarity_learning_rate']
    
    def _calculate_performance_trend(self) -> float:
        """Calculate whether performance is improving (+) or declining (-)."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_values = list(self.performance_history)
        first_half = np.mean(recent_values[:len(recent_values)//2])
        second_half = np.mean(recent_values[len(recent_values)//2:])
        
        return second_half - first_half
    
    def _calculate_stability_score(self, learning_context: Dict[str, Any]) -> float:
        """Calculate how stable the learning process is (0.0-1.0)."""
        
        # Check performance variance (lower variance = more stable)
        if len(self.performance_history) >= 5:
            performance_variance = np.var(list(self.performance_history)[-10:])
            variance_stability = max(0.0, 1.0 - performance_variance * 5)  # Scale to 0-1
        else:
            variance_stability = 0.5
        
        # Check learning context stability
        context_stability = learning_context.get('stability_score', 0.5)
        
        # Combine measures
        overall_stability = (variance_stability + context_stability) / 2
        
        self.stability_history.append(overall_stability)
        return overall_stability
    
    def get_all_current_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get all current adaptive bounds."""
        return self.current_bounds.copy()
    
    def get_bound_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about bound adaptations."""
        
        stats = {
            'total_adaptations': self.bound_adaptations,
            'current_bounds': self.current_bounds.copy(),
            'performance_trend': self._calculate_performance_trend() if len(self.performance_history) >= 10 else 0.0,
            'recent_stability': np.mean(list(self.stability_history)[-5:]) if len(self.stability_history) >= 5 else 0.5,
            'adaptation_rate': self.bound_adaptation_rate
        }
        
        # Calculate bound ranges (how wide bounds are)
        for param_name, (min_val, max_val) in self.current_bounds.items():
            stats[f'{param_name}_range'] = max_val - min_val
            stats[f'{param_name}_center'] = (max_val + min_val) / 2
        
        return stats
    
    def reset_bounds(self):
        """Reset all bounds to safe defaults."""
        self.current_bounds = {
            'learning_rate': (0.001, 0.1),
            'decay_rate': (0.005, 0.05),
            'activation_threshold': (0.05, 0.8),
            'similarity_learning_rate': (0.005, 0.05)
        }
        
        self.performance_history.clear()
        self.stability_history.clear()
        self.learning_effectiveness.clear()
        self.bound_adaptations = 0
        
        print("🔄 Adaptive bounds reset to safe defaults")


class PerformanceMonitor:
    """
    Monitors system performance to ensure adaptations don't hurt speed.
    
    If adaptive parameters start hurting performance, this can trigger
    rollback to safer settings.
    """
    
    def __init__(self):
        """Initialize performance monitoring."""
        
        # Performance baselines
        self.baseline_cycle_time = None
        self.baseline_accuracy = None
        self.baseline_memory_usage = None
        
        # Current performance tracking
        self.recent_cycle_times = deque(maxlen=100)
        self.recent_accuracies = deque(maxlen=50)
        self.recent_memory_usage = deque(maxlen=30)
        
        # Performance degradation detection
        self.performance_degradation_threshold = 1.0  # Biologically plausible (100% vs 20%)
        self.consecutive_bad_cycles = 0
        self.max_bad_cycles = 50  # Biologically plausible patience (50 vs 10)
        
        print("📈 PerformanceMonitor initialized - watching for degradation")
    
    def record_performance(self, 
                          cycle_time: float,
                          accuracy: float,
                          memory_usage: float):
        """Record current performance metrics."""
        
        self.recent_cycle_times.append(cycle_time)
        self.recent_accuracies.append(accuracy)
        self.recent_memory_usage.append(memory_usage)
        
        # Set baselines if not established
        if self.baseline_cycle_time is None and len(self.recent_cycle_times) >= 20:
            self.baseline_cycle_time = np.mean(list(self.recent_cycle_times)[-20:])
            self.baseline_accuracy = np.mean(list(self.recent_accuracies)[-20:])
            self.baseline_memory_usage = np.mean(list(self.recent_memory_usage)[-20:])
            print(f"📊 Performance baselines established: {self.baseline_cycle_time:.3f}s cycle, {self.baseline_accuracy:.3f} accuracy")
    
    def check_performance_degradation(self) -> bool:
        """
        Check if performance has degraded significantly.
        
        Returns:
            True if performance degradation detected
        """
        if self.baseline_cycle_time is None or len(self.recent_cycle_times) < 10:
            return False
        
        # Check recent performance vs baseline
        recent_cycle_time = np.mean(list(self.recent_cycle_times)[-10:])
        recent_accuracy = np.mean(list(self.recent_accuracies)[-10:])
        
        # Performance degradation conditions
        cycle_degradation = (recent_cycle_time - self.baseline_cycle_time) / self.baseline_cycle_time
        accuracy_degradation = (self.baseline_accuracy - recent_accuracy) / max(0.1, self.baseline_accuracy)
        
        degraded = (cycle_degradation > self.performance_degradation_threshold or 
                   accuracy_degradation > self.performance_degradation_threshold)
        
        if degraded:
            self.consecutive_bad_cycles += 1
        else:
            self.consecutive_bad_cycles = 0
        
        # Trigger rollback if consistent degradation
        if self.consecutive_bad_cycles >= self.max_bad_cycles:
            print(f"⚠️  Performance degradation detected: cycle_time +{cycle_degradation:.1%}, accuracy -{accuracy_degradation:.1%}")
            return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.recent_cycle_times:
            return {}
        
        return {
            'current_cycle_time': np.mean(list(self.recent_cycle_times)[-5:]),
            'current_accuracy': np.mean(list(self.recent_accuracies)[-5:]),
            'baseline_cycle_time': self.baseline_cycle_time,
            'baseline_accuracy': self.baseline_accuracy,
            'performance_stable': self.consecutive_bad_cycles == 0,
            'consecutive_bad_cycles': self.consecutive_bad_cycles
        }