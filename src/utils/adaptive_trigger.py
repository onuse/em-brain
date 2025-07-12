"""
Event-Driven Adaptation Triggers

Replaces arbitrary "adapt every N experiences" with natural adaptation triggers.
The system adapts when information events indicate it should, not on fixed schedules.

Key principle: Adaptation emerges from prediction error patterns, not engineering convenience.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import deque
import time


class AdaptiveTrigger:
    """
    Detects when the system naturally needs to adapt based on information events.
    
    Instead of fixed schedules, adaptation is triggered by:
    - Sudden changes in prediction error gradients
    - Sustained poor performance periods  
    - Information surprise events
    - Performance plateau detection
    """
    
    def __init__(self, 
                 min_experiences_between_adaptations: int = 5,
                 gradient_change_threshold: float = 0.3,
                 plateau_detection_window: int = 20,
                 surprise_threshold: float = 0.7):
        """
        Initialize adaptive trigger system.
        
        Args:
            min_experiences_between_adaptations: Minimum gap to prevent thrashing
            gradient_change_threshold: How much gradient change triggers adaptation
            plateau_detection_window: Window size for plateau detection
            surprise_threshold: Prediction error level that counts as "surprising"
        """
        self.min_experiences_between_adaptations = min_experiences_between_adaptations
        
        # Strategy 5: Make thresholds adaptive
        self.initial_gradient_change_threshold = gradient_change_threshold
        self.gradient_change_threshold = gradient_change_threshold  # Will adapt
        self.plateau_detection_window = plateau_detection_window
        self.initial_surprise_threshold = surprise_threshold
        self.surprise_threshold = surprise_threshold  # Will adapt
        
        # Meta-learning parameters for adaptive thresholds
        self.threshold_adaptation_rate = 0.1
        self.adaptation_success_tracking = []  # Track trigger success/failure
        
        # Tracking state
        self.prediction_errors = deque(maxlen=50)  # Recent prediction errors
        self.last_adaptation_experience = -min_experiences_between_adaptations
        self.adaptation_triggers_fired = []  # Track what triggers fire
        
        # Gradient tracking
        self.error_gradients = deque(maxlen=10)  # Recent error gradients
        self.last_gradient = None
        
        # Performance tracking  
        self.performance_windows = deque(maxlen=5)  # Recent performance windows
        
        print("AdaptiveTrigger initialized - adaptation will emerge from information events")
    
    def record_prediction_outcome(self, prediction_error: float, experience_count: int):
        """
        Record a prediction outcome and check for adaptation triggers.
        
        Args:
            prediction_error: Current prediction error (0.0-1.0)
            experience_count: Current experience count
        """
        self.prediction_errors.append(prediction_error)
        
        # Compute prediction error gradient if we have enough data
        if len(self.prediction_errors) >= 5:
            recent_errors = list(self.prediction_errors)[-5:]
            current_gradient = self._compute_error_gradient(recent_errors)
            self.error_gradients.append(current_gradient)
            
            # Update performance windows
            if len(self.prediction_errors) >= 10:
                self._update_performance_windows()
    
    def check_adaptation_triggers(self, experience_count: int, 
                                additional_context: Dict = None) -> List[Tuple[str, str, Dict]]:
        """
        Check if any adaptation triggers should fire.
        
        Args:
            experience_count: Current experience count
            additional_context: Additional context for trigger decisions
            
        Returns:
            List of (trigger_type, reason, evidence) tuples for fired triggers
        """
        # Prevent adaptation thrashing
        if experience_count - self.last_adaptation_experience < self.min_experiences_between_adaptations:
            return []
        
        if len(self.prediction_errors) < 10:
            return []  # Need sufficient data
        
        triggered = []
        
        # Trigger 1: Sudden gradient change
        gradient_trigger = self._check_gradient_change_trigger()
        if gradient_trigger:
            triggered.append(gradient_trigger)
        
        # Trigger 2: Sustained poor performance
        performance_trigger = self._check_poor_performance_trigger()
        if performance_trigger:
            triggered.append(performance_trigger)
        
        # Trigger 3: High surprise period
        surprise_trigger = self._check_surprise_trigger()
        if surprise_trigger:
            triggered.append(surprise_trigger)
        
        # Trigger 4: Performance plateau
        plateau_trigger = self._check_plateau_trigger()
        if plateau_trigger:
            triggered.append(plateau_trigger)
        
        # Update last adaptation time if any triggers fired
        if triggered:
            self.last_adaptation_experience = experience_count
            self.adaptation_triggers_fired.extend([t[0] for t in triggered])
        
        return triggered
    
    def _compute_error_gradient(self, errors: List[float]) -> float:
        """Compute prediction error gradient (trend)."""
        if len(errors) < 3:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(errors))
        y = np.array(errors)
        
        # Compute slope (gradient)
        n = len(errors)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def _update_performance_windows(self):
        """Update performance tracking windows."""
        recent_errors = list(self.prediction_errors)[-10:]
        avg_error = np.mean(recent_errors)
        error_variance = np.var(recent_errors)
        
        performance_window = {
            'avg_error': avg_error,
            'error_variance': error_variance,
            'timestamp': time.time()
        }
        
        self.performance_windows.append(performance_window)
    
    def _check_gradient_change_trigger(self) -> Optional[Tuple[str, str, Dict]]:
        """Check for sudden changes in prediction error gradient."""
        if len(self.error_gradients) < 3:
            return None
        
        current_gradient = self.error_gradients[-1]
        prev_gradient = self.error_gradients[-2]
        
        gradient_change = abs(current_gradient - prev_gradient)
        
        if gradient_change > self.gradient_change_threshold:
            return (
                "gradient_change", 
                f"prediction_error_gradient_changed_by_{gradient_change:.3f}",
                {
                    "current_gradient": current_gradient,
                    "previous_gradient": prev_gradient,
                    "gradient_change": gradient_change,
                    "threshold": self.gradient_change_threshold
                }
            )
        
        return None
    
    def _check_poor_performance_trigger(self) -> Optional[Tuple[str, str, Dict]]:
        """Check for sustained poor performance periods."""
        if len(self.prediction_errors) < 15:
            return None
        
        recent_errors = list(self.prediction_errors)[-15:]
        avg_recent_error = np.mean(recent_errors)
        
        # Compare to longer-term average
        all_errors = list(self.prediction_errors)
        overall_avg = np.mean(all_errors)
        
        # Trigger if recent performance is significantly worse
        performance_degradation = avg_recent_error - overall_avg
        
        if performance_degradation > 0.2:  # 20% worse than average
            return (
                "poor_performance",
                f"sustained_poor_performance_degradation_{performance_degradation:.3f}",
                {
                    "recent_avg_error": avg_recent_error,
                    "overall_avg_error": overall_avg,
                    "performance_degradation": performance_degradation,
                    "window_size": len(recent_errors)
                }
            )
        
        return None
    
    def _check_surprise_trigger(self) -> Optional[Tuple[str, str, Dict]]:
        """Check for periods of high prediction surprise."""
        if len(self.prediction_errors) < 10:
            return None
        
        recent_errors = list(self.prediction_errors)[-10:]
        surprise_count = sum(1 for error in recent_errors if error > self.surprise_threshold)
        surprise_rate = surprise_count / len(recent_errors)
        
        if surprise_rate > 0.6:  # More than 60% surprising predictions
            return (
                "high_surprise",
                f"high_surprise_rate_{surprise_rate:.3f}",
                {
                    "surprise_rate": surprise_rate,
                    "surprise_count": surprise_count,
                    "window_size": len(recent_errors),
                    "surprise_threshold": self.surprise_threshold
                }
            )
        
        return None
    
    def _check_plateau_trigger(self) -> Optional[Tuple[str, str, Dict]]:
        """Check for performance plateaus (no improvement)."""
        if len(self.performance_windows) < 3:
            return None
        
        # Check if performance has plateaued (no significant change)
        recent_windows = list(self.performance_windows)[-3:]
        error_means = [w['avg_error'] for w in recent_windows]
        
        # Check if error means are very similar (plateau)
        error_variance_across_windows = np.var(error_means)
        
        if error_variance_across_windows < 0.01:  # Very low variance = plateau
            # Also check that we're not already at very low error
            current_error = error_means[-1]
            if current_error > 0.3:  # Only trigger if error is not already very low
                return (
                    "performance_plateau",
                    f"performance_plateau_variance_{error_variance_across_windows:.4f}",
                    {
                        "error_variance_across_windows": error_variance_across_windows,
                        "recent_error_means": error_means,
                        "current_error": current_error,
                        "window_count": len(recent_windows)
                    }
                )
        
        return None
    
    def get_trigger_statistics(self) -> Dict:
        """Get statistics about adaptation triggers."""
        from collections import Counter
        trigger_counts = Counter(self.adaptation_triggers_fired)
        
        return {
            "total_triggers_fired": len(self.adaptation_triggers_fired),
            "trigger_type_counts": dict(trigger_counts),
            "most_common_trigger": trigger_counts.most_common(1)[0] if trigger_counts else None,
            "prediction_errors_tracked": len(self.prediction_errors),
            "error_gradients_tracked": len(self.error_gradients),
            "performance_windows_tracked": len(self.performance_windows),
            "current_avg_error": np.mean(list(self.prediction_errors)) if self.prediction_errors else 0.0
        }
    
    def reset_tracking(self):
        """Reset all tracking state."""
        self.prediction_errors.clear()
        self.error_gradients.clear()
        self.performance_windows.clear()
        self.adaptation_triggers_fired.clear()
        self.last_adaptation_experience = -self.min_experiences_between_adaptations
        print("AdaptiveTrigger tracking reset")