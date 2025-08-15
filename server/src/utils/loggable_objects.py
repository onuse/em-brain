#!/usr/bin/env python3
"""
Loggable Objects for Brain State

Implementations of LoggableObject for different brain components.
These objects capture state from the brain thread quickly (<1ms),
then perform heavy serialization on the background logger thread.
"""

import time
from typing import Dict, Any, List, Optional
from .async_logger import LoggableObject, LogLevel


class LoggableBrainState(LoggableObject):
    """
    Brain state snapshot for async logging.
    
    Captures brain state quickly on brain thread,
    defers heavy serialization to background thread.
    """
    
    def __init__(self, brain, experience_count: int, 
                 additional_data: Optional[Dict] = None,
                 log_level: LogLevel = LogLevel.DEBUG):
        super().__init__(log_level=log_level)
        
        # Quick capture on brain thread (should be <1ms)
        self.experience_count = experience_count
        self.brain_uptime = time.time() - brain.brain_start_time
        
        # Capture key metrics quickly
        self.total_cycles = brain.total_cycles
        self.total_experiences = brain.total_experiences
        self.total_predictions = brain.total_predictions
        
        # Capture confidence if available
        if hasattr(brain, 'field_brain_adapter') and hasattr(brain.field_brain_adapter, 'emergent_confidence'):
            confidence_system = brain.field_brain_adapter.emergent_confidence
            self.confidence_data = {
                'current_confidence': confidence_system.current_confidence,
                'volatility_confidence': confidence_system.volatility_confidence,
                'coherence_confidence': confidence_system.coherence_confidence,
                'meta_confidence': confidence_system.meta_confidence,
                'total_updates': confidence_system.total_updates,
                'emergent_pattern': confidence_system._detect_current_pattern()
            }
        else:
            self.confidence_data = None
        
        # Store references for background serialization
        self._brain_ref = brain  # Will be used in serialize()
        self._additional_data = additional_data or {}
    
    def serialize(self) -> Dict[str, Any]:
        """
        Heavy serialization work done on background thread.
        
        This can be expensive since it's not on the brain thread.
        """
        try:
            # Get comprehensive brain stats (expensive)
            brain_stats = self._brain_ref.get_brain_stats()
            
            # Compute intrinsic reward (can be expensive)
            intrinsic_reward = self._brain_ref.compute_intrinsic_reward()
            
            # Build comprehensive state
            state_data = {
                "experience_count": self.experience_count,
                "brain_uptime": self.brain_uptime,
                "brain_state": {
                    "total_experiences": self.total_experiences,
                    "total_predictions": self.total_predictions,
                    "optimal_prediction_error": self._brain_ref.optimal_prediction_error,
                    "current_intrinsic_reward": intrinsic_reward,
                    "learning_outcomes_tracked": len(self._brain_ref.recent_learning_outcomes)
                },
                "system_stats": brain_stats,
                "additional_data": self._additional_data
            }
            
            # Add confidence dynamics if available
            if self.confidence_data:
                state_data["confidence_dynamics"] = self.confidence_data
            
            return state_data
            
        except Exception as e:
            # Fallback to minimal data if full serialization fails
            return {
                "experience_count": self.experience_count,
                "brain_uptime": self.brain_uptime,
                "total_cycles": self.total_cycles,
                "error": f"Serialization failed: {e}",
                "confidence_dynamics": self.confidence_data
            }
    
    def get_log_category(self) -> str:
        return "brain_state"


class LoggableEmergenceEvent(LoggableObject):
    """
    Emergence event for async logging.
    
    Captures emergence events quickly and logs them asynchronously.
    """
    
    def __init__(self, event_type: str, description: str, 
                 evidence: Dict[str, Any], significance: float,
                 experience_count: int,
                 log_level: LogLevel = LogLevel.VITAL):
        super().__init__(log_level=log_level)
        
        # Quick capture
        self.event_type = event_type
        self.description = description
        self.evidence = evidence.copy()  # Shallow copy for safety
        self.significance = significance
        self.experience_count = experience_count
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize emergence event."""
        return {
            "event_type": self.event_type,
            "description": self.description,
            "evidence": self.evidence,
            "significance": self.significance,
            "experience_count": self.experience_count
        }
    
    def get_log_category(self) -> str:
        return "emergence"


class LoggablePerformanceMetrics(LoggableObject):
    """
    Performance metrics for async logging.
    
    Tracks brain performance without impacting brain cycles.
    """
    
    def __init__(self, cycle_time_ms: float, 
                 hardware_limits: Dict[str, Any],
                 cognitive_pressure: float,
                 log_level: LogLevel = LogLevel.VITAL):
        super().__init__(log_level=log_level)
        
        # Quick capture
        self.cycle_time_ms = cycle_time_ms
        self.hardware_limits = hardware_limits.copy()
        self.cognitive_pressure = cognitive_pressure
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize performance metrics."""
        return {
            "cycle_time_ms": self.cycle_time_ms,
            "hardware_limits": self.hardware_limits,
            "cognitive_pressure": self.cognitive_pressure,
            "performance_category": self._categorize_performance()
        }
    
    def _categorize_performance(self) -> str:
        """Categorize performance for analysis."""
        if self.cycle_time_ms < 10:
            return "optimal"
        elif self.cycle_time_ms < 25:
            return "good"
        elif self.cycle_time_ms < 50:
            return "acceptable"
        else:
            return "degraded"
    
    def get_log_category(self) -> str:
        return "performance"


class LoggableConfidenceProgression(LoggableObject):
    """
    Confidence progression tracking for async logging.
    
    Specialized for tracking confidence evolution over time.
    """
    
    def __init__(self, confidence_history: List[float],
                 pattern_history: List[str],
                 confidence_components: Dict[str, float],
                 cycle_range: tuple,
                 log_level: LogLevel = LogLevel.DEBUG):
        super().__init__(log_level=log_level)
        
        # Quick capture
        self.confidence_history = confidence_history.copy()
        self.pattern_history = pattern_history.copy()
        self.confidence_components = confidence_components.copy()
        self.cycle_range = cycle_range
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize confidence progression."""
        return {
            "confidence_history": self.confidence_history,
            "pattern_history": self.pattern_history,
            "confidence_components": self.confidence_components,
            "cycle_range": {
                "start": self.cycle_range[0],
                "end": self.cycle_range[1],
                "duration": self.cycle_range[1] - self.cycle_range[0]
            },
            "analysis": self._analyze_progression()
        }
    
    def _analyze_progression(self) -> Dict[str, Any]:
        """Analyze confidence progression patterns."""
        if len(self.confidence_history) < 2:
            return {"insufficient_data": True}
        
        start_confidence = self.confidence_history[0]
        end_confidence = self.confidence_history[-1]
        change = end_confidence - start_confidence
        
        # Detect Dunning-Kruger pattern
        if len(self.confidence_history) >= 10:
            early = sum(self.confidence_history[:3]) / 3
            middle = sum(self.confidence_history[len(self.confidence_history)//3:2*len(self.confidence_history)//3]) / max(1, len(self.confidence_history)//3)
            late = sum(self.confidence_history[-3:]) / 3
            
            dunning_kruger = early > middle and late > middle
        else:
            dunning_kruger = False
        
        return {
            "start_confidence": start_confidence,
            "end_confidence": end_confidence,
            "total_change": change,
            "trend": "increasing" if change > 0.05 else "decreasing" if change < -0.05 else "stable",
            "dunning_kruger_detected": dunning_kruger,
            "pattern_diversity": len(set(self.pattern_history)),
            "dominant_pattern": max(set(self.pattern_history), key=self.pattern_history.count) if self.pattern_history else None
        }
    
    def get_log_category(self) -> str:
        return "confidence"


class LoggableVitalSigns(LoggableObject):
    """
    Minimal vital signs for always-on production logging.
    
    Designed for <0.1ms overhead on brain thread.
    """
    
    def __init__(self, confidence: float, cycle_time_ms: float, 
                 error_count: int = 0,
                 log_level: LogLevel = LogLevel.VITAL):
        super().__init__(log_level=log_level)
        
        # Minimal capture
        self.confidence = confidence
        self.cycle_time_ms = cycle_time_ms
        self.error_count = error_count
    
    def serialize(self) -> Dict[str, Any]:
        """Minimal serialization for vital signs."""
        return {
            "confidence": self.confidence,
            "cycle_time_ms": self.cycle_time_ms,
            "error_count": self.error_count,
            "health_status": self._assess_health()
        }
    
    def _assess_health(self) -> str:
        """Quick health assessment."""
        if self.error_count > 0:
            return "error"
        elif self.cycle_time_ms > 100:
            return "slow"
        elif self.confidence < 0.1:
            return "low_confidence"
        else:
            return "healthy"
    
    def get_log_category(self) -> str:
        return "vital_signs"


class LoggableSystemEvent(LoggableObject):
    """
    General system events for async logging.
    
    For important events that need to be logged but aren't frequent.
    """
    
    def __init__(self, event_name: str, details: Dict[str, Any],
                 severity: str = "info",
                 log_level: LogLevel = LogLevel.DEBUG):
        super().__init__(log_level=log_level)
        
        self.event_name = event_name
        self.details = details.copy()
        self.severity = severity
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize system event."""
        return {
            "event_name": self.event_name,
            "details": self.details,
            "severity": self.severity
        }
    
    def get_log_category(self) -> str:
        return "system_events"