"""
Learning Velocity Monitor - Advanced system for tracking and analyzing learning patterns.

This monitor provides deep insights into how the brain learns over time:
- Learning velocity (rate of error reduction)
- Learning acceleration (change in learning rate)
- Learning patterns (steady, accelerating, plateauing)
- Prediction confidence trends
- Knowledge consolidation rates
- Adaptive learning recommendations

Designed for continuous optimization of brain learning parameters.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import math


@dataclass
class LearningSnapshot:
    """Snapshot of learning metrics at a specific moment."""
    timestamp: float
    step_count: int
    
    # Core learning metrics
    prediction_error: float
    learning_velocity: float      # Rate of error reduction
    learning_acceleration: float  # Change in learning velocity
    
    # Confidence and stability
    confidence_level: float
    confidence_trend: float       # Rate of change in confidence
    prediction_stability: float   # Variance in recent predictions
    
    # Knowledge consolidation
    knowledge_growth_rate: float  # Rate of new knowledge acquisition
    consolidation_efficiency: float  # How well knowledge is being consolidated
    forgetting_rate: float        # Rate of knowledge loss
    
    # Adaptive learning indicators
    exploration_effectiveness: float  # How well exploration leads to learning
    exploitation_efficiency: float   # How well existing knowledge is used
    learning_plateau_risk: float     # Risk of learning stagnation
    
    # Performance metrics
    thinking_speed: float         # How fast decisions are made
    memory_efficiency: float      # How efficiently memory is used
    
    # Context
    brain_size: int              # Number of experiences/nodes
    recent_experiences: List[float]  # Recent prediction errors for pattern analysis


class LearningVelocityMonitor:
    """
    Advanced learning velocity monitoring system.
    
    Tracks learning patterns and provides optimization insights.
    """
    
    def __init__(self, session_name: str = None, history_size: int = 1000):
        """Initialize the learning velocity monitor."""
        self.session_name = session_name or f"learning_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history_size = history_size
        
        # Storage
        self.snapshots: List[LearningSnapshot] = []
        self.snapshot_history = deque(maxlen=history_size)
        
        # Analysis windows
        self.short_term_window = deque(maxlen=20)    # Last 20 snapshots
        self.medium_term_window = deque(maxlen=100)  # Last 100 snapshots
        self.long_term_window = deque(maxlen=500)    # Last 500 snapshots
        
        # Learning pattern detection
        self.error_history = deque(maxlen=200)
        self.velocity_history = deque(maxlen=200)
        self.acceleration_history = deque(maxlen=200)
        
        # Adaptive thresholds
        self.plateau_threshold = 0.001  # Error change threshold for plateau detection
        self.acceleration_threshold = 0.01  # Acceleration threshold for trend detection
        
        # Output paths
        self.output_dir = Path("learning_velocity_logs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Learning Velocity Monitor initialized: {self.session_name}")
        print(f"   History size: {history_size}")
        print(f"   Output directory: {self.output_dir}")
    
    def capture_learning_snapshot(self, brain_stats: Dict[str, Any], 
                                decision_context: Dict[str, Any],
                                step_count: int) -> Optional[LearningSnapshot]:
        """Capture a comprehensive learning velocity snapshot."""
        current_time = time.time()
        
        try:
            # Extract core learning metrics
            prediction_error = decision_context.get('recent_prediction_error', 0.0)
            learning_velocity = self._calculate_learning_velocity(prediction_error)
            learning_acceleration = self._calculate_learning_acceleration(learning_velocity)
            
            # Extract confidence metrics
            confidence_level = decision_context.get('confidence', 0.0)
            confidence_trend = self._calculate_confidence_trend(confidence_level)
            prediction_stability = self._calculate_prediction_stability(prediction_error)
            
            # Extract knowledge consolidation metrics
            knowledge_growth_rate = self._calculate_knowledge_growth_rate(brain_stats)
            consolidation_efficiency = self._calculate_consolidation_efficiency(brain_stats)
            forgetting_rate = self._calculate_forgetting_rate(brain_stats)
            
            # Extract adaptive learning indicators
            exploration_effectiveness = self._calculate_exploration_effectiveness(decision_context)
            exploitation_efficiency = self._calculate_exploitation_efficiency(decision_context)
            learning_plateau_risk = self._calculate_plateau_risk()
            
            # Extract performance metrics
            thinking_speed = self._calculate_thinking_speed(brain_stats)
            memory_efficiency = self._calculate_memory_efficiency(brain_stats)
            
            # Extract context
            brain_size = brain_stats.get('graph_stats', {}).get('total_nodes', 0)
            recent_experiences = list(self.error_history)[-10:]  # Last 10 errors
            
            # Create snapshot
            snapshot = LearningSnapshot(
                timestamp=current_time,
                step_count=step_count,
                prediction_error=prediction_error,
                learning_velocity=learning_velocity,
                learning_acceleration=learning_acceleration,
                confidence_level=confidence_level,
                confidence_trend=confidence_trend,
                prediction_stability=prediction_stability,
                knowledge_growth_rate=knowledge_growth_rate,
                consolidation_efficiency=consolidation_efficiency,
                forgetting_rate=forgetting_rate,
                exploration_effectiveness=exploration_effectiveness,
                exploitation_efficiency=exploitation_efficiency,
                learning_plateau_risk=learning_plateau_risk,
                thinking_speed=thinking_speed,
                memory_efficiency=memory_efficiency,
                brain_size=brain_size,
                recent_experiences=recent_experiences
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.snapshot_history.append(snapshot)
            
            # Update analysis windows
            self._update_analysis_windows(snapshot)
            
            return snapshot
            
        except Exception as e:
            print(f"Error capturing learning snapshot: {e}")
            return None
    
    def _calculate_learning_velocity(self, current_error: float) -> float:
        """Calculate learning velocity (rate of error reduction)."""
        if len(self.error_history) < 2:
            self.error_history.append(current_error)
            return 0.0
        
        # Calculate velocity as rate of error improvement
        prev_error = self.error_history[-1]
        time_diff = time.time() - (self.snapshots[-1].timestamp if self.snapshots else time.time())
        
        if time_diff > 0:
            velocity = (prev_error - current_error) / time_diff  # Positive = learning
            self.velocity_history.append(velocity)
            self.error_history.append(current_error)
            return velocity
        
        self.error_history.append(current_error)
        return 0.0
    
    def _calculate_learning_acceleration(self, current_velocity: float) -> float:
        """Calculate learning acceleration (change in learning velocity)."""
        if len(self.velocity_history) < 2:
            return 0.0
        
        prev_velocity = self.velocity_history[-2]
        time_diff = time.time() - (self.snapshots[-1].timestamp if self.snapshots else time.time())
        
        if time_diff > 0:
            acceleration = (current_velocity - prev_velocity) / time_diff
            self.acceleration_history.append(acceleration)
            return acceleration
        
        return 0.0
    
    def _calculate_confidence_trend(self, current_confidence: float) -> float:
        """Calculate trend in prediction confidence."""
        if len(self.short_term_window) < 2:
            return 0.0
        
        prev_confidence = self.short_term_window[-1].confidence_level
        time_diff = time.time() - self.short_term_window[-1].timestamp
        
        if time_diff > 0:
            return (current_confidence - prev_confidence) / time_diff
        return 0.0
    
    def _calculate_prediction_stability(self, current_error: float) -> float:
        """Calculate stability of recent predictions."""
        if len(self.error_history) < 10:
            return 0.0
        
        recent_errors = list(self.error_history)[-10:]
        return 1.0 / (1.0 + np.std(recent_errors))  # High stability = low variance
    
    def _calculate_knowledge_growth_rate(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate rate of knowledge acquisition."""
        if len(self.snapshots) < 2:
            return 0.0
        
        current_nodes = brain_stats.get('graph_stats', {}).get('total_nodes', 0)
        prev_nodes = self.snapshots[-1].brain_size
        time_diff = time.time() - self.snapshots[-1].timestamp
        
        if time_diff > 0:
            return (current_nodes - prev_nodes) / time_diff
        return 0.0
    
    def _calculate_consolidation_efficiency(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate how efficiently knowledge is being consolidated."""
        consolidation_stats = brain_stats.get('consolidation_stats', {})
        consolidation_rate = consolidation_stats.get('consolidation_rate', 0.0)
        
        # Normalize consolidation rate (higher = more efficient)
        return min(1.0, consolidation_rate * 10.0)
    
    def _calculate_forgetting_rate(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate rate of knowledge loss/forgetting."""
        # This would require specific forgetting metrics from the brain
        # For now, return 0 (no forgetting implemented)
        return 0.0
    
    def _calculate_exploration_effectiveness(self, decision_context: Dict[str, Any]) -> float:
        """Calculate how effective exploration is for learning."""
        # Look at recent exploration actions and their learning outcomes
        if len(self.snapshots) < 10:
            return 0.5  # Default effectiveness
        
        # Calculate correlation between exploration and learning velocity
        recent_snapshots = self.snapshots[-10:]
        exploration_levels = []
        learning_velocities = []
        
        for snapshot in recent_snapshots:
            # Estimate exploration level from action diversity and confidence
            exploration_level = 1.0 - snapshot.confidence_level
            exploration_levels.append(exploration_level)
            learning_velocities.append(snapshot.learning_velocity)
        
        # Calculate correlation (handle zero variance case)
        if len(exploration_levels) > 1:
            try:
                # Check for zero variance to avoid numpy warnings
                if np.var(exploration_levels) == 0 or np.var(learning_velocities) == 0:
                    return 0.5  # No correlation possible with constant values
                correlation = np.corrcoef(exploration_levels, learning_velocities)[0, 1]
                # Handle NaN case (shouldn't happen with variance check, but safety)
                if np.isnan(correlation):
                    return 0.5
                return max(0.0, correlation)  # Positive correlation = effective exploration
            except Exception:
                return 0.5  # Fallback on any calculation error
        
        return 0.5
    
    def _calculate_exploitation_efficiency(self, decision_context: Dict[str, Any]) -> float:
        """Calculate how efficiently existing knowledge is used."""
        confidence_level = decision_context.get('confidence', 0.0)
        prediction_error = decision_context.get('recent_prediction_error', 0.0)
        
        # High confidence with low error = efficient exploitation
        if confidence_level > 0.7 and prediction_error < 0.3:
            return min(1.0, confidence_level * (1.0 - prediction_error))
        
        return confidence_level * 0.5
    
    def _calculate_plateau_risk(self) -> float:
        """Calculate risk of learning plateau."""
        if len(self.velocity_history) < 20:
            return 0.0
        
        # Look at recent learning velocities
        recent_velocities = list(self.velocity_history)[-20:]
        
        # Calculate variance in learning velocity
        velocity_variance = np.var(recent_velocities)
        
        # Calculate trend in learning velocity
        if len(recent_velocities) > 1:
            velocity_trend = np.polyfit(range(len(recent_velocities)), recent_velocities, 1)[0]
        else:
            velocity_trend = 0.0
        
        # High plateau risk = low variance and negative/zero trend
        if velocity_variance < self.plateau_threshold and velocity_trend <= 0:
            return 1.0 - velocity_variance / self.plateau_threshold
        
        return 0.0
    
    def _calculate_thinking_speed(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate thinking speed (inverse of thinking time)."""
        thinking_time = brain_stats.get('predictor_stats', {}).get('average_thinking_time', 0.1)
        return 1.0 / max(0.001, thinking_time)  # Avoid division by zero
    
    def _calculate_memory_efficiency(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate memory system efficiency."""
        graph_stats = brain_stats.get('graph_stats', {})
        total_nodes = graph_stats.get('total_nodes', 0)
        total_merges = graph_stats.get('total_merges', 0)
        
        if total_nodes == 0:
            return 0.0
        
        return total_merges / total_nodes  # Higher merge rate = more efficient
    
    def _update_analysis_windows(self, snapshot: LearningSnapshot):
        """Update analysis windows with new snapshot."""
        self.short_term_window.append(snapshot)
        self.medium_term_window.append(snapshot)
        self.long_term_window.append(snapshot)
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns and provide insights."""
        if len(self.snapshots) < 10:
            return {"error": "Insufficient data for learning pattern analysis (need at least 10 snapshots)"}
        
        analysis = {
            "session_metadata": {
                "session_name": self.session_name,
                "total_snapshots": len(self.snapshots),
                "time_span": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
                "step_span": self.snapshots[-1].step_count - self.snapshots[0].step_count
            },
            
            "learning_velocity_analysis": self._analyze_learning_velocity(),
            "learning_acceleration_analysis": self._analyze_learning_acceleration(),
            "confidence_evolution": self._analyze_confidence_evolution(),
            "knowledge_consolidation": self._analyze_knowledge_consolidation(),
            "adaptive_learning_indicators": self._analyze_adaptive_indicators(),
            "performance_optimization": self._analyze_performance_optimization(),
            
            "learning_insights": self._generate_learning_insights(),
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
        
        return analysis
    
    def _analyze_learning_velocity(self) -> Dict[str, Any]:
        """Analyze learning velocity patterns."""
        velocities = [s.learning_velocity for s in self.snapshots]
        errors = [s.prediction_error for s in self.snapshots]
        
        return {
            "initial_velocity": velocities[0],
            "final_velocity": velocities[-1],
            "average_velocity": np.mean(velocities),
            "velocity_trend": "increasing" if velocities[-1] > velocities[0] else "decreasing",
            "velocity_stability": 1.0 / (1.0 + np.std(velocities)),
            "peak_velocity": max(velocities),
            "velocity_consistency": 1.0 - (np.std(velocities) / max(0.001, np.mean(velocities))),
            "error_reduction_rate": (errors[0] - errors[-1]) / len(self.snapshots) if len(self.snapshots) > 0 else 0.0
        }
    
    def _analyze_learning_acceleration(self) -> Dict[str, Any]:
        """Analyze learning acceleration patterns."""
        accelerations = [s.learning_acceleration for s in self.snapshots]
        
        return {
            "average_acceleration": np.mean(accelerations),
            "acceleration_trend": "increasing" if accelerations[-1] > accelerations[0] else "decreasing",
            "acceleration_stability": 1.0 / (1.0 + np.std(accelerations)),
            "positive_acceleration_ratio": sum(1 for a in accelerations if a > 0) / len(accelerations),
            "acceleration_momentum": np.mean(accelerations[-5:]) if len(accelerations) >= 5 else 0.0
        }
    
    def _analyze_confidence_evolution(self) -> Dict[str, Any]:
        """Analyze confidence evolution patterns."""
        confidences = [s.confidence_level for s in self.snapshots]
        confidence_trends = [s.confidence_trend for s in self.snapshots]
        
        return {
            "initial_confidence": confidences[0],
            "final_confidence": confidences[-1],
            "average_confidence": np.mean(confidences),
            "confidence_growth": confidences[-1] - confidences[0],
            "confidence_stability": 1.0 / (1.0 + np.std(confidences)),
            "confidence_trend_direction": "increasing" if np.mean(confidence_trends) > 0 else "decreasing",
            "peak_confidence": max(confidences),
            "confidence_maturity": min(1.0, np.mean(confidences) * 1.2)
        }
    
    def _analyze_knowledge_consolidation(self) -> Dict[str, Any]:
        """Analyze knowledge consolidation patterns."""
        growth_rates = [s.knowledge_growth_rate for s in self.snapshots]
        consolidation_efficiencies = [s.consolidation_efficiency for s in self.snapshots]
        
        return {
            "average_growth_rate": np.mean(growth_rates),
            "growth_trend": "increasing" if growth_rates[-1] > growth_rates[0] else "decreasing",
            "average_consolidation_efficiency": np.mean(consolidation_efficiencies),
            "consolidation_trend": "improving" if consolidation_efficiencies[-1] > consolidation_efficiencies[0] else "declining",
            "knowledge_acceleration": np.mean(np.diff(growth_rates)) if len(growth_rates) > 1 else 0.0,
            "consolidation_stability": 1.0 / (1.0 + np.std(consolidation_efficiencies))
        }
    
    def _analyze_adaptive_indicators(self) -> Dict[str, Any]:
        """Analyze adaptive learning indicators."""
        exploration_effectiveness = [s.exploration_effectiveness for s in self.snapshots]
        exploitation_efficiency = [s.exploitation_efficiency for s in self.snapshots]
        plateau_risks = [s.learning_plateau_risk for s in self.snapshots]
        
        return {
            "exploration_effectiveness": np.mean(exploration_effectiveness),
            "exploitation_efficiency": np.mean(exploitation_efficiency),
            "current_plateau_risk": plateau_risks[-1],
            "average_plateau_risk": np.mean(plateau_risks),
            "exploration_trend": "improving" if exploration_effectiveness[-1] > exploration_effectiveness[0] else "declining",
            "exploitation_trend": "improving" if exploitation_efficiency[-1] > exploitation_efficiency[0] else "declining",
            "learning_balance": abs(np.mean(exploration_effectiveness) - np.mean(exploitation_efficiency))
        }
    
    def _analyze_performance_optimization(self) -> Dict[str, Any]:
        """Analyze performance optimization patterns."""
        thinking_speeds = [s.thinking_speed for s in self.snapshots]
        memory_efficiencies = [s.memory_efficiency for s in self.snapshots]
        
        return {
            "thinking_speed_improvement": thinking_speeds[-1] - thinking_speeds[0],
            "average_thinking_speed": np.mean(thinking_speeds),
            "memory_efficiency_improvement": memory_efficiencies[-1] - memory_efficiencies[0],
            "average_memory_efficiency": np.mean(memory_efficiencies),
            "performance_trend": "improving" if thinking_speeds[-1] > thinking_speeds[0] else "declining",
            "efficiency_optimization": "active" if np.mean(memory_efficiencies) > 0.1 else "minimal"
        }
    
    def _generate_learning_insights(self) -> List[str]:
        """Generate high-level learning insights."""
        insights = []
        
        if len(self.snapshots) < 10:
            insights.append("⚠️ Insufficient data for meaningful learning insights")
            return insights
        
        # Velocity insights
        velocities = [s.learning_velocity for s in self.snapshots]
        if np.mean(velocities) > 0.5:
            insights.append("🚀 Strong learning velocity - rapid error reduction")
        elif np.mean(velocities) < 0.1:
            insights.append("🐌 Slow learning velocity - consider parameter adjustments")
        
        # Acceleration insights
        accelerations = [s.learning_acceleration for s in self.snapshots]
        if np.mean(accelerations) > 0.1:
            insights.append("📈 Learning acceleration detected - knowledge acquisition speeding up")
        elif np.mean(accelerations) < -0.1:
            insights.append("📉 Learning deceleration detected - may be approaching plateau")
        
        # Confidence insights
        confidences = [s.confidence_level for s in self.snapshots]
        if confidences[-1] > confidences[0] * 1.5:
            insights.append("💪 Confidence significantly improved - brain becoming more certain")
        
        # Plateau risk insights
        plateau_risks = [s.learning_plateau_risk for s in self.snapshots]
        if plateau_risks[-1] > 0.7:
            insights.append("⚠️ High plateau risk - learning may be stagnating")
        
        # Performance insights
        thinking_speeds = [s.thinking_speed for s in self.snapshots]
        if thinking_speeds[-1] > thinking_speeds[0] * 1.3:
            insights.append("⚡ Thinking speed improved - faster decision making")
        
        return insights
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on learning patterns."""
        recommendations = []
        
        if len(self.snapshots) < 10:
            return recommendations
        
        # Analyze current state
        latest = self.snapshots[-1]
        
        # Velocity recommendations
        if latest.learning_velocity < 0.1:
            recommendations.append("🔧 Consider increasing exploration rate to improve learning velocity")
        
        # Acceleration recommendations
        if latest.learning_acceleration < -0.05:
            recommendations.append("🔧 Learning is decelerating - consider adjusting novelty thresholds")
        
        # Plateau risk recommendations
        if latest.learning_plateau_risk > 0.6:
            recommendations.append("🔧 High plateau risk - introduce new challenges or reset exploration")
        
        # Balance recommendations
        if latest.exploration_effectiveness < 0.3:
            recommendations.append("🔧 Exploration is ineffective - optimize exploration strategy")
        
        if latest.exploitation_efficiency < 0.3:
            recommendations.append("🔧 Exploitation is inefficient - improve confidence mechanisms")
        
        # Performance recommendations
        if latest.thinking_speed < 10.0:
            recommendations.append("🔧 Slow thinking speed - consider performance optimizations")
        
        return recommendations
    
    def save_learning_report(self) -> str:
        """Save comprehensive learning analysis report."""
        analysis = self.analyze_learning_patterns()
        
        report_file = self.output_dir / f"{self.session_name}_learning_report.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📊 Learning velocity report saved: {report_file}")
        return str(report_file)
    
    def save_learning_snapshots(self) -> str:
        """Save all learning snapshots."""
        snapshots_file = self.output_dir / f"{self.session_name}_learning_snapshots.json"
        
        snapshot_data = {
            "session_name": self.session_name,
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots]
        }
        
        with open(snapshots_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        
        print(f"📸 Learning snapshots saved: {snapshots_file}")
        return str(snapshots_file)
    
    def print_learning_status(self):
        """Print current learning status."""
        if not self.snapshots:
            print("🎯 No learning snapshots captured yet")
            return
        
        latest = self.snapshots[-1]
        print(f"\n🎯 LEARNING VELOCITY STATUS")
        print(f"   Session: {self.session_name}")
        print(f"   Step: {latest.step_count} | Brain Size: {latest.brain_size}")
        print(f"   Learning Velocity: {latest.learning_velocity:.3f} | Acceleration: {latest.learning_acceleration:.3f}")
        print(f"   Prediction Error: {latest.prediction_error:.3f} | Confidence: {latest.confidence_level:.3f}")
        print(f"   Exploration Effectiveness: {latest.exploration_effectiveness:.3f}")
        print(f"   Plateau Risk: {latest.learning_plateau_risk:.3f}")
        print(f"   Thinking Speed: {latest.thinking_speed:.1f} | Memory Efficiency: {latest.memory_efficiency:.3f}")