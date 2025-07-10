"""
Drive Interaction Correlation Analyzer - Advanced system for analyzing drive system dynamics.

This analyzer provides deep insights into how different drives interact:
- Drive correlation matrices (how drives influence each other)
- Drive switching patterns (when and why drives change dominance)
- Drive stability analysis (how consistent each drive is)
- Drive competition analysis (which drives compete vs cooperate)
- Drive effectiveness analysis (which drives lead to better outcomes)
- Drive balance recommendations (optimal drive weight distributions)

Designed for optimizing the multi-drive motivation system for emergent behavior.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math
try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - using numpy correlation instead")


@dataclass
class DriveSnapshot:
    """Snapshot of drive states at a specific moment."""
    timestamp: float
    step_count: int
    
    # Drive weights and pressures
    motivator_weights: Dict[str, float]
    total_motivator_pressure: float
    dominant_motivator: str
    
    # Drive dynamics
    drive_entropy: float           # Diversity of drive activation
    drive_stability: float         # How stable the drive system is
    drive_switching_rate: float    # How often dominance changes
    
    # Drive effectiveness
    drive_outcome_correlation: Dict[str, float]  # How well each drive predicts good outcomes
    drive_learning_correlation: Dict[str, float]  # How well each drive correlates with learning
    
    # Context
    prediction_error: float
    confidence_level: float
    robot_health: float
    robot_energy: float
    robot_position: Tuple[float, float]
    
    # Outcome measures
    learning_velocity: float
    behavioral_complexity: float
    survival_success: float


class DriveCorrelationAnalyzer:
    """
    Advanced drive interaction correlation analyzer.
    
    Analyzes how drives interact and provides optimization insights.
    """
    
    def __init__(self, session_name: str = None, history_size: int = 1000):
        """Initialize the drive correlation analyzer."""
        self.session_name = session_name or f"drive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history_size = history_size
        
        # Storage
        self.snapshots: List[DriveSnapshot] = []
        self.snapshot_history = deque(maxlen=history_size)
        
        # Drive tracking
        self.drive_names = set()
        self.drive_weight_history = defaultdict(lambda: deque(maxlen=history_size))
        self.drive_dominance_history = deque(maxlen=history_size)
        
        # Analysis matrices
        self.correlation_matrix = {}
        self.competition_matrix = {}
        self.cooperation_matrix = {}
        
        # Pattern detection
        self.switching_patterns = deque(maxlen=200)
        self.stability_windows = deque(maxlen=100)
        
        # Outcome tracking
        self.drive_outcome_tracking = defaultdict(list)
        self.drive_learning_tracking = defaultdict(list)
        
        # Output paths
        self.output_dir = Path("drive_correlation_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Drive Correlation Analyzer initialized: {self.session_name}")
        print(f"   History size: {history_size}")
        print(f"   Output directory: {self.output_dir}")
    
    def capture_drive_snapshot(self, brain_stats: Dict[str, Any], 
                             decision_context: Dict[str, Any],
                             step_count: int) -> Optional[DriveSnapshot]:
        """Capture a comprehensive drive interaction snapshot."""
        current_time = time.time()
        
        try:
            # Extract drive information
            motivator_weights = decision_context.get('motivator_weights', {})
            if not motivator_weights:
                return None  # No drive information available
            
            # Update drive tracking
            self.drive_names.update(motivator_weights.keys())
            for drive_name, weight in motivator_weights.items():
                self.drive_weight_history[drive_name].append(weight)
            
            # Calculate drive dynamics
            total_motivator_pressure = sum(motivator_weights.values())
            dominant_motivator = max(motivator_weights.keys(), key=lambda k: motivator_weights[k]) if motivator_weights else 'unknown'
            
            drive_entropy = self._calculate_drive_entropy(motivator_weights)
            drive_stability = self._calculate_drive_stability(motivator_weights)
            drive_switching_rate = self._calculate_switching_rate(dominant_motivator)
            
            # Calculate drive effectiveness
            drive_outcome_correlation = self._calculate_drive_outcome_correlation(motivator_weights, decision_context)
            drive_learning_correlation = self._calculate_drive_learning_correlation(motivator_weights, decision_context)
            
            # Extract context and outcomes
            prediction_error = decision_context.get('recent_prediction_error', 0.0)
            confidence_level = decision_context.get('confidence', 0.0)
            robot_health = decision_context.get('robot_health', 0.0)
            robot_energy = decision_context.get('robot_energy', 0.0)
            robot_position = decision_context.get('robot_position', (0.0, 0.0))
            
            # Calculate outcome measures
            learning_velocity = self._calculate_learning_velocity(prediction_error)
            behavioral_complexity = self._calculate_behavioral_complexity(decision_context)
            survival_success = self._calculate_survival_success(robot_health, robot_energy)
            
            # Create snapshot
            snapshot = DriveSnapshot(
                timestamp=current_time,
                step_count=step_count,
                motivator_weights=motivator_weights,
                total_motivator_pressure=total_motivator_pressure,
                dominant_motivator=dominant_motivator,
                drive_entropy=drive_entropy,
                drive_stability=drive_stability,
                drive_switching_rate=drive_switching_rate,
                drive_outcome_correlation=drive_outcome_correlation,
                drive_learning_correlation=drive_learning_correlation,
                prediction_error=prediction_error,
                confidence_level=confidence_level,
                robot_health=robot_health,
                robot_energy=robot_energy,
                robot_position=robot_position,
                learning_velocity=learning_velocity,
                behavioral_complexity=behavioral_complexity,
                survival_success=survival_success
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.snapshot_history.append(snapshot)
            
            # Update analysis matrices
            self._update_correlation_matrices(snapshot)
            
            # Track drive outcomes
            self._track_drive_outcomes(snapshot)
            
            return snapshot
            
        except Exception as e:
            print(f"Error capturing drive snapshot: {e}")
            return None
    
    def _calculate_drive_entropy(self, motivator_weights: Dict[str, float]) -> float:
        """Calculate entropy of drive system (higher = more diverse activation)."""
        if not motivator_weights:
            return 0.0
        
        total_weight = sum(motivator_weights.values())
        if total_weight == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for weight in motivator_weights.values():
            if weight > 0:
                p = weight / total_weight
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_drive_stability(self, current_weights: Dict[str, float]) -> float:
        """Calculate stability of drive system."""
        if len(self.snapshots) < 10:
            return 0.5  # Default stability
        
        # Calculate variance in drive weights over recent history
        recent_snapshots = self.snapshots[-10:]
        drive_variances = []
        
        for drive_name in current_weights.keys():
            recent_weights = [s.motivator_weights.get(drive_name, 0.0) for s in recent_snapshots]
            drive_variances.append(np.var(recent_weights))
        
        # Higher variance = lower stability
        avg_variance = np.mean(drive_variances)
        return 1.0 / (1.0 + avg_variance)
    
    def _calculate_switching_rate(self, current_dominant: str) -> float:
        """Calculate rate of drive dominance switching."""
        self.drive_dominance_history.append(current_dominant)
        
        if len(self.drive_dominance_history) < 2:
            return 0.0
        
        # Count switches in recent history
        recent_history = list(self.drive_dominance_history)[-20:]  # Last 20 steps
        switches = sum(1 for i in range(1, len(recent_history)) 
                      if recent_history[i] != recent_history[i-1])
        
        return switches / max(1, len(recent_history) - 1)
    
    def _calculate_drive_outcome_correlation(self, motivator_weights: Dict[str, float], 
                                           decision_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation between each drive and positive outcomes."""
        correlations = {}
        
        # Define positive outcome (low error, high confidence, good health)
        error = decision_context.get('recent_prediction_error', 0.0)
        confidence = decision_context.get('confidence', 0.0)
        health = decision_context.get('robot_health', 0.0)
        
        outcome_score = (1.0 - error) * confidence * health
        
        # Track outcomes for each drive
        for drive_name, weight in motivator_weights.items():
            self.drive_outcome_tracking[drive_name].append((weight, outcome_score))
            
            # Calculate correlation if we have enough data
            if len(self.drive_outcome_tracking[drive_name]) >= 10:
                weights = [w for w, o in self.drive_outcome_tracking[drive_name][-20:]]
                outcomes = [o for w, o in self.drive_outcome_tracking[drive_name][-20:]]
                
                if len(weights) > 1 and np.std(weights) > 0:
                    if SCIPY_AVAILABLE:
                        correlation, _ = pearsonr(weights, outcomes)
                        correlations[drive_name] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        # Fallback to numpy correlation (with variance check)
                        if np.var(weights) == 0 or np.var(outcomes) == 0:
                            correlation = 0.0  # No correlation possible with constant values
                        else:
                            correlation = np.corrcoef(weights, outcomes)[0, 1]
                        correlations[drive_name] = correlation if not np.isnan(correlation) else 0.0
                else:
                    correlations[drive_name] = 0.0
            else:
                correlations[drive_name] = 0.0
        
        return correlations
    
    def _calculate_drive_learning_correlation(self, motivator_weights: Dict[str, float], 
                                            decision_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlation between each drive and learning progress."""
        correlations = {}
        
        # Define learning progress (error reduction over time)
        current_error = decision_context.get('recent_prediction_error', 0.0)
        learning_score = 1.0 - current_error  # Simple learning metric
        
        # Track learning for each drive
        for drive_name, weight in motivator_weights.items():
            self.drive_learning_tracking[drive_name].append((weight, learning_score))
            
            # Calculate correlation if we have enough data
            if len(self.drive_learning_tracking[drive_name]) >= 10:
                weights = [w for w, l in self.drive_learning_tracking[drive_name][-20:]]
                learning_scores = [l for w, l in self.drive_learning_tracking[drive_name][-20:]]
                
                if len(weights) > 1 and np.std(weights) > 0:
                    if SCIPY_AVAILABLE:
                        correlation, _ = pearsonr(weights, learning_scores)
                        correlations[drive_name] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        # Fallback to numpy correlation (with variance check)
                        if np.var(weights) == 0 or np.var(learning_scores) == 0:
                            correlation = 0.0  # No correlation possible with constant values
                        else:
                            correlation = np.corrcoef(weights, learning_scores)[0, 1]
                        correlations[drive_name] = correlation if not np.isnan(correlation) else 0.0
                else:
                    correlations[drive_name] = 0.0
            else:
                correlations[drive_name] = 0.0
        
        return correlations
    
    def _calculate_learning_velocity(self, current_error: float) -> float:
        """Calculate learning velocity from error reduction."""
        if len(self.snapshots) < 2:
            return 0.0
        
        prev_error = self.snapshots[-1].prediction_error
        time_diff = time.time() - self.snapshots[-1].timestamp
        
        if time_diff > 0:
            return (prev_error - current_error) / time_diff
        return 0.0
    
    def _calculate_behavioral_complexity(self, decision_context: Dict[str, Any]) -> float:
        """Calculate behavioral complexity from action diversity."""
        chosen_action = decision_context.get('chosen_action', {})
        if not chosen_action:
            return 0.0
        
        # Calculate coefficient of variation in actions
        action_values = [abs(v) for v in chosen_action.values() if isinstance(v, (int, float))]
        if not action_values:
            return 0.0
        
        mean_action = np.mean(action_values)
        std_action = np.std(action_values)
        
        return std_action / mean_action if mean_action > 0 else 0.0
    
    def _calculate_survival_success(self, health: float, energy: float) -> float:
        """Calculate survival success metric."""
        return (health + energy) / 2.0
    
    def _update_correlation_matrices(self, snapshot: DriveSnapshot):
        """Update correlation matrices with new snapshot."""
        if len(self.snapshots) < 20:
            return  # Need more data for meaningful correlations
        
        # Get recent drive weight history
        recent_snapshots = self.snapshots[-20:]
        
        # Calculate pairwise correlations
        for drive1 in self.drive_names:
            for drive2 in self.drive_names:
                if drive1 != drive2:
                    weights1 = [s.motivator_weights.get(drive1, 0.0) for s in recent_snapshots]
                    weights2 = [s.motivator_weights.get(drive2, 0.0) for s in recent_snapshots]
                    
                    if np.std(weights1) > 0 and np.std(weights2) > 0:
                        if SCIPY_AVAILABLE:
                            correlation, _ = pearsonr(weights1, weights2)
                        else:
                            # Fallback to numpy correlation (with variance check)
                            if np.var(weights1) == 0 or np.var(weights2) == 0:
                                correlation = 0.0  # No correlation possible with constant values
                            else:
                                correlation = np.corrcoef(weights1, weights2)[0, 1]
                        
                        if not np.isnan(correlation):
                            self.correlation_matrix[f"{drive1}-{drive2}"] = correlation
        
        # Calculate competition vs cooperation
        self._update_competition_cooperation_matrices()
    
    def _update_competition_cooperation_matrices(self):
        """Update competition and cooperation matrices."""
        # Competition: negative correlation + alternating dominance
        # Cooperation: positive correlation + simultaneous high activation
        
        for drive1 in self.drive_names:
            for drive2 in self.drive_names:
                if drive1 != drive2:
                    correlation_key = f"{drive1}-{drive2}"
                    correlation = self.correlation_matrix.get(correlation_key, 0.0)
                    
                    # Competition score: negative correlation indicates competition
                    competition_score = max(0.0, -correlation)
                    self.competition_matrix[correlation_key] = competition_score
                    
                    # Cooperation score: positive correlation indicates cooperation
                    cooperation_score = max(0.0, correlation)
                    self.cooperation_matrix[correlation_key] = cooperation_score
    
    def _track_drive_outcomes(self, snapshot: DriveSnapshot):
        """Track outcomes for each drive activation pattern."""
        # This is handled in the correlation calculation methods
        pass
    
    def analyze_drive_interactions(self) -> Dict[str, Any]:
        """Analyze drive interactions and provide insights."""
        if len(self.snapshots) < 20:
            return {"error": "Insufficient data for drive interaction analysis (need at least 20 snapshots)"}
        
        analysis = {
            "session_metadata": {
                "session_name": self.session_name,
                "total_snapshots": len(self.snapshots),
                "time_span": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
                "step_span": self.snapshots[-1].step_count - self.snapshots[0].step_count,
                "drive_names": list(self.drive_names)
            },
            
            "drive_correlation_analysis": self._analyze_drive_correlations(),
            "drive_competition_analysis": self._analyze_drive_competition(),
            "drive_cooperation_analysis": self._analyze_drive_cooperation(),
            "drive_stability_analysis": self._analyze_drive_stability(),
            "drive_effectiveness_analysis": self._analyze_drive_effectiveness(),
            "drive_switching_patterns": self._analyze_switching_patterns(),
            
            "drive_insights": self._generate_drive_insights(),
            "drive_optimization_recommendations": self._generate_drive_recommendations()
        }
        
        return analysis
    
    def _analyze_drive_correlations(self) -> Dict[str, Any]:
        \"\"\"Analyze correlations between drives.\"\"\"
        if not self.correlation_matrix:
            return {\"error\": \"No correlation data available\"}
        
        # Find strongest correlations
        sorted_correlations = sorted(self.correlation_matrix.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        return {
            \"strongest_correlations\": sorted_correlations[:5],
            \"average_correlation\": np.mean(list(self.correlation_matrix.values())),
            \"correlation_variance\": np.var(list(self.correlation_matrix.values())),
            \"positive_correlations\": {k: v for k, v in self.correlation_matrix.items() if v > 0.3},
            \"negative_correlations\": {k: v for k, v in self.correlation_matrix.items() if v < -0.3}
        }
    
    def _analyze_drive_competition(self) -> Dict[str, Any]:
        \"\"\"Analyze competition between drives.\"\"\"
        if not self.competition_matrix:
            return {\"error\": \"No competition data available\"}
        
        # Find strongest competitions
        sorted_competitions = sorted(self.competition_matrix.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return {
            \"strongest_competitions\": sorted_competitions[:5],
            \"average_competition\": np.mean(list(self.competition_matrix.values())),
            \"high_competition_pairs\": {k: v for k, v in self.competition_matrix.items() if v > 0.5}
        }
    
    def _analyze_drive_cooperation(self) -> Dict[str, Any]:
        \"\"\"Analyze cooperation between drives.\"\"\"
        if not self.cooperation_matrix:
            return {\"error\": \"No cooperation data available\"}
        
        # Find strongest cooperations
        sorted_cooperations = sorted(self.cooperation_matrix.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return {
            \"strongest_cooperations\": sorted_cooperations[:5],
            \"average_cooperation\": np.mean(list(self.cooperation_matrix.values())),
            \"high_cooperation_pairs\": {k: v for k, v in self.cooperation_matrix.items() if v > 0.5}
        }
    
    def _analyze_drive_stability(self) -> Dict[str, Any]:
        \"\"\"Analyze stability of drive system.\"\"\"
        stabilities = [s.drive_stability for s in self.snapshots]
        switching_rates = [s.drive_switching_rate for s in self.snapshots]
        
        return {
            \"average_stability\": np.mean(stabilities),
            \"stability_trend\": \"increasing\" if stabilities[-1] > stabilities[0] else \"decreasing\",
            \"stability_variance\": np.var(stabilities),
            \"average_switching_rate\": np.mean(switching_rates),
            \"switching_trend\": \"increasing\" if switching_rates[-1] > switching_rates[0] else \"decreasing\"
        }
    
    def _analyze_drive_effectiveness(self) -> Dict[str, Any]:
        \"\"\"Analyze effectiveness of each drive.\"\"\"
        if not self.snapshots:
            return {\"error\": \"No drive effectiveness data available\"}
        
        # Aggregate effectiveness metrics
        drive_effectiveness = {}
        for drive_name in self.drive_names:
            outcome_correlations = [s.drive_outcome_correlation.get(drive_name, 0.0) for s in self.snapshots]
            learning_correlations = [s.drive_learning_correlation.get(drive_name, 0.0) for s in self.snapshots]
            
            drive_effectiveness[drive_name] = {
                \"average_outcome_correlation\": np.mean(outcome_correlations),
                \"average_learning_correlation\": np.mean(learning_correlations),
                \"overall_effectiveness\": (np.mean(outcome_correlations) + np.mean(learning_correlations)) / 2
            }
        
        # Rank drives by effectiveness
        ranked_drives = sorted(drive_effectiveness.items(), 
                             key=lambda x: x[1]['overall_effectiveness'], reverse=True)
        
        return {
            \"drive_effectiveness\": drive_effectiveness,
            \"ranked_drives\": ranked_drives,
            \"most_effective_drive\": ranked_drives[0][0] if ranked_drives else None,
            \"least_effective_drive\": ranked_drives[-1][0] if ranked_drives else None
        }
    
    def _analyze_switching_patterns(self) -> Dict[str, Any]:
        \"\"\"Analyze drive switching patterns.\"\"\"
        if len(self.snapshots) < 10:
            return {\"error\": \"Insufficient data for switching pattern analysis\"}
        
        # Count dominance sequences
        dominance_sequence = [s.dominant_motivator for s in self.snapshots]
        
        # Count transitions
        transitions = defaultdict(int)
        for i in range(1, len(dominance_sequence)):
            transition = f"{dominance_sequence[i-1]}->{dominance_sequence[i]}"
            transitions[transition] += 1
        
        # Find most common transitions
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            \"dominance_sequence\": dominance_sequence[-20:],  # Last 20 dominance changes
            \"most_common_transitions\": sorted_transitions[:5],
            \"total_transitions\": len(transitions),
            \"switching_frequency\": len(transitions) / len(dominance_sequence)
        }
    
    def _generate_drive_insights(self) -> List[str]:
        \"\"\"Generate high-level drive interaction insights.\"\"\"
        insights = []
        
        if len(self.snapshots) < 20:
            insights.append(\"⚠️ Insufficient data for meaningful drive insights\")
            return insights
        
        # Stability insights
        stabilities = [s.drive_stability for s in self.snapshots]
        if np.mean(stabilities) > 0.7:
            insights.append(\"🔒 Drive system is highly stable - consistent behavior patterns\")
        elif np.mean(stabilities) < 0.3:
            insights.append(\"🌪️ Drive system is unstable - chaotic behavior patterns\")
        
        # Competition insights
        if self.competition_matrix:
            max_competition = max(self.competition_matrix.values())
            if max_competition > 0.7:
                insights.append(\"⚔️ High drive competition detected - drives strongly oppose each other\")
        
        # Cooperation insights
        if self.cooperation_matrix:
            max_cooperation = max(self.cooperation_matrix.values())
            if max_cooperation > 0.7:
                insights.append(\"🤝 Strong drive cooperation detected - drives work together effectively\")
        
        # Switching insights
        switching_rates = [s.drive_switching_rate for s in self.snapshots]
        if np.mean(switching_rates) > 0.5:
            insights.append(\"🔄 High drive switching rate - brain rapidly changes priorities\")
        
        # Effectiveness insights
        if self.snapshots:
            latest = self.snapshots[-1]
            if latest.drive_outcome_correlation:
                best_drive = max(latest.drive_outcome_correlation.keys(), 
                               key=lambda k: latest.drive_outcome_correlation[k])
                insights.append(f"🏆 {best_drive} drive is most effective for positive outcomes")
        
        return insights
    
    def _generate_drive_recommendations(self) -> List[str]:
        \"\"\"Generate drive optimization recommendations.\"\"\"
        recommendations = []
        
        if len(self.snapshots) < 20:
            return recommendations
        
        # Stability recommendations
        stabilities = [s.drive_stability for s in self.snapshots]
        if np.mean(stabilities) < 0.3:
            recommendations.append(\"🔧 Drive system is too unstable - consider increasing drive weight persistence\")
        
        # Competition recommendations
        if self.competition_matrix:
            max_competition = max(self.competition_matrix.values())
            if max_competition > 0.8:
                recommendations.append(\"🔧 Excessive drive competition - consider balancing drive weight calculations\")
        
        # Effectiveness recommendations
        if self.snapshots:
            latest = self.snapshots[-1]
            if latest.drive_outcome_correlation:
                ineffective_drives = [k for k, v in latest.drive_outcome_correlation.items() if v < 0.1]
                if ineffective_drives:
                    recommendations.append(f"🔧 {', '.join(ineffective_drives)} drives are ineffective - consider tuning parameters")
        
        # Switching recommendations
        switching_rates = [s.drive_switching_rate for s in self.snapshots]
        if np.mean(switching_rates) > 0.7:
            recommendations.append(\"🔧 Drive switching too frequent - consider increasing decision persistence\")
        
        return recommendations
    
    def save_drive_analysis_report(self) -> str:
        \"\"\"Save comprehensive drive analysis report.\"\"\"
        analysis = self.analyze_drive_interactions()
        
        report_file = self.output_dir / f"{self.session_name}_drive_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📊 Drive correlation analysis saved: {report_file}")
        return str(report_file)
    
    def save_drive_snapshots(self) -> str:
        \"\"\"Save all drive snapshots.\"\"\"
        snapshots_file = self.output_dir / f"{self.session_name}_drive_snapshots.json"
        
        snapshot_data = {
            \"session_name\": self.session_name,
            \"snapshots\": [asdict(snapshot) for snapshot in self.snapshots]
        }
        
        with open(snapshots_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        
        print(f"📸 Drive snapshots saved: {snapshots_file}")
        return str(snapshots_file)
    
    def print_drive_status(self):
        \"\"\"Print current drive interaction status.\"\"\"
        if not self.snapshots:
            print(\"🔄 No drive snapshots captured yet\")
            return
        
        latest = self.snapshots[-1]
        print(f"\n🔄 DRIVE INTERACTION STATUS")
        print(f"   Session: {self.session_name}")
        print(f"   Step: {latest.step_count} | Dominant: {latest.dominant_motivator}")
        print(f"   Drive Pressure: {latest.total_motivator_pressure:.3f} | Stability: {latest.drive_stability:.3f}")
        print(f"   Switching Rate: {latest.drive_switching_rate:.3f} | Entropy: {latest.drive_entropy:.3f}")
        print(f"   Drive Weights: {', '.join(f'{k}:{v:.2f}' for k, v in latest.motivator_weights.items())}")
        
        if latest.drive_outcome_correlation:
            best_drive = max(latest.drive_outcome_correlation.keys(), 
                           key=lambda k: latest.drive_outcome_correlation[k])
            print(f"   Most Effective: {best_drive} ({latest.drive_outcome_correlation[best_drive]:.3f})")