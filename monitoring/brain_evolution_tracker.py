"""
Brain State Evolution Tracker - Advanced debugging system for analyzing brain development.

This tracker monitors how the brain evolves over time, tracking:
- Neural network growth patterns
- Drive system maturation
- Learning velocity and acceleration
- Behavioral complexity emergence
- Memory consolidation patterns
- Prediction accuracy evolution

Designed for radical brain optimization through deep insight into cognitive development.
"""

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


@dataclass
class BrainSnapshot:
    """Snapshot of brain state at a specific moment."""
    timestamp: float
    step_count: int
    
    # Neural structure
    total_nodes: int
    node_growth_rate: float
    network_density: float
    
    # Drive system
    motivator_weights: Dict[str, float]
    motivator_pressure: float
    dominant_motivator: str
    drive_entropy: float  # Measure of drive system complexity
    
    # Learning metrics
    prediction_error: float
    learning_velocity: float  # Rate of error improvement
    learning_acceleration: float  # Change in learning velocity
    
    # Behavioral complexity
    action_diversity: float
    decision_confidence: float
    exploration_radius: float
    
    # Memory patterns
    memory_efficiency: float
    consolidation_rate: float
    forgetting_rate: float
    
    # Performance metrics
    thinking_time: float
    fps: float
    
    # Robot state
    robot_position: Tuple[float, float]
    robot_health: float
    robot_energy: float


class BrainEvolutionTracker:
    """
    Advanced brain evolution monitoring system.
    
    Tracks brain development patterns and provides insights for optimization.
    """
    
    def __init__(self, session_name: str = None, track_every_n_steps: int = 10):
        """Initialize the brain evolution tracker."""
        self.session_name = session_name or f"brain_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.track_every_n_steps = track_every_n_steps
        
        # Storage
        self.snapshots: List[BrainSnapshot] = []
        self.snapshot_history = deque(maxlen=1000)  # Keep last 1000 snapshots
        
        # Analysis caches
        self.learning_velocity_history = deque(maxlen=100)
        self.drive_correlation_matrix = defaultdict(list)
        self.behavioral_complexity_trend = deque(maxlen=200)
        
        # Performance tracking
        self.last_snapshot_time = time.time()
        self.step_counter = 0
        
        # Output paths
        self.output_dir = Path("logs/brain_evolution")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 Brain Evolution Tracker initialized: {self.session_name}")
        print(f"   Tracking every {track_every_n_steps} steps")
        print(f"   Output directory: {self.output_dir}")
    
    def capture_snapshot(self, brain_stats: Dict[str, Any], decision_context: Dict[str, Any],
                        step_count: int) -> Optional[BrainSnapshot]:
        """Capture a comprehensive brain state snapshot."""
        self.step_counter += 1
        
        # Only capture every N steps to avoid overhead
        if self.step_counter % self.track_every_n_steps != 0:
            return None
        
        current_time = time.time()
        
        try:
            # Extract neural structure metrics
            graph_stats = brain_stats.get('graph_stats', {})
            total_nodes = graph_stats.get('total_nodes', 0)
            
            # Calculate node growth rate
            node_growth_rate = self._calculate_node_growth_rate(total_nodes)
            network_density = self._calculate_network_density(graph_stats)
            
            # Extract drive system metrics
            motivator_weights = decision_context.get('motivator_weights', {})
            motivator_pressure = decision_context.get('total_motivator_pressure', 0.0)
            dominant_motivator = decision_context.get('dominant_motivator', 'unknown')
            drive_entropy = self._calculate_drive_entropy(motivator_weights)
            
            # Extract learning metrics
            prediction_error = decision_context.get('recent_prediction_error', 0.0)
            learning_velocity = self._calculate_learning_velocity(prediction_error)
            learning_acceleration = self._calculate_learning_acceleration(learning_velocity)
            
            # Extract behavioral complexity metrics
            action_diversity = self._calculate_action_diversity(decision_context)
            decision_confidence = decision_context.get('confidence', 0.0)
            exploration_radius = self._calculate_exploration_radius(decision_context)
            
            # Extract memory patterns
            memory_efficiency = self._calculate_memory_efficiency(brain_stats)
            consolidation_rate = self._calculate_consolidation_rate(brain_stats)
            forgetting_rate = self._calculate_forgetting_rate(brain_stats)
            
            # Extract performance metrics
            thinking_time = brain_stats.get('predictor_stats', {}).get('average_thinking_time', 0.0)
            fps = 1.0 / (current_time - self.last_snapshot_time) if self.last_snapshot_time else 0.0
            
            # Extract robot state
            robot_position = decision_context.get('robot_position', (0, 0))
            robot_health = decision_context.get('robot_health', 0.0)
            robot_energy = decision_context.get('robot_energy', 0.0)
            
            # Create snapshot
            snapshot = BrainSnapshot(
                timestamp=current_time,
                step_count=step_count,
                total_nodes=total_nodes,
                node_growth_rate=node_growth_rate,
                network_density=network_density,
                motivator_weights=motivator_weights,
                motivator_pressure=motivator_pressure,
                dominant_motivator=dominant_motivator,
                drive_entropy=drive_entropy,
                prediction_error=prediction_error,
                learning_velocity=learning_velocity,
                learning_acceleration=learning_acceleration,
                action_diversity=action_diversity,
                decision_confidence=decision_confidence,
                exploration_radius=exploration_radius,
                memory_efficiency=memory_efficiency,
                consolidation_rate=consolidation_rate,
                forgetting_rate=forgetting_rate,
                thinking_time=thinking_time,
                fps=fps,
                robot_position=robot_position,
                robot_health=robot_health,
                robot_energy=robot_energy
            )
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.snapshot_history.append(snapshot)
            
            # Update analysis caches
            self._update_analysis_caches(snapshot)
            
            self.last_snapshot_time = current_time
            return snapshot
            
        except Exception as e:
            print(f"Error capturing brain snapshot: {e}")
            return None
    
    def _calculate_node_growth_rate(self, current_nodes: int) -> float:
        """Calculate the rate of neural network growth."""
        if len(self.snapshots) < 2:
            return 0.0
        
        prev_nodes = self.snapshots[-1].total_nodes
        time_diff = time.time() - self.snapshots[-1].timestamp
        
        if time_diff > 0:
            return (current_nodes - prev_nodes) / time_diff
        return 0.0
    
    def _calculate_network_density(self, graph_stats: Dict[str, Any]) -> float:
        """Calculate network density (connections per node)."""
        total_nodes = graph_stats.get('total_nodes', 0)
        if total_nodes == 0:
            return 0.0
        
        # Estimate connections from average access count
        avg_access_count = graph_stats.get('avg_access_count', 0)
        return avg_access_count / total_nodes if total_nodes > 0 else 0.0
    
    def _calculate_drive_entropy(self, motivator_weights: Dict[str, float]) -> float:
        """Calculate entropy of drive system (higher = more complex behavior)."""
        if not motivator_weights:
            return 0.0
        
        # Normalize weights
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
    
    def _calculate_learning_velocity(self, current_error: float) -> float:
        """Calculate learning velocity (rate of error improvement)."""
        if len(self.snapshots) < 2:
            return 0.0
        
        prev_error = self.snapshots[-1].prediction_error
        time_diff = time.time() - self.snapshots[-1].timestamp
        
        if time_diff > 0:
            velocity = (prev_error - current_error) / time_diff  # Positive = learning
            self.learning_velocity_history.append(velocity)
            return velocity
        return 0.0
    
    def _calculate_learning_acceleration(self, current_velocity: float) -> float:
        """Calculate learning acceleration (change in learning velocity)."""
        if len(self.learning_velocity_history) < 2:
            return 0.0
        
        prev_velocity = self.learning_velocity_history[-2]
        time_diff = time.time() - self.snapshots[-1].timestamp
        
        if time_diff > 0:
            return (current_velocity - prev_velocity) / time_diff
        return 0.0
    
    def _calculate_action_diversity(self, decision_context: Dict[str, Any]) -> float:
        """Calculate diversity of actions taken."""
        chosen_action = decision_context.get('chosen_action', {})
        if not chosen_action:
            return 0.0
        
        # Calculate action magnitude and diversity
        action_magnitudes = [abs(v) for v in chosen_action.values() if isinstance(v, (int, float))]
        if not action_magnitudes:
            return 0.0
        
        # Use coefficient of variation as diversity measure
        mean_mag = np.mean(action_magnitudes)
        std_mag = np.std(action_magnitudes)
        
        return std_mag / mean_mag if mean_mag > 0 else 0.0
    
    def _calculate_exploration_radius(self, decision_context: Dict[str, Any]) -> float:
        """Calculate exploration radius (how far robot has traveled)."""
        current_pos = decision_context.get('robot_position', (0, 0))
        
        if len(self.snapshots) == 0:
            return 0.0
        
        # Calculate distance from starting position
        start_pos = self.snapshots[0].robot_position
        return np.sqrt((current_pos[0] - start_pos[0])**2 + (current_pos[1] - start_pos[1])**2)
    
    def _calculate_memory_efficiency(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate memory system efficiency."""
        graph_stats = brain_stats.get('graph_stats', {})
        total_nodes = graph_stats.get('total_nodes', 0)
        total_merges = graph_stats.get('total_merges', 0)
        
        if total_nodes == 0:
            return 0.0
        
        # Efficiency = how much consolidation is happening
        return total_merges / total_nodes if total_nodes > 0 else 0.0
    
    def _calculate_consolidation_rate(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate memory consolidation rate."""
        consolidation_stats = brain_stats.get('consolidation_stats', {})
        return consolidation_stats.get('consolidation_rate', 0.0)
    
    def _calculate_forgetting_rate(self, brain_stats: Dict[str, Any]) -> float:
        """Calculate forgetting rate (how much old information is discarded)."""
        # This would need specific forgetting metrics from the brain
        # For now, return 0 (no forgetting implemented)
        return 0.0
    
    def _update_analysis_caches(self, snapshot: BrainSnapshot):
        """Update analysis caches with new snapshot."""
        # Update behavioral complexity trend
        complexity_score = (
            snapshot.drive_entropy * 0.3 +
            snapshot.action_diversity * 0.3 +
            snapshot.decision_confidence * 0.2 +
            snapshot.exploration_radius / 100.0 * 0.2  # Normalize exploration
        )
        self.behavioral_complexity_trend.append(complexity_score)
        
        # Update drive correlation matrix
        for drive_name, weight in snapshot.motivator_weights.items():
            self.drive_correlation_matrix[drive_name].append(weight)
    
    def analyze_brain_evolution(self) -> Dict[str, Any]:
        """Analyze brain evolution patterns and return insights."""
        if len(self.snapshots) < 10:
            return {"error": "Insufficient data for analysis (need at least 10 snapshots)"}
        
        analysis = {
            "session_metadata": {
                "session_name": self.session_name,
                "total_snapshots": len(self.snapshots),
                "time_span": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
                "step_span": self.snapshots[-1].step_count - self.snapshots[0].step_count
            },
            
            "neural_development": self._analyze_neural_development(),
            "drive_system_evolution": self._analyze_drive_system_evolution(),
            "learning_dynamics": self._analyze_learning_dynamics(),
            "behavioral_emergence": self._analyze_behavioral_emergence(),
            "memory_patterns": self._analyze_memory_patterns(),
            "performance_trends": self._analyze_performance_trends(),
            "robot_development": self._analyze_robot_development(),
            
            "insights": self._generate_evolution_insights()
        }
        
        return analysis
    
    def _analyze_neural_development(self) -> Dict[str, Any]:
        """Analyze neural network growth patterns."""
        node_counts = [s.total_nodes for s in self.snapshots]
        growth_rates = [s.node_growth_rate for s in self.snapshots]
        
        return {
            "initial_nodes": node_counts[0],
            "final_nodes": node_counts[-1],
            "total_growth": node_counts[-1] - node_counts[0],
            "average_growth_rate": np.mean(growth_rates),
            "growth_acceleration": np.mean(np.diff(growth_rates)),
            "growth_pattern": "exponential" if np.mean(np.diff(growth_rates)) > 0 else "linear"
        }
    
    def _analyze_drive_system_evolution(self) -> Dict[str, Any]:
        """Analyze how the drive system evolves over time."""
        motivator_pressures = [s.motivator_pressure for s in self.snapshots]
        drive_entropies = [s.drive_entropy for s in self.snapshots]
        
        # Analyze drive dominance patterns
        drive_dominance = defaultdict(int)
        for snapshot in self.snapshots:
            drive_dominance[snapshot.dominant_motivator] += 1
        
        return {
            "pressure_trend": "increasing" if motivator_pressures[-1] > motivator_pressures[0] else "decreasing",
            "average_pressure": np.mean(motivator_pressures),
            "pressure_stability": 1.0 / (1.0 + np.std(motivator_pressures)),
            "entropy_trend": "increasing" if drive_entropies[-1] > drive_entropies[0] else "decreasing",
            "average_entropy": np.mean(drive_entropies),
            "dominant_motivator_distribution": dict(drive_dominance),
            "drive_switching_frequency": len(set(s.dominant_motivator for s in self.snapshots))
        }
    
    def _analyze_learning_dynamics(self) -> Dict[str, Any]:
        """Analyze learning velocity and acceleration patterns."""
        errors = [s.prediction_error for s in self.snapshots]
        velocities = [s.learning_velocity for s in self.snapshots]
        accelerations = [s.learning_acceleration for s in self.snapshots]
        
        return {
            "initial_error": errors[0],
            "final_error": errors[-1],
            "total_improvement": errors[0] - errors[-1],
            "average_velocity": np.mean(velocities),
            "velocity_stability": 1.0 / (1.0 + np.std(velocities)),
            "average_acceleration": np.mean(accelerations),
            "learning_plateau_detected": np.std(errors[-10:]) < 0.001 if len(errors) >= 10 else False,
            "learning_efficiency": (errors[0] - errors[-1]) / len(self.snapshots) if len(self.snapshots) > 0 else 0.0
        }
    
    def _analyze_behavioral_emergence(self) -> Dict[str, Any]:
        """Analyze emergence of complex behaviors."""
        complexities = list(self.behavioral_complexity_trend)
        confidences = [s.decision_confidence for s in self.snapshots]
        diversities = [s.action_diversity for s in self.snapshots]
        
        return {
            "complexity_trend": "increasing" if complexities[-1] > complexities[0] else "decreasing",
            "average_complexity": np.mean(complexities),
            "complexity_growth_rate": (complexities[-1] - complexities[0]) / len(complexities),
            "confidence_trend": "increasing" if confidences[-1] > confidences[0] else "decreasing",
            "average_confidence": np.mean(confidences),
            "action_diversity_trend": "increasing" if diversities[-1] > diversities[0] else "decreasing",
            "behavior_maturity": min(1.0, np.mean(complexities) * 2.0)  # Normalize to 0-1
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory consolidation and efficiency patterns."""
        efficiencies = [s.memory_efficiency for s in self.snapshots]
        consolidation_rates = [s.consolidation_rate for s in self.snapshots]
        
        return {
            "average_efficiency": np.mean(efficiencies),
            "efficiency_trend": "increasing" if efficiencies[-1] > efficiencies[0] else "decreasing",
            "average_consolidation_rate": np.mean(consolidation_rates),
            "consolidation_stability": 1.0 / (1.0 + np.std(consolidation_rates)),
            "memory_optimization": "active" if np.mean(efficiencies) > 0.1 else "minimal"
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze computational performance trends."""
        thinking_times = [s.thinking_time for s in self.snapshots]
        fps_values = [s.fps for s in self.snapshots]
        
        return {
            "average_thinking_time": np.mean(thinking_times),
            "thinking_time_trend": "increasing" if thinking_times[-1] > thinking_times[0] else "decreasing",
            "average_fps": np.mean(fps_values),
            "fps_stability": 1.0 / (1.0 + np.std(fps_values)),
            "performance_efficiency": 1.0 / (1.0 + np.mean(thinking_times))
        }
    
    def _analyze_robot_development(self) -> Dict[str, Any]:
        """Analyze robot's physical development and exploration."""
        positions = [s.robot_position for s in self.snapshots]
        healths = [s.robot_health for s in self.snapshots]
        energies = [s.robot_energy for s in self.snapshots]
        exploration_radii = [s.exploration_radius for s in self.snapshots]
        
        return {
            "starting_position": positions[0],
            "final_position": positions[-1],
            "total_distance_traveled": sum(np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                                 (positions[i][1] - positions[i-1][1])**2) 
                                         for i in range(1, len(positions))),
            "exploration_radius": exploration_radii[-1],
            "average_health": np.mean(healths),
            "health_trend": "improving" if healths[-1] > healths[0] else "declining",
            "average_energy": np.mean(energies),
            "energy_trend": "improving" if energies[-1] > energies[0] else "declining",
            "survival_success": np.mean(healths) > 0.5 and np.mean(energies) > 0.3
        }
    
    def _generate_evolution_insights(self) -> List[str]:
        """Generate high-level insights about brain evolution."""
        insights = []
        
        if len(self.snapshots) < 10:
            insights.append("⚠️ Insufficient data for meaningful insights")
            return insights
        
        # Learning insights
        errors = [s.prediction_error for s in self.snapshots]
        if errors[-1] < errors[0] * 0.5:
            insights.append("🎯 Strong learning progress - prediction error reduced by >50%")
        elif errors[-1] > errors[0] * 1.5:
            insights.append("⚠️ Learning regression detected - error increased significantly")
        
        # Drive system insights
        motivator_pressures = [s.motivator_pressure for s in self.snapshots]
        if min(motivator_pressures) < 0.1:
            insights.append("😴 Homeostatic rest achieved - drive pressure reached low levels")
        elif np.std(motivator_pressures) > 0.3:
            insights.append("🔄 Highly dynamic drive system - pressure varies significantly")
        
        # Behavioral insights
        complexities = list(self.behavioral_complexity_trend)
        if complexities[-1] > complexities[0] * 1.5:
            insights.append("🌟 Behavioral complexity emerging - actions becoming more sophisticated")
        
        # Performance insights
        thinking_times = [s.thinking_time for s in self.snapshots]
        if thinking_times[-1] < thinking_times[0] * 0.7:
            insights.append("⚡ Thinking efficiency improved - faster decision making")
        
        # Neural development insights
        node_counts = [s.total_nodes for s in self.snapshots]
        if node_counts[-1] > node_counts[0] * 3:
            insights.append("🧠 Rapid neural growth - network expanded significantly")
        
        return insights
    
    def save_evolution_report(self) -> str:
        """Save comprehensive evolution report to file."""
        analysis = self.analyze_brain_evolution()
        
        report_file = self.output_dir / f"{self.session_name}_evolution_report.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📊 Brain evolution report saved: {report_file}")
        return str(report_file)
    
    def save_snapshots(self) -> str:
        """Save all snapshots to file."""
        snapshots_file = self.output_dir / f"{self.session_name}_snapshots.json"
        
        snapshot_data = {
            "session_name": self.session_name,
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots]
        }
        
        with open(snapshots_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2, default=str)
        
        print(f"📸 Brain snapshots saved: {snapshots_file}")
        return str(snapshots_file)
    
    def print_live_status(self):
        """Print current brain evolution status."""
        if not self.snapshots:
            print("🧠 No brain snapshots captured yet")
            return
        
        latest = self.snapshots[-1]
        print(f"\n🧠 BRAIN EVOLUTION STATUS")
        print(f"   Session: {self.session_name}")
        print(f"   Step: {latest.step_count} | Nodes: {latest.total_nodes}")
        print(f"   Drive Pressure: {latest.motivator_pressure:.3f} | Dominant: {latest.dominant_motivator}")
        print(f"   Learning Velocity: {latest.learning_velocity:.3f} | Error: {latest.prediction_error:.3f}")
        print(f"   Behavioral Complexity: {list(self.behavioral_complexity_trend)[-1]:.3f}")
        print(f"   Position: ({latest.robot_position[0]:.1f}, {latest.robot_position[1]:.1f})")
        print(f"   Health: {latest.robot_health:.1%} | Energy: {latest.robot_energy:.1%}")