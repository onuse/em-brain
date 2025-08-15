#!/usr/bin/env python3
"""
Standalone Embodied Learning Experiment

A simplified version of the biological embodied learning experiment that runs
the brain directly without client-server complexity. This tests whether the
brain actually learns and improves over time in an embodied environment.

Key Questions:
1. Does prediction error decrease over time?
2. Does the brain develop better navigation strategies?
3. Do emergent behaviors actually emerge?
4. Is there evidence of learning consolidation?
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add paths for brain import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.brain import MinimalBrain

@dataclass
class LearningMetrics:
    """Metrics to track learning progress."""
    cycle: int
    timestamp: float
    prediction_error: float
    prediction_confidence: float
    position_x: float
    position_y: float
    target_distance: float
    collision: bool
    action_taken: List[float]
    brain_architecture: str

class SimpleEmbodiedWorld:
    """A simple 2D world for embodied learning experiments."""
    
    def __init__(self, size: float = 10.0, random_seed: int = 42):
        np.random.seed(random_seed)
        self.size = size
        self.reset()
        
        # Fixed obstacles for consistent testing
        self.obstacles = [
            (2.0, 2.0, 1.0),  # (x, y, radius)
            (7.0, 7.0, 1.0),
            (3.0, 8.0, 0.8),
            (8.0, 3.0, 0.8),
        ]
        
        # Light source (target)
        self.target_x = 8.5
        self.target_y = 8.5
        
        print(f"🌍 Created {size}x{size} world with {len(self.obstacles)} obstacles")
        print(f"   Target at ({self.target_x:.1f}, {self.target_y:.1f})")
    
    def reset(self):
        """Reset robot to starting position."""
        self.robot_x = 1.5
        self.robot_y = 1.5
        self.robot_heading = 0.0  # radians
        
    def get_sensory_input(self) -> List[float]:
        """Get 8-channel sensory input: distance sensors + target info."""
        # 6 distance sensors in different directions
        angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]  # 60° apart
        
        sensors = []
        for angle in angles:
            abs_angle = self.robot_heading + angle
            distance = self._raycast_distance(abs_angle)
            sensors.append(min(distance / 5.0, 1.0))  # Normalize to 0-1, max range 5m
        
        # Add target direction and distance
        target_dx = self.target_x - self.robot_x
        target_dy = self.target_y - self.robot_y
        target_distance = np.sqrt(target_dx**2 + target_dy**2)
        target_angle = np.arctan2(target_dy, target_dx) - self.robot_heading
        
        # Normalize target info
        sensors.append(np.sin(target_angle))  # Target direction X
        sensors.append(np.cos(target_angle))  # Target direction Y
        
        return sensors
    
    def _raycast_distance(self, angle: float) -> float:
        """Cast ray and return distance to nearest obstacle or wall."""
        step_size = 0.1
        max_distance = 5.0
        
        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size
        
        x, y = self.robot_x, self.robot_y
        distance = 0.0
        
        while distance < max_distance:
            x += dx
            y += dy
            distance += step_size
            
            # Check wall collision
            if x <= 0 or x >= self.size or y <= 0 or y >= self.size:
                return distance
            
            # Check obstacle collision
            for obs_x, obs_y, radius in self.obstacles:
                if np.sqrt((x - obs_x)**2 + (y - obs_y)**2) <= radius:
                    return distance
        
        return max_distance
    
    def execute_action(self, action: List[float]) -> Tuple[bool, float]:
        """Execute action and return (collision, target_distance)."""
        # Interpret action as [forward_speed, turn_rate]
        if len(action) < 2:
            action = action + [0.0] * (2 - len(action))
        
        forward_speed = np.clip(action[0], -1.0, 1.0)
        turn_rate = np.clip(action[1], -1.0, 1.0)
        
        # Update robot state
        dt = 0.1  # 100ms time step
        self.robot_heading += turn_rate * dt * 2.0  # Max 2 rad/s turn rate
        
        # Move forward
        dx = np.cos(self.robot_heading) * forward_speed * dt * 2.0  # Max 2 m/s speed
        dy = np.sin(self.robot_heading) * forward_speed * dt * 2.0
        
        new_x = self.robot_x + dx
        new_y = self.robot_y + dy
        
        # Check for collisions before moving
        collision = False
        
        # Wall collision
        if new_x <= 0.2 or new_x >= self.size - 0.2 or new_y <= 0.2 or new_y >= self.size - 0.2:
            collision = True
        
        # Obstacle collision
        for obs_x, obs_y, radius in self.obstacles:
            if np.sqrt((new_x - obs_x)**2 + (new_y - obs_y)**2) <= radius + 0.2:
                collision = True
                break
        
        # Update position only if no collision
        if not collision:
            self.robot_x = new_x
            self.robot_y = new_y
        
        # Calculate distance to target
        target_distance = np.sqrt((self.robot_x - self.target_x)**2 + 
                                (self.robot_y - self.target_y)**2)
        
        return collision, target_distance

class StandaloneEmbodiedLearning:
    """Standalone embodied learning experiment."""
    
    def __init__(self, brain_type: str = "sparse_goldilocks", random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize brain
        print(f"🧠 Initializing {brain_type} brain...")
        self.brain = MinimalBrain(
            brain_type=brain_type,
            sensory_dim=8,  # 6 distance sensors + 2 target info
            motor_dim=2,    # forward_speed, turn_rate
            temporal_dim=4,
            quiet_mode=False
        )
        
        # Initialize world
        self.world = SimpleEmbodiedWorld(size=10.0, random_seed=random_seed)
        
        # Metrics tracking
        self.metrics: List[LearningMetrics] = []
        self.cycle_count = 0
        
        print(f"✅ Standalone embodied learning experiment ready")
    
    def run_experiment(self, duration_minutes: float = 30, save_results: bool = True) -> Dict:
        """Run the embodied learning experiment."""
        print(f"🔬 Starting {duration_minutes}-minute embodied learning experiment...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        last_report_time = start_time
        report_interval = 60  # Report every minute
        
        try:
            while time.time() < end_time:
                # Get current sensory input
                sensory_input = self.world.get_sensory_input()
                
                # Process through brain
                action, brain_state = self.brain.process_sensory_input(sensory_input, action_dimensions=2)
                
                # Execute action in world
                collision, target_distance = self.world.execute_action(action)
                
                # Store experience in brain (for learning)
                outcome = sensory_input.copy()  # Next sensory state as outcome
                self.brain.store_experience(sensory_input, action, outcome, action)
                
                # Record metrics
                metrics = LearningMetrics(
                    cycle=self.cycle_count,
                    timestamp=time.time() - start_time,
                    prediction_error=brain_state.get('prediction_error', 0.5),
                    prediction_confidence=brain_state.get('prediction_confidence', 0.5),
                    position_x=self.world.robot_x,
                    position_y=self.world.robot_y,
                    target_distance=target_distance,
                    collision=collision,
                    action_taken=action.copy(),
                    brain_architecture=brain_state.get('architecture', 'unknown')
                )
                self.metrics.append(metrics)
                
                self.cycle_count += 1
                
                # Reset robot if it gets too close to target or stuck
                if target_distance < 0.5 or self.cycle_count % 500 == 0:
                    self.world.reset()
                
                # Progress report
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    self._print_progress_report(current_time - start_time, duration_minutes * 60)
                    last_report_time = current_time
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\\n⚠️ Experiment interrupted by user")
        
        print(f"\\n✅ Experiment completed: {self.cycle_count} cycles in {time.time() - start_time:.1f}s")
        
        # Analyze results
        results = self._analyze_results()
        
        if save_results:
            self._save_results(results)
        
        return results
    
    def _print_progress_report(self, elapsed_seconds: float, total_seconds: float):
        """Print progress report."""
        if not self.metrics:
            return
        
        # Get recent metrics (last 60 cycles)
        recent_metrics = self.metrics[-60:] if len(self.metrics) >= 60 else self.metrics
        
        avg_error = np.mean([m.prediction_error for m in recent_metrics])
        avg_confidence = np.mean([m.prediction_confidence for m in recent_metrics])
        avg_distance = np.mean([m.target_distance for m in recent_metrics])
        collision_rate = np.mean([m.collision for m in recent_metrics])
        
        progress_pct = (elapsed_seconds / total_seconds) * 100
        
        print(f"[{elapsed_seconds/60:.1f}min/{total_seconds/60:.0f}min {progress_pct:.0f}%] " +
              f"Cycles: {self.cycle_count:,} | " +
              f"Error: {avg_error:.3f} | " +
              f"Confidence: {avg_confidence:.3f} | " +
              f"Target dist: {avg_distance:.2f}m | " +
              f"Collisions: {collision_rate:.1%}")
    
    def _analyze_results(self) -> Dict:
        """Analyze learning results."""
        print("\\n📊 Analyzing learning results...")
        
        if len(self.metrics) < 10:
            return {"error": "Insufficient data for analysis"}
        
        # Convert to numpy arrays for analysis
        cycles = np.array([m.cycle for m in self.metrics])
        timestamps = np.array([m.timestamp for m in self.metrics])
        errors = np.array([m.prediction_error for m in self.metrics])
        confidences = np.array([m.prediction_confidence for m in self.metrics])
        distances = np.array([m.target_distance for m in self.metrics])
        collisions = np.array([m.collision for m in self.metrics], dtype=float)
        
        # Learning trend analysis
        def calculate_trend(values, window_size=100):
            if len(values) < window_size * 2:
                return 0.0
            early = np.mean(values[:window_size])
            late = np.mean(values[-window_size:])
            return (late - early) / early if early != 0 else 0.0
        
        error_trend = calculate_trend(errors)
        confidence_trend = calculate_trend(confidences)
        distance_trend = calculate_trend(distances)
        collision_trend = calculate_trend(collisions)
        
        # Performance metrics
        final_performance = {
            "avg_prediction_error": float(np.mean(errors[-100:])) if len(errors) >= 100 else float(np.mean(errors)),
            "avg_confidence": float(np.mean(confidences[-100:])) if len(confidences) >= 100 else float(np.mean(confidences)),
            "avg_target_distance": float(np.mean(distances[-100:])) if len(distances) >= 100 else float(np.mean(distances)),
            "collision_rate": float(np.mean(collisions[-100:])) if len(collisions) >= 100 else float(np.mean(collisions))
        }
        
        # Learning evidence
        learning_evidence = {
            "prediction_error_trend": float(error_trend),  # Negative = improving
            "confidence_trend": float(confidence_trend),    # Positive = improving
            "navigation_trend": float(distance_trend),      # Negative = improving (closer to target)
            "collision_trend": float(collision_trend),      # Negative = improving (fewer collisions)
        }
        
        # Overall learning score (-1 to +1, positive = learning detected)
        learning_score = (
            -error_trend * 0.3 +      # Lower error = better
            confidence_trend * 0.3 +   # Higher confidence = better
            -distance_trend * 0.2 +    # Closer to target = better
            -collision_trend * 0.2     # Fewer collisions = better
        )
        
        results = {
            "experiment_info": {
                "total_cycles": int(self.cycle_count),
                "duration_minutes": float(timestamps[-1] / 60) if len(timestamps) > 0 else 0,
                "brain_architecture": self.metrics[0].brain_architecture if self.metrics else "unknown",
                "random_seed": self.random_seed
            },
            "final_performance": final_performance,
            "learning_evidence": learning_evidence,
            "learning_score": float(learning_score),
            "learning_detected": learning_score > 0.1,  # Threshold for claiming learning
        }
        
        # Print summary
        print(f"\\n🎯 LEARNING ANALYSIS SUMMARY:")
        print(f"   Total cycles: {results['experiment_info']['total_cycles']:,}")
        print(f"   Duration: {results['experiment_info']['duration_minutes']:.1f} minutes")
        print(f"   Brain: {results['experiment_info']['brain_architecture']}")
        print(f"\\n📈 LEARNING EVIDENCE:")
        print(f"   Prediction error trend: {learning_evidence['prediction_error_trend']:+.3f} (negative = improving)")
        print(f"   Confidence trend: {learning_evidence['confidence_trend']:+.3f} (positive = improving)")
        print(f"   Navigation trend: {learning_evidence['navigation_trend']:+.3f} (negative = improving)")
        print(f"   Collision trend: {learning_evidence['collision_trend']:+.3f} (negative = improving)")
        print(f"\\n🏆 OVERALL LEARNING SCORE: {learning_score:+.3f}")
        
        if results['learning_detected']:
            print("   ✅ LEARNING DETECTED - Brain shows improvement over time!")
        else:
            print("   ❌ NO CLEAR LEARNING - Brain performance appears static")
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results to file."""
        # Create results directory
        results_dir = Path("validation/embodied_learning/reports")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"standalone_embodied_learning_{timestamp}.json"
        
        # Add raw metrics to results (convert numpy types to Python types for JSON)
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj
        
        raw_metrics = []
        for m in self.metrics:
            metric_dict = asdict(m)
            # Convert all values to JSON-serializable types
            for key, value in metric_dict.items():
                metric_dict[key] = convert_for_json(value)
            raw_metrics.append(metric_dict)
        
        results["raw_metrics"] = raw_metrics
        
        # Convert entire results dict for JSON compatibility
        def convert_dict_for_json(d):
            if isinstance(d, dict):
                return {k: convert_dict_for_json(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict_for_json(v) for v in d]
            else:
                return convert_for_json(d)
        
        json_results = convert_dict_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Generate plots if we have enough data
        if len(self.metrics) > 50:
            self._generate_plots(results_dir / f"learning_plots_{timestamp}.png")
    
    def _generate_plots(self, output_file: Path):
        """Generate learning progression plots."""
        try:
            import matplotlib.pyplot as plt
            
            timestamps = [m.timestamp / 60 for m in self.metrics]  # Convert to minutes
            errors = [m.prediction_error for m in self.metrics]
            confidences = [m.prediction_confidence for m in self.metrics]
            distances = [m.target_distance for m in self.metrics]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Embodied Learning Progression', fontsize=16)
            
            # Plot 1: Prediction Error
            ax1.plot(timestamps, errors, alpha=0.7, color='red')
            ax1.set_title('Prediction Error Over Time')
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Prediction Error')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Prediction Confidence
            ax2.plot(timestamps, confidences, alpha=0.7, color='blue')
            ax2.set_title('Prediction Confidence Over Time')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Confidence')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Target Distance
            ax3.plot(timestamps, distances, alpha=0.7, color='green')
            ax3.set_title('Distance to Target Over Time')
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('Distance (m)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Robot Trajectory
            positions_x = [m.position_x for m in self.metrics[::10]]  # Sample every 10th point
            positions_y = [m.position_y for m in self.metrics[::10]]
            ax4.plot(positions_x, positions_y, alpha=0.6, color='purple')
            ax4.scatter([self.world.target_x], [self.world.target_y], color='gold', s=100, marker='*', label='Target')
            for obs_x, obs_y, radius in self.world.obstacles:
                circle = plt.Circle((obs_x, obs_y), radius, color='black', alpha=0.3)
                ax4.add_patch(circle)
            ax4.set_title('Robot Trajectory')
            ax4.set_xlabel('X Position (m)')
            ax4.set_ylabel('Y Position (m)')
            ax4.set_xlim(0, self.world.size)
            ax4.set_ylim(0, self.world.size)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Learning plots saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")

def main():
    """Run standalone embodied learning experiment."""
    print("🤖 STANDALONE EMBODIED LEARNING EXPERIMENT")
    print("=" * 60)
    
    # Test minimal brain first (sparse_goldilocks has dimension issues)
    for brain_type in ["minimal"]:  # "sparse_goldilocks" has tensor dimension bugs
        print(f"\\n🧠 Testing {brain_type} brain architecture...")
        
        experiment = StandaloneEmbodiedLearning(
            brain_type=brain_type,
            random_seed=42
        )
        
        # Run short experiment (1 minute for quick validation)
        results = experiment.run_experiment(duration_minutes=1.0, save_results=True)
        
        # Print key results
        if results.get('learning_detected'):
            print(f"✅ {brain_type}: Learning detected (score: {results['learning_score']:+.3f})")
        else:
            print(f"❌ {brain_type}: No clear learning (score: {results['learning_score']:+.3f})")
        
        print()

if __name__ == "__main__":
    main()