#!/usr/bin/env python3
"""
Full Brain Exploration Simulation

Simulate a robot exploring an environment with:
- Realistic sensory inputs (distance sensors, light, temperature)
- Natural behavior patterns (exploration, goal-seeking, obstacle avoidance)
- Emergent learning from experience
- Real-time decision making

This tests the full brain in a natural setting to see how all optimizations
work together during normal exploration behavior.
"""

import sys
import os
import time
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from enum import Enum

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain
from biological_laziness_strategies import BiologicalLazinessManager
from biologically_realistic_fuzzyness import BiologicallyRealisticFuzzyness


class ExplorationBehavior(Enum):
    """Different exploration behaviors."""
    EXPLORE = "explore"          # Random exploration
    APPROACH_LIGHT = "approach"  # Move toward light
    AVOID_OBSTACLE = "avoid"     # Avoid obstacles
    STUCK = "stuck"              # Stuck in corner


class VirtualEnvironment:
    """Simulated 2D environment for robot exploration."""
    
    def __init__(self, width: int = 20, height: int = 20):
        """Initialize virtual environment."""
        self.width = width
        self.height = height
        
        # Environment features
        self.obstacles = self._generate_obstacles()
        self.light_sources = self._generate_light_sources()
        self.temperature_zones = self._generate_temperature_zones()
        
        # Robot state
        self.robot_x = width // 2
        self.robot_y = height // 2
        self.robot_heading = 0.0  # radians
        
        # Environment dynamics
        self.time_step = 0
        self.light_intensity_base = 0.3
        
        print(f"🌍 Virtual environment: {width}x{height}")
        print(f"   Obstacles: {len(self.obstacles)}")
        print(f"   Light sources: {len(self.light_sources)}")
        print(f"   Temperature zones: {len(self.temperature_zones)}")
    
    def _generate_obstacles(self) -> List[Tuple[int, int]]:
        """Generate random obstacles."""
        obstacles = []
        num_obstacles = random.randint(8, 15)
        
        for _ in range(num_obstacles):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            
            # Don't place obstacle on center (robot start)
            if abs(x - self.width//2) > 2 or abs(y - self.height//2) > 2:
                obstacles.append((x, y))
        
        # Add border obstacles
        for x in range(self.width):
            obstacles.extend([(x, 0), (x, self.height-1)])
        for y in range(self.height):
            obstacles.extend([(0, y), (self.width-1, y)])
        
        return obstacles
    
    def _generate_light_sources(self) -> List[Tuple[int, int, float]]:
        """Generate light sources (x, y, intensity)."""
        sources = []
        num_sources = random.randint(2, 4)
        
        for _ in range(num_sources):
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)
            intensity = random.uniform(0.5, 1.0)
            sources.append((x, y, intensity))
        
        return sources
    
    def _generate_temperature_zones(self) -> List[Tuple[int, int, float]]:
        """Generate temperature zones (x, y, temperature)."""
        zones = []
        num_zones = random.randint(3, 6)
        
        for _ in range(num_zones):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            temp = random.uniform(0.2, 0.8)
            zones.append((x, y, temp))
        
        return zones
    
    def get_sensory_input(self) -> List[float]:
        """
        Get robot's sensory input in current position.
        
        Returns 4D sensory vector:
        [distance_front, light_intensity, temperature, motion_comfort]
        """
        # Distance sensor (forward direction)
        front_x = self.robot_x + int(np.cos(self.robot_heading) * 2)
        front_y = self.robot_y + int(np.sin(self.robot_heading) * 2)
        
        distance = 1.0  # Default max distance
        for step in range(1, 6):  # Check up to 5 steps ahead
            check_x = self.robot_x + int(np.cos(self.robot_heading) * step)
            check_y = self.robot_y + int(np.sin(self.robot_heading) * step)
            
            if (check_x, check_y) in self.obstacles:
                distance = step / 5.0  # Normalize to 0-1
                break
        
        # Light intensity (combined from all sources)
        light_intensity = self.light_intensity_base
        for lx, ly, intensity in self.light_sources:
            dist = np.sqrt((self.robot_x - lx)**2 + (self.robot_y - ly)**2)
            light_contribution = intensity / (1 + dist * 0.3)
            light_intensity += light_contribution
        
        # Add time-based light variation
        light_variation = 0.1 * np.sin(self.time_step * 0.1)
        light_intensity = max(0, min(1, light_intensity + light_variation))
        
        # Temperature (from nearest zone)
        temperature = 0.5  # Default temperature
        min_temp_dist = float('inf')
        for tx, ty, temp in self.temperature_zones:
            dist = np.sqrt((self.robot_x - tx)**2 + (self.robot_y - ty)**2)
            if dist < min_temp_dist:
                min_temp_dist = dist
                temperature = temp / (1 + dist * 0.2)
        
        # Motion comfort (how "stuck" the robot feels)
        # Based on nearby obstacles
        nearby_obstacles = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (self.robot_x + dx, self.robot_y + dy) in self.obstacles:
                    nearby_obstacles += 1
        
        motion_comfort = max(0.1, 1.0 - nearby_obstacles / 8.0)
        
        # Add small amount of sensor noise
        noise = np.random.normal(0, 0.02, 4)
        
        sensory = [
            max(0, min(1, distance + noise[0])),
            max(0, min(1, light_intensity + noise[1])), 
            max(0, min(1, temperature + noise[2])),
            max(0, min(1, motion_comfort + noise[3]))
        ]
        
        return sensory
    
    def execute_action(self, action: List[float]) -> Tuple[List[float], ExplorationBehavior]:
        """
        Execute robot action and return outcome.
        
        Action format: [forward_speed, turn_rate, light_seeking, exploration_drive]
        Returns: (outcome, detected_behavior)
        """
        forward_speed = np.clip(action[0], -0.5, 1.0)
        turn_rate = np.clip(action[1], -1.0, 1.0)
        
        # Update heading
        self.robot_heading += turn_rate * 0.3
        self.robot_heading = self.robot_heading % (2 * np.pi)
        
        # Calculate new position
        new_x = self.robot_x + int(forward_speed * np.cos(self.robot_heading) * 2)
        new_y = self.robot_y + int(forward_speed * np.sin(self.robot_heading) * 2)
        
        # Check for collision
        movement_success = 1.0
        if (new_x, new_y) in self.obstacles:
            # Collision - don't move
            movement_success = 0.0
            behavior = ExplorationBehavior.AVOID_OBSTACLE
        else:
            # Valid move
            self.robot_x = max(0, min(self.width-1, new_x))
            self.robot_y = max(0, min(self.height-1, new_y))
            
            # Determine behavior based on action pattern
            if action[2] > 0.6:  # light_seeking high
                behavior = ExplorationBehavior.APPROACH_LIGHT
            elif forward_speed < 0.1 and abs(turn_rate) > 0.5:
                behavior = ExplorationBehavior.STUCK
            else:
                behavior = ExplorationBehavior.EXPLORE
        
        # Calculate outcome based on environmental rewards
        light_reward = self.get_sensory_input()[1] * 0.3  # Reward for being in light
        motion_reward = movement_success * 0.2
        exploration_reward = 0.1 if behavior == ExplorationBehavior.EXPLORE else 0.0
        
        # Outcome represents success/satisfaction
        outcome = [
            movement_success,
            light_reward,
            motion_reward + exploration_reward,
            action[3] * 0.5  # Maintain exploration drive
        ]
        
        self.time_step += 1
        return outcome, behavior
    
    def get_robot_position(self) -> Tuple[int, int, float]:
        """Get current robot position and heading."""
        return self.robot_x, self.robot_y, self.robot_heading
    
    def is_near_light(self) -> bool:
        """Check if robot is near any light source."""
        for lx, ly, _ in self.light_sources:
            if np.sqrt((self.robot_x - lx)**2 + (self.robot_y - ly)**2) < 2:
                return True
        return False


class ExplorationSimulation:
    """Full brain exploration simulation."""
    
    def __init__(self, use_optimizations: bool = True):
        """Initialize exploration simulation."""
        self.use_optimizations = use_optimizations
        
        # Create brain with or without optimizations
        self.brain = MinimalBrain(
            enable_logging=False, 
            enable_persistence=False, 
            quiet_mode=True
        )
        
        # Add optimizations if requested
        if use_optimizations:
            self.laziness = BiologicalLazinessManager()
            self.fuzzyness = BiologicallyRealisticFuzzyness()
        else:
            self.laziness = None
            self.fuzzyness = None
        
        # Environment
        self.environment = VirtualEnvironment()
        
        # Simulation state
        self.step_count = 0
        self.behavior_history = deque(maxlen=20)
        self.performance_history = deque(maxlen=50)
        
        # Metrics
        self.cycle_times = []
        self.prediction_errors = []
        self.behaviors_detected = {behavior: 0 for behavior in ExplorationBehavior}
        self.light_discoveries = 0
        self.exploration_coverage = set()
        
        print(f"🤖 Exploration simulation initialized")
        print(f"   Optimizations: {'Enabled' if use_optimizations else 'Disabled'}")
    
    def run_exploration_step(self) -> Dict:
        """Run one step of exploration."""
        step_start = time.time()
        
        # Get current sensory input
        sensory_input = self.environment.get_sensory_input()
        
        # Brain prediction (with potential fuzzy caching)
        if self.fuzzyness:
            cached_result = self.fuzzyness.should_use_cache(sensory_input)
            if cached_result:
                predicted_action, brain_state = cached_result
                cache_used = True
            else:
                predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
                self.fuzzyness.cache_prediction(sensory_input, predicted_action, brain_state)
                cache_used = False
        else:
            predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
            cache_used = False
        
        # Execute action in environment
        outcome, behavior = self.environment.execute_action(predicted_action)
        
        # Calculate prediction error
        prediction_error = np.mean([(p - o) ** 2 for p, o in zip(predicted_action, outcome)])
        self.prediction_errors.append(prediction_error)
        
        # Apply biological laziness for learning decision
        experience_processed = True
        learning_decision = "learn"
        
        if self.laziness:
            confidence = brain_state.get('prediction_confidence', 0.5)
            decision = self.laziness.should_process_experience(
                sensory_input, prediction_error, confidence
            )
            learning_decision = decision['action']
            
            if decision['action'] == 'learn':
                # Full learning
                self.brain.store_experience(sensory_input, predicted_action, outcome, predicted_action)
                
                # Record learning outcome
                prediction_success = 1.0 - min(1.0, prediction_error)
                learn_time = (time.time() - step_start) * 1000
                self.laziness.record_learning_outcome(learn_time, prediction_success)
                
            elif decision['action'] == 'buffer':
                # Minimal storage
                experience_processed = False
            else:  # ignore
                experience_processed = False
            
            # Update experience context
            self.laziness.add_experience_to_context(sensory_input, prediction_error)
        else:
            # Normal learning
            self.brain.store_experience(sensory_input, predicted_action, outcome, predicted_action)
        
        # Update performance tracking
        cycle_time = (time.time() - step_start) * 1000
        self.cycle_times.append(cycle_time)
        
        if self.fuzzyness:
            self.fuzzyness.update_performance_tier(cycle_time)
        
        # Track exploration metrics
        pos_x, pos_y, heading = self.environment.get_robot_position()
        self.exploration_coverage.add((pos_x, pos_y))
        self.behaviors_detected[behavior] += 1
        self.behavior_history.append(behavior)
        
        # Check for light discovery
        if self.environment.is_near_light() and behavior == ExplorationBehavior.APPROACH_LIGHT:
            self.light_discoveries += 1
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            behavior, prediction_error, cycle_time, outcome
        )
        self.performance_history.append(performance_score)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'position': (pos_x, pos_y),
            'behavior': behavior,
            'sensory_input': sensory_input,
            'predicted_action': predicted_action,
            'outcome': outcome,
            'prediction_error': prediction_error,
            'cycle_time': cycle_time,
            'cache_used': cache_used,
            'experience_processed': experience_processed,
            'learning_decision': learning_decision,
            'performance_score': performance_score
        }
    
    def _calculate_performance_score(self, behavior: ExplorationBehavior, 
                                   prediction_error: float, 
                                   cycle_time: float,
                                   outcome: List[float]) -> float:
        """Calculate overall performance score for this step."""
        # Behavior quality
        behavior_scores = {
            ExplorationBehavior.EXPLORE: 0.7,
            ExplorationBehavior.APPROACH_LIGHT: 0.9,
            ExplorationBehavior.AVOID_OBSTACLE: 0.8,
            ExplorationBehavior.STUCK: 0.2
        }
        behavior_score = behavior_scores[behavior]
        
        # Prediction accuracy (lower error = better)
        accuracy_score = max(0, 1.0 - prediction_error)
        
        # Speed efficiency (faster cycles = better, within reason)
        speed_score = max(0.2, 1.0 - (cycle_time - 1.0) / 20.0)  # Target ~1ms
        
        # Environmental success
        env_score = np.mean(outcome)
        
        # Combined weighted score
        total_score = (
            behavior_score * 0.3 +
            accuracy_score * 0.2 +
            speed_score * 0.2 +
            env_score * 0.3
        )
        
        return max(0, min(1, total_score))
    
    def get_simulation_statistics(self) -> Dict:
        """Get comprehensive simulation statistics."""
        if len(self.cycle_times) == 0:
            return {'status': 'no_data'}
        
        # Basic performance
        avg_cycle_time = np.mean(self.cycle_times)
        avg_prediction_error = np.mean(self.prediction_errors)
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0
        
        # Exploration metrics
        coverage_percentage = (len(self.exploration_coverage) / 
                             (self.environment.width * self.environment.height)) * 100
        
        # Behavior distribution
        total_behaviors = sum(self.behaviors_detected.values())
        behavior_distribution = {
            behavior.value: (count / max(1, total_behaviors)) * 100
            for behavior, count in self.behaviors_detected.items()
        }
        
        # Recent behavior pattern
        recent_behaviors = list(self.behavior_history)[-10:]
        behavior_diversity = len(set(recent_behaviors)) / max(1, len(recent_behaviors))
        
        stats = {
            'steps_completed': self.step_count,
            'avg_cycle_time': avg_cycle_time,
            'avg_prediction_error': avg_prediction_error,
            'avg_performance_score': avg_performance,
            'exploration_coverage': coverage_percentage,
            'light_discoveries': self.light_discoveries,
            'behavior_distribution': behavior_distribution,
            'behavior_diversity': behavior_diversity,
            'experiences_per_second': self.step_count / (sum(self.cycle_times) / 1000) if self.cycle_times else 0
        }
        
        # Add optimization-specific stats
        if self.laziness:
            laziness_stats = self.laziness.get_statistics()
            stats.update({
                'experiences_filtered': laziness_stats['filter_rate'],
                'computational_savings': (laziness_stats['filter_rate'] + 
                                        laziness_stats['delay_rate'] * 0.8) / 100
            })
        
        if self.fuzzyness:
            fuzzy_stats = self.fuzzyness.get_statistics()
            stats.update({
                'cache_hit_rate': fuzzy_stats['cache_hit_rate'],
                'performance_tier': fuzzy_stats['current_tier']
            })
        
        return stats


def run_full_brain_exploration():
    """Run complete exploration simulation."""
    print("🧭 FULL BRAIN EXPLORATION SIMULATION")
    print("=" * 60)
    print("Simulating robot exploration with realistic behavior patterns")
    print()
    
    # Run simulations with and without optimizations
    configurations = [
        ("Baseline Brain", False),
        ("Optimized Brain", True)
    ]
    
    simulation_results = {}
    
    for config_name, use_optimizations in configurations:
        print(f"\n{'='*50}")
        print(f"🤖 Running: {config_name}")
        print(f"{'='*50}")
        
        # Create simulation
        sim = ExplorationSimulation(use_optimizations=use_optimizations)
        
        # Run exploration steps
        steps_to_run = 100
        
        for step in range(steps_to_run):
            step_result = sim.run_exploration_step()
            
            # Progress updates
            if (step + 1) % 20 == 0:
                stats = sim.get_simulation_statistics()
                print(f"  Step {step+1}/{steps_to_run}: "
                      f"cycle={stats['avg_cycle_time']:.1f}ms, "
                      f"coverage={stats['exploration_coverage']:.1f}%, "
                      f"performance={stats['avg_performance_score']:.2f}")
        
        # Final statistics
        final_stats = sim.get_simulation_statistics()
        simulation_results[config_name] = final_stats
        
        print(f"\n📊 {config_name} Results:")
        print(f"   Avg cycle time: {final_stats['avg_cycle_time']:.1f}ms")
        print(f"   Exploration coverage: {final_stats['exploration_coverage']:.1f}%")
        print(f"   Light discoveries: {final_stats['light_discoveries']}")
        print(f"   Performance score: {final_stats['avg_performance_score']:.3f}")
        print(f"   Experiences/sec: {final_stats['experiences_per_second']:.1f}")
        
        if 'computational_savings' in final_stats:
            print(f"   Computational savings: {final_stats['computational_savings']*100:.1f}%")
        if 'cache_hit_rate' in final_stats:
            print(f"   Cache hit rate: {final_stats['cache_hit_rate']:.1f}%")
        
        # Behavior analysis
        print(f"   Behavior distribution:")
        for behavior, percentage in final_stats['behavior_distribution'].items():
            print(f"     {behavior}: {percentage:.1f}%")
    
    # Compare results
    if len(simulation_results) == 2:
        print("\n" + "="*60)
        print("🎯 OPTIMIZATION IMPACT ON EXPLORATION")
        print("="*60)
        
        baseline = simulation_results["Baseline Brain"]
        optimized = simulation_results["Optimized Brain"]
        
        # Performance comparison
        speed_improvement = (optimized['experiences_per_second'] / 
                           baseline['experiences_per_second'] - 1) * 100
        cycle_improvement = (1 - optimized['avg_cycle_time'] / 
                           baseline['avg_cycle_time']) * 100
        
        print(f"\n⚡ Performance Improvements:")
        print(f"   Processing speed: {speed_improvement:+.1f}%")
        print(f"   Cycle time: {cycle_improvement:+.1f}% faster")
        
        # Exploration effectiveness
        coverage_diff = optimized['exploration_coverage'] - baseline['exploration_coverage']
        discovery_diff = optimized['light_discoveries'] - baseline['light_discoveries']
        performance_diff = optimized['avg_performance_score'] - baseline['avg_performance_score']
        
        print(f"\n🧭 Exploration Effectiveness:")
        print(f"   Coverage difference: {coverage_diff:+.1f}%")
        print(f"   Light discoveries: {discovery_diff:+d}")
        print(f"   Performance score: {performance_diff:+.3f}")
        
        # System efficiency
        if 'computational_savings' in optimized:
            print(f"\n💡 System Efficiency:")
            print(f"   Computational savings: {optimized['computational_savings']*100:.1f}%")
            print(f"   Experience filtering: {optimized['experiences_filtered']:.1f}%")
            print(f"   Performance tier: {optimized.get('performance_tier', 'N/A')}")
        
        print(f"\n🎯 Overall Assessment:")
        if speed_improvement > 50 and performance_diff > -0.1:
            print("   ✅ EXCELLENT: Major speed gains with preserved exploration quality")
        elif speed_improvement > 20 and performance_diff > -0.2:
            print("   📊 GOOD: Solid improvements with acceptable trade-offs")
        elif speed_improvement > 0:
            print("   ⚡ MODERATE: Some improvements, optimization working")
        else:
            print("   ⚠️  NEEDS WORK: Limited benefits from optimizations")


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    run_full_brain_exploration()