#!/usr/bin/env python3
"""
Fixed Behavioral Test Framework for Dynamic Brain

This is a simplified version that works with the new dynamic brain architecture.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.dynamic_brain_factory import DynamicBrainFactory


class IntelligenceMetric(Enum):
    """Core intelligence metrics we want to achieve"""
    PREDICTION_LEARNING = "prediction_learning"
    EXPLORATION_EXPLOITATION = "exploration_exploitation" 
    FIELD_STABILIZATION = "field_stabilization"
    PATTERN_RECOGNITION = "pattern_recognition"
    GOAL_SEEKING = "goal_seeking"
    BIOLOGICAL_REALISM = "biological_realism"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


@dataclass
class TestResult:
    """Result from a single intelligence test"""
    metric: IntelligenceMetric
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: float
    

class BehavioralTestFramework:
    """Simplified framework for testing brain intelligence with dynamic brain"""
    
    def __init__(self, quiet_mode: bool = True):
        self.quiet_mode = quiet_mode
        self.test_results_history = []
        
    def create_brain(self, config: Dict[str, Any] = None):
        """Create a brain for testing with optional configuration"""
        # Default configuration for dynamic brain
        default_config = {
            'use_dynamic_brain': True,
            'use_full_features': True,
            'quiet_mode': self.quiet_mode,
            'temporal_window': 10.0,
            'field_evolution_rate': 0.1,
            'constraint_discovery_rate': 0.15,
            'complexity_factor': 6.0
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
            
        # Create factory
        factory = DynamicBrainFactory(default_config)
        
        # Create brain with correct PiCar-X profile (from picarx_profile.json)
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=4,
            sensory_dim=16,
            motor_dim=5
        )
        
        return brain_wrapper
    
    def test_prediction_learning(self, brain_wrapper, cycles: int = 100) -> float:
        """Test how well the brain learns to predict patterns"""
        brain = brain_wrapper.brain
        
        # Reset confidence history for fresh learning
        if hasattr(brain, '_prediction_confidence_history'):
            brain._prediction_confidence_history.clear()
            brain._improvement_rate_history.clear()
        
        # Simple repeating pattern
        pattern = [0.3, 0.7, 0.5, 0.9, 0.1, 0.6, 0.4, 0.8]
        pattern_length = len(pattern)
        
        confidences = []
        for i in range(cycles):
            # Create sensory input from pattern
            sensory_input = [pattern[i % pattern_length]] * 24
            
            # Process cycle
            _, brain_state = brain.process_robot_cycle(sensory_input)
            
            # Track confidence
            confidence = brain_state.get('prediction_confidence', 0.5)
            confidences.append(confidence)
        
        # Score based on confidence improvement
        if len(confidences) > 20:
            early_confidence = np.mean(confidences[:20])
            late_confidence = np.mean(confidences[-20:])
            improvement = late_confidence - early_confidence
            score = min(1.0, max(0.0, improvement * 2 + 0.5))
        else:
            score = 0.5
            
        if not self.quiet_mode:
            print(f"Prediction Learning Score: {score:.2f}")
            
        return score
    
    def test_exploration_exploitation_balance(self, brain_wrapper, cycles: int = 100) -> float:
        """Test balance between exploring new areas and exploiting known rewards"""
        brain = brain_wrapper.brain
        
        # Track motor variety
        motor_outputs = []
        rewards_given = []
        
        for i in range(cycles):
            # Varied sensory input to create gradients
            # PiCar-X has 16 sensory channels
            sensory_input = [0.5 + 0.1 * np.sin(i * 0.1 + j * 0.2) for j in range(16)]
            
            # Give rewards for certain motor patterns
            if i > 0 and len(motor_outputs) > 0:
                last_motor = motor_outputs[-1]
                # Reward movement in positive X direction
                if last_motor[0] > 0.1:
                    # Use one of the sensory channels as reward signal
                    sensory_input[15] = 0.8  # High reward (use last channel)
                    rewards_given.append(1)
                else:
                    sensory_input[15] = 0.0  # No reward
                    rewards_given.append(0)
            
            # Process cycle
            motor_output, _ = brain.process_robot_cycle(sensory_input)
            motor_outputs.append(motor_output)
        
        # Analyze exploration vs exploitation
        if len(motor_outputs) > 20:
            # Check motor variance (exploration)
            motor_variance = np.var([m[0] for m in motor_outputs])
            
            # Check reward acquisition (exploitation)
            reward_rate = np.mean(rewards_given) if rewards_given else 0
            
            # Good balance: some exploration but also learns to get rewards
            score = min(1.0, motor_variance * 10) * 0.5 + reward_rate * 0.5
            
            # DEBUG: Print detailed stats
            if True:  # Always show debug
                print(f"\n[DEBUG] Exploration/Exploitation Details:")
                print(f"  Motor outputs (first 10): {[m[0] for m in motor_outputs[:10]]}")
                print(f"  Motor variance: {motor_variance:.6f}")
                print(f"  Rewards given: {sum(rewards_given)}/{len(rewards_given)}")
                print(f"  Reward rate: {reward_rate:.3f}")
                print(f"  Exploration component: {min(1.0, motor_variance * 10):.3f}")
                print(f"  Final score: {score:.3f}")
        else:
            score = 0.5
            
        if not self.quiet_mode:
            print(f"Exploration/Exploitation Score: {score:.2f}")
            
        return score
    
    def test_computational_efficiency(self, brain_wrapper, cycles: int = 50) -> float:
        """Test computational performance"""
        brain = brain_wrapper.brain
        
        cycle_times = []
        
        for i in range(cycles):
            sensory_input = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
            
            start = time.perf_counter()
            brain.process_robot_cycle(sensory_input)
            elapsed = time.perf_counter() - start
            
            cycle_times.append(elapsed)
        
        # Skip first few cycles (warmup)
        avg_time = np.mean(cycle_times[5:]) * 1000  # Convert to ms
        
        # Score based on performance targets
        if avg_time < 10:
            score = 1.0
        elif avg_time < 50:
            score = 0.8
        elif avg_time < 100:
            score = 0.6
        elif avg_time < 200:
            score = 0.4
        else:
            score = 0.2
            
        if not self.quiet_mode:
            print(f"Computational Efficiency Score: {score:.2f} (avg: {avg_time:.1f}ms)")
            
        return score
    
    def run_quick_assessment(self, brain_wrapper) -> Dict[str, float]:
        """Run a quick assessment of key metrics"""
        results = {}
        
        if not self.quiet_mode:
            print("\n🧪 Running Quick Intelligence Assessment")
            print("=" * 50)
        
        # Test key metrics
        results['prediction'] = self.test_prediction_learning(brain_wrapper, cycles=50)
        results['exploration'] = self.test_exploration_exploitation_balance(brain_wrapper, cycles=50)
        results['efficiency'] = self.test_computational_efficiency(brain_wrapper, cycles=30)
        
        # Overall score
        results['overall'] = np.mean(list(results.values()))
        
        if not self.quiet_mode:
            print(f"\nOverall Intelligence Score: {results['overall']:.2f}")
            
        return results


def main():
    """Run behavioral tests on the dynamic brain"""
    framework = BehavioralTestFramework(quiet_mode=False)
    
    print("🧠 Behavioral Test Framework - Dynamic Brain")
    print("=" * 50)
    
    # Create brain
    print("\nCreating brain...")
    brain_wrapper = framework.create_brain()
    
    # Run assessment
    results = framework.run_quick_assessment(brain_wrapper)
    
    print("\n📊 Summary:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.2f}")
    
    # Performance info
    brain = brain_wrapper.brain
    print(f"\nBrain Info:")
    print(f"  Conceptual dimensions: {brain.total_dimensions}D")
    print(f"  Tensor shape: {brain.tensor_shape}")
    print(f"  Memory usage: {brain._calculate_memory_usage():.1f}MB")
    print(f"  Device: {brain.device}")


if __name__ == "__main__":
    main()