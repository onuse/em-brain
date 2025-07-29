#!/usr/bin/env python3
"""
Behavioral Test for Optimized Brain

Tests that optimizations maintain proper behavioral characteristics.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


class OptimizedBehavioralTest:
    """Test behavioral characteristics of optimized brain."""
    
    def __init__(self, use_optimized: bool = True):
        """Initialize test with optimized or non-optimized brain."""
        self.use_optimized = use_optimized
        print(f"\n{'=' * 60}")
        print(f"BEHAVIORAL TEST - {'OPTIMIZED' if use_optimized else 'NON-OPTIMIZED'} BRAIN")
        print(f"{'=' * 60}")
        
        # Create brain
        self.brain = SimplifiedUnifiedBrain(
            sensory_dim=24,
            motor_dim=4,
            spatial_resolution=32,
            quiet_mode=True,
            use_optimized=use_optimized
        )
        
        if use_optimized:
            self.brain.enable_predictive_actions(False)
            self.brain.set_pattern_extraction_limit(3)
        
        self.results = {}
        
    def test_response_variability(self, cycles: int = 20) -> Dict:
        """Test that brain produces varied responses."""
        print("\n1. Testing Response Variability...")
        
        # Same input repeated
        test_input = [0.5] * 24
        responses = []
        
        for i in range(cycles):
            motors, state = self.brain.process_robot_cycle(test_input)
            responses.append(motors)
        
        # Analyze variability
        responses_array = np.array(responses)
        std_per_motor = np.std(responses_array, axis=0)
        mean_std = np.mean(std_per_motor)
        
        result = {
            'mean_variability': mean_std,
            'per_motor_std': std_per_motor.tolist(),
            'passes': mean_std > 0.01  # Should have some variability
        }
        
        print(f"   Mean variability: {mean_std:.4f}")
        print(f"   Status: {'✅ PASS' if result['passes'] else '❌ FAIL'} (threshold: 0.01)")
        
        return result
    
    def test_stimulus_differentiation(self) -> Dict:
        """Test that brain responds differently to different stimuli."""
        print("\n2. Testing Stimulus Differentiation...")
        
        # Different stimuli
        stimulus_a = [1.0] * 12 + [0.0] * 12  # Strong in first half
        stimulus_b = [0.0] * 12 + [1.0] * 12  # Strong in second half
        stimulus_c = [0.1] * 24  # Weak uniform
        
        # Get responses (average over a few cycles for stability)
        responses = {}
        for name, stimulus in [('A', stimulus_a), ('B', stimulus_b), ('C', stimulus_c)]:
            motor_sum = np.zeros(3)
            for _ in range(5):
                motors, _ = self.brain.process_robot_cycle(stimulus)
                motor_sum += np.array(motors)
            responses[name] = motor_sum / 5
        
        # Calculate differences
        diff_ab = np.linalg.norm(responses['A'] - responses['B'])
        diff_ac = np.linalg.norm(responses['A'] - responses['C'])
        diff_bc = np.linalg.norm(responses['B'] - responses['C'])
        mean_diff = (diff_ab + diff_ac + diff_bc) / 3
        
        result = {
            'diff_ab': diff_ab,
            'diff_ac': diff_ac,
            'diff_bc': diff_bc,
            'mean_difference': mean_diff,
            'passes': mean_diff > 0.05  # Should differentiate stimuli
        }
        
        print(f"   Mean difference: {mean_diff:.4f}")
        print(f"   A-B: {diff_ab:.4f}, A-C: {diff_ac:.4f}, B-C: {diff_bc:.4f}")
        print(f"   Status: {'✅ PASS' if result['passes'] else '❌ FAIL'} (threshold: 0.05)")
        
        return result
    
    def test_reward_influence(self) -> Dict:
        """Test that rewards influence behavior."""
        print("\n3. Testing Reward Influence...")
        
        # Baseline behavior
        neutral_input = [0.5] * 23 + [0.0]  # No reward
        baseline_motors = []
        for _ in range(10):
            motors, _ = self.brain.process_robot_cycle(neutral_input)
            baseline_motors.append(motors)
        baseline_mean = np.mean(baseline_motors, axis=0)
        
        # Positive reward
        reward_input = [0.5] * 23 + [1.0]  # Strong positive reward
        for _ in range(10):
            self.brain.process_robot_cycle(reward_input)
        
        # Test if behavior changed
        post_reward_motors = []
        for _ in range(10):
            motors, _ = self.brain.process_robot_cycle(neutral_input)
            post_reward_motors.append(motors)
        post_reward_mean = np.mean(post_reward_motors, axis=0)
        
        # Calculate change
        behavior_change = np.linalg.norm(post_reward_mean - baseline_mean)
        
        result = {
            'baseline_mean': baseline_mean.tolist(),
            'post_reward_mean': post_reward_mean.tolist(),
            'behavior_change': behavior_change,
            'passes': behavior_change > 0.02  # Should show some change
        }
        
        print(f"   Behavior change: {behavior_change:.4f}")
        print(f"   Status: {'✅ PASS' if result['passes'] else '❌ FAIL'} (threshold: 0.02)")
        
        return result
    
    def test_temporal_dynamics(self) -> Dict:
        """Test that brain maintains temporal dynamics."""
        print("\n4. Testing Temporal Dynamics...")
        
        # Oscillating input
        responses = []
        for i in range(30):
            phase = i * 0.2
            sensory = [np.sin(phase), np.cos(phase)] + [0.1] * 22
            motors, state = self.brain.process_robot_cycle(sensory)
            responses.append({
                'motors': motors,
                'energy': state.get('field_energy', 0),
                'confidence': state.get('prediction_confidence', 0)
            })
        
        # Analyze dynamics
        motor_array = np.array([r['motors'] for r in responses])
        energies = [r['energy'] for r in responses]
        
        # Check for temporal patterns
        motor_autocorr = []
        for lag in [1, 2, 3]:
            if len(motor_array) > lag:
                corr = np.corrcoef(motor_array[:-lag].flatten(), 
                                  motor_array[lag:].flatten())[0, 1]
                motor_autocorr.append(corr)
        
        # Energy should vary
        energy_std = np.std(energies)
        
        result = {
            'motor_autocorrelation': motor_autocorr,
            'energy_variation': energy_std,
            'passes': energy_std > 0.0001 and len(motor_autocorr) > 0
        }
        
        print(f"   Energy variation: {energy_std:.6f}")
        print(f"   Autocorrelation: {[f'{c:.3f}' for c in motor_autocorr]}")
        print(f"   Status: {'✅ PASS' if result['passes'] else '❌ FAIL'}")
        
        return result
    
    def test_performance_metrics(self) -> Dict:
        """Test performance characteristics."""
        print("\n5. Testing Performance Metrics...")
        
        # Measure cycle times
        cycle_times = []
        sensory_input = [0.5] * 24
        
        # Warmup
        for _ in range(5):
            self.brain.process_robot_cycle(sensory_input)
        
        # Measure
        for _ in range(20):
            start = time.perf_counter()
            _, state = self.brain.process_robot_cycle(sensory_input)
            cycle_times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(cycle_times[5:])  # Skip initial cycles
        std_time = np.std(cycle_times[5:])
        
        result = {
            'avg_cycle_ms': avg_time,
            'std_cycle_ms': std_time,
            'min_cycle_ms': np.min(cycle_times),
            'max_cycle_ms': np.max(cycle_times),
            'passes': avg_time < 500  # Should be reasonably fast
        }
        
        print(f"   Average cycle: {avg_time:.1f}ms ± {std_time:.1f}ms")
        print(f"   Range: {result['min_cycle_ms']:.1f}ms - {result['max_cycle_ms']:.1f}ms")
        print(f"   Status: {'✅ PASS' if result['passes'] else '❌ FAIL'} (threshold: 500ms)")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Run all behavioral tests."""
        all_results = {
            'response_variability': self.test_response_variability(),
            'stimulus_differentiation': self.test_stimulus_differentiation(),
            'reward_influence': self.test_reward_influence(),
            'temporal_dynamics': self.test_temporal_dynamics(),
            'performance_metrics': self.test_performance_metrics()
        }
        
        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        
        passed = sum(1 for r in all_results.values() if r['passes'])
        total = len(all_results)
        
        print(f"\nTests passed: {passed}/{total}")
        for test_name, result in all_results.items():
            status = '✅' if result['passes'] else '❌'
            print(f"{status} {test_name}")
        
        all_results['summary'] = {
            'passed': passed,
            'total': total,
            'success_rate': passed / total
        }
        
        return all_results


def compare_optimized_vs_normal():
    """Compare behavioral characteristics of optimized vs non-optimized brain."""
    print("\nCOMPARING OPTIMIZED VS NON-OPTIMIZED BRAIN BEHAVIOR")
    print("=" * 70)
    
    # Test non-optimized
    test_normal = OptimizedBehavioralTest(use_optimized=False)
    results_normal = test_normal.run_all_tests()
    
    # Test optimized
    test_optimized = OptimizedBehavioralTest(use_optimized=True)
    results_optimized = test_optimized.run_all_tests()
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\nNon-optimized: {results_normal['summary']['passed']}/{results_normal['summary']['total']} tests passed")
    print(f"Optimized: {results_optimized['summary']['passed']}/{results_optimized['summary']['total']} tests passed")
    
    # Performance comparison
    normal_time = results_normal['performance_metrics']['avg_cycle_ms']
    optimized_time = results_optimized['performance_metrics']['avg_cycle_ms']
    speedup = normal_time / optimized_time
    
    print(f"\nPerformance:")
    print(f"  Non-optimized: {normal_time:.1f}ms per cycle")
    print(f"  Optimized: {optimized_time:.1f}ms per cycle")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Behavioral comparison
    print(f"\nBehavioral characteristics maintained:")
    for test in ['response_variability', 'stimulus_differentiation', 'reward_influence', 'temporal_dynamics']:
        normal_pass = results_normal[test]['passes']
        opt_pass = results_optimized[test]['passes']
        if normal_pass and opt_pass:
            print(f"  ✅ {test}: Both pass")
        elif not normal_pass and not opt_pass:
            print(f"  ⚠️  {test}: Both fail (expected)")
        else:
            print(f"  ❌ {test}: Inconsistent (normal={normal_pass}, opt={opt_pass})")
    
    # Overall verdict
    if results_optimized['summary']['success_rate'] >= results_normal['summary']['success_rate'] * 0.8:
        print(f"\n✅ OPTIMIZATION SUCCESSFUL: Behavior maintained with {speedup:.2f}x speedup!")
    else:
        print(f"\n❌ OPTIMIZATION ISSUE: Behavioral degradation detected")


if __name__ == "__main__":
    compare_optimized_vs_normal()