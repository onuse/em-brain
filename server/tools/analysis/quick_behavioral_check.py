#!/usr/bin/env python3
"""
Quick Behavioral Check

Fast verification that optimized brain maintains behavioral characteristics.
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def quick_check():
    """Run quick behavioral check."""
    print("Quick Behavioral Check")
    print("-" * 50)
    
    # Create optimized brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=True,
        use_optimized=True
    )
    brain.enable_predictive_actions(False)
    
    # Test 1: Response to different stimuli
    print("\n1. Stimulus differentiation:")
    stimulus_a = [1.0] * 12 + [0.0] * 12
    stimulus_b = [0.0] * 12 + [1.0] * 12
    
    motors_a, _ = brain.process_robot_cycle(stimulus_a)
    motors_b, _ = brain.process_robot_cycle(stimulus_b)
    
    diff = np.linalg.norm(np.array(motors_a) - np.array(motors_b))
    print(f"   Response difference: {diff:.4f}")
    print(f"   {'✅ PASS' if diff > 0.01 else '❌ FAIL'}")
    
    # Test 2: Temporal consistency
    print("\n2. Temporal dynamics:")
    energies = []
    for i in range(10):
        sensory = [np.sin(i * 0.5)] * 24
        _, state = brain.process_robot_cycle(sensory)
        energies.append(state.get('field_energy', 0))
    
    energy_var = np.var(energies)
    print(f"   Energy variance: {energy_var:.6f}")
    print(f"   {'✅ PASS' if energy_var > 0 else '❌ FAIL'}")
    
    # Test 3: Performance
    print("\n3. Performance:")
    times = []
    for _ in range(5):
        start = time.perf_counter()
        brain.process_robot_cycle([0.1] * 24)
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   Average cycle: {avg_time:.1f}ms")
    print(f"   {'✅ PASS' if avg_time < 300 else '❌ FAIL'}")
    
    print("\n✅ Quick check complete!")


if __name__ == "__main__":
    quick_check()