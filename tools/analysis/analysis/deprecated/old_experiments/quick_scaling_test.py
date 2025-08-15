#!/usr/bin/env python3
"""
Quick Scaling Test

Identifies where the brain performance breaks down as experiences accumulate.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def quick_scaling_test():
    """Test performance at different experience counts."""
    print("⚡ QUICK SCALING TEST")
    print("=" * 40)
    
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Test at different experience counts
    test_points = [10, 50, 100, 200]
    
    for target in test_points:
        # Add experiences quickly
        for i in range(len(brain.experience_storage._experiences), target):
            sensory = [0.1 + 0.001 * i, 0.2 + 0.001 * i, 0.3 + 0.001 * i, 0.4 + 0.001 * i]
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Test one cycle
        sensory = [0.5, 0.4, 0.6, 0.3]
        start_time = time.time()
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        cycle_time = (time.time() - start_time) * 1000
        
        print(f"   {target} experiences: {cycle_time:.1f}ms")
        
        # Stop if performance gets too bad
        if cycle_time > 500:
            print(f"   ❌ STOPPING: Performance too slow at {target} experiences")
            break
    
    brain.finalize_session()

if __name__ == "__main__":
    quick_scaling_test()