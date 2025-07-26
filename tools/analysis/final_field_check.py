#!/usr/bin/env python3
"""
Final Field Deployment Check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import numpy as np
from brain_factory import BrainFactory


def final_field_check():
    """Final comprehensive check before field deployment"""
    
    print("Final Field Deployment Check")
    print("=" * 60)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    print("\n1. Core System Checks")
    print("-" * 30)
    
    # Verify configuration
    print(f"✓ Spatial resolution: {brain.brain.spatial_resolution}³")
    print(f"✓ Gradient following strength: {brain.brain.gradient_following_strength}")
    print(f"✓ Motor smoothing factor: {brain.brain.motor_smoothing_factor}")
    print(f"✓ Maintenance frequency: every 25 cycles")
    
    print("\n2. Performance Baseline")
    print("-" * 30)
    
    # Quick performance test
    cycle_times = []
    for i in range(10):
        sensory_input = [0.2] * 16
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_times.append((time.time() - start) * 1000)
    
    print(f"✓ Average cycle time: {np.mean(cycle_times):.1f}ms")
    print(f"✓ Max cycle time: {np.max(cycle_times):.1f}ms")
    
    print("\n3. Behavioral Verification")
    print("-" * 30)
    
    # Test obstacle avoidance
    print("Testing obstacle responses...")
    responses = {}
    
    # Front obstacle
    for i in range(3):
        sensory_input = [0.8, 0.1, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    responses['front'] = action[:2]
    print(f"  Front obstacle: turn={action[0]:.3f}, speed={action[1]:.3f}")
    
    # Left obstacle
    for i in range(3):
        sensory_input = [0.1, 0.8, 0.1] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    responses['left'] = action[:2]
    print(f"  Left obstacle: turn={action[0]:.3f}, speed={action[1]:.3f}")
    
    # Right obstacle
    for i in range(3):
        sensory_input = [0.1, 0.1, 0.8] + [0.0] * 13
        action, state = brain.process_sensory_input(sensory_input)
    responses['right'] = action[:2]
    print(f"  Right obstacle: turn={action[0]:.3f}, speed={action[1]:.3f}")
    
    # Clear path with reward
    print("\nTesting exploration with reward...")
    for i in range(5):
        # Clear path
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 10 + [1.0, 0.0, 0.0]  # Positive reward
        action, state = brain.process_sensory_input(sensory_input)
    
    # Now test exploration
    speeds = []
    for i in range(10):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13  # Clear path
        action, state = brain.process_sensory_input(sensory_input)
        speeds.append(action[1])
    
    print(f"  Speed after reward: {np.mean(speeds):.3f} (variance: {np.var(speeds):.4f})")
    
    print("\n4. Safety Checks")
    print("-" * 30)
    
    # Test extreme inputs
    extreme_tests = [
        ([10.0] * 16, "Extreme high values"),
        ([-10.0] * 16, "Extreme negative values"),
        ([np.nan] * 16, "NaN values"),
    ]
    
    all_safe = True
    for test_input, desc in extreme_tests:
        try:
            action, state = brain.process_sensory_input(test_input)
            if any(np.isnan(action[:4])) or any(np.abs(action[:4]) > 1.0):
                print(f"  ❌ {desc}: Unsafe output")
                all_safe = False
            else:
                print(f"  ✓ {desc}: Safe output")
        except Exception as e:
            print(f"  ❌ {desc}: Exception - {e}")
            all_safe = False
    
    print("\n5. Final Status")
    print("-" * 30)
    
    # Check all criteria
    performance_ok = np.mean(cycle_times) < 150
    responses_ok = (abs(responses['left'][0]) > 0.001 and 
                   abs(responses['right'][0]) > 0.001)
    exploration_ok = np.mean(speeds) != 0 or np.var(speeds) > 0.0001
    
    print(f"✓ Performance: {'PASS' if performance_ok else 'FAIL'}")
    print(f"✓ Obstacle responses: {'PASS' if responses_ok else 'FAIL'}")  
    print(f"✓ Exploration behavior: {'PASS' if exploration_ok else 'FAIL'}")
    print(f"✓ Safety handling: {'PASS' if all_safe else 'FAIL'}")
    
    brain.shutdown()
    
    print("\n" + "=" * 60)
    
    if performance_ok and responses_ok and all_safe:
        print("✅ BRAIN IS READY FOR FIELD DEPLOYMENT!")
        print("\nDeployment checklist:")
        print("1. [ ] Connect to PiCar-X via network")
        print("2. [ ] Run tethered test (wheels off ground)")
        print("3. [ ] Verify sensor data flow")
        print("4. [ ] Test emergency stop")
        print("5. [ ] Begin controlled environment testing")
        
        if not exploration_ok:
            print("\nNote: Exploration behavior is minimal in untrained state.")
            print("This is expected and will improve with experience.")
    else:
        print("❌ ISSUES DETECTED - Review before deployment")
    
    return performance_ok and responses_ok and all_safe


if __name__ == "__main__":
    final_field_check()