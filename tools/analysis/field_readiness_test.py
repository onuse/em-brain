#!/usr/bin/env python3
"""
Field Readiness Test - Comprehensive check before deployment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import numpy as np
from brain_factory import BrainFactory


def test_field_readiness():
    """Comprehensive test for field deployment readiness"""
    
    print("Field Readiness Test")
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
    
    results = {
        'performance': {'passed': False, 'details': []},
        'stability': {'passed': False, 'details': []},
        'exploration': {'passed': False, 'details': []},
        'smoothing': {'passed': False, 'details': []},
        'memory': {'passed': False, 'details': []}
    }
    
    # Test 1: Performance under load
    print("\n1. Testing performance under realistic load...")
    cycle_times = []
    
    for i in range(100):
        # Simulate varied sensor data
        sensory_input = [
            np.random.rand() * 0.5,  # Front sensor
            np.random.rand() * 0.3,  # Left sensor  
            np.random.rand() * 0.3,  # Right sensor
            np.sin(i * 0.1),  # IMU pitch
            np.cos(i * 0.1),  # IMU roll
            0.0,  # IMU yaw
        ] + [np.random.rand() * 0.1 for _ in range(10)]  # Other sensors
        
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        cycle_times.append(cycle_time)
    
    avg_cycle = np.mean(cycle_times)
    max_cycle = np.max(cycle_times)
    p95_cycle = np.percentile(cycle_times, 95)
    
    results['performance']['details'].append(f"Avg cycle: {avg_cycle:.1f}ms")
    results['performance']['details'].append(f"95th percentile: {p95_cycle:.1f}ms")
    results['performance']['details'].append(f"Max cycle: {max_cycle:.1f}ms")
    results['performance']['passed'] = p95_cycle < 150
    
    # Test 2: Stability over extended run
    print("\n2. Testing stability over extended operation...")
    memory_growth = []
    initial_regions = len(brain.brain.topology_regions)
    
    for i in range(200):
        sensory_input = [np.random.rand() * 0.3 for _ in range(16)]
        action, state = brain.process_sensory_input(sensory_input)
        
        if i % 50 == 0:
            regions = len(brain.brain.topology_regions)
            memory_growth.append(regions)
    
    final_regions = len(brain.brain.topology_regions)
    growth_rate = (final_regions - initial_regions) / 200
    
    results['stability']['details'].append(f"Memory growth rate: {growth_rate:.2f} regions/cycle")
    results['stability']['details'].append(f"Final region count: {final_regions}")
    results['stability']['passed'] = growth_rate < 0.5  # Less than 0.5 new regions per cycle
    
    # Test 3: Exploration behavior
    print("\n3. Testing exploration vs exploitation...")
    positions = []
    
    # Reset with no obstacles
    for i in range(50):
        sensory_input = [0.1, 0.1, 0.1] + [0.0] * 13  # Clear path
        action, state = brain.process_sensory_input(sensory_input)
        positions.append(action[1])  # Forward speed
    
    speed_variance = np.var(positions)
    exploration_score = np.mean(np.abs(np.diff(positions)))
    
    results['exploration']['details'].append(f"Speed variance: {speed_variance:.3f}")
    results['exploration']['details'].append(f"Exploration score: {exploration_score:.3f}")
    results['exploration']['passed'] = exploration_score > 0.001  # Some variation
    
    # Test 4: Motor smoothing verification
    print("\n4. Testing motor smoothing...")
    motor_changes = []
    prev_motors = None
    
    for i in range(30):
        # Alternating obstacles
        if i % 10 < 5:
            sensory_input = [0.1, 0.8, 0.1] + [0.0] * 13
        else:
            sensory_input = [0.1, 0.1, 0.8] + [0.0] * 13
            
        action, state = brain.process_sensory_input(sensory_input)
        
        if prev_motors is not None:
            change = np.max(np.abs(action[:4] - prev_motors))
            motor_changes.append(change)
        
        prev_motors = action[:4].copy()
    
    avg_change = np.mean(motor_changes)
    max_change = np.max(motor_changes)
    
    results['smoothing']['details'].append(f"Avg motor change: {avg_change:.3f}")
    results['smoothing']['details'].append(f"Max motor change: {max_change:.3f}")
    results['smoothing']['passed'] = max_change < 0.2
    
    # Test 5: Memory and learning
    print("\n5. Testing memory formation and recall...")
    
    # Train on specific pattern
    pattern_actions = []
    for i in range(10):
        sensory_input = [0.5, 0.2, 0.3] + [0.1] * 13
        action, state = brain.process_sensory_input(sensory_input)
        pattern_actions.append(action[:2])
    
    # Test different pattern
    for i in range(5):
        sensory_input = [0.3, 0.5, 0.2] + [0.1] * 13
        action, state = brain.process_sensory_input(sensory_input)
    
    # Recall original pattern
    recall_actions = []
    for i in range(5):
        sensory_input = [0.5, 0.2, 0.3] + [0.1] * 13
        action, state = brain.process_sensory_input(sensory_input)
        recall_actions.append(action[:2])
    
    # Check consistency
    initial_avg = np.mean(pattern_actions[-3:], axis=0)
    recall_avg = np.mean(recall_actions, axis=0)
    consistency = 1 - np.mean(np.abs(initial_avg - recall_avg))
    
    results['memory']['details'].append(f"Pattern consistency: {consistency:.3f}")
    results['memory']['details'].append(f"Active regions: {len(brain.brain.topology_regions)}")
    results['memory']['passed'] = consistency > 0.7
    
    brain.shutdown()
    
    # Generate report
    print("\n" + "=" * 60)
    print("FIELD READINESS REPORT")
    print("=" * 60)
    
    all_passed = True
    
    for test_name, test_result in results.items():
        status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
        print(f"\n{test_name.upper()}: {status}")
        for detail in test_result['details']:
            print(f"  - {detail}")
        
        if not test_result['passed']:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ BRAIN IS READY FOR FIELD TESTING!")
        print("\nRecommended deployment steps:")
        print("1. Start with tethered testing (wheels off ground)")
        print("2. Test in controlled environment (small enclosed area)")
        print("3. Gradually increase environment complexity")
        print("4. Monitor cycle times and adjust if needed")
    else:
        print("❌ BRAIN NEEDS ADJUSTMENTS BEFORE FIELD TESTING")
        print("\nIssues to address:")
        for test_name, test_result in results.items():
            if not test_result['passed']:
                print(f"  - Fix {test_name} issues")
    
    return all_passed


if __name__ == "__main__":
    test_field_readiness()