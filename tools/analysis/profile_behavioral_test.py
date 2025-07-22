#!/usr/bin/env python3
"""
Profile Behavioral Test Framework to Identify Bottlenecks
Step-by-step timing analysis of the behavioral test components.
"""

import sys
import time
sys.path.append('server/tools/testing')

from behavioral_test_framework import BehavioralTestFramework

def profile_framework_initialization():
    """Profile the framework initialization."""
    print("🔍 Profiling Framework Initialization")
    
    start = time.time()
    framework = BehavioralTestFramework(quiet_mode=True)
    init_time = time.time() - start
    print(f"  Framework init: {init_time:.3f}s")
    
    return framework

def profile_brain_creation(framework):
    """Profile brain creation process."""
    print("\n🔍 Profiling Brain Creation")
    
    # Test with minimal config first
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 16,
            'motor_dim': 4,
            'spatial_resolution': 6,  # Very small for speed
        },
        'memory': {'enable_persistence': False}
    }
    
    start = time.time()
    brain = framework.create_brain(config)
    creation_time = time.time() - start
    print(f"  Brain creation: {creation_time:.3f}s")
    
    return brain

def profile_individual_test_methods(framework, brain):
    """Profile each test method individually."""
    print("\n🔍 Profiling Individual Test Methods")
    
    test_methods = [
        ('prediction_learning', lambda: framework.test_prediction_learning(brain, cycles=10)),
        ('exploration_exploitation', lambda: framework.test_exploration_exploitation_balance(brain, cycles=20)),
        ('field_stabilization', lambda: framework.test_field_stabilization(brain, cycles=10)),
        ('computational_efficiency', lambda: framework.test_computational_efficiency(brain, cycles=10))
    ]
    
    results = {}
    for test_name, test_func in test_methods:
        try:
            print(f"  Testing {test_name}...")
            start = time.time()
            score = test_func()
            elapsed = time.time() - start
            results[test_name] = {'time': elapsed, 'score': score, 'success': True}
            print(f"    {test_name}: {elapsed:.3f}s (score: {score:.3f})")
            
            if elapsed > 30:  # More than 30 seconds
                print(f"    🚨 BOTTLENECK: {test_name} is very slow!")
                break
                
        except Exception as e:
            elapsed = time.time() - start
            results[test_name] = {'time': elapsed, 'score': 0, 'success': False, 'error': str(e)}
            print(f"    ❌ {test_name} failed after {elapsed:.3f}s: {e}")
            break
    
    return results

def profile_single_cycle(brain):
    """Profile a single brain cycle to understand base performance."""
    print("\n🔍 Profiling Single Brain Cycle")
    
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D
    
    # Warm up
    brain.process_sensory_input(pattern)
    
    # Time single cycle
    start = time.time()
    action, brain_state = brain.process_sensory_input(pattern)
    cycle_time = time.time() - start
    
    print(f"  Single cycle: {cycle_time*1000:.1f}ms")
    print(f"  Projected 100 cycles: {cycle_time*100:.1f}s")
    print(f"  Learning active: {brain_state.get('learning_addiction_modifier', 1.0):.2f}")
    
    return cycle_time

def main():
    """Run complete bottleneck profiling."""
    print("🚀 Behavioral Test Framework Bottleneck Analysis")
    print("=" * 60)
    
    try:
        # Step 1: Framework initialization
        framework = profile_framework_initialization()
        
        # Step 2: Brain creation
        brain = profile_brain_creation(framework)
        
        # Step 3: Single cycle performance
        cycle_time = profile_single_cycle(brain)
        
        # Step 4: Individual test methods
        if cycle_time < 1.0:  # Only proceed if cycles are reasonable
            test_results = profile_individual_test_methods(framework, brain)
            
            print(f"\n📊 Bottleneck Analysis Summary:")
            total_test_time = sum(r['time'] for r in test_results.values())
            print(f"  Total test time: {total_test_time:.1f}s")
            
            # Identify the slowest component
            slowest = max(test_results.items(), key=lambda x: x[1]['time'])
            print(f"  Slowest component: {slowest[0]} ({slowest[1]['time']:.1f}s)")
            
            if total_test_time > 60:
                print(f"  🎯 RECOMMENDATION: Focus on optimizing {slowest[0]}")
            else:
                print(f"  ✅ Performance acceptable for development")
                
        else:
            print(f"\n🚨 CRITICAL: Single cycle too slow ({cycle_time*1000:.0f}ms)")
            print(f"  Cannot run full behavioral tests with this performance")
    
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()