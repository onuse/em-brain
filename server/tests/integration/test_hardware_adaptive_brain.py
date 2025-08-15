#!/usr/bin/env python3

"""
Test brain with hardware adaptation integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.src.brain_factory import MinimalBrain
import time
import random

def test_hardware_adaptive_brain():
    print("Testing Brain with Hardware Adaptation")
    print("=" * 50)
    
    # Create brain with hardware adaptation
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    print()
    print("Brain initialized. Running test cycles...")
    
    # Run several brain cycles to test adaptation
    sensory_input = [0.5, 0.3, 0.8, 0.2] * 4  # 16-dimensional input
    
    for cycle in range(20):
        start_time = time.time()
        
        # Process sensory input
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        # Simulate some sensory changes
        sensory_input = [max(0.0, min(1.0, s + random.uniform(-0.1, 0.1))) for s in sensory_input]
        
        # Store experience (simulate outcome)
        outcome = [s + random.uniform(-0.05, 0.05) for s in sensory_input]
        brain.store_experience(sensory_input, action, outcome, action)
        
        cycle_time_ms = brain_state.get('cycle_time_ms', 0)
        hardware_limits = brain_state.get('hardware_adaptive_limits', {})
        
        if cycle % 5 == 0:
            print(f"Cycle {cycle}: {cycle_time_ms:.1f}ms, WM limit: {hardware_limits.get('working_memory_limit', 'N/A')}")
        
        # Brief pause to simulate real time
        time.sleep(0.01)
    
    print()
    print("Final brain statistics:")
    
    # Get final hardware profile
    hardware_profile = brain.hardware_adaptation.get_hardware_profile()
    print(f"  Total experiences: {brain.total_experiences}")
    print(f"  Total predictions: {brain.total_predictions}")
    print(f"  Hardware adaptations: {hardware_profile.get('adaptation_count', 0)}")
    print(f"  Current WM limit: {hardware_profile.get('working_memory_limit', 'N/A')}")
    print(f"  Current energy budget: {hardware_profile.get('cognitive_energy_budget', 'N/A')}")
    
    if 'performance_stats' in hardware_profile:
        perf = hardware_profile['performance_stats']
        print(f"  Avg cycle time: {perf.get('recent_avg_cycle_time_ms', 'N/A'):.1f}ms")
        print(f"  Min cycle time: {perf.get('recent_min_cycle_time_ms', 'N/A'):.1f}ms")
        print(f"  Max cycle time: {perf.get('recent_max_cycle_time_ms', 'N/A'):.1f}ms")
    
    print()
    print("Hardware-adaptive brain test completed successfully!")

if __name__ == "__main__":
    test_hardware_adaptive_brain()