#!/usr/bin/env python3
"""
Test Long-term Performance Degradation

Simulates the conditions that cause severe performance degradation after
many experiences are stored.
"""

import sys
import os
import time
import threading
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def test_long_term_degradation():
    """Test performance degradation with many experiences."""
    print("⏳ TESTING LONG-TERM PERFORMANCE DEGRADATION")
    print("=" * 60)
    
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,  # Disable checkpointing
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,  # Use best config
        quiet_mode=True
    )
    
    # Track performance over time
    performance_data = []
    
    # Add many experiences to simulate long-term operation
    print("📊 Testing performance as experiences accumulate...")
    
    test_points = [10, 50, 100, 200, 300, 500]
    
    for target_experiences in test_points:
        # Add experiences up to target
        while len(brain.experience_storage._experiences) < target_experiences:
            i = len(brain.experience_storage._experiences)
            sensory = [0.1 + 0.001 * i, 0.2 + 0.001 * i, 0.3 + 0.001 * i, 0.4 + 0.001 * i]
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Test current performance
        cycle_times = []
        for i in range(10):
            sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            end_time = time.time()
            
            cycle_time = (end_time - start_time) * 1000
            cycle_times.append(cycle_time)
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        working_memory_size = brain.activation_dynamics.get_working_memory_size()
        
        performance_data.append({
            'experiences': target_experiences,
            'avg_cycle_time': avg_cycle_time,
            'working_memory_size': working_memory_size
        })
        
        print(f"   {target_experiences} experiences: {avg_cycle_time:.1f}ms (WM: {working_memory_size})")
    
    # Analyze performance degradation
    print(f"\n📈 PERFORMANCE DEGRADATION ANALYSIS:")
    first_performance = performance_data[0]['avg_cycle_time']
    last_performance = performance_data[-1]['avg_cycle_time']
    
    degradation = ((last_performance - first_performance) / first_performance) * 100
    
    print(f"   Initial performance (10 exp): {first_performance:.1f}ms")
    print(f"   Final performance (500 exp): {last_performance:.1f}ms")
    print(f"   Performance degradation: {degradation:+.1f}%")
    
    # Check if this explains the 2230% degradation
    if degradation > 1000:
        print(f"   🚨 SEVERE DEGRADATION CONFIRMED: {degradation:.1f}%")
        print("   This explains the 2230% degradation seen in production!")
    elif degradation > 100:
        print(f"   ⚠️  SIGNIFICANT DEGRADATION: {degradation:.1f}%")
        print("   This partially explains the production issues")
    else:
        print(f"   ✅ REASONABLE DEGRADATION: {degradation:.1f}%")
        print("   Production issues may be from other factors")
    
    # Show scaling behavior
    print(f"\n🔍 SCALING BEHAVIOR:")
    for i, data in enumerate(performance_data):
        if i > 0:
            prev_data = performance_data[i-1]
            exp_ratio = data['experiences'] / prev_data['experiences']
            time_ratio = data['avg_cycle_time'] / prev_data['avg_cycle_time']
            print(f"   {prev_data['experiences']}→{data['experiences']} exp ({exp_ratio:.1f}x): {time_ratio:.1f}x slower")
    
    brain.finalize_session()
    
    return performance_data

if __name__ == "__main__":
    test_long_term_degradation()