#!/usr/bin/env python3

"""
Test script for hardware adaptation system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils.hardware_adaptation import get_hardware_adaptation
import time

def test_hardware_adaptation():
    print("Testing Hardware Adaptation System")
    print("=" * 50)
    
    # Test hardware adaptation
    adapter = get_hardware_adaptation()
    print()
    print('Hardware Profile:')
    profile = adapter.get_hardware_profile()
    for key, value in profile.items():
        if key != 'performance_stats':
            print(f'  {key}: {value}')

    print()
    print('Adaptive Cognitive Limits:')
    limits = adapter.get_cognitive_limits()
    for key, value in limits.items():
        print(f'  {key}: {value}')

    # Test performance recording
    print()
    print('Testing performance recording...')
    for i in range(5):
        # Simulate different cycle times
        cycle_time = 45.0 + (i * 10)  # 45, 55, 65, 75, 85 ms
        adapter.record_cycle_performance(cycle_time, 100.0)
        print(f'  Recorded cycle: {cycle_time}ms')
        time.sleep(0.1)

    print()
    print('Updated limits after performance recording:')
    updated_limits = adapter.get_cognitive_limits()
    for key, value in updated_limits.items():
        print(f'  {key}: {value}')
    
    print()
    print("Hardware adaptation test completed!")

if __name__ == "__main__":
    test_hardware_adaptation()