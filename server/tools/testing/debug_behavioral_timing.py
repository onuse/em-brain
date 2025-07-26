#!/usr/bin/env python3
"""
Debug timing issues in behavioral test
"""

import sys
import os
from pathlib import Path
import time

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory

def main():
    """Test basic cycle timing"""
    print("🔍 Debugging Behavioral Test Timing")
    print("=" * 60)
    
    # Create brain with minimal config
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    print("Creating brain...")
    start = time.time()
    brain = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    print(f"Brain created in {time.time() - start:.1f}s")
    
    # Test a few cycles
    print("\nTesting cycles:")
    cycle_times = []
    
    for i in range(20):
        sensory_input = [0.1] * 16
        
        start = time.time()
        action, state = brain.process_sensory_input(sensory_input)
        cycle_time = (time.time() - start) * 1000
        
        cycle_times.append(cycle_time)
        
        if i % 5 == 0:
            print(f"  Cycle {i}: {cycle_time:.1f}ms")
    
    print(f"\nAverage cycle time: {sum(cycle_times)/len(cycle_times):.1f}ms")
    print(f"Min: {min(cycle_times):.1f}ms, Max: {max(cycle_times):.1f}ms")
    
    # Test 100 cycles (like prediction test)
    print("\nTesting 100 cycles (like prediction learning test):")
    start = time.time()
    
    for i in range(100):
        sensory_input = [0.1] * 16
        action, state = brain.process_sensory_input(sensory_input)
        
        if i % 20 == 0:
            elapsed = time.time() - start
            print(f"  {i} cycles completed in {elapsed:.1f}s")
    
    total_time = time.time() - start
    print(f"\n100 cycles completed in {total_time:.1f}s")
    print(f"Average: {total_time/100*1000:.1f}ms per cycle")
    
    brain.shutdown()
    print("\n✅ Debug test completed!")


if __name__ == "__main__":
    main()