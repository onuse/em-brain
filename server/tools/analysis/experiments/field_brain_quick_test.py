#!/usr/bin/env python3
"""
Quick Field Brain Test - Identify performance bottlenecks
"""

import sys
import os
import torch
import time

# Add necessary paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)

from brains.field.core_brain import UnifiedFieldBrain

def quick_test():
    """Quick test to identify issues."""
    print("🧪 Quick Field Brain Test")
    print("=" * 40)
    
    # 1. Test instantiation with tiny brain
    print("\n1. Testing tiny brain instantiation...")
    start = time.time()
    brain = UnifiedFieldBrain(
        spatial_resolution=3,  # Tiny brain
        temporal_window=2.0,
        quiet_mode=True
    )
    print(f"✅ Instantiation took: {time.time() - start:.3f}s")
    print(f"   Field shape: {brain.unified_field.shape}")
    
    # 2. Test single cycle
    print("\n2. Testing single cycle...")
    start = time.time()
    action, state = brain.process_robot_cycle([0.5] * 24)
    cycle_time = time.time() - start
    print(f"✅ Single cycle took: {cycle_time:.3f}s")
    print(f"   Action: {[f'{x:.3f}' for x in action]}")
    
    # 3. Test 10 cycles
    print("\n3. Testing 10 cycles...")
    start = time.time()
    for i in range(10):
        action, state = brain.process_robot_cycle([0.5] * 24)
    total_time = time.time() - start
    print(f"✅ 10 cycles took: {total_time:.3f}s (avg: {total_time/10:.3f}s per cycle)")
    
    # 4. Check key issues
    print("\n4. Checking for issues...")
    
    # Check placeholder values
    if brain.field_experiences:
        last_exp = brain.field_experiences[-1]
        coords = last_exp.field_coordinates
        placeholders = []
        for idx, val in enumerate(coords):
            if abs(val - 0.5) < 0.01:
                placeholders.append(idx)
        if placeholders:
            print(f"⚠️  Placeholder dimensions found: {placeholders}")
    
    # Check maintenance
    print(f"   Maintenance interval: {brain.maintenance_interval}")
    print(f"   Last maintenance: {brain.last_maintenance_cycle}")
    
    # Check memory usage
    print(f"   Experiences: {len(brain.field_experiences)}")
    print(f"   Actions: {len(brain.field_actions)}")
    print(f"   Topology regions: {len(brain.topology_regions)}")
    
    print("\n✅ Quick test complete!")

if __name__ == "__main__":
    quick_test()