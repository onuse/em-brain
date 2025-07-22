#!/usr/bin/env python3
"""
Absolute minimal test to isolate CPU performance issue
"""

import sys
import time
sys.path.append('server/src')

try:
    print("🔍 Testing UnifiedFieldBrain import...")
    from server.src.brains.field.core_brain import UnifiedFieldBrain
    print("✅ Import successful")
    
    print("🔍 Testing minimal brain creation...")
    start = time.time()
    brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)  # Tiny 3x3 = 9 elements
    creation_time = time.time() - start
    print(f"✅ Brain created in {creation_time:.2f}s")
    
    print("🔍 Testing single cycle...")
    start = time.time()
    actions, state = brain.process_robot_cycle([0.1, 0.2])
    cycle_time = time.time() - start
    print(f"✅ Single cycle: {cycle_time:.2f}s")
    
    if cycle_time > 5.0:
        print("❌ CRITICAL: Single cycle > 5s - CPU path has fundamental performance issue")
    elif cycle_time > 1.0:
        print("⚠️ WARNING: Single cycle > 1s - very slow but might work")
    else:
        print("✅ CPU performance acceptable for testing")
        
        print("🔍 Testing 3 cycles...")
        start = time.time()
        for i in range(3):
            actions, state = brain.process_robot_cycle([0.1 + i*0.1, 0.2])
        total_time = time.time() - start
        print(f"✅ 3 cycles: {total_time:.2f}s ({3/total_time:.1f} cps)")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()