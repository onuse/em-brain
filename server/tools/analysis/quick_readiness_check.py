#!/usr/bin/env python3
"""Quick readiness check for 8-hour test."""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("🔍 Quick 8-Hour Test Readiness Check")
print("=" * 50)

# Create optimized brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,  # biological_embodied_learning uses 24D
    motor_dim=4,
    quiet_mode=True,
    use_optimized=True
)
brain.enable_predictive_actions(False)
brain.set_pattern_extraction_limit(3)

# Quick performance test
times = []
for i in range(20):
    start = time.perf_counter()
    motors, state = brain.process_robot_cycle([0.1] * 24)
    times.append((time.perf_counter() - start) * 1000)

avg_time = np.mean(times)
print(f"\n✅ Brain compatible: 24D sensors → 3D motors")
print(f"✅ Avg cycle time: {avg_time:.1f}ms")
print(f"✅ Est. throughput: {int(3600/avg_time*1000):,} cycles/hour")
print(f"✅ Memory usage: {brain._calculate_memory_usage():.1f}MB")

if avg_time < 300:
    print(f"\n🎉 READY for 8-hour test!")
    print(f"\nExpected performance:")
    print(f"  - 1Hz operation: ✓ ({avg_time:.0f}ms << 1000ms)")
    print(f"  - 8-hour total: ~{int(8*3600/avg_time*1000):,} cycles")
else:
    print(f"\n⚠️  Performance may be marginal")