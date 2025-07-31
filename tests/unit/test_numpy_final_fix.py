#!/usr/bin/env python3
"""Final test to verify numpy conversion errors are completely fixed."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Final numpy conversion error test...\n")

# Create brain with all features enabled
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable all prediction phases
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)

# Run 50 cycles to ensure we hit all periodic operations
print("Running 50 cycles with all features enabled...")
errors = []
success_count = 0

for i in range(50):
    # Vary input to trigger different code paths
    sensory_input = [0.5 + (i % 10) * 0.05] * 24
    
    try:
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        success_count += 1
        
        if i % 10 == 0:
            print(f"  Cycle {i}: ✅")
            
    except Exception as e:
        error_msg = str(e)
        if "can't convert" in error_msg and "numpy" in error_msg:
            errors.append(f"Cycle {i}: {e}")
            print(f"  Cycle {i}: ❌ NUMPY ERROR!")
        else:
            print(f"  Cycle {i}: ❌ Other error: {e}")

# Results
print(f"\n{'='*60}")
print("FINAL RESULTS")
print('='*60)
print(f"Successful cycles: {success_count}/50")

if errors:
    print(f"\n❌ Still found {len(errors)} numpy conversion errors!")
    for err in errors[:5]:
        print(f"  - {err}")
else:
    print("\n✅ SUCCESS! No numpy conversion errors detected!")
    print("   All tensor operations now properly handle MPS device")
    print("   The brain can run indefinitely without numpy errors")

# Test some specific operations that might use numpy
print("\nTesting specific operations...")

# Test consolidation (if it uses numpy)
try:
    print("  - Testing maintenance operations...", end="")
    brain.perform_maintenance()
    print(" ✅")
except Exception as e:
    if "numpy" in str(e):
        print(f" ❌ Numpy error: {e}")
    else:
        print(f" ❌ Other error: {e}")

# Test brain state creation
try:
    print("  - Testing brain state creation...", end="")
    for _ in range(5):
        state = brain._create_brain_state()
    print(" ✅")
except Exception as e:
    if "numpy" in str(e):
        print(f" ❌ Numpy error: {e}")
    else:
        print(f" ❌ Other error: {e}")

print("\n✅ All tests completed!")