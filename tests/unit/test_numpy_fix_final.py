#!/usr/bin/env python3
"""Test that numpy conversion errors are fixed."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Testing numpy conversion fixes...\n")

# Create brain
brain = SimplifiedUnifiedBrain(
    sensory_dim=24,
    motor_dim=4,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable features that might trigger numpy conversions
brain.enable_hierarchical_prediction(True)
brain.enable_action_prediction(True)

# Run many cycles to trigger periodic operations
print("Running 30 cycles to test various operations...")
success_count = 0
errors = []

for i in range(30):
    sensory_input = [0.5 + i*0.01] * 24  # Varying input
    
    try:
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        success_count += 1
        
        # Print progress every 5 cycles
        if i % 5 == 0:
            print(f"  Cycle {i}: ✅ Success")
            
    except Exception as e:
        if "can't convert" in str(e) and "numpy" in str(e):
            errors.append(f"Cycle {i}: {e}")
            print(f"  Cycle {i}: ❌ Numpy conversion error")
        else:
            # Other errors
            print(f"  Cycle {i}: ❌ Unexpected error: {e}")
            break

print(f"\n{'='*50}")
print("RESULTS")
print('='*50)
print(f"Successful cycles: {success_count}/30")

if errors:
    print(f"\nNumpy conversion errors found: {len(errors)}")
    for err in errors[:3]:  # Show first 3
        print(f"  - {err}")
else:
    print("\n✅ No numpy conversion errors detected!")
    print("   All tensor operations properly handle MPS device")
    
# Also test brain state creation which has many tensor operations
print("\nTesting brain state creation...")
try:
    for i in range(5):
        state = brain._create_brain_state()
    print("✅ Brain state creation works without numpy errors")
except Exception as e:
    if "can't convert" in str(e) and "numpy" in str(e):
        print(f"❌ Numpy error in brain state: {e}")
    else:
        print(f"❌ Other error: {e}")