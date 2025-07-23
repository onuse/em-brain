#!/usr/bin/env python3
"""Test what error occurs with constraint system enabled."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

from brains.field.core_brain import UnifiedFieldBrain

print("Testing constraint system...")

try:
    brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)
    
    # Process a few cycles to trigger constraint evolution
    for i in range(6):  # Need at least 5 cycles to trigger constraint discovery
        brain.process_robot_cycle([0.5] * 24)
        
    print("✅ No errors! Constraint system might be working.")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()