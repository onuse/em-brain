#!/usr/bin/env python3
"""Debug full cycle with all steps."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

# Patch to see each step
original_cycle = UnifiedFieldBrain.process_robot_cycle

def debug_cycle(self, sensory_input):
    print(f"\n=== CYCLE {self.brain_cycles} ===")
    print(f"Input: {sensory_input[:3]}... (first 3 values)")
    
    # Before
    print(f"Field before: max={self.unified_field.max():.4f}, mean={self.unified_field.mean():.6f}")
    
    # Call original
    result = original_cycle(self, sensory_input)
    
    # After
    print(f"Field after: max={self.unified_field.max():.4f}, mean={self.unified_field.mean():.6f}")
    print(f"Action: {result[0][:3]}... (first 3 values)")
    
    return result

UnifiedFieldBrain.process_robot_cycle = debug_cycle

# Also patch apply_field_experience
original_apply = UnifiedFieldBrain._apply_field_experience

def debug_apply(self, experience):
    print("  Applying experience:")
    print(f"    Before: max={self.unified_field.max():.4f}")
    print(f"    Intensity: {experience.field_intensity}")
    original_apply(self, experience)
    print(f"    After: max={self.unified_field.max():.4f}")

UnifiedFieldBrain._apply_field_experience = debug_apply

# Also patch evolution to see what happens
original_evolve = UnifiedFieldBrain._evolve_unified_field

def debug_evolve(self):
    print("  Evolution:")
    print(f"    Before: max={self.unified_field.max():.4f}")
    original_evolve(self)
    print(f"    After: max={self.unified_field.max():.4f}")

UnifiedFieldBrain._evolve_unified_field = debug_evolve

# Create and test
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

# Run a few cycles
for i in range(3):
    brain.process_robot_cycle([0.9] * 24)