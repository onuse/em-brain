#!/usr/bin/env python3
"""Test imprint directly."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)

print("Testing direct field modification...")

# Directly set a value
brain.unified_field[2, 2, 2, 5, 7, 1, 1, 1, 1, 1, 1] = 0.5
print(f"After direct set: max={brain.unified_field.max():.4f}")

# Apply evolution
brain._evolve_unified_field()
print(f"After evolution: max={brain.unified_field.max():.4f}")

# Try the imprint function directly
indices = (2, 2, 2, 5, 7, 1, 1, 1, 1, 1, 1)
brain._apply_multidimensional_imprint_new(indices, 0.3)
print(f"After imprint: max={brain.unified_field.max():.4f}")

# Check that specific location
val = brain.unified_field[indices].item()
print(f"Value at imprint location: {val:.4f}")

# One more evolution
brain._evolve_unified_field()
print(f"After 2nd evolution: max={brain.unified_field.max():.4f}")