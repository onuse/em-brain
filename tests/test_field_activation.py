#!/usr/bin/env python3
"""Debug field activation."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=True)
print(f"Initial field: min={brain.unified_field.min():.4f}, max={brain.unified_field.max():.4f}, mean={brain.unified_field.mean():.6f}")

# Create a test experience directly
import time
from brains.field.field_types import UnifiedFieldExperience

# Strong input pattern
raw_input = [0.9] * 24
field_coords = brain._map_sensory_to_field_coordinates(raw_input)
print(f"\nField coordinates norm: {torch.norm(field_coords).item():.4f}")

experience = UnifiedFieldExperience(
    timestamp=time.time(),
    raw_sensory_input=raw_input,
    field_coordinates=field_coords,
    field_intensity=0.5,  # Strong intensity
    experience_id="test"
)

print(f"Experience intensity: {experience.field_intensity}")

# Apply experience
brain._apply_field_experience(experience)

print(f"\nAfter experience: min={brain.unified_field.min():.4f}, max={brain.unified_field.max():.4f}, mean={brain.unified_field.mean():.6f}")

# Check specific locations
center = brain.spatial_resolution // 2
print(f"\nCenter region value: {brain.unified_field[center, center, center, 5, 7].item():.4f}")

# Now evolve field
brain._evolve_unified_field()
print(f"\nAfter evolution: min={brain.unified_field.min():.4f}, max={brain.unified_field.max():.4f}, mean={brain.unified_field.mean():.6f}")