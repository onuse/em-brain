#!/usr/bin/env python3
"""Test brain initialization."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server/src'))

print("Importing...")
from brains.field.core_brain import UnifiedFieldBrain

print("Creating brain...")
brain = UnifiedFieldBrain(spatial_resolution=5, quiet_mode=False)
print("Brain created successfully!")
print(f"Field shape: {list(brain.unified_field.shape)}")