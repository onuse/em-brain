#!/usr/bin/env python3
"""
Quick test to verify GPU utilization in field brain
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.brains.field.generic_brain import GenericFieldBrain, DEVICE, GPU_AVAILABLE

print("🔧 GPU Field Brain Test")
print(f"Device selected: {DEVICE}")
print(f"GPU available: {GPU_AVAILABLE}")

# Create field brain with verbose logging
field_brain = GenericFieldBrain(
    spatial_resolution=10,  # Smaller for testing
    temporal_window=5.0,
    field_evolution_rate=0.1,
    constraint_discovery_rate=0.15,
    quiet_mode=False  # Show GPU device info
)

print(f"✅ Field brain created successfully")
print(f"Unified field device: {field_brain.unified_field.device}")
print(f"Field device: {field_brain.field_device}")