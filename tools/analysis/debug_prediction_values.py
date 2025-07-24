#!/usr/bin/env python3
"""Debug why prediction confidence is always 1.0"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brains.field.core_brain import UnifiedFieldBrain

print("=== DEBUGGING PREDICTION VALUES ===\n")

# Create brain
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=True)

# First cycle - no prediction yet
sensors1 = [0.5] * 24
action1, state1 = brain.process_robot_cycle(sensors1)
print(f"Cycle 1: confidence={state1['prediction_confidence']:.6f}")

# Check field values
print(f"\nField shape: {brain.unified_field.shape}")
print(f"Field max: {torch.max(brain.unified_field):.6f}")
print(f"Field mean: {torch.mean(brain.unified_field):.6f}")
print(f"Field min: {torch.min(brain.unified_field):.6f}")

# Second cycle - should have prediction
sensors2 = [0.6] * 24  # Different input
action2, state2 = brain.process_robot_cycle(sensors2)
print(f"\nCycle 2: confidence={state2['prediction_confidence']:.6f}")

# Check if we have the post evolution field
if hasattr(brain, '_post_evolution_field'):
    print("\n✓ Post-evolution field exists")
    
    # Look at the local region
    center = brain.spatial_resolution // 2
    region_slice = slice(center-1, center+2)
    
    current_region = brain.unified_field[region_slice, region_slice, region_slice]
    post_evo_region = brain._post_evolution_field[region_slice, region_slice, region_slice]
    
    print(f"\nCurrent region shape: {current_region.shape}")
    print(f"Current region max: {torch.max(current_region):.6f}")
    print(f"Post-evo region max: {torch.max(post_evo_region):.6f}")
    
    # Calculate difference
    diff = torch.abs(current_region - post_evo_region)
    print(f"\nDifference max: {torch.max(diff):.6f}")
    print(f"Difference mean: {torch.mean(diff):.6f}")
    
    # Manual calculation
    prediction_error = torch.mean(diff).item()
    manual_confidence = 1.0 / (1.0 + prediction_error * 10.0)
    print(f"\nManual calculation:")
    print(f"Prediction error: {prediction_error:.6f}")
    print(f"Manual confidence: {manual_confidence:.6f}")

# Third cycle with very different input
sensors3 = [0.1, 0.9, 0.2] + [0.5] * 21  # Very different
action3, state3 = brain.process_robot_cycle(sensors3)
print(f"\nCycle 3 (different): confidence={state3['prediction_confidence']:.6f}")

# Check field evolution parameters
print(f"\nField parameters:")
print(f"Decay rate: {brain.field_decay_rate}")
print(f"Evolution rate: {brain.field_evolution_rate}")
print(f"Diffusion rate: {brain.field_diffusion_rate}")

brain.shutdown()