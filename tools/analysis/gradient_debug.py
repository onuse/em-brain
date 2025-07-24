#!/usr/bin/env python3
"""Debug why gradients are too weak."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brain_factory import BrainFactory

print("=== GRADIENT DEBUG ===\n")

# Create brain with some output
config = {'type': 'unified_field', 'spatial_resolution': 5, 'quiet_mode': False}
brain = BrainFactory(config)
field_brain = brain.brain

print("\n1. INITIAL FIELD STATE:")
print(f"Field max: {torch.max(field_brain.unified_field):.6f}")
print(f"Field mean: {torch.mean(field_brain.unified_field):.6f}")
print(f"Field min: {torch.min(field_brain.unified_field):.6f}")

print("\n2. PROCESSING STRONG PATTERN:")
# Very strong obstacle pattern
strong_pattern = [0.1, 0.1, 0.9] + [0.8] * 21 + [-1.0]  # Strong values + negative reward
action, state = brain.process_sensory_input(strong_pattern)

print(f"\nAfter processing:")
print(f"Field max: {torch.max(field_brain.unified_field):.6f}")
print(f"Field mean: {torch.mean(field_brain.unified_field):.6f}")

print("\n3. CHECKING GRADIENTS:")
if field_brain.gradient_flows:
    for name, grad in field_brain.gradient_flows.items():
        grad_max = torch.max(torch.abs(grad)).item()
        grad_mean = torch.mean(torch.abs(grad)).item()
        print(f"{name}: max={grad_max:.6f}, mean={grad_mean:.6f}")
else:
    print("No gradients found!")

print("\n4. CHECKING GRADIENT CALCULATION:")
# Force gradient calculation
field_brain._calculate_gradient_flows()

if field_brain.gradient_flows:
    print("\nAfter forced calculation:")
    for name, grad in field_brain.gradient_flows.items():
        grad_max = torch.max(torch.abs(grad)).item()
        grad_mean = torch.mean(torch.abs(grad)).item()
        print(f"{name}: max={grad_max:.6f}, mean={grad_mean:.6f}")

print("\n5. FIELD DECAY ANALYSIS:")
print(f"Field decay rate: {field_brain.field_decay_rate}")
print(f"Field diffusion rate: {field_brain.field_diffusion_rate}")
print(f"Gradient following strength: {field_brain.gradient_following_strength}")

# Check the local region
center = field_brain.spatial_resolution // 2
print(f"\nCenter index: {center}")
print(f"Checking local region around [{center}, {center}, {center}]")

# Extract local region
local_region = field_brain.unified_field[center-1:center+2, center-1:center+2, center-1:center+2]
print(f"Local region shape: {local_region.shape}")
print(f"Local region max: {torch.max(local_region):.6f}")
print(f"Local region values > 0.01: {(local_region > 0.01).sum().item()}")

# Process multiple times to build up field
print("\n6. BUILDING UP FIELD:")
for i in range(5):
    action, state = brain.process_sensory_input(strong_pattern)
    field_max = torch.max(field_brain.unified_field).item()
    print(f"Cycle {i+1}: field_max={field_max:.3f}, action={action}")

brain.shutdown()