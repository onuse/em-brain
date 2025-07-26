#!/usr/bin/env python3
"""
Trace the prediction mechanism to understand why it's not working
"""

import sys
import os
from pathlib import Path
import torch

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def trace_prediction_mechanism():
    """Trace through the prediction mechanism step by step"""
    print("🔍 TRACING PREDICTION MECHANISM")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    ub = brain.brain
    
    print("\n📋 How prediction SHOULD work:")
    print("1. Cycle N: Compare predicted field (from N-1) with actual field after sensory input")
    print("2. Calculate prediction error and confidence")
    print("3. Evolve field forward in time")
    print("4. Store evolved field as prediction for cycle N+1")
    
    print("\n🔬 Let's trace what actually happens:")
    print("-" * 40)
    
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    
    # Cycle 1
    print("\n🔄 Cycle 1:")
    print("   - No _predicted_field exists yet")
    _, state1 = brain.process_sensory_input(pattern)
    print(f"   - Prediction confidence: {state1['prediction_confidence']:.3f} (default)")
    print(f"   - Has _predicted_field now: {hasattr(ub, '_predicted_field')}")
    
    # Check the predicted field
    if hasattr(ub, '_predicted_field'):
        print(f"   - Predicted field shape: {ub._predicted_field.shape}")
        print(f"   - Predicted field max: {ub._predicted_field.max().item():.6f}")
        print(f"   - Current field max: {ub.unified_field.max().item():.6f}")
        # Are they the same?
        field_diff = torch.mean(torch.abs(ub._predicted_field - ub.unified_field)).item()
        print(f"   - Difference between predicted and current: {field_diff:.8f}")
    
    # Cycle 2
    print("\n🔄 Cycle 2:")
    print("   - Now we have _predicted_field from cycle 1")
    
    # Let's manually check what the prediction error will be
    center = ub.spatial_resolution // 2
    region_slice = slice(center-1, center+2)
    
    # Store the predicted region before processing
    predicted_region_before = ub._predicted_field[region_slice, region_slice, region_slice].clone()
    
    _, state2 = brain.process_sensory_input(pattern)
    
    # Check what happened
    print(f"   - Prediction confidence: {state2['prediction_confidence']:.6f}")
    
    # Manually calculate what the error was
    actual_region_after = ub.unified_field[region_slice, region_slice, region_slice]
    manual_error = torch.mean(torch.abs(predicted_region_before - actual_region_after)).item()
    print(f"   - Manual prediction error: {manual_error:.8f}")
    print(f"   - This would give confidence: {1.0 / (1.0 + manual_error * 5000):.6f}")
    
    # The problem analysis
    print("\n❌ PROBLEM IDENTIFIED:")
    print("-" * 40)
    
    # Check if field evolution actually changes the field
    field_before_evolution = ub.unified_field.clone()
    ub._evolve_unified_field()
    field_after_evolution = ub.unified_field
    evolution_change = torch.mean(torch.abs(field_after_evolution - field_before_evolution)).item()
    
    print(f"1. Field evolution change: {evolution_change:.8f}")
    if evolution_change < 1e-6:
        print("   → Field evolution is not changing the field!")
    
    # Check if sensory input changes the field
    print("\n2. Let's check sensory input impact:")
    field_before_sensory = ub.unified_field.clone()
    from src.brains.field.core_brain import UnifiedFieldExperience
    experience = ub._robot_sensors_to_field_experience(pattern)
    ub._apply_field_experience(experience)
    sensory_change = torch.mean(torch.abs(ub.unified_field - field_before_sensory)).item()
    print(f"   Sensory input change: {sensory_change:.8f}")
    
    print("\n📊 SUMMARY:")
    print("-" * 40)
    print("The prediction mechanism exists but may have issues:")
    print("1. If field evolution doesn't change the field, predictions are trivial")
    print("2. If sensory input has minimal impact, there's nothing to predict")
    print("3. The 3x3x3 region might be too small to capture meaningful changes")

def investigate_field_sparsity():
    """Investigate why the field is so sparse"""
    print("\n\n🔍 INVESTIGATING FIELD SPARSITY")
    print("=" * 60)
    
    # Create brain
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    ub = brain.brain
    
    print("\n📊 Field Statistics:")
    print("-" * 40)
    
    # Field size
    total_elements = ub.unified_field.numel()
    print(f"Total field elements: {total_elements:,}")
    
    # Run a few cycles
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    for i in range(5):
        brain.process_sensory_input(pattern)
    
    # Analyze sparsity
    nonzero_mask = torch.abs(ub.unified_field) > 0.01
    active_elements = nonzero_mask.sum().item()
    sparsity = active_elements / total_elements * 100
    
    print(f"Active elements (>0.01): {active_elements:,} ({sparsity:.4f}%)")
    
    # Where are the active elements?
    active_indices = torch.nonzero(nonzero_mask)
    print(f"\nActive element distribution:")
    print(f"Number of active positions: {len(active_indices)}")
    
    if len(active_indices) > 0:
        # Check spatial distribution
        spatial_indices = active_indices[:, :3]  # First 3 dimensions are spatial
        unique_spatial = torch.unique(spatial_indices, dim=0)
        print(f"Unique spatial locations: {len(unique_spatial)}")
        print(f"Example spatial locations: {unique_spatial[:5].tolist()}")
    
    print("\n🔍 Why is the field sparse?")
    print("-" * 40)
    print("1. Imprinting might be too localized")
    print("2. Field evolution might not spread activation")
    print("3. Energy dissipation might be too aggressive")
    print("4. Initial field state starts at zero")

if __name__ == "__main__":
    trace_prediction_mechanism()
    investigate_field_sparsity()