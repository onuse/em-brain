#!/usr/bin/env python3
"""
Test Temporal Persistence Working Memory

Demonstrates how temporal persistence creates working memory emergence
through differential decay rates in the 4D field architecture.
"""

import sys
import os
import time
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_working_memory_persistence():
    """Test that temporal features persist longer than spatial features."""
    print("🧠 Testing Temporal Persistence Working Memory")
    print("=" * 60)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=False
    )
    
    print("\n1. Testing Differential Decay Rates")
    print("-" * 40)
    
    # Create a strong pattern
    test_pattern = [1.0] * 12 + [0.0] * 12  # Strong sensory input
    
    # Process the pattern to imprint it
    print("  Imprinting strong pattern...")
    motors, state = brain.process_robot_cycle(test_pattern)
    initial_energy = state['field_energy']
    print(f"  Initial field energy: {initial_energy:.4f}")
    
    # Extract initial working memory state
    wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
    initial_wm_activation = wm_state['mean_activation'].item()
    print(f"  Initial working memory activation: {initial_wm_activation:.4f}")
    
    # Now process empty cycles and track decay
    print("\n  Tracking decay over empty cycles:")
    spatial_activations = []
    temporal_activations = []
    
    for i in range(20):
        # Process empty input
        motors, state = brain.process_robot_cycle([0.0] * 24)
        
        # Get field state
        field = brain.unified_field
        
        # Get the correct split from field dynamics
        spatial_features = brain.field_dynamics.spatial_features
        temporal_features = brain.field_dynamics.temporal_features
        
        # Measure spatial features
        spatial_field = field[:, :, :, :spatial_features]
        spatial_activation = torch.mean(torch.abs(spatial_field)).item()
        spatial_activations.append(spatial_activation)
        
        # Measure temporal features
        temporal_field = field[:, :, :, spatial_features:]
        temporal_activation = torch.mean(torch.abs(temporal_field)).item()
        temporal_activations.append(temporal_activation)
        
        if i % 5 == 0:
            print(f"    Cycle {i}: spatial={spatial_activation:.4f}, temporal={temporal_activation:.4f}")
    
    # Calculate decay rates
    spatial_decay_rate = spatial_activations[-1] / (spatial_activations[0] + 1e-8)
    temporal_decay_rate = temporal_activations[-1] / (temporal_activations[0] + 1e-8)
    
    print(f"\n  Decay analysis:")
    print(f"    Spatial features retained: {spatial_decay_rate:.1%}")
    print(f"    Temporal features retained: {temporal_decay_rate:.1%}")
    print(f"    Temporal persistence advantage: {temporal_decay_rate/spatial_decay_rate:.1f}x")
    
    print("\n2. Testing Sequential Pattern Memory")
    print("-" * 40)
    
    # Present a sequence of patterns
    patterns = [
        [1.0, 0.0, 0.0] + [0.0] * 21,  # Pattern A
        [0.0, 1.0, 0.0] + [0.0] * 21,  # Pattern B
        [0.0, 0.0, 1.0] + [0.0] * 21,  # Pattern C
    ]
    
    print("  Presenting sequence A → B → C...")
    for i, pattern in enumerate(patterns):
        motors, state = brain.process_robot_cycle(pattern)
        wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
        print(f"    Pattern {chr(65+i)}: {wm_state['n_patterns']} patterns in working memory")
        time.sleep(0.1)
    
    # Check if sequence is retained
    print("\n  Checking working memory after sequence:")
    wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
    print(f"    Patterns in memory: {wm_state['n_patterns']}")
    print(f"    Temporal coherence: {wm_state['temporal_coherence'].item():.3f}")
    
    # Process empty cycles and check persistence
    print("\n  Testing sequence persistence:")
    for i in range(10):
        motors, state = brain.process_robot_cycle([0.0] * 24)
        if i % 3 == 0:
            wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
            print(f"    Cycle {i}: {wm_state['n_patterns']} patterns, "
                  f"coherence={wm_state['temporal_coherence'].item():.3f}")
    
    print("\n3. Testing Temporal Momentum")
    print("-" * 40)
    
    # Create a moving pattern
    print("  Creating moving pattern...")
    for position in range(5):
        pattern = [0.0] * 24
        pattern[position] = 1.0  # Moving light
        
        motors, state = brain.process_robot_cycle(pattern)
        
        # Check temporal position
        temporal_pos = brain.field_dynamics.temporal_position
        print(f"    Position {position}: temporal_position={temporal_pos}")
        time.sleep(0.05)
    
    # Check if momentum predicts next position
    print("\n  Checking predictive momentum:")
    
    # Get field state before and after evolution
    field_before = brain.unified_field.clone()
    brain._evolve_field()
    field_after = brain.unified_field
    
    # Check if temporal features show forward propagation
    spatial_features = brain.field_dynamics.spatial_features
    temporal_before = field_before[:, :, :, spatial_features:].mean().item()
    temporal_after = field_after[:, :, :, spatial_features:].mean().item()
    
    if temporal_after > temporal_before * 0.9:
        print("    ✓ Temporal momentum maintains activation")
    else:
        print("    - Temporal momentum weak")
    
    print("\n4. Testing Working Memory Capacity")
    print("-" * 40)
    
    # Test how many distinct patterns can be held
    print("  Loading multiple patterns...")
    
    distinct_patterns = []
    for i in range(8):
        pattern = [0.0] * 24
        pattern[i*3:(i+1)*3] = [1.0, 0.5, 0.25]  # Unique pattern
        distinct_patterns.append(pattern)
    
    # Present all patterns
    for i, pattern in enumerate(distinct_patterns):
        motors, state = brain.process_robot_cycle(pattern)
        if i % 2 == 0:
            wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
            print(f"    After pattern {i+1}: {wm_state['n_patterns']} in memory")
    
    # Final memory state
    wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
    print(f"\n  Final working memory state:")
    print(f"    Patterns retained: {wm_state['n_patterns']}")
    print(f"    Mean activation: {wm_state['mean_activation'].item():.4f}")
    print(f"    Temporal coherence: {wm_state['temporal_coherence'].item():.3f}")
    
    # Test recall by presenting partial cue
    print("\n5. Testing Pattern Recall")
    print("-" * 40)
    
    # Present partial cue from first pattern
    cue_pattern = [1.0, 0.0, 0.0] + [0.0] * 21
    print("  Presenting partial cue from first pattern...")
    
    # Check field response
    field_before_cue = brain.unified_field.clone()
    motors, state = brain.process_robot_cycle(cue_pattern)
    field_after_cue = brain.unified_field
    
    # Measure recall strength
    field_change = torch.mean(torch.abs(field_after_cue - field_before_cue)).item()
    print(f"    Field response to cue: {field_change:.4f}")
    
    # Check if temporal features show stronger response
    spatial_features = brain.field_dynamics.spatial_features
    temporal_change = torch.mean(
        torch.abs(field_after_cue[:, :, :, spatial_features:] - field_before_cue[:, :, :, spatial_features:])
    ).item()
    spatial_change = torch.mean(
        torch.abs(field_after_cue[:, :, :, :spatial_features] - field_before_cue[:, :, :, :spatial_features])
    ).item()
    
    print(f"    Spatial response: {spatial_change:.4f}")
    print(f"    Temporal response: {temporal_change:.4f}")
    
    if temporal_change > spatial_change:
        print("    ✓ Working memory shows enhanced recall")
    
    print("\n✅ Temporal persistence test complete!")
    
    return brain


def test_temporal_imprinting():
    """Test direct temporal pattern imprinting."""
    print("\n\n🎯 Testing Direct Temporal Imprinting")
    print("=" * 60)
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,
        use_optimized=False
    )
    
    print("\n1. Creating and imprinting test pattern")
    print("-" * 40)
    
    # Create a distinctive pattern
    test_pattern = torch.randn(32, 32, 32, 64, device=brain.device) * 0.5
    
    # Imprint it into temporal features
    print("  Imprinting pattern into temporal features...")
    brain.unified_field = brain.field_dynamics.imprint_temporal_pattern(
        brain.unified_field,
        test_pattern,
        strength=0.8
    )
    
    # Check imprint
    wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
    print(f"  Working memory after imprint: {wm_state['n_patterns']} patterns")
    print(f"  Mean activation: {wm_state['mean_activation'].item():.4f}")
    
    # Test persistence
    print("\n2. Testing imprint persistence")
    print("-" * 40)
    
    activations = []
    for i in range(15):
        brain._evolve_field()
        wm_state = brain.field_dynamics.get_working_memory_state(brain.unified_field)
        activation = wm_state['mean_activation'].item()
        activations.append(activation)
        
        if i % 5 == 0:
            print(f"  Evolution {i}: activation={activation:.4f}")
    
    # Check decay
    retention = activations[-1] / (activations[0] + 1e-8)
    print(f"\n  Pattern retention after 15 cycles: {retention:.1%}")
    
    print("\n✅ Temporal imprinting test complete!")


if __name__ == "__main__":
    # Run both tests
    brain = test_working_memory_persistence()
    test_temporal_imprinting()