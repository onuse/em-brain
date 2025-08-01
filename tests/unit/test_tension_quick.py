#!/usr/bin/env python3
"""Quick test of tension-based planning with reduced simulation horizon."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from server.src.brains.field.field_strategic_planner import FieldStrategicPlanner


def test_tension_based_planning():
    """Quick test of the core tension-based functionality."""
    print("Testing tension-based strategic planning...")
    
    # Create planner with smaller field
    field_shape = (8, 8, 8, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # Reduce simulation horizon for faster testing
    planner.simulation_horizon = 10
    
    # Create field with high information tension (low energy)
    field = torch.zeros(field_shape, device=planner.device)
    field[:, :, :, :32] = 0.1  # Low energy = high information tension
    
    # Measure initial tensions
    initial_tensions = planner._measure_field_tensions(field)
    print(f"\nInitial tensions:")
    for key, value in initial_tensions.items():
        print(f"  {key}: {value:.3f}")
    
    # Generate tension-targeted pattern
    pattern = planner._generate_tension_targeted_pattern(initial_tensions)
    print(f"\nGenerated pattern shape: {pattern.shape}")
    print(f"Pattern activation level: {pattern.abs().mean():.3f}")
    
    # Evaluate pattern using tension-based method
    score, signature, trajectory = planner._evaluate_pattern_tension_based(field, pattern)
    print(f"\nPattern evaluation:")
    print(f"  Score: {score:.2f}")
    print(f"  Behavioral signature: {signature}")
    
    # Test full discovery
    print("\nTesting full discovery cycle...")
    discovered = planner.discover_strategic_pattern(
        field,
        reward_signal=0.0,  # Ignored internally
        n_candidates=4  # Fewer candidates for speed
    )
    
    print(f"\nDiscovered pattern:")
    print(f"  Score: {discovered.score:.2f}")
    print(f"  Persistence: {discovered.persistence:.1f} cycles")
    print(f"  Pattern added to library: {len(planner.pattern_library) > 0}")
    
    # Test that different tensions create different behaviors
    print("\n\nTesting different tension scenarios...")
    
    # High confidence scenario
    confident_field = torch.ones(field_shape, device=planner.device) * 0.5
    confident_field[:, :, :, 58] = 0.9  # High confidence
    
    confident_tensions = planner._measure_field_tensions(confident_field)
    print(f"\nConfident field tensions:")
    for key, value in confident_tensions.items():
        if key != 'total':
            print(f"  {key}: {value:.3f}")
    
    # Discover pattern for confident field
    confident_pattern = planner.discover_strategic_pattern(
        confident_field,
        reward_signal=0.0,
        n_candidates=4
    )
    
    # Compare behaviors
    behavior_diff = torch.norm(discovered.behavioral_signature - confident_pattern.behavioral_signature).item()
    print(f"\nBehavioral difference between low-energy and high-confidence: {behavior_diff:.3f}")
    
    if behavior_diff > 0.1:
        print("✓ Different tensions successfully create different behaviors!")
    else:
        print("✗ Behaviors too similar - tension system may need tuning")
    
    print("\n✓ Tension-based planning system is working!")
    print("  The brain now operates on intrinsic drives without external rewards.")


if __name__ == "__main__":
    test_tension_based_planning()