#!/usr/bin/env python3
"""
Test novelty computation specifically
"""

import sys
import os
from pathlib import Path

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.brains.field.evolved_field_dynamics import EvolvedFieldDynamics
import torch
import numpy as np


def test_novelty_computation():
    """Test that novelty computation works correctly"""
    print("\n=== Testing Novelty Computation ===")
    
    # Create field dynamics
    field_shape = (4, 4, 4, 64)
    dynamics = EvolvedFieldDynamics(
        field_shape=field_shape,
        device=torch.device('cpu')
    )
    
    # Create a test field
    field = torch.randn(field_shape) * 0.1
    
    # Test first pattern (should be high novelty)
    novelty1 = dynamics.compute_novelty(field)
    print(f"First pattern novelty: {novelty1:.3f} (should be 1.0)")
    assert novelty1 == 1.0, f"First pattern should have novelty 1.0, got {novelty1}"
    
    # Test same pattern (should be low novelty)
    novelty2 = dynamics.compute_novelty(field)
    print(f"Same pattern novelty: {novelty2:.3f} (should be low)")
    assert novelty2 < 0.5, f"Same pattern should have low novelty, got {novelty2}"
    assert novelty2 >= 0.1, f"Novelty should have minimum floor of 0.1, got {novelty2}"
    
    # Test different pattern (should be higher novelty)
    field2 = torch.randn(field_shape) * 0.2
    novelty3 = dynamics.compute_novelty(field2)
    print(f"Different pattern novelty: {novelty3:.3f} (should be higher)")
    assert novelty3 > novelty2, f"Different pattern should have higher novelty"
    
    # Test temporal forgetting
    # Advance time significantly
    dynamics.evolution_count = 1500
    novelty4 = dynamics.compute_novelty(field)
    print(f"\nAfter 1500 cycles, same pattern novelty: {novelty4:.3f} (should be higher due to forgetting)")
    assert novelty4 > novelty2, f"Old patterns should be forgotten over time"
    
    print("\n✓ Novelty computation working correctly!")


def test_novelty_in_brain():
    """Test novelty computation in full brain context"""
    print("\n=== Testing Novelty in Brain Context ===")
    
    from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_resolution=8,  # Small for speed
        device='cpu',
        quiet_mode=True
    )
    
    novelty_values = []
    
    # Process some cycles
    for i in range(20):
        # Vary input to create novelty
        if i % 5 == 0:
            sensory_input = [np.random.randn() * 0.5 for _ in range(16)]
        else:
            sensory_input = [0.1] * 16
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        novelty = brain_state['energy_state'].get('novelty', 0)
        novelty_values.append(novelty)
        
        if i < 5 or i % 5 == 0:
            print(f"Cycle {i}: Novelty={novelty:.3f}")
    
    # Check results
    print(f"\nNovelty range: {min(novelty_values):.3f} - {max(novelty_values):.3f}")
    assert max(novelty_values) > 0.1, "Should have some non-zero novelty"
    assert min(novelty_values) >= 0.1, "Novelty should have minimum floor"
    
    print("✓ Novelty working in brain context!")


if __name__ == "__main__":
    print("Testing novelty computation...")
    
    test_novelty_computation()
    test_novelty_in_brain()
    
    print("\n✅ All novelty tests passed!")