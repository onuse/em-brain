#!/usr/bin/env python3
"""
Test integrated spontaneous dynamics in UnifiedFieldDynamics
"""

import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.unified_field_dynamics import UnifiedFieldDynamics


def test_spontaneous_activity():
    """Test that spontaneous activity is generated in field evolution."""
    print("Testing Integrated Spontaneous Dynamics")
    print("=" * 50)
    
    # Create field
    field_shape = (8, 8, 8, 16)
    field = torch.zeros(field_shape, dtype=torch.float32)
    
    # Create unified dynamics
    dynamics = UnifiedFieldDynamics(
        field_shape=field_shape,
        spontaneous_rate=0.01,
        resting_potential=0.01,
        device=torch.device('cpu')
    )
    
    # Test 1: Field should gain activity even without input
    print("\n1. Testing spontaneous activity generation...")
    initial_energy = torch.mean(torch.abs(field)).item()
    print(f"   Initial field energy: {initial_energy:.6f}")
    
    # Evolve field multiple times
    for i in range(10):
        field = dynamics.evolve_field(field)
    
    final_energy = torch.mean(torch.abs(field)).item()
    print(f"   Final field energy: {final_energy:.6f}")
    print(f"   ✓ Spontaneous activity: {'YES' if final_energy > initial_energy else 'NO'}")
    
    # Test 2: Modulation affects spontaneous activity
    print("\n2. Testing spontaneous modulation...")
    
    # High internal drive (more spontaneous)
    dynamics._last_modulation = {'spontaneous_weight': 0.9, 'decay_rate': 0.999}
    field_high = field.clone()
    for i in range(5):
        field_high = dynamics.evolve_field(field_high)
    energy_high = torch.mean(torch.abs(field_high)).item()
    
    # Low internal drive (less spontaneous)
    dynamics._last_modulation = {'spontaneous_weight': 0.1, 'decay_rate': 0.999}
    field_low = field.clone()
    for i in range(5):
        field_low = dynamics.evolve_field(field_low)
    energy_low = torch.mean(torch.abs(field_low)).item()
    
    print(f"   High spontaneous weight energy: {energy_high:.6f}")
    print(f"   Low spontaneous weight energy: {energy_low:.6f}")
    print(f"   ✓ Modulation works: {'YES' if energy_high > energy_low else 'NO'}")
    
    # Test 3: Traveling waves
    print("\n3. Testing traveling wave patterns...")
    
    # Reset field
    field = torch.zeros(field_shape, dtype=torch.float32)
    dynamics._last_modulation = {'spontaneous_weight': 1.0, 'decay_rate': 1.0}
    
    # Evolve and check for spatial patterns
    phases = []
    for i in range(20):
        field = dynamics.evolve_field(field)
        # Check center slice
        center_slice = field[4, 4, :, 0]
        phases.append(center_slice.clone())
    
    # Check if patterns change over time (traveling)
    pattern_changes = 0
    for i in range(1, len(phases)):
        diff = torch.mean(torch.abs(phases[i] - phases[i-1])).item()
        if diff > 0.001:
            pattern_changes += 1
    
    print(f"   Pattern changes: {pattern_changes}/19")
    print(f"   ✓ Traveling waves: {'YES' if pattern_changes > 10 else 'NO'}")
    
    # Test 4: Homeostatic regulation
    print("\n4. Testing homeostatic regulation...")
    
    # Start with high activity
    field = torch.randn(field_shape) * 0.5
    initial = torch.mean(torch.abs(field)).item()
    
    # Evolve many times
    for i in range(50):
        field = dynamics.evolve_field(field)
    
    final = torch.mean(torch.abs(field)).item()
    target = dynamics.resting_potential
    
    print(f"   Initial activity: {initial:.6f}")
    print(f"   Final activity: {final:.6f}")
    print(f"   Target (resting): {target:.6f}")
    print(f"   ✓ Homeostasis: {'YES' if abs(final - target) < abs(initial - target) else 'NO'}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    test_spontaneous_activity()