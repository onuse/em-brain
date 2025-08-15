#!/usr/bin/env python3
"""
Test field decay and energy accumulation
"""

import sys
import os
from pathlib import Path

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
import numpy as np
import torch


def test_field_decay():
    """Test how quickly field accumulates energy"""
    print("\n=== Testing Field Decay and Accumulation ===")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # Check initial field statistics
    field_abs = torch.abs(brain.unified_field)
    print(f"Initial field stats:")
    print(f"  Mean: {field_abs.mean():.3f}")
    print(f"  Max: {field_abs.max():.3f}")
    print(f"  Energy: {field_abs.mean():.3f}")
    
    # Single strong input
    print("\nApplying single strong input...")
    sensory_input = [1.0] * 4
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    field_abs = torch.abs(brain.unified_field)
    print(f"After input:")
    print(f"  Mean: {field_abs.mean():.3f}")
    print(f"  Max: {field_abs.max():.3f}")
    print(f"  Field energy: {brain_state['field_energy']:.3f}")
    print(f"  Normalized energy: {brain_state['energy_state']['energy']:.3f}")
    
    # Let it decay
    print("\nDecaying with no input...")
    for i in range(10):
        sensory_input = [0.0] * 4
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        if i % 2 == 0:
            print(f"  Cycle {i+1}: Field energy={brain_state['field_energy']:.3f}, "
                  f"Exploration={brain_state['energy_state']['exploration_drive']:.3f}")
    
    # Check decay rate
    print("\n=== Checking Decay Parameters ===")
    print(f"Field decay rate: {brain.field_decay_rate}")
    print(f"Modulation decay rate: {brain.modulation.get('decay_rate', 'not set')}")
    
    # Test imprint strength
    print("\n=== Testing Imprint Strength ===")
    brain2 = SimplifiedUnifiedBrain(
        sensory_dim=4,
        motor_dim=3,
        spatial_resolution=8,
        device='cpu',
        quiet_mode=True
    )
    
    # Weak input
    sensory_input = [0.1] * 4
    motor_output, brain_state = brain2.process_robot_cycle(sensory_input)
    print(f"Weak input (0.1): Field energy={brain_state['field_energy']:.3f}")
    
    # Medium input
    sensory_input = [0.5] * 4
    motor_output, brain_state = brain2.process_robot_cycle(sensory_input)
    print(f"Medium input (0.5): Field energy={brain_state['field_energy']:.3f}")
    
    # Strong input
    sensory_input = [1.0] * 4
    motor_output, brain_state = brain2.process_robot_cycle(sensory_input)
    print(f"Strong input (1.0): Field energy={brain_state['field_energy']:.3f}")
    
    # Check imprint parameters
    print(f"\nImprint strength: {brain2._last_imprint_strength:.3f}")
    print(f"Modulated imprint: {brain2.modulation.get('imprint_strength', 'not set')}")


if __name__ == "__main__":
    print("Testing field decay dynamics...")
    test_field_decay()
    print("\n✅ Test complete!")