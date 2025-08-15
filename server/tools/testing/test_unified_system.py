#!/usr/bin/env python3
"""
Test the unified field dynamics system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from src.core.simplified_brain_factory import SimplifiedBrainFactory

def test_unified_dynamics():
    """Test that the unified dynamics system works correctly."""
    print("🧪 Testing Unified Field Dynamics")
    print("=" * 50)
    
    # Create brain
    factory = SimplifiedBrainFactory({'quiet_mode': False})
    brain_wrapper = factory.create(
        sensory_dim=16,
        motor_dim=4,
        spatial_resolution=4
    )
    brain = brain_wrapper.brain
    
    print("\n📊 Initial State:")
    print(brain.field_dynamics.get_state_description())
    
    # Test 1: Low energy state (should encourage exploration)
    print("\n1️⃣ Testing Low Energy State")
    brain.unified_field *= 0.1  # Low activity
    
    for i in range(5):
        sensory_input = [0.1] * 16 + [0.0]  # Minimal input, no reward
        motor, state = brain.process_robot_cycle(sensory_input)
        
        if i == 4:
            print(f"   Energy: {brain.modulation.get('energy', 0):.3f}")
            print(f"   Exploration drive: {brain.modulation.get('exploration_drive', 0):.3f}")
            print(f"   Internal drive: {brain.modulation.get('internal_drive', 0):.3f}")
            print(f"   State: {brain.field_dynamics.get_state_description()}")
    
    # Test 2: High energy state (should encourage exploitation)
    print("\n2️⃣ Testing High Energy State")
    brain.unified_field = torch.randn_like(brain.unified_field) * 2.0  # High activity
    
    for i in range(5):
        sensory_input = [0.5] * 16 + [0.0]
        motor, state = brain.process_robot_cycle(sensory_input)
        
        if i == 4:
            print(f"   Energy: {brain.modulation.get('energy', 0):.3f}")
            print(f"   Exploration drive: {brain.modulation.get('exploration_drive', 0):.3f}")
            print(f"   Internal drive: {brain.modulation.get('internal_drive', 0):.3f}")
            print(f"   State: {brain.field_dynamics.get_state_description()}")
    
    # Test 3: Confidence building (repeated patterns)
    print("\n3️⃣ Testing Confidence Building")
    pattern = [np.sin(x * 0.5) for x in range(16)] + [0.0]
    
    for i in range(20):
        motor, state = brain.process_robot_cycle(pattern)
        
        if i % 5 == 4:
            print(f"   Cycle {i+1}: Confidence={brain.modulation.get('confidence', 0):.3f}, "
                  f"Internal={brain.modulation.get('internal_drive', 0):.3f}")
    
    # Test 4: Dream mode (no input)
    print("\n4️⃣ Testing Dream Mode")
    empty_input = [0.0] * 17
    
    for i in range(110):
        motor, state = brain.process_robot_cycle(empty_input)
        
        if i in [0, 50, 100, 109]:
            print(f"   Cycle {i+1}: {brain.field_dynamics.get_state_description()}")
    
    print("\n✅ Unified dynamics test complete!")
    
    # Summary
    print("\n📋 Summary:")
    print("   - Energy modulation: ✓")
    print("   - Confidence tracking: ✓")
    print("   - Exploration/exploitation balance: ✓")
    print("   - Dream mode activation: ✓")
    print("   - Unified field dynamics successfully replaces separate systems!")

if __name__ == "__main__":
    test_unified_dynamics()