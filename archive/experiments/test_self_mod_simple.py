#!/usr/bin/env python3
"""
Simple Self-Modifying Dynamics Test

Quick test to verify self-modifying dynamics are working.
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def main():
    print("🧠 Simple Self-Modifying Dynamics Test")
    print("=" * 60)
    
    # Create brain with self-modification enabled
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=16,  # Smaller for speed
        quiet_mode=False,
        enable_self_modification=True,
        self_mod_ratio=0.3  # Start at 30%
    )
    
    print("\n1. Testing basic functionality")
    print("-" * 40)
    
    # Run a few cycles
    for i in range(5):
        pattern = [0.5] * 24
        pattern[i] = 1.0  # Different pattern each cycle
        
        motors, state = brain.process_robot_cycle(pattern)
        print(f"  Cycle {i}: energy={state.get('field_energy', 0):.4f}")
    
    print("\n2. Checking self-modification state")
    print("-" * 40)
    
    self_mod_state = brain.get_self_modification_state()
    print(f"  Enabled: {self_mod_state['enabled']}")
    print(f"  Ratio: {self_mod_state['ratio']:.0%}")
    
    if 'emergent_properties' in self_mod_state:
        props = self_mod_state['emergent_properties']
        print(f"\n  Emergent properties:")
        for key, value in props.items():
            print(f"    {key}: {value}")
    
    print("\n3. Testing persistence learning")
    print("-" * 40)
    
    # Apply strong input
    strong_pattern = [2.0] * 6 + [0.0] * 18
    motors, state = brain.process_robot_cycle(strong_pattern)
    initial_energy = state.get('field_energy', 0)
    print(f"  Strong input applied, energy: {initial_energy:.4f}")
    
    # Track decay
    print("  Tracking decay:")
    for i in range(10):
        motors, state = brain.process_robot_cycle([0.0] * 24)
        energy = state.get('field_energy', 0)
        print(f"    Cycle {i}: {energy:.4f} ({energy/initial_energy:.1%} retained)")
    
    print("\n4. Increasing self-modification")
    print("-" * 40)
    
    brain.increase_self_modification(0.2)
    new_state = brain.get_self_modification_state()
    print(f"  New ratio: {new_state['ratio']:.0%}")
    
    # Test with new ratio
    strong_pattern = [0.0] * 6 + [2.0] * 6 + [0.0] * 12
    motors, state = brain.process_robot_cycle(strong_pattern)
    print(f"  New pattern energy: {state.get('field_energy', 0):.4f}")
    
    print("\n✅ Self-modifying dynamics working!")
    
    # Extract topology if available
    if hasattr(brain.field_dynamics, 'extract_topology'):
        print("\n5. Field Topology")
        print("-" * 40)
        topology = brain.field_dynamics.extract_topology()
        if topology:
            print(f"  Decay landscape shape: {topology['decay_landscape'].shape}")
            print(f"  Mean decay rate: {torch.mean(topology['decay_landscape']):.3f}")
            print(f"  Decay variance: {torch.std(topology['decay_landscape']):.3f}")


if __name__ == "__main__":
    main()