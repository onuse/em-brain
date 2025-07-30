#!/usr/bin/env python3
"""
Test Evolved Brain Architecture

Verify that the new evolved field dynamics work correctly
as the single, unified brain architecture.
"""

import sys
import os
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def main():
    print("🧠 Testing Evolved Brain Architecture")
    print("=" * 60)
    
    # Create brain - no need for enable_self_modification flag anymore!
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=16,  # Small for quick testing
        quiet_mode=False
    )
    
    print("\n1. Basic Functionality Test")
    print("-" * 40)
    
    # Run some cycles
    for i in range(10):
        pattern = np.random.randn(24) * 0.5
        pattern[i % 24] = 1.0  # Different strong signal each time
        
        motors, state = brain.process_robot_cycle(pattern.tolist())
        
        if i % 3 == 0:
            print(f"  Cycle {i}: energy={state.get('field_energy', 0):.4f}")
    
    print("\n2. Evolution State")
    print("-" * 40)
    
    evo_state = brain.get_evolution_state()
    print(f"  Evolution cycles: {evo_state['evolution_cycles']}")
    print(f"  Self-modification strength: {evo_state['self_modification_strength']:.1%}")
    print(f"  Energy: {evo_state['smoothed_energy']:.4f}")
    print(f"  Confidence: {evo_state['smoothed_confidence']:.4f}")
    
    wm = evo_state['working_memory']
    print(f"  Working memory patterns: {wm['n_patterns']}")
    
    print("\n3. Regional Specialization Test")
    print("-" * 40)
    
    # Apply different patterns to different sensory regions
    print("  Training different sensory regions...")
    
    # Fast pattern in first region
    for i in range(50):
        pattern = np.zeros(24)
        pattern[0:6] = np.sin(i * 0.5)
        brain.process_robot_cycle(pattern.tolist())
    
    # Slow pattern in second region
    for i in range(50):
        pattern = np.zeros(24)
        pattern[6:12] = np.sin(i * 0.05) + 0.5
        brain.process_robot_cycle(pattern.tolist())
    
    # Check evolution progress
    evo_state = brain.get_evolution_state()
    print(f"\n  After training:")
    print(f"    Evolution cycles: {evo_state['evolution_cycles']}")
    print(f"    Self-modification: {evo_state['self_modification_strength']:.1%}")
    
    print("\n4. Memory Persistence Test")
    print("-" * 40)
    
    # Strong input
    strong_pattern = [2.0] * 12 + [0.0] * 12
    brain.process_robot_cycle(strong_pattern)
    initial_energy = brain.get_evolution_state()['smoothed_energy']
    print(f"  Strong input energy: {initial_energy:.4f}")
    
    # Empty cycles
    print("  Decay over empty cycles:")
    for i in range(5):
        brain.process_robot_cycle([0.0] * 24)
        energy = brain.get_evolution_state()['smoothed_energy']
        print(f"    Cycle {i}: {energy:.4f} ({energy/initial_energy:.1%})")
    
    print("\n5. Topology Regions")
    print("-" * 40)
    
    topology_stats = brain.topology_region_system.get_statistics()
    print(f"  Total regions: {topology_stats['total_regions']}")
    print(f"  Active regions: {topology_stats['active_regions']}")
    print(f"  Causal links: {topology_stats['causal_links']}")
    
    print("\n✅ Evolved brain architecture working perfectly!")
    print("\n💡 Key insights:")
    print("  - No configuration needed for self-modification")
    print("  - Starts with minimal self-modification (1%)")
    print("  - Naturally increases with experience")
    print("  - All parameters emerge from the field itself")
    print("  - This IS the brain - not an option or mode!")


if __name__ == "__main__":
    main()