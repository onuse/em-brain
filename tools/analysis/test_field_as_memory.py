#!/usr/bin/env python3
"""
Test the enhanced field-as-memory implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
import time
from brains.field.core_brain import create_unified_field_brain


def test_memory_formation_and_recall():
    """Test if experiences create persistent memories."""
    print("Testing Field-as-Memory: Formation and Recall")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=False
    )
    
    # Create distinct experiences
    experiences = [
        ([0.9, 0.1, 0.5] + [0.8] * 21, "Strong Left Turn"),
        ([0.1, 0.9, 0.5] + [0.8] * 21, "Strong Right Turn"),
        ([0.5, 0.5, 0.9] + [0.9] * 21, "High Vertical"),
        ([0.5, 0.5, 0.1] + [0.2] * 21, "Low Energy")
    ]
    
    print("\n1. MEMORY FORMATION:")
    print("-" * 40)
    
    # Form memories by repeating experiences
    for input_data, label in experiences:
        print(f"\nForming memory: {label}")
        for i in range(5):  # Repeat to strengthen
            action, state = brain.process_robot_cycle(input_data)
            if i == 0:
                print(f"  Initial topology regions: {state['topology_regions_count']}")
        print(f"  Final topology regions: {state['topology_regions_count']}")
    
    # Show formed memories
    print(f"\n\nFormed {len(brain.topology_regions)} memory regions:")
    for key, region in brain.topology_regions.items():
        print(f"  {key}: importance={region['importance']:.2f}, "
              f"activations={region['activation_count']}, "
              f"consolidation={region['consolidation_level']}")
    
    # Test recall through resonance
    print("\n\n2. MEMORY RECALL (Resonance):")
    print("-" * 40)
    
    # Present similar but not identical stimuli
    test_inputs = [
        ([0.85, 0.15, 0.5] + [0.7] * 21, "Similar to Left Turn"),
        ([0.15, 0.85, 0.5] + [0.7] * 21, "Similar to Right Turn"),
        ([0.5, 0.5, 0.5] + [0.5] * 21, "Neutral Input")
    ]
    
    for input_data, label in test_inputs:
        print(f"\nPresenting: {label}")
        
        # Track which memories get activated
        initial_importances = {k: v['importance'] for k, v in brain.topology_regions.items()}
        
        action, state = brain.process_robot_cycle(input_data)
        
        # Check which memories were strengthened
        activated = []
        for key, region in brain.topology_regions.items():
            if region['importance'] > initial_importances.get(key, 0):
                activated.append(key)
        
        if activated:
            print(f"  Activated memories: {activated}")
        else:
            print(f"  No strong memory activation")
        
        print(f"  Output action: {[f'{a:.3f}' for a in action]}")


def test_memory_persistence():
    """Test if important memories persist longer."""
    print("\n\nTesting Memory Persistence")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Create memories with different importance
    print("Creating memories with different frequencies...")
    
    # Important memory (repeated many times)
    important_input = [0.9, 0.1, 0.5] + [0.8] * 21
    for _ in range(10):
        brain.process_robot_cycle(important_input)
    
    # Less important memory (repeated few times)
    casual_input = [0.1, 0.9, 0.5] + [0.3] * 21
    for _ in range(2):
        brain.process_robot_cycle(casual_input)
    
    print(f"\nInitial state:")
    for key, region in brain.topology_regions.items():
        print(f"  {key}: importance={region['importance']:.2f}, "
              f"activations={region['activation_count']}")
    
    # Run many empty cycles to test decay
    print("\nRunning 200 neutral cycles...")
    for _ in range(200):
        brain.process_robot_cycle([0.5] * 24)
        
        # Trigger maintenance periodically
        if brain.brain_cycles % 50 == 0:
            brain._run_field_maintenance()
    
    print(f"\nAfter decay and consolidation:")
    surviving_regions = []
    for key, region in brain.topology_regions.items():
        print(f"  {key}: importance={region['importance']:.2f}, "
              f"consolidation={region['consolidation_level']}, "
              f"decay_rate={region['decay_rate']:.4f}")
        if region['importance'] > 0.5:
            surviving_regions.append(key)
    
    print(f"\nSurviving memories: {surviving_regions}")
    
    if len(surviving_regions) > 0:
        print("✅ SUCCESS: Important memories persist!")
    else:
        print("⚠️  WARNING: All memories decayed")


def test_memory_interference():
    """Test how new experiences affect existing memories."""
    print("\n\nTesting Memory Interference")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=12,
        quiet_mode=True
    )
    
    # Create initial memory
    print("Creating initial memory...")
    initial_input = [0.8, 0.2, 0.5] + [0.7] * 21
    for _ in range(5):
        brain.process_robot_cycle(initial_input)
    
    initial_regions = len(brain.topology_regions)
    print(f"Initial topology regions: {initial_regions}")
    
    # Create interfering memory (similar but different)
    print("\nCreating interfering memory...")
    interfering_input = [0.7, 0.3, 0.5] + [0.6] * 21
    for _ in range(5):
        brain.process_robot_cycle(interfering_input)
    
    final_regions = len(brain.topology_regions)
    print(f"Final topology regions: {final_regions}")
    
    # Check if memories merged or remained separate
    if final_regions == initial_regions + 1:
        print("✅ Memories remain distinct")
    elif final_regions == initial_regions:
        print("🔄 Memories merged or interfered")
    else:
        print(f"🆕 Created {final_regions - initial_regions} new regions")
    
    # Test recall of original memory
    print("\nTesting recall of original memory...")
    action1, _ = brain.process_robot_cycle(initial_input)
    action2, _ = brain.process_robot_cycle(interfering_input)
    
    action_diff = sum(abs(a - b) for a, b in zip(action1, action2))
    print(f"Action difference: {action_diff:.3f}")
    
    if action_diff > 0.1:
        print("✅ Different memories produce different outputs")
    else:
        print("⚠️  Memories produce similar outputs")


def test_memory_capacity():
    """Test the natural capacity limits of field memory."""
    print("\n\nTesting Memory Capacity")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    print("Creating many different memories...")
    
    # Create many distinct memories
    memories_created = 0
    for i in range(50):
        # Create unique input pattern
        input_data = [
            0.5 + 0.4 * np.sin(i * 0.3),
            0.5 + 0.4 * np.cos(i * 0.4),
            0.5 + 0.3 * np.sin(i * 0.2)
        ] + [0.5 + 0.1 * np.sin(i * 0.1 + j) for j in range(21)]
        
        brain.process_robot_cycle(input_data)
        memories_created += 1
        
        # Run maintenance occasionally
        if i % 10 == 0:
            brain._run_field_maintenance()
            print(f"  After {memories_created} memories: {len(brain.topology_regions)} regions active")
    
    print(f"\n\nCapacity Analysis:")
    print(f"  Memories created: {memories_created}")
    print(f"  Regions maintained: {len(brain.topology_regions)}")
    print(f"  Retention rate: {len(brain.topology_regions) / memories_created:.1%}")
    
    # Analyze importance distribution
    importances = [r['importance'] for r in brain.topology_regions.values()]
    if importances:
        print(f"\nImportance distribution:")
        print(f"  Min: {min(importances):.2f}")
        print(f"  Max: {max(importances):.2f}")
        print(f"  Mean: {np.mean(importances):.2f}")


def analyze_memory_statistics():
    """Analyze overall memory system behavior."""
    print("\n\nMemory System Statistics")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Run a typical session
    print("Running typical usage session...")
    
    for cycle in range(100):
        # Vary input patterns
        if cycle % 20 == 0:
            # Strong memorable event
            input_data = [0.9, 0.1, 0.5] + [0.8] * 21
        elif cycle % 5 == 0:
            # Moderate event
            input_data = [0.7, 0.3, 0.5] + [0.6] * 21
        else:
            # Normal variation
            input_data = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
        
        brain.process_robot_cycle(input_data)
    
    print(f"\n\nSession Statistics:")
    print(f"  Total cycles: {brain.brain_cycles}")
    print(f"  Topology regions: {len(brain.topology_regions)}")
    print(f"  Field evolution cycles: {brain.field_evolution_cycles}")
    
    # Memory characteristics
    if brain.topology_regions:
        activations = [r['activation_count'] for r in brain.topology_regions.values()]
        consolidations = [r['consolidation_level'] for r in brain.topology_regions.values()]
        
        print(f"\nMemory Characteristics:")
        print(f"  Most activated: {max(activations)} times")
        print(f"  Most consolidated: level {max(consolidations)}")
        print(f"  Average activation: {np.mean(activations):.1f}")
    
    # Field energy analysis
    field_energy = torch.sum(torch.abs(brain.unified_field)).item()
    print(f"\nField State:")
    print(f"  Total energy: {field_energy:.3f}")
    print(f"  Energy per cycle: {field_energy / brain.brain_cycles:.4f}")


if __name__ == "__main__":
    test_memory_formation_and_recall()
    test_memory_persistence()
    test_memory_interference()
    test_memory_capacity()
    analyze_memory_statistics()
    
    print("\n\n✅ Field-as-Memory testing complete!")
    print("\nThe brain IS the memory - no separation needed!")