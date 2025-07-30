#!/usr/bin/env python3
"""
Test Topology Region System

Tests the new topology region detection, abstraction formation,
and causal tracking capabilities.
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_topology_detection():
    """Test basic topology region detection."""
    print("🧠 Testing Topology Region Detection")
    print("=" * 60)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=False,
        use_optimized=False  # Disable optimization for clearer testing
    )
    
    print("\n1. Creating distinctive patterns")
    print("-" * 40)
    
    # Create some distinctive sensory patterns
    patterns = [
        # Pattern 1: Light on left
        [1.0] * 6 + [0.0] * 18,
        # Pattern 2: Light on right  
        [0.0] * 6 + [1.0] * 6 + [0.0] * 12,
        # Pattern 3: Obstacle in front
        [0.0] * 12 + [1.0] * 6 + [0.0] * 6,
        # Pattern 4: Reward
        [0.5] * 23 + [2.0],
    ]
    
    # Process patterns multiple times to establish regions
    for cycle in range(3):
        print(f"\n  Cycle {cycle + 1}:")
        for i, pattern in enumerate(patterns):
            motors, state = brain.process_robot_cycle(pattern)
            topology_info = state.get('topology_regions', {})
            print(f"    Pattern {i+1}: {topology_info.get('activated_now', 0)} regions activated")
            time.sleep(0.1)
    
    # Check established regions
    stats = brain.topology_region_system.get_statistics()
    print(f"\n2. Topology Statistics:")
    print(f"   Total regions discovered: {stats['total_regions']}")
    print(f"   Currently active regions: {stats['active_regions']}")
    print(f"   Average connections: {stats['avg_connections']:.2f}")
    
    return brain


def test_causal_relationships(brain):
    """Test causal relationship detection."""
    print("\n\n🔗 Testing Causal Relationships")
    print("=" * 60)
    
    print("1. Creating causal sequence: Light → Move → Reward")
    print("-" * 40)
    
    # Simulate causal sequence multiple times
    for trial in range(5):
        print(f"\n  Trial {trial + 1}:")
        
        # Step 1: Light stimulus
        light_pattern = [1.0] * 6 + [0.0] * 18
        motors, state = brain.process_robot_cycle(light_pattern)
        light_regions = getattr(brain, '_last_activated_regions', [])
        print(f"    Light → {len(light_regions)} regions")
        time.sleep(0.2)
        
        # Step 2: Movement (pretend it's a response)
        move_pattern = [0.5] * 24
        motors, state = brain.process_robot_cycle(move_pattern)  
        move_regions = getattr(brain, '_last_activated_regions', [])
        print(f"    Move → {len(move_regions)} regions")
        time.sleep(0.2)
        
        # Step 3: Reward
        reward_pattern = [0.3] * 23 + [1.0]
        motors, state = brain.process_robot_cycle(reward_pattern)
        reward_regions = getattr(brain, '_last_activated_regions', [])
        print(f"    Reward → {len(reward_regions)} regions")
        time.sleep(0.5)
    
    # Analyze causal chains
    print("\n2. Detected Causal Chains:")
    print("-" * 40)
    
    regions = brain.topology_region_system.regions
    for region_id, region in list(regions.items())[:5]:  # Show first 5
        if region.causal_successors:
            print(f"\n  {region_id}:")
            print(f"    Activation count: {region.activation_count}")
            print(f"    Causal successors: {list(region.causal_successors.keys())[:3]}")
            
            # Get causal chains
            chains = brain.topology_region_system.get_causal_chain(region_id, max_depth=3)
            if chains:
                print(f"    Example chain: {' → '.join(chains[0])}")


def test_abstraction_formation(brain):
    """Test abstraction formation from co-activated regions."""
    print("\n\n🎯 Testing Abstraction Formation")
    print("=" * 60)
    
    print("1. Creating compound patterns for abstraction")
    print("-" * 40)
    
    # Create compound patterns that should form abstractions
    for cycle in range(10):
        # Compound pattern: Light + Sound + Touch
        compound_pattern = (
            [1.0] * 6 +      # Light
            [0.8] * 6 +      # Sound  
            [0.6] * 6 +      # Touch
            [0.0] * 6        # Nothing
        )
        
        motors, state = brain.process_robot_cycle(compound_pattern)
        
        if cycle % 3 == 0:
            stats = brain.topology_region_system.get_statistics()
            print(f"  Cycle {cycle}: {stats['abstract_regions']} abstract regions")
    
    # Check for abstract regions
    abstract_regions = brain.topology_region_system.get_active_abstractions()
    print(f"\n2. Abstract Regions Found: {len(abstract_regions)}")
    
    for i, region in enumerate(abstract_regions[:3]):  # Show first 3
        print(f"\n  Abstract Region {i+1}:")
        print(f"    Components: {len(region.component_regions)}")
        print(f"    Abstraction level: {region.abstraction_level}")
        print(f"    Associated regions: {len(region.associated_regions)}")


def test_memory_consolidation(brain):
    """Test how topology regions are consolidated."""
    print("\n\n💾 Testing Memory Consolidation")
    print("=" * 60)
    
    # Get initial state
    initial_stats = brain.topology_region_system.get_statistics()
    print(f"Before consolidation: {initial_stats['total_regions']} regions")
    
    # Run maintenance
    print("\nRunning maintenance...")
    brain.perform_maintenance()
    
    # Check consolidated regions
    consolidated_count = 0
    for region in brain.topology_region_system.regions.values():
        if region.consolidation_level > 0:
            consolidated_count += 1
    
    print(f"\nConsolidated regions: {consolidated_count}")
    
    # Show some consolidated regions
    print("\nExample consolidated regions:")
    for region_id, region in list(brain.topology_region_system.regions.items())[:5]:
        if region.consolidation_level > 0:
            print(f"  {region_id}: level={region.consolidation_level}, "
                  f"importance={region.importance:.2f}, "
                  f"activations={region.activation_count}")


def main():
    """Run all topology tests."""
    print("🔬 Topology Region System Test Suite")
    print("=" * 80)
    
    # Test 1: Basic detection
    brain = test_topology_detection()
    
    # Test 2: Causal relationships
    test_causal_relationships(brain)
    
    # Test 3: Abstraction formation
    test_abstraction_formation(brain)
    
    # Test 4: Memory consolidation
    test_memory_consolidation(brain)
    
    # Final statistics
    print("\n\n📊 Final Statistics")
    print("=" * 60)
    final_stats = brain.topology_region_system.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Topology region system test complete!")


if __name__ == "__main__":
    main()