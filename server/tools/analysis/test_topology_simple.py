#!/usr/bin/env python3
"""
Simple Topology Region Test

A cleaner test of the topology region system focusing on core functionality.
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def main():
    print("🧠 Simple Topology Region Test")
    print("=" * 60)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=True,  # Less output
        use_optimized=False
    )
    
    print("\n1. Testing Pattern Recognition")
    print("-" * 40)
    
    # Define test patterns
    patterns = {
        'light_left': [1.0] * 6 + [0.0] * 18,
        'light_right': [0.0] * 6 + [1.0] * 6 + [0.0] * 12,
        'obstacle': [0.0] * 12 + [1.0] * 6 + [0.0] * 6,
        'reward': [0.3] * 23 + [1.0],
    }
    
    # Process each pattern multiple times
    pattern_regions = {}
    for name, pattern in patterns.items():
        print(f"\n  Pattern: {name}")
        region_ids = set()
        
        for i in range(3):
            motors, state = brain.process_robot_cycle(pattern)
            activated = getattr(brain, '_last_activated_regions', [])
            region_ids.update(activated)
            print(f"    Cycle {i+1}: {len(activated)} regions activated")
            time.sleep(0.05)
        
        pattern_regions[name] = region_ids
        print(f"    Total unique regions: {len(region_ids)}")
    
    # Check if patterns create distinct regions
    print("\n2. Region Distinctiveness")
    print("-" * 40)
    
    for name1, regions1 in pattern_regions.items():
        for name2, regions2 in pattern_regions.items():
            if name1 < name2:
                overlap = len(regions1 & regions2)
                total = len(regions1 | regions2)
                similarity = overlap / total if total > 0 else 0
                print(f"  {name1} vs {name2}: {similarity:.1%} overlap")
    
    # Test temporal sequences
    print("\n3. Testing Causal Sequences")
    print("-" * 40)
    
    for trial in range(3):
        print(f"\n  Trial {trial + 1}: Light → Move → Reward")
        
        # Light
        motors, state = brain.process_robot_cycle(patterns['light_left'])
        light_regions = getattr(brain, '_last_activated_regions', [])
        
        # Movement
        time.sleep(0.1)
        motors, state = brain.process_robot_cycle([0.5] * 24)
        
        # Reward
        time.sleep(0.1)
        motors, state = brain.process_robot_cycle(patterns['reward'])
        reward_regions = getattr(brain, '_last_activated_regions', [])
        
        # Check for causal links
        causal_found = False
        for light_id in light_regions:
            if light_id in brain.topology_region_system.regions:
                light_region = brain.topology_region_system.regions[light_id]
                for reward_id in reward_regions:
                    if reward_id in light_region.causal_successors:
                        print(f"    ✓ Causal link: {light_id} → {reward_id}")
                        causal_found = True
                        break
                if causal_found:
                    break
        
        if not causal_found:
            print("    - No direct causal link yet")
    
    # Test abstraction
    print("\n4. Testing Abstraction Formation")
    print("-" * 40)
    
    # Create compound pattern repeatedly
    for i in range(10):
        compound = [0.0] * 24
        compound[0:6] = [0.8] * 6    # Light
        compound[6:12] = [0.6] * 6   # Sound
        compound[12:18] = [0.4] * 6  # Touch
        
        motors, state = brain.process_robot_cycle(compound)
        
        if i % 3 == 0:
            abstract_count = state['topology_regions']['abstract']
            print(f"  Cycle {i}: {abstract_count} abstract regions")
    
    # Final statistics
    print("\n5. Final Statistics")
    print("-" * 40)
    
    stats = brain.topology_region_system.get_statistics()
    print(f"  Total regions: {stats['total_regions']}")
    print(f"  Active regions: {stats['active_regions']}")
    print(f"  Abstract regions: {stats['abstract_regions']}")
    print(f"  Causal links: {stats['causal_links']}")
    print(f"  Avg connections: {stats['avg_connections']:.2f}")
    
    # Show example regions
    print("\n6. Example Regions")
    print("-" * 40)
    
    regions = list(brain.topology_region_system.regions.values())[:3]
    for region in regions:
        print(f"\n  {region.region_id}:")
        print(f"    Location: {region.spatial_center}")
        print(f"    Activations: {region.activation_count}")
        print(f"    Importance: {region.importance:.2f}")
        print(f"    Causal links: {len(region.causal_successors)} successors, "
              f"{len(region.causal_predecessors)} predecessors")
        
        if region.abstraction_level > 0:
            print(f"    Abstraction level: {region.abstraction_level}")
            print(f"    Components: {len(region.component_regions)}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()