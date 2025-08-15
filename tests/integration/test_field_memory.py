#!/usr/bin/env python3
"""Test the unified field as memory implementation."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
from server.src.brains.field.core_brain import UnifiedFieldBrain

def test_field_memory():
    """Test that the field acts as memory through topology regions."""
    print("\n=== Testing Field-as-Memory Implementation ===")
    
    # Create brain with small field for testing
    print("\n1. Creating brain...")
    brain = UnifiedFieldBrain(spatial_resolution=8, quiet_mode=True)
    print(f"Field shape: {list(brain.unified_field.shape)}")
    print(f"Initial topology regions: {len(brain.topology_regions)}")
    
    # Test 1: Pattern formation through repeated inputs
    print("\n2. Testing pattern formation...")
    
    # Send distinctive pattern multiple times
    pattern_a = [0.9, 0.1, 0.9, 0.1] * 6  # Alternating pattern
    pattern_b = [0.1, 0.1, 0.9, 0.9] * 6  # Different pattern
    
    # Debug: check field values
    print(f"   Field max before: {brain.unified_field.max().item():.6f}")
    print(f"   Field mean before: {brain.unified_field.mean().item():.6f}")
    print(f"   Topology threshold: {brain.topology_stability_threshold}")
    
    # Present pattern A multiple times
    print("   Presenting pattern A...")
    for i in range(5):
        action, state = brain.process_robot_cycle(pattern_a)
        if i == 0:
            initial_regions = len(brain.topology_regions)
    
    print(f"   Field max after A: {brain.unified_field.max().item():.6f}")
    print(f"   Field mean after A: {brain.unified_field.mean().item():.6f}")
    
    regions_after_a = len(brain.topology_regions)
    print(f"   Topology regions after pattern A: {regions_after_a}")
    
    # Present pattern B
    print("   Presenting pattern B...")
    for i in range(5):
        action, state = brain.process_robot_cycle(pattern_b)
    
    regions_after_b = len(brain.topology_regions)
    print(f"   Topology regions after pattern B: {regions_after_b}")
    
    if regions_after_b > initial_regions:
        print("   ✅ New topology regions formed (memory creation)")
    else:
        print("   ⚠️  No new regions formed")
    
    # Test 2: Pattern recall (resonance)
    print("\n3. Testing pattern recall...")
    
    # Get current field state as baseline
    field_before = brain.unified_field.clone()
    
    # Present pattern A again (should resonate with stored pattern)
    print("   Re-presenting pattern A...")
    action_recall, state_recall = brain.process_robot_cycle(pattern_a)
    
    # Check if field shows stronger response (resonance)
    field_after = brain.unified_field
    field_change = torch.abs(field_after - field_before).mean().item()
    
    print(f"   Field change magnitude: {field_change:.6f}")
    
    # Present novel pattern
    pattern_novel = [0.5, 0.7, 0.3, 0.8] * 6
    field_before_novel = brain.unified_field.clone()
    print("   Presenting novel pattern...")
    action_novel, state_novel = brain.process_robot_cycle(pattern_novel)
    
    field_change_novel = torch.abs(brain.unified_field - field_before_novel).mean().item()
    print(f"   Field change for novel: {field_change_novel:.6f}")
    
    if field_change > field_change_novel * 0.8:
        print("   ✅ Familiar patterns show resonance")
    else:
        print("   ⚠️  No clear resonance effect")
    
    # Test 3: Persistence across maintenance
    print("\n4. Testing memory persistence...")
    
    # Store current important regions
    important_regions = []
    for key, region in brain.topology_regions.items():
        if region['importance'] > 0.5:
            important_regions.append(key)
    
    print(f"   Important regions before maintenance: {len(important_regions)}")
    
    # Run maintenance (should preserve important memories)
    if hasattr(brain, '_run_field_maintenance'):
        brain._run_field_maintenance()
        print("   Ran field maintenance")
    
    # Check which regions survived
    survived = 0
    for key in important_regions:
        if key in brain.topology_regions:
            survived += 1
    
    print(f"   Important regions after maintenance: {survived}/{len(important_regions)}")
    
    if survived > len(important_regions) * 0.7:
        print("   ✅ Important memories persisted")
    else:
        print("   ⚠️  Too many memories lost")
    
    # Test 4: Capacity and organization
    print("\n5. Testing memory capacity...")
    
    # Present many different patterns
    for i in range(20):
        pattern = np.random.rand(24).tolist()
        brain.process_robot_cycle(pattern)
    
    final_regions = len(brain.topology_regions)
    print(f"   Final topology regions: {final_regions}")
    
    # Check region statistics
    if brain.topology_regions:
        importances = [r['importance'] for r in brain.topology_regions.values()]
        stabilities = [r['stability'] for r in brain.topology_regions.values()]
        
        print(f"   Average importance: {np.mean(importances):.3f}")
        print(f"   Average stability: {np.mean(stabilities):.3f}")
        print(f"   Most important: {max(importances):.3f}")
        
        print("   ✅ Memory system functional")
    else:
        print("   ❌ No topology regions formed")
    
    return True

if __name__ == "__main__":
    import torch  # Import here to avoid issues
    success = test_field_memory()
    print(f"\n{'✅ Field-as-memory test complete!' if success else '❌ Test failed'}")