#!/usr/bin/env python3
"""
Verification Test: 37D Field Brain After Dimension Cleanup
=========================================================
Tests that the 37D field brain works correctly after fixing
the 36D/37D documentation inconsistency.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

import torch
import numpy as np
from brains.field.core_brain import UnifiedFieldBrain

def test_37d_field_brain():
    """Test that the 37D field brain initializes and processes correctly."""
    
    print("🧠 Testing 37D UnifiedFieldBrain after dimension cleanup...")
    print("=" * 60)
    
    # Test 1: Initialization
    print("\n1️⃣ Testing brain initialization...")
    try:
        brain = UnifiedFieldBrain(
            spatial_resolution=15,  # Smaller for faster testing
            temporal_window=5.0,
            quiet_mode=False
        )
        print(f"   ✅ Brain initialized successfully")
        print(f"   📐 Field dimensions: {brain.total_dimensions}D")
        print(f"   🔧 Field shape: {brain.unified_field.shape}")
        print(f"   💻 Device: {brain.device}")
        
        # Verify it's exactly 37D
        assert brain.total_dimensions == 37, f"Expected 37D, got {brain.total_dimensions}D"
        print(f"   ✅ Dimension count verified: 37D")
        
    except Exception as e:
        print(f"   ❌ Brain initialization failed: {e}")
        return False
    
    # Test 2: Sensory Processing
    print("\n2️⃣ Testing sensory processing...")
    try:
        # Create test sensory input (24D as expected by robot interface)
        test_input = [0.1 * np.sin(i * 0.5) + 0.05 * np.random.randn() for i in range(24)]
        
        print(f"   📊 Input: {len(test_input)}D sensor data")
        
        # Process sensory input
        action, brain_state = brain.process_robot_cycle(test_input)
        
        print(f"   ✅ Processing successful")
        print(f"   🎯 Output: {len(action)}D action vector")
        print(f"   🧠 Brain cycles: {brain.brain_cycles}")
        print(f"   📈 State keys: {list(brain_state.keys())}")
        
    except Exception as e:
        print(f"   ❌ Sensory processing failed: {e}")
        return False
    
    # Test 3: Field Evolution
    print("\n3️⃣ Testing field evolution across multiple cycles...")
    try:
        initial_field_energy = torch.sum(brain.unified_field).item()
        
        for cycle in range(5):
            # Vary input slightly
            varied_input = [val + 0.02 * np.sin(cycle) for val in test_input]
            action, state = brain.process_robot_cycle(varied_input)
            
            field_energy = torch.sum(brain.unified_field).item()
            confidence = state.get('last_action_confidence', 0.0)
            
            print(f"   🔄 Cycle {cycle + 1}: Energy={field_energy:.1f}, Confidence={confidence:.3f}")
        
        print(f"   ✅ Field evolution successful across {brain.brain_cycles} total cycles")
        
    except Exception as e:
        print(f"   ❌ Field evolution failed: {e}")
        return False
    
    # Test 4: Dimension Family Verification
    print("\n4️⃣ Verifying dimension family structure...")
    try:
        from collections import defaultdict
        
        family_counts = defaultdict(int)
        for dim in brain.field_dimensions:
            family_counts[dim.family] += 1
        
        expected_families = {
            'spatial': 5,
            'oscillatory': 6, 
            'flow': 8,
            'topology': 6,
            'energy': 4,
            'coupling': 5,
            'emergence': 3
        }
        
        print("   📊 Dimension families:")
        total_dims = 0
        for family, expected_count in expected_families.items():
            actual_count = family_counts[getattr(brain.field_dimensions[0].family.__class__, family.upper())]
            total_dims += actual_count
            status = "✅" if actual_count == expected_count else "❌"
            print(f"      {status} {family}: {actual_count}D (expected {expected_count}D)")
        
        print(f"   📐 Total: {total_dims}D (expected 37D)")
        assert total_dims == 37, f"Dimension mismatch: {total_dims} != 37"
        
    except Exception as e:
        print(f"   ❌ Dimension verification failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ 37D UnifiedFieldBrain is working correctly after cleanup")
    print("🌊 Field-native intelligence system verified")
    return True

if __name__ == "__main__":
    success = test_37d_field_brain()
    exit(0 if success else 1)