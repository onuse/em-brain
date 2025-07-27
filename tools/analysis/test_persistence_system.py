#!/usr/bin/env python3
"""
Test Persistence System

Verify that brain state persistence is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import torch
import shutil
from pathlib import Path
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.persistence.integrated_persistence import initialize_persistence, get_persistence


def test_persistence_system():
    """Test brain state persistence and recovery."""
    print("\n=== Testing Persistence System ===\n")
    
    # Set up test directory
    test_memory_path = "./test_brain_memory"
    if Path(test_memory_path).exists():
        shutil.rmtree(test_memory_path)
    
    # Initialize persistence
    print("1️⃣ Initializing persistence system...")
    try:
        persistence = initialize_persistence(
            memory_path=test_memory_path,
            save_interval_cycles=10,  # Save every 10 cycles
            auto_save=True
        )
        print(f"   ✅ Persistence initialized at: {test_memory_path}")
        print(f"   Save interval: 10 cycles")
        print(f"   Auto-save: enabled")
    except Exception as e:
        print(f"   ❌ Failed to initialize persistence: {e}")
        return
    
    # Create first brain
    print("\n2️⃣ Creating initial brain...")
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True,
    })
    
    brain_wrapper1 = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=16,
        motor_dim=4
    )
    brain1 = brain_wrapper1.brain
    
    print(f"   Brain created with field shape: {brain1.unified_field.shape}")
    
    # Process some patterns to establish brain state
    print("\n3️⃣ Processing patterns to establish brain state...")
    patterns = [
        [1.0] * 16 + [0.0],
        [0.0] * 16 + [0.0],
        [0.5, 1.0] * 8 + [0.0],
    ]
    
    for i in range(15):
        pattern = patterns[i % len(patterns)]
        motor_output, brain_state = brain1.process_robot_cycle(pattern)
        
        if i % 5 == 0:
            print(f"   Cycle {i}: Field energy = {brain_state.get('field_energy', 0):.4f}")
    
    # Get final state
    final_energy1 = float(torch.mean(torch.abs(brain1.unified_field)))
    final_constraints1 = len(brain1.constraint_field.active_constraints) if hasattr(brain1, 'constraint_field') else 0
    
    print(f"\n   Final state:")
    print(f"   - Field energy: {final_energy1:.6f}")
    print(f"   - Active constraints: {final_constraints1}")
    print(f"   - Brain cycles: {brain1.brain_cycles}")
    
    # Save brain state
    print("\n4️⃣ Saving brain state...")
    try:
        save_success = persistence.save_brain_state(brain1, blocking=True)
        if save_success:
            print(f"   ✅ Brain state saved successfully")
        else:
            print(f"   ❌ Save returned False")
    except Exception as e:
        print(f"   ❌ Failed to save brain state: {e}")
    
    # Create new brain to test recovery
    print("\n5️⃣ Creating new brain for recovery test...")
    brain_wrapper2 = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=16,
        motor_dim=4
    )
    brain2 = brain_wrapper2.brain
    
    # Check initial state
    initial_energy2 = float(torch.mean(torch.abs(brain2.unified_field)))
    print(f"   New brain initial field energy: {initial_energy2:.6f}")
    
    # Check initial state before recovery
    initial_cycles = brain2.brain_cycles
    
    # Recover state
    print("\n6️⃣ Recovering brain state...")
    try:
        recovery_success = persistence.recover_brain_state(brain2)
        if recovery_success:
            print(f"   ✅ Brain state recovered successfully")
        else:
            print(f"   ⚠️  Recovery returned False")
    except Exception as e:
        print(f"   ❌ Failed to recover brain state: {e}")
        recovery_success = False
    
    # Verify recovery immediately
    if recovery_success:
        recovered_energy = float(torch.mean(torch.abs(brain2.unified_field)))
        recovered_constraints = len(brain2.constraint_field.active_constraints) if hasattr(brain2, 'constraint_field') else 0
        
        print(f"\n   Recovered state immediately after restoration:")
        print(f"   - Field energy: {recovered_energy:.6f}")
        print(f"   - Active constraints: {recovered_constraints}")
        print(f"   - Brain cycles: {brain2.brain_cycles} (was {initial_cycles})")
        
        # Compare
        energy_diff = abs(recovered_energy - final_energy1)
        print(f"\n   Comparison:")
        print(f"   - Energy difference: {energy_diff:.6f}")
        print(f"   - Constraints match: {recovered_constraints == final_constraints1}")
        
        if energy_diff < 0.001:
            print("\n✅ Persistence system is WORKING!")
            print("   - Brain state saved successfully")
            print("   - Brain state recovered successfully")
            print("   - Field values match within tolerance")
        else:
            print("\n⚠️  Persistence system has issues:")
            print("   - State was saved and loaded")
            print("   - But field values don't match exactly")
    
    # Check what files were created
    print("\n7️⃣ Checking persistence files:")
    memory_path = Path(test_memory_path)
    if memory_path.exists():
        files = list(memory_path.rglob("*"))
        print(f"   Found {len(files)} files:")
        for f in files[:5]:  # Show first 5
            if f.is_file():
                print(f"   - {f.relative_to(memory_path)} ({f.stat().st_size} bytes)")
    
    # Cleanup
    if Path(test_memory_path).exists():
        shutil.rmtree(test_memory_path)
        print("\n🧹 Cleaned up test files")


if __name__ == "__main__":
    test_persistence_system()