#!/usr/bin/env python3
"""
Test Integrated Persistence System

Verifies that the brain can save and restore its state across sessions,
maintaining learning continuity.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import shutil
import tempfile
from pathlib import Path

from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.persistence.integrated_persistence import IntegratedPersistence


def test_persistence_basic():
    """Test basic save and restore functionality."""
    
    print("\n🧪 Testing Basic Persistence")
    print("=" * 60)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persistence manager
        persistence = IntegratedPersistence(
            memory_path=temp_dir,
            save_interval_cycles=10,
            auto_save=False  # Manual control for testing
        )
        
        # Create brain
        factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'quiet_mode': True
        })
        
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Process some cycles to create state
        print("\n📝 Creating brain state...")
        for i in range(20):
            sensors = [0.5 + 0.1 * i] * 7 + [0.1]  # Gradual change
            motors, state = brain_wrapper.brain.process_robot_cycle(sensors)
            
            if i % 5 == 0:
                print(f"   Cycle {i}: confidence={state['prediction_confidence']:.2f}")
        
        # Save state
        print("\n💾 Saving brain state...")
        success = persistence.save_brain_state(brain_wrapper, blocking=True)
        assert success, "Failed to save brain state"
        
        # Get state info before clearing
        original_cycles = brain_wrapper.brain.brain_cycles
        original_energy = state['field_energy']
        original_confidence = state['prediction_confidence']
        
        print(f"   Saved at cycle {original_cycles}")
        print(f"   Field energy: {original_energy:.4f}")
        print(f"   Confidence: {original_confidence:.2f}")
        
        # Create new brain
        print("\n🔄 Creating new brain and recovering state...")
        brain_wrapper2 = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Verify new brain starts fresh
        assert brain_wrapper2.brain.brain_cycles == 0, "New brain should start at cycle 0"
        
        # Recover state
        success = persistence.recover_brain_state(brain_wrapper2)
        assert success, "Failed to recover brain state"
        
        # Verify state was restored
        print(f"\n✅ State recovered:")
        print(f"   Brain cycles: {brain_wrapper2.brain.brain_cycles} (was {original_cycles})")
        print(f"   Has memory regions: {len(brain_wrapper2.brain.topology_regions) > 0}")
        
        # Process a cycle to verify brain works
        sensors = [0.7] * 7 + [0.1]
        motors, state = brain_wrapper2.brain.process_robot_cycle(sensors)
        print(f"   Post-recovery confidence: {state['prediction_confidence']:.2f}")
        
        assert brain_wrapper2.brain.brain_cycles == original_cycles + 1, "Brain cycles not restored correctly"
        
        print("\n✅ Basic persistence test passed!")


def test_persistence_auto_save():
    """Test automatic saving during operation."""
    
    print("\n\n🧪 Testing Auto-Save Functionality")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persistence with auto-save
        persistence = IntegratedPersistence(
            memory_path=temp_dir,
            save_interval_cycles=25,  # Save every 25 cycles
            auto_save=True
        )
        
        # Create brain
        factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'quiet_mode': True
        })
        
        brain_wrapper = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        print("\n📝 Processing with auto-save enabled...")
        
        # Process many cycles
        for i in range(100):
            sensors = [0.5 + 0.2 * (i % 10) / 10] * 7 + [0.0]
            motors, state = brain_wrapper.brain.process_robot_cycle(sensors)
            
            # Check for auto-save
            if persistence.check_auto_save(brain_wrapper):
                print(f"   Auto-saved at cycle {i}")
        
        # Check how many saves occurred
        stats = persistence.get_persistence_stats()
        print(f"\n📊 Auto-save stats:")
        print(f"   Total saves: {stats['save_count']}")
        print(f"   State files: {stats['state_files']}")
        
        assert stats['save_count'] >= 3, "Should have auto-saved at least 3 times"
        
        print("\n✅ Auto-save test passed!")


def test_persistence_continuity():
    """Test learning continuity across sessions."""
    
    print("\n\n🧪 Testing Learning Continuity")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Session 1: Initial learning
        print("\n📚 Session 1: Initial learning")
        persistence1 = IntegratedPersistence(memory_path=temp_dir)
        
        factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'quiet_mode': True
        })
        
        brain1 = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Train on pattern A
        pattern_a = [0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.5]
        for i in range(50):
            motors, state = brain1.brain.process_robot_cycle(pattern_a)
        
        confidence_a = state['prediction_confidence']
        print(f"   Learned pattern A: confidence={confidence_a:.2f}")
        
        # Save and close
        persistence1.shutdown_save(brain1)
        
        # Session 2: Resume learning
        print("\n📚 Session 2: Resume with memory")
        persistence2 = IntegratedPersistence(memory_path=temp_dir)
        
        brain2 = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Recover state
        persistence2.recover_brain_state(brain2)
        
        # Test on pattern A again
        motors, state = brain2.brain.process_robot_cycle(pattern_a)
        confidence_a_recalled = state['prediction_confidence']
        print(f"   Recalled pattern A: confidence={confidence_a_recalled:.2f}")
        
        # Learn new pattern B
        pattern_b = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.5]
        for i in range(50):
            motors, state = brain2.brain.process_robot_cycle(pattern_b)
        
        confidence_b = state['prediction_confidence']
        print(f"   Learned pattern B: confidence={confidence_b:.2f}")
        
        # Test pattern A again (should still remember)
        motors, state = brain2.brain.process_robot_cycle(pattern_a)
        confidence_a_retained = state['prediction_confidence']
        print(f"   Pattern A retained: confidence={confidence_a_retained:.2f}")
        
        # The brain should show it remembers pattern A from session 1
        assert confidence_a_recalled > 0.5, "Brain forgot pattern A"
        assert brain2.brain.brain_cycles > 50, "Brain cycles not continued from session 1"
        
        print("\n✅ Learning continuity test passed!")


def test_persistence_crash_recovery():
    """Test recovery from incomplete saves."""
    
    print("\n\n🧪 Testing Crash Recovery")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        persistence = IntegratedPersistence(memory_path=temp_dir)
        
        factory = DynamicBrainFactory({
            'use_dynamic_brain': True,
            'use_full_features': True,
            'quiet_mode': True
        })
        
        brain = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Create some state
        print("\n📝 Creating state with multiple saves...")
        for i in range(75):
            sensors = [0.5] * 7 + [0.0]
            brain.brain.process_robot_cycle(sensors)
            
            if i % 25 == 24:
                persistence.save_brain_state(brain, blocking=True)
                print(f"   Saved at cycle {i+1}")
        
        # Simulate crash (no shutdown save)
        print("\n💥 Simulating crash (no shutdown save)")
        
        # New session
        print("\n🔄 Starting new session after crash...")
        persistence2 = IntegratedPersistence(memory_path=temp_dir)
        
        brain2 = factory.create(
            field_dimensions=None,
            spatial_resolution=3,
            sensory_dim=8,
            motor_dim=2
        )
        
        # Recover
        success = persistence2.recover_brain_state(brain2)
        assert success, "Failed to recover after crash"
        
        print(f"✅ Recovered to cycle {brain2.brain.brain_cycles}")
        assert brain2.brain.brain_cycles >= 50, "Should have recovered to at least cycle 50"
        
        print("\n✅ Crash recovery test passed!")


def main():
    """Run all persistence tests."""
    print("🧠 INTEGRATED PERSISTENCE SYSTEM TESTS")
    print("=" * 80)
    
    test_persistence_basic()
    test_persistence_auto_save()
    test_persistence_continuity()
    test_persistence_crash_recovery()
    
    print("\n\n🎉 ALL PERSISTENCE TESTS PASSED!")
    print("\nThe brain can now:")
    print("  • Save its complete state")
    print("  • Recover state on startup")
    print("  • Continue learning across sessions")
    print("  • Auto-save during operation")
    print("  • Recover from crashes")
    print("\nNo more amnesia! 🧠💾")


if __name__ == "__main__":
    main()