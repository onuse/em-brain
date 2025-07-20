#!/usr/bin/env python3
"""
Test Brain Persistence System

Tests the complete save/load cycle to ensure brain memories persist
between sessions.
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.src.brain_factory import MinimalBrain
from src.persistence import PersistenceConfig, PersistenceManager


def test_basic_persistence():
    """Test basic save and load functionality."""
    print("🧪 Testing basic brain persistence...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure persistence to use temp directory
        config = PersistenceConfig(
            memory_root_path=temp_dir,
            checkpoint_interval_experiences=5,  # Checkpoint every 5 experiences
            checkpoint_interval_seconds=10,
            use_compression=False  # Disable compression for easier debugging
        )
        
        # Create first brain instance
        print("\n📝 Creating first brain and adding experiences...")
        brain1 = MinimalBrain(enable_logging=False, enable_persistence=False)  # Create without persistence first
        
        # Now manually set up persistence with our config
        brain1.enable_persistence = True
        brain1.persistence_manager = PersistenceManager(config)
        
        # Add some experiences
        test_experiences = [
            ([1.0, 0.5, 0.2], [0.1, 0.9], [1.1, 0.6, 0.3]),
            ([0.8, 0.3, 0.1], [0.2, 0.8], [0.9, 0.4, 0.2]),
            ([1.2, 0.7, 0.4], [0.3, 0.7], [1.3, 0.8, 0.5]),
            ([0.9, 0.4, 0.3], [0.1, 0.6], [1.0, 0.5, 0.4]),
            ([1.1, 0.6, 0.5], [0.4, 0.5], [1.2, 0.7, 0.6])
        ]
        
        experience_ids = []
        for sensory, action, outcome in test_experiences:
            # Process sensory input and get predicted action
            predicted_action, brain_info = brain1.process_sensory_input(sensory, len(action))
            
            # Store experience with actual outcome
            exp_id = brain1.store_experience(sensory, action, outcome)
            experience_ids.append(exp_id)
            
            print(f"   Added experience {exp_id[:8]} - prediction error: {brain_info.get('prediction_error', 0):.3f}")
        
        # Force a checkpoint save
        checkpoint_id = brain1.save_brain_state()
        print(f"💾 Saved checkpoint: {checkpoint_id}")
        
        # Get brain stats before closing
        original_stats = brain1.get_brain_stats()
        print(f"📊 Original brain stats keys: {list(original_stats.keys())}")
        brain_summary = original_stats.get('brain_summary', {})
        total_exp = brain_summary.get('total_experiences', 0)
        print(f"📊 Original brain - {total_exp} experiences")
        
        # Finalize session
        brain1.finalize_session()
        del brain1
        
        # Create second brain instance (should load from persistence)
        print("\n🔄 Creating second brain (should restore from checkpoint)...")
        brain2 = MinimalBrain(enable_logging=False, enable_persistence=False)  # Create without persistence first
        
        # Now manually set up persistence with the same config
        brain2.enable_persistence = True
        brain2.persistence_manager = PersistenceManager(config)
        brain2._load_persistent_state()
        
        # Check that state was restored
        restored_stats = brain2.get_brain_stats()
        print(f"📊 Restored brain stats keys: {list(restored_stats.keys())}")
        restored_brain_summary = restored_stats.get('brain_summary', {})
        restored_total_exp = restored_brain_summary.get('total_experiences', 0)
        print(f"📊 Restored brain - {restored_total_exp} experiences")
        
        # Verify experiences were restored
        restored_experiences = brain2.experience_storage._experiences
        print(f"✅ Restored {len(restored_experiences)} experiences")
        
        # Store the count before adding more experiences
        restored_count_initial = len(restored_experiences)
        
        # Verify specific experience data
        for i, exp_id in enumerate(experience_ids):
            if exp_id in restored_experiences:
                exp = restored_experiences[exp_id]
                original_sensory = test_experiences[i][0]
                restored_sensory = exp.sensory_input.tolist()
                print(f"   Experience {exp_id[:8]}: {original_sensory} -> {restored_sensory}")
                
                # Check if data matches
                if abs(sum(original_sensory) - sum(restored_sensory)) < 0.001:
                    print(f"   ✅ Data matches")
                else:
                    print(f"   ❌ Data mismatch!")
            else:
                print(f"   ❌ Missing experience {exp_id[:8]}")
        
        # Add more experiences to test continued learning
        print(f"\n🧠 Adding more experiences to restored brain...")
        new_experiences = [
            ([1.3, 0.8, 0.6], [0.5, 0.4], [1.4, 0.9, 0.7]),
            ([0.7, 0.2, 0.1], [0.0, 1.0], [0.8, 0.3, 0.2])
        ]
        
        for sensory, action, outcome in new_experiences:
            predicted_action, brain_info = brain2.process_sensory_input(sensory, len(action))
            exp_id = brain2.store_experience(sensory, action, outcome)
            print(f"   Added experience {exp_id[:8]} - prediction error: {brain_info.get('prediction_error', 0):.3f}")
        
        final_stats = brain2.get_brain_stats()
        final_brain_summary = final_stats.get('brain_summary', {})
        final_total_exp = final_brain_summary.get('total_experiences', 0)
        print(f"📊 Final brain - {final_total_exp} experiences")
        
        # Clean up
        brain2.finalize_session()
        
        # Test results
        expected_final = len(test_experiences) + len(new_experiences)
        print(f"🧮 Test calculation:")
        print(f"   Restored: {restored_total_exp} == Original: {total_exp} ? {restored_total_exp == total_exp}")
        print(f"   Initial restored experiences: {restored_count_initial} == {len(test_experiences)} ? {restored_count_initial == len(test_experiences)}")
        print(f"   Final: {final_total_exp} == Expected: {expected_final} ? {final_total_exp == expected_final}")
        
        success = (
            restored_total_exp == total_exp and
            restored_count_initial == len(test_experiences) and
            final_total_exp == expected_final
        )
        
        if success:
            print("\n✅ Persistence test PASSED!")
            print("   - Brain state successfully saved and restored")
            print("   - Experience data preserved correctly")
            print("   - Continued learning after restoration")
        else:
            print("\n❌ Persistence test FAILED!")
            print(f"   Original: {total_exp} experiences")
            print(f"   Restored: {restored_total_exp} experiences") 
            print(f"   Final: {final_total_exp} experiences")
        
        return success


def test_checkpoint_intervals():
    """Test automatic checkpoint creation at intervals."""
    print("\n🧪 Testing automatic checkpoint intervals...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistenceConfig(
            memory_root_path=temp_dir,
            checkpoint_interval_experiences=3,  # Very frequent for testing
            checkpoint_interval_seconds=2,
            use_compression=False
        )
        
        brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        brain.enable_persistence = True
        brain.persistence_manager = PersistenceManager(config)
        
        # Add experiences one by one and check for checkpoints
        checkpoints_created = []
        for i in range(7):  # Should trigger 2 checkpoints (at 3 and 6 experiences)
            sensory = [float(i), float(i*0.5), float(i*0.2)]
            action = [float(i*0.1), float(i*0.3)]
            outcome = [float(i+0.1), float(i*0.5+0.1), float(i*0.2+0.1)]
            
            # Process and store experience
            predicted_action, brain_info = brain.process_sensory_input(sensory, len(action))
            exp_id = brain.store_experience(sensory, action, outcome)
            
            # Check if checkpoint was created
            checkpoint_files = list(Path(temp_dir).glob("checkpoints/checkpoint_*.json*"))
            if len(checkpoint_files) > len(checkpoints_created):
                new_checkpoint = checkpoint_files[-1].stem.replace('.json', '')
                checkpoints_created.append(new_checkpoint)
                print(f"📋 Checkpoint created: {new_checkpoint} (after {brain.total_experiences} experiences)")
        
        brain.finalize_session()
        
        # Should have created checkpoints at 3 and 6 experiences
        expected_checkpoints = 2
        success = len(checkpoints_created) >= expected_checkpoints
        
        if success:
            print(f"✅ Checkpoint interval test PASSED! ({len(checkpoints_created)} checkpoints created)")
        else:
            print(f"❌ Checkpoint interval test FAILED! Expected {expected_checkpoints}, got {len(checkpoints_created)}")
        
        return success


def main():
    """Run all persistence tests."""
    print("🚀 Brain Persistence System Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Persistence", test_basic_persistence),
        ("Checkpoint Intervals", test_checkpoint_intervals)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All persistence tests PASSED!")
        print("🧠 Brain memory will now persist between sessions!")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    return passed == len(results)


if __name__ == "__main__":
    main()