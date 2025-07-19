#!/usr/bin/env python3
"""
Test script to validate persistence system fixes.

Tests:
1. SerializedBrainState schema compatibility with old data containing 'filepath'
2. File movement error handling
3. Backward compatibility for unknown fields
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

import tempfile
import json
import time
from persistence.brain_serializer import BrainSerializer, SerializedBrainState
from persistence.storage_backend import StorageBackend
from persistence.persistence_config import create_default_config


def test_backward_compatibility():
    """Test that old save data with unknown fields is handled properly."""
    print("🧪 Testing backward compatibility with old save data...")
    
    serializer = BrainSerializer()
    
    # Simulate old save data that contains 'filepath' and other unknown fields
    old_save_data = {
        'version': '1.0',
        'session_count': 1,
        'total_cycles': 100,
        'total_experiences': 50,
        'save_timestamp': time.time(),
        'patterns': [],
        'confidence_state': {},
        'hardware_adaptations': {},
        'cross_stream_associations': {},
        'brain_type': 'sparse_goldilocks',
        'sensory_dim': 16,
        'motor_dim': 4,
        'temporal_dim': 4,
        'learning_history': [],
        'emergence_events': [],
        # These are unknown fields that might exist in old save data
        'filepath': '/some/old/path/brain_state.json',
        'unknown_field': 'some_value',
        'legacy_data': {'old': 'structure'}
    }
    
    try:
        # This should work now with the fix
        brain_state = serializer.from_dict(old_save_data)
        
        # Verify the brain state was created successfully
        assert brain_state.version == '1.0'
        assert brain_state.session_count == 1
        assert brain_state.total_cycles == 100
        assert brain_state.total_experiences == 50
        assert brain_state.brain_type == 'sparse_goldilocks'
        
        # Verify unknown fields were filtered out
        # (they shouldn't cause an error when creating SerializedBrainState)
        print("✅ Backward compatibility test passed - old data loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def test_file_movement_error_handling():
    """Test improved error handling in file movement operations."""
    print("🧪 Testing file movement error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_default_config(temp_dir)
        storage = StorageBackend(config)
        
        # Test 1: Moving non-existent file
        print("  Testing move of non-existent file...")
        non_existent = os.path.join(temp_dir, "does_not_exist.json")
        destination = os.path.join(temp_dir, "moved.json")
        
        result = storage.move_file(non_existent, destination)
        if not result:
            print("  ✅ Correctly handled non-existent source file")
        else:
            print("  ❌ Should have failed for non-existent file")
            return False
        
        # Test 2: Moving to existing destination
        print("  Testing move with existing destination...")
        
        # Create source and destination files
        source_file = os.path.join(temp_dir, "source.json")
        dest_file = os.path.join(temp_dir, "dest.json")
        
        with open(source_file, 'w') as f:
            json.dump({"test": "data"}, f)
        
        with open(dest_file, 'w') as f:
            json.dump({"existing": "data"}, f)
        
        # Move should succeed by overwriting destination
        result = storage.move_file(source_file, dest_file)
        if result and os.path.exists(dest_file) and not os.path.exists(source_file):
            print("  ✅ Successfully handled existing destination file")
        else:
            print("  ❌ Failed to handle existing destination file")
            return False
        
        # Test 3: Moving to subdirectory that doesn't exist
        print("  Testing move to non-existent subdirectory...")
        
        source_file2 = os.path.join(temp_dir, "source2.json")
        dest_subdir = os.path.join(temp_dir, "subdir", "nested", "dest.json")
        
        with open(source_file2, 'w') as f:
            json.dump({"test": "data2"}, f)
        
        result = storage.move_file(source_file2, dest_subdir)
        if result and os.path.exists(dest_subdir):
            print("  ✅ Successfully created subdirectories and moved file")
        else:
            print("  ❌ Failed to create subdirectories or move file")
            return False
    
    print("✅ File movement error handling tests passed")
    return True


def test_serialization_roundtrip():
    """Test complete serialization/deserialization roundtrip."""
    print("🧪 Testing serialization roundtrip...")
    
    serializer = BrainSerializer()
    
    # Create test data
    test_data = {
        'version': '1.0',
        'session_count': 5,
        'total_cycles': 1000,
        'total_experiences': 500,
        'save_timestamp': time.time(),
        'patterns': [
            {
                'pattern_id': 'test_pattern_1',
                'stream_type': 'sensory',
                'pattern_data': {'test': [1, 2, 3]},
                'activation_count': 10,
                'last_accessed': time.time(),
                'success_rate': 0.8,
                'energy_level': 0.9,
                'creation_time': time.time(),
                'importance_score': 0.7
            }
        ],
        'confidence_state': {'current_confidence': 0.75},
        'hardware_adaptations': {'working_memory_limit': 671},
        'cross_stream_associations': {'stream_connections': {}},
        'brain_type': 'sparse_goldilocks',
        'sensory_dim': 16,
        'motor_dim': 4,
        'temporal_dim': 4,
        'learning_history': [{'event': 'test'}],
        'emergence_events': []
    }
    
    try:
        # Deserialize
        brain_state = serializer.from_dict(test_data)
        
        # Serialize back to dict
        serialized_dict = serializer.to_dict(brain_state)
        
        # Deserialize again
        brain_state2 = serializer.from_dict(serialized_dict)
        
        # Verify data integrity
        assert brain_state2.session_count == 5
        assert brain_state2.total_cycles == 1000
        assert brain_state2.total_experiences == 500
        assert len(brain_state2.patterns) == 1
        assert brain_state2.patterns[0].pattern_id == 'test_pattern_1'
        
        print("✅ Serialization roundtrip test passed")
        return True
        
    except Exception as e:
        print(f"❌ Serialization roundtrip test failed: {e}")
        return False


def main():
    """Run all persistence fix tests."""
    print("🚀 Running persistence system fix validation tests...")
    print()
    
    tests = [
        test_backward_compatibility,
        test_file_movement_error_handling,
        test_serialization_roundtrip
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All persistence fix tests passed!")
        return True
    else:
        print("💥 Some tests failed - fixes may need adjustment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)