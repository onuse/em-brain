#!/usr/bin/env python3
"""
Test Consolidation Fix

Tests the persistence consolidation system with the new error handling
to ensure KeyError issues are resolved.
"""

import sys
import os
import time
import tempfile
import json
import gzip

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.persistence.consolidation_engine import ConsolidationEngine
from server.src.persistence.persistence_config import PersistenceConfig
from server.src.brain_factory import MinimalBrain


def create_test_delta_file(temp_dir: str, filename: str, patterns_data=None):
    """Create a test delta file with given patterns data."""
    if patterns_data is None:
        patterns_data = []
    
    delta_data = {
        "version": "1.0",
        "session_count": 1,
        "total_cycles": 10,
        "total_experiences": 10,
        "save_timestamp": time.time(),
        "patterns": patterns_data,
        "confidence_state": {
            "current_confidence": 0.7,
            "confidence_history": [],
            "total_updates": 10,
            "volatility_history": [],
            "coherence_history": []
        },
        "hardware_adaptations": {
            "working_memory_limit": 671,
            "similarity_search_limit": 16777,
            "cognitive_energy_budget": 20800,
            "cycle_time_history": [],
            "adaptation_events": []
        },
        "cross_stream_associations": {},
        "brain_type": "sparse_goldilocks",
        "sensory_dim": 16,
        "motor_dim": 4,
        "temporal_dim": 4,
        "learning_history": [],
        "emergence_events": [],
        "incremental_metadata": {
            "request_id": "test_delta",
            "save_type": "incremental",
            "cycles_at_save": 10,
            "request_timestamp": time.time(),
            "save_timestamp": time.time()
        }
    }
    
    incremental_dir = os.path.join(temp_dir, "incremental")
    os.makedirs(incremental_dir, exist_ok=True)
    
    file_path = os.path.join(incremental_dir, filename)
    
    # Write as compressed JSON
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(delta_data, f, indent=2)
    
    return file_path


def test_consolidation_with_valid_files():
    """Test consolidation with valid delta files."""
    print("🔬 Testing consolidation with valid delta files...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persistence config
        config = PersistenceConfig(
            memory_root_path=temp_dir,
            incremental_save_interval_cycles=10,
            enable_compression=True
        )
        
        # Create test delta files
        create_test_delta_file(temp_dir, "delta_test_1.json.gz", [])
        create_test_delta_file(temp_dir, "delta_test_2.json.gz", [])
        
        # Create consolidation engine
        engine = ConsolidationEngine(config)
        
        try:
            success = engine.consolidate_incremental_files()
            if success:
                print("✅ Consolidation with valid files succeeded")
                return True
            else:
                print("❌ Consolidation with valid files failed")
                return False
        except Exception as e:
            print(f"❌ Consolidation with valid files threw exception: {e}")
            return False


def test_consolidation_with_missing_patterns_key():
    """Test consolidation with delta file missing patterns key."""
    print("🔬 Testing consolidation with missing patterns key...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persistence config
        config = PersistenceConfig(
            memory_root_path=temp_dir,
            incremental_save_interval_cycles=10,
            enable_compression=True
        )
        
        # Create a delta file missing the patterns key
        incremental_dir = os.path.join(temp_dir, "incremental")
        os.makedirs(incremental_dir, exist_ok=True)
        
        invalid_data = {
            "version": "1.0",
            "session_count": 1,
            "total_cycles": 10,
            "total_experiences": 10,
            "save_timestamp": time.time(),
            # Missing 'patterns' key!
            "confidence_state": {},
            "hardware_adaptations": {},
            "cross_stream_associations": {},
            "brain_type": "sparse_goldilocks",
            "sensory_dim": 16,
            "motor_dim": 4,
            "temporal_dim": 4,
            "learning_history": [],
            "emergence_events": []
        }
        
        file_path = os.path.join(incremental_dir, "delta_invalid.json.gz")
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(invalid_data, f, indent=2)
        
        # Create consolidation engine
        engine = ConsolidationEngine(config)
        
        try:
            success = engine.consolidate_incremental_files()
            print(f"✅ Consolidation with missing patterns key handled gracefully: {success}")
            return True
        except Exception as e:
            print(f"❌ Consolidation with missing patterns key threw exception: {e}")
            return False


def test_consolidation_with_corrupted_patterns():
    """Test consolidation with corrupted patterns data."""
    print("🔬 Testing consolidation with corrupted patterns data...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create persistence config
        config = PersistenceConfig(
            memory_root_path=temp_dir,
            incremental_save_interval_cycles=10,
            enable_compression=True
        )
        
        # Create a delta file with corrupted patterns data
        incremental_dir = os.path.join(temp_dir, "incremental")
        os.makedirs(incremental_dir, exist_ok=True)
        
        corrupted_data = {
            "version": "1.0",
            "session_count": 1,
            "total_cycles": 10,
            "total_experiences": 10,
            "save_timestamp": time.time(),
            "patterns": "not_a_list",  # Should be a list
            "confidence_state": {},
            "hardware_adaptations": {},
            "cross_stream_associations": {},
            "brain_type": "sparse_goldilocks",
            "sensory_dim": 16,
            "motor_dim": 4,
            "temporal_dim": 4,
            "learning_history": [],
            "emergence_events": []
        }
        
        file_path = os.path.join(incremental_dir, "delta_corrupted.json.gz")
        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(corrupted_data, f, indent=2)
        
        # Create consolidation engine
        engine = ConsolidationEngine(config)
        
        try:
            success = engine.consolidate_incremental_files()
            print(f"✅ Consolidation with corrupted patterns handled gracefully: {success}")
            return True
        except Exception as e:
            print(f"❌ Consolidation with corrupted patterns threw exception: {e}")
            return False


def test_real_brain_consolidation():
    """Test consolidation with a real brain that generates delta files."""
    print("🔬 Testing consolidation with real brain...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration for brain with persistence
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': True,
                'save_interval_cycles': 5
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 8,
                'motor_dim': 4,
                'enable_biological_timing': False  # Disable for faster testing
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with persistence...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        # Run brain for several cycles to generate delta files
        print("🧠 Running brain cycles to generate delta files...")
        for i in range(20):
            sensory_input = [0.1 * i] * 8
            brain.process_sensory_input(sensory_input)
        
        # Force a save to create delta files
        if brain.persistence_manager and hasattr(brain.persistence_manager, 'incremental_engine'):
            brain.persistence_manager.incremental_engine.force_incremental_save(brain)
        
        brain.finalize_session()
        
        # Now test consolidation
        print("🔄 Testing consolidation of real delta files...")
        persistence_config = PersistenceConfig(
            memory_root_path=temp_dir,
            incremental_save_interval_cycles=10,
            enable_compression=True
        )
        
        engine = ConsolidationEngine(persistence_config)
        
        try:
            success = engine.consolidate_incremental_files()
            if success:
                print("✅ Real brain consolidation succeeded")
                return True
            else:
                print("❌ Real brain consolidation failed")
                return False
        except Exception as e:
            print(f"❌ Real brain consolidation threw exception: {e}")
            return False


def main():
    """Run all consolidation tests."""
    print("🚀 Consolidation Fix Test Suite")
    print("=" * 50)
    
    tests = [
        ("Valid Files", test_consolidation_with_valid_files),
        ("Missing Patterns Key", test_consolidation_with_missing_patterns_key),
        ("Corrupted Patterns", test_consolidation_with_corrupted_patterns),
        ("Real Brain", test_real_brain_consolidation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All consolidation fix tests passed!")
        return True
    else:
        print("⚠️ Some consolidation fix tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)