#!/usr/bin/env python3
"""
Test Async Logging Integration with Brain System

Tests that the updated BrainLogger works correctly with async logging
and maintains backward compatibility.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
import tempfile
import shutil
from pathlib import Path
import json

from utils.brain_logger import BrainLogger


class MockBrain:
    """Mock brain class for testing."""
    
    def __init__(self):
        self.brain_start_time = time.time()
        self.total_cycles = 0
        self.total_experiences = 0
        self.total_predictions = 0
        self.optimal_prediction_error = 0.5
        self.recent_learning_outcomes = []
        
        # Mock vector brain with confidence system
        self.vector_brain = MockVectorBrain()
    
    def get_brain_stats(self):
        """Mock brain stats."""
        return {
            "working_memory_size": 8,
            "consolidated_memory_size": 25,
            "memory_pressure": 0.3,
            "cognitive_pressure": 0.4,
            "hardware_limits": {"cpu": 4, "memory": 1024}
        }
    
    def compute_intrinsic_reward(self):
        """Mock intrinsic reward computation."""
        return 0.75


class MockVectorBrain:
    """Mock vector brain with confidence system."""
    
    def __init__(self):
        self.emergent_confidence = MockConfidenceSystem()


class MockConfidenceSystem:
    """Mock confidence system."""
    
    def __init__(self):
        self.current_confidence = 0.65
        self.volatility_confidence = 0.7
        self.coherence_confidence = 0.6
        self.meta_confidence = 0.8
        self.total_updates = 42
    
    def _detect_current_pattern(self):
        return "learning"


def test_async_brain_logging():
    """Test async brain logging functionality."""
    print("🧠 Testing Async Brain Logging Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test async-enabled brain logger
        brain_logger = BrainLogger(
            session_name="test_async_brain",
            log_dir=temp_dir,
            enable_async=True,
            quiet_mode=True
        )
        
        # Create mock brain
        mock_brain = MockBrain()
        
        # Test brain state logging
        for i in range(5):
            brain_logger.log_brain_state(
                brain=mock_brain,
                experience_count=i * 10,
                additional_data={"test_iteration": i}
            )
            mock_brain.total_experiences += 10
            time.sleep(0.01)  # Small delay
        
        # Test emergence event logging
        brain_logger.log_emergence_event(
            event_type="test_emergence",
            description="Test emergence event",
            evidence={"score": 0.8, "duration": 5.0},
            significance=0.75
        )
        
        # Allow async processing
        time.sleep(1.0)
        
        # Check async logger stats
        if brain_logger.async_logger:
            stats = brain_logger.async_logger.get_stats()
            print(f"   📊 Async Stats:")
            print(f"      Queued: {stats['total_logs_queued']}")
            print(f"      Written: {stats['total_logs_written']}")
            print(f"      Dropped: {stats['total_logs_dropped']}")
            
            assert stats['total_logs_queued'] >= 6, f"Expected at least 6 logs, got {stats['total_logs_queued']}"
            assert stats['total_logs_dropped'] == 0, f"Expected no drops, got {stats['total_logs_dropped']}"
        
        # Close session and check files
        brain_logger.close_session()
        
        # Verify log files were created by async logger
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        print(f"   📁 Log files created: {len(log_files)}")
        
        # Check specific log categories
        brain_state_files = [f for f in log_files if "brain_state" in f.name]
        emergence_files = [f for f in log_files if "emergence" in f.name]
        
        assert len(brain_state_files) > 0, "No brain state log files found"
        assert len(emergence_files) > 0, "No emergence log files found"
        
        # Verify log content
        total_entries = 0
        for log_file in log_files:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                total_entries += len(lines)
                
                # Verify JSON format
                for line in lines:
                    data = json.loads(line.strip())
                    assert 'timestamp' in data
                    assert 'log_level' in data
                    assert 'category' in data
                    assert 'data' in data
        
        print(f"   ✅ Total log entries: {total_entries}")
        print(f"   ✅ Async brain logging successful")
        
        return True


def test_synchronous_fallback():
    """Test synchronous fallback functionality."""
    print("🔄 Testing Synchronous Fallback...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test sync-only brain logger
        brain_logger = BrainLogger(
            session_name="test_sync_brain",
            log_dir=temp_dir,
            enable_async=False,
            quiet_mode=True
        )
        
        # Create mock brain
        mock_brain = MockBrain()
        
        # Test brain state logging (sync mode)
        brain_logger.log_brain_state(
            brain=mock_brain,
            experience_count=100,
            additional_data={"sync_test": True}
        )
        
        # Test emergence event logging (sync mode)
        brain_logger.log_emergence_event(
            event_type="sync_emergence",
            description="Sync emergence test",
            evidence={"sync_mode": True},
            significance=0.6
        )
        
        # Close session
        brain_logger.close_session()
        
        # Verify traditional log files exist
        traditional_files = [
            "test_sync_brain_brain_state.jsonl",
            "test_sync_brain_emergence.jsonl"
        ]
        
        for filename in traditional_files:
            file_path = Path(temp_dir) / filename
            assert file_path.exists(), f"Traditional log file not found: {filename}"
            
            # Verify content
            with open(file_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0, f"No content in {filename}"
                
                # Verify JSON format
                for line in lines:
                    data = json.loads(line.strip())
                    assert 'timestamp' in data
        
        print(f"   ✅ Synchronous fallback working correctly")
        return True


def test_performance_comparison():
    """Compare async vs sync logging performance."""
    print("⚡ Testing Performance Comparison...")
    
    num_logs = 100
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test async performance
        start_time = time.perf_counter()
        
        brain_logger_async = BrainLogger(
            session_name="perf_test_async",
            log_dir=temp_dir,
            enable_async=True,
            quiet_mode=True
        )
        
        mock_brain = MockBrain()
        
        for i in range(num_logs):
            brain_logger_async.log_brain_state(
                brain=mock_brain,
                experience_count=i,
                additional_data={"perf_test": i}
            )
        
        async_time = time.perf_counter() - start_time
        brain_logger_async.close_session()
        
        # Test sync performance
        start_time = time.perf_counter()
        
        brain_logger_sync = BrainLogger(
            session_name="perf_test_sync",
            log_dir=temp_dir,
            enable_async=False,
            quiet_mode=True
        )
        
        for i in range(num_logs):
            brain_logger_sync.log_brain_state(
                brain=mock_brain,
                experience_count=i,
                additional_data={"perf_test": i}
            )
        
        sync_time = time.perf_counter() - start_time
        brain_logger_sync.close_session()
        
        # Compare performance
        speedup = sync_time / async_time if async_time > 0 else float('inf')
        async_ms_per_log = (async_time / num_logs) * 1000
        sync_ms_per_log = (sync_time / num_logs) * 1000
        
        print(f"   📊 Performance Results ({num_logs} logs):")
        print(f"      Async: {async_time:.4f}s ({async_ms_per_log:.3f}ms/log)")
        print(f"      Sync:  {sync_time:.4f}s ({sync_ms_per_log:.3f}ms/log)")
        print(f"      Speedup: {speedup:.2f}x")
        
        # Verify async is significantly faster
        assert async_ms_per_log < 1.0, f"Async logging should be <1ms per log, got {async_ms_per_log:.3f}ms"
        assert speedup > 2.0, f"Async should be >2x faster, got {speedup:.2f}x"
        
        print(f"   ✅ Async logging significantly faster")
        return speedup


def run_integration_tests():
    """Run complete integration test suite."""
    print("🧪 Brain Async Logging Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_async_brain_logging,
        test_synchronous_fallback,
        test_performance_comparison
    ]
    
    results = []
    
    for test in tests:
        try:
            print()
            result = test()
            results.append(("✅", test.__name__, result))
        except Exception as e:
            print(f"   ❌ {test.__name__} failed: {e}")
            results.append(("❌", test.__name__, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Integration Test Results:")
    
    passed = 0
    for status, test_name, result in results:
        print(f"   {status} {test_name}")
        if status == "✅":
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All integration tests passed! Brain async logging ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Review integration before deployment.")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)