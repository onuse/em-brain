#!/usr/bin/env python3
"""
Test Async Logging Framework

Validates that the async logging system provides <1ms overhead
while maintaining data integrity and reliability.
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
import threading

from utils.async_logger import AsyncLogger, LoggerConfig, LogLevel
from utils.loggable_objects import (
    LoggableVitalSigns, LoggableSystemEvent, LoggablePerformanceMetrics
)


def test_basic_async_logging():
    """Test basic async logging functionality."""
    print("🧪 Testing Basic Async Logging...")
    
    # Create temporary log directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggerConfig(
            log_directory=Path(temp_dir),
            max_queue_size=1000,
            batch_size=10,
            flush_interval_seconds=0.5
        )
        
        logger = AsyncLogger(config, quiet_mode=True)
        
        # Test logging different types
        vital_signs = LoggableVitalSigns(0.75, 15.2)
        system_event = LoggableSystemEvent("test_event", {"detail": "test_detail"})
        performance = LoggablePerformanceMetrics(12.5, {"limit": 100}, 0.3)
        
        # Log objects
        assert logger.please_log(vital_signs), "Failed to log vital signs"
        assert logger.please_log(system_event), "Failed to log system event"
        assert logger.please_log(performance), "Failed to log performance"
        
        # Wait for background processing
        time.sleep(1.0)
        
        # Check stats
        stats = logger.get_stats()
        assert stats['total_logs_queued'] == 3, f"Expected 3 queued, got {stats['total_logs_queued']}"
        
        # Shutdown and check files
        logger.shutdown()
        
        # Verify log files were created
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) >= 1, f"Expected log files, found: {log_files}"
        
        # Verify content
        total_lines = 0
        for log_file in log_files:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                total_lines += len(lines)
                
                # Verify JSON format
                for line in lines:
                    data = json.loads(line.strip())
                    assert 'timestamp' in data
                    assert 'log_level' in data
                    assert 'category' in data
                    assert 'data' in data
        
        print(f"   ✅ Logged {total_lines} entries to {len(log_files)} files")
        return True


def test_performance_overhead():
    """Test that logging overhead is <1ms per call."""
    print("🚀 Testing Performance Overhead...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggerConfig(
            log_directory=Path(temp_dir),
            max_queue_size=10000,
            batch_size=50
        )
        
        logger = AsyncLogger(config, quiet_mode=True)
        
        # Test performance with many quick logs
        num_tests = 1000
        total_time = 0
        
        for i in range(num_tests):
            start_time = time.perf_counter()
            
            # Create and log vital signs (minimal object)
            vital_signs = LoggableVitalSigns(0.5 + i * 0.0001, 10.0 + i * 0.01)
            success = logger.please_log(vital_signs)
            
            end_time = time.perf_counter()
            
            if success:
                total_time += (end_time - start_time)
            else:
                print(f"   ⚠️ Failed to log at iteration {i}")
        
        avg_time_ms = (total_time / num_tests) * 1000
        
        print(f"   📊 Performance Results:")
        print(f"      Tests: {num_tests}")
        print(f"      Average time: {avg_time_ms:.4f}ms")
        print(f"      Total time: {total_time:.4f}s")
        
        # Verify <1ms overhead
        assert avg_time_ms < 1.0, f"Logging overhead {avg_time_ms:.4f}ms exceeds 1ms target"
        
        # Wait for background processing
        time.sleep(2.0)
        
        stats = logger.get_stats()
        print(f"      Queued: {stats['total_logs_queued']}")
        print(f"      Written: {stats['total_logs_written']}")
        print(f"      Dropped: {stats['total_logs_dropped']}")
        
        logger.shutdown()
        
        print(f"   ✅ Average overhead: {avg_time_ms:.4f}ms (< 1ms target)")
        return avg_time_ms


def test_backpressure_handling():
    """Test backpressure handling when queue fills up."""
    print("🔄 Testing Backpressure Handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggerConfig(
            log_directory=Path(temp_dir),
            max_queue_size=10,  # Small queue to trigger backpressure
            batch_size=5,
            drop_policy="drop_oldest"
        )
        
        logger = AsyncLogger(config, quiet_mode=True)
        
        # Flood the queue
        success_count = 0
        for i in range(50):  # Much more than queue size
            vital_signs = LoggableVitalSigns(float(i), float(i * 2))
            if logger.please_log(vital_signs):
                success_count += 1
        
        # Wait for processing
        time.sleep(1.0)
        
        stats = logger.get_stats()
        print(f"   📊 Backpressure Results:")
        print(f"      Attempted: 50")
        print(f"      Successful: {success_count}")
        print(f"      Queued: {stats['total_logs_queued']}")
        print(f"      Dropped: {stats['total_logs_dropped']}")
        print(f"      Queue full events: {stats['queue_full_events']}")
        
        # Verify some drops occurred (expected with small queue)
        assert stats['total_logs_dropped'] > 0, "Expected some logs to be dropped"
        assert stats['queue_full_events'] > 0, "Expected queue full events"
        
        logger.shutdown()
        
        print(f"   ✅ Backpressure handled correctly")
        return True


def test_concurrent_logging():
    """Test concurrent logging from multiple threads."""
    print("🧵 Testing Concurrent Logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggerConfig(
            log_directory=Path(temp_dir),
            max_queue_size=5000,
            batch_size=20
        )
        
        logger = AsyncLogger(config, quiet_mode=True)
        
        # Thread worker function
        def worker_thread(thread_id: int, num_logs: int):
            for i in range(num_logs):
                event = LoggableSystemEvent(
                    f"thread_{thread_id}_event_{i}",
                    {"thread_id": thread_id, "event_num": i}
                )
                logger.please_log(event)
        
        # Start multiple threads
        num_threads = 5
        logs_per_thread = 100
        threads = []
        
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(thread_id, logs_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Wait for background processing
        time.sleep(2.0)
        
        stats = logger.get_stats()
        expected_logs = num_threads * logs_per_thread
        
        print(f"   📊 Concurrent Results:")
        print(f"      Threads: {num_threads}")
        print(f"      Logs per thread: {logs_per_thread}")
        print(f"      Expected total: {expected_logs}")
        print(f"      Queued: {stats['total_logs_queued']}")
        print(f"      Written: {stats['total_logs_written']}")
        print(f"      Time: {end_time - start_time:.2f}s")
        
        # Verify most logs were processed
        assert stats['total_logs_queued'] >= expected_logs * 0.9, "Lost too many logs in concurrent test"
        
        logger.shutdown()
        
        print(f"   ✅ Concurrent logging successful")
        return True


def test_data_integrity():
    """Test that serialized data maintains integrity."""
    print("🔍 Testing Data Integrity...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LoggerConfig(
            log_directory=Path(temp_dir),
            max_queue_size=100,
            batch_size=5,
            flush_interval_seconds=0.1
        )
        
        logger = AsyncLogger(config, quiet_mode=True)
        
        # Create test data with known values
        test_data = [
            LoggableVitalSigns(0.123456, 15.789),
            LoggableSystemEvent("test_integrity", {"number": 42, "text": "hello"}),
            LoggablePerformanceMetrics(25.5, {"mem": 1024, "cpu": 8}, 0.75)
        ]
        
        # Log test data
        for obj in test_data:
            logger.please_log(obj)
        
        # Wait for processing
        time.sleep(1.0)
        logger.shutdown()
        
        # Read back and verify
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        logged_data = []
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    logged_data.append(entry)
        
        print(f"   📊 Integrity Results:")
        print(f"      Original objects: {len(test_data)}")
        print(f"      Logged entries: {len(logged_data)}")
        
        # Verify vital signs data
        vital_entries = [e for e in logged_data if e['category'] == 'vital_signs']
        if vital_entries:
            vital_data = vital_entries[0]['data']
            assert vital_data['confidence'] == 0.123456, "Confidence value corrupted"
            assert vital_data['cycle_time_ms'] == 15.789, "Cycle time value corrupted"
        
        # Verify system event data
        system_entries = [e for e in logged_data if e['category'] == 'system_events']
        if system_entries:
            system_data = system_entries[0]['data']
            assert system_data['details']['number'] == 42, "System event number corrupted"
            assert system_data['details']['text'] == "hello", "System event text corrupted"
        
        assert len(logged_data) == len(test_data), "Data count mismatch"
        
        print(f"   ✅ Data integrity verified")
        return True


def run_all_tests():
    """Run complete async logging test suite."""
    print("🧪 Async Logging Framework Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_async_logging,
        test_performance_overhead, 
        test_backpressure_handling,
        test_concurrent_logging,
        test_data_integrity
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
    print("📋 Test Results Summary:")
    
    passed = 0
    for status, test_name, result in results:
        print(f"   {status} {test_name}")
        if status == "✅":
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Async logging framework ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Review implementation before deployment.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)