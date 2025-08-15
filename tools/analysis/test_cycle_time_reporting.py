#!/usr/bin/env python3
"""
Test Cycle Time Reporting

Verifies that the hardware adaptation system correctly reports
cycle times every 60 seconds during brain operation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import time
from utils.hardware_adaptation import get_hardware_adaptation, record_brain_cycle_performance


def test_cycle_time_reporting():
    """Test cycle time reporting functionality."""
    print("⏱️ Testing Cycle Time Reporting...")
    
    # Get hardware adaptation instance
    adapter = get_hardware_adaptation()
    
    # Simulate brain cycles with varying performance
    print("🔄 Simulating brain cycles...")
    
    # Simulate good performance (under 50ms)
    for i in range(15):
        cycle_time = 25.0 + (i * 2.0)  # 25ms to 53ms
        record_brain_cycle_performance(cycle_time, 50.0)
        time.sleep(0.1)  # Small delay between cycles
    
    print(f"📊 Recorded {len(adapter.performance_history)} cycles")
    
    # Force a performance report by setting last_report to trigger it
    adapter.last_cycle_report = time.time() - 61.0  # Force report trigger
    
    # Record one more cycle to trigger the report
    record_brain_cycle_performance(30.0, 50.0)
    
    print("✅ Cycle time reporting test complete")
    
    # Test with degraded performance
    print("\n🚨 Testing with degraded performance...")
    
    # Simulate slow cycles (over 100ms)
    for i in range(10):
        cycle_time = 120.0 + (i * 5.0)  # 120ms to 165ms
        record_brain_cycle_performance(cycle_time, 75.0)
        time.sleep(0.05)
    
    # Force another report
    adapter.last_cycle_report = time.time() - 61.0
    record_brain_cycle_performance(150.0, 75.0)
    
    print("✅ Degraded performance reporting test complete")


if __name__ == "__main__":
    test_cycle_time_reporting()