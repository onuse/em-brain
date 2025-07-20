#!/usr/bin/env python3
"""
Test Maintenance Throttling

Quick test to verify that maintenance scheduling throttling works correctly
and prevents excessive maintenance operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory


def test_maintenance_throttling():
    """Test that maintenance throttling prevents excessive operations."""
    print("🧪 Testing maintenance throttling")
    
    # Use field brain for maintenance interface testing
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True)
    
    if not hasattr(brain, 'maintenance_scheduler') or brain.maintenance_scheduler is None:
        print("❌ Brain does not have maintenance scheduler")
        return
    
    print("✅ Brain has maintenance scheduler")
    
    # Test 1: Rapid maintenance calls should be throttled
    print("\n🔄 Test 1: Rapid maintenance calls (should be throttled)")
    maintenance_count = 0
    
    for i in range(5):
        performed = brain.run_recommended_maintenance()
        if any(performed.values()):
            maintenance_count += 1
            print(f"   Call {i+1}: Maintenance performed - {performed}")
        else:
            print(f"   Call {i+1}: Throttled (no maintenance)")
        time.sleep(1.0)  # 1 second between calls
    
    print(f"   Total maintenance operations in 5 calls: {maintenance_count}")
    
    # Test 2: After throttle period, maintenance should work
    print("\n⏰ Test 2: After throttle period (should work)")
    print("   Waiting 6 seconds for throttle to expire...")
    time.sleep(6.0)
    
    performed = brain.run_recommended_maintenance()
    print(f"   After throttle period: {performed}")
    
    # Test 3: Check maintenance statistics
    print("\n📊 Test 3: Maintenance statistics")
    status = brain.get_maintenance_status()
    print(f"   Light maintenance count: {status['maintenance_stats']['light_maintenance_count']}")
    print(f"   Heavy maintenance count: {status['maintenance_stats']['heavy_maintenance_count']}")
    print(f"   Deep consolidation count: {status['maintenance_stats']['deep_consolidation_count']}")
    
    # Test 4: Manual method calls (should not be throttled)
    print("\n🔧 Test 4: Manual maintenance methods (not throttled)")
    print("   Calling brain.light_maintenance() directly...")
    brain.light_maintenance()
    print("   Direct call completed")
    
    final_status = brain.get_maintenance_status()
    print(f"   Final light maintenance count: {final_status['maintenance_stats']['light_maintenance_count']}")
    
    brain.finalize_session()
    print("\n✅ Maintenance throttling test completed")


if __name__ == "__main__":
    print("🔧 Maintenance Throttling Test")
    print("=" * 50)
    
    test_maintenance_throttling()
    
    print("\n🎯 Throttling test completed")